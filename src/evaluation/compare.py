"""
Fase CRISP-DM: Evaluation.
Compara as estratégias de recomendação sob o MESMO protocolo:

  * Split temporal: treino = 80% mais antigo dos eventos, teste = 20% mais recente.
  * Item relevante = item que o usuário interagiu no período de teste, que existe no
    vocabulário de treino e que ele AINDA NÃO tinha visto no treino (descoberta).
    É a tarefa real de um recomendador: sugerir algo novo, não repetir o que já viu.
  * Todas as estratégias recebem os mesmos usuários (amostra com semente fixa) e nenhuma
    recomenda itens que o usuário já viu no treino.
  * Os usuários são divididos em dois segmentos (com/sem histórico), pois é nessa
    divisão que a estratégia híbrida deveria se justificar.
"""
import logging
import time

import numpy as np
import pandas as pd

from src.config import EVENT_WEIGHTS, HYBRID_MIN_HISTORY, RANDOM_STATE, TOP_K_DEFAULT
from src.evaluation.metrics import evaluate_recommender
from src.models.hybrid import hybrid_recommend

logger = logging.getLogger("recsys.compare")

POPULARITY_POOL = 200
HISTORY_SEGMENT_THRESHOLD = 3  # >= 3 itens distintos no treino = "com histórico"
LEGACY_HYBRID_THRESHOLD = 3    # regra original do híbrido, mantida na comparação como evidência


def build_relevant_items(train_events: pd.DataFrame, test_events: pd.DataFrame, interaction_matrix) -> dict:
    """{user_id: set(itens novos que o usuário consumiu no período de teste)}."""
    vocab = set(interaction_matrix.item_to_idx)
    known_users = set(interaction_matrix.user_to_idx)
    seen = train_events.groupby("visitorid")["itemid"].apply(set).to_dict()
    test_items = test_events[test_events["visitorid"].isin(known_users)].groupby("visitorid")["itemid"].apply(set)
    relevant = {}
    for user_id, items in test_items.items():
        new_items = (items & vocab) - seen.get(user_id, set())
        if new_items:
            relevant[user_id] = new_items
    return relevant


def split_users_by_history(relevant: dict, interaction_matrix) -> dict:
    """Divide os usuários avaliáveis em 'com_historico' (>= 3 itens distintos no treino) e 'pouco_historico'."""
    n_items_per_user = np.diff(interaction_matrix.matrix.indptr)
    user_to_idx = interaction_matrix.user_to_idx
    return {
        "com_historico": [u for u in relevant if n_items_per_user[user_to_idx[u]] >= HISTORY_SEGMENT_THRESHOLD],
        "pouco_historico": [u for u in relevant if n_items_per_user[user_to_idx[u]] < HISTORY_SEGMENT_THRESHOLD],
    }


def popularity_ranking(train_events: pd.DataFrame, top_n: int = POPULARITY_POOL) -> list:
    """Itens mais populares (soma dos pesos de evento) -> baseline que todo recomendador precisa superar."""
    weights = train_events["event"].map(EVENT_WEIGHTS)
    return weights.groupby(train_events["itemid"]).sum().sort_values(ascending=False).head(top_n).index.tolist()


def _timed(strategy_fn):
    """Envolve a função de recomendação para medir a latência média por usuário (ms)."""
    stats = {"total": 0.0, "calls": 0}

    def wrapped(user_id):
        start = time.perf_counter()
        result = strategy_fn(user_id)
        stats["total"] += time.perf_counter() - start
        stats["calls"] += 1
        return result

    wrapped.stats = stats
    return wrapped


def compare_strategies(
    train_events,
    test_events,
    interaction_matrix,
    als_model,
    memory_model,
    clustered_users,
    cluster_top_items,
    k: int = TOP_K_DEFAULT,
    n_users_per_segment: int = 1000,
    seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    idx_to_item = interaction_matrix.idx_to_item
    user_to_idx = interaction_matrix.user_to_idx
    n_catalog = len(interaction_matrix.item_to_idx)

    relevant_all = build_relevant_items(train_events, test_events, interaction_matrix)
    seen_by_user = train_events.groupby("visitorid")["itemid"].apply(set).to_dict()
    popular = popularity_ranking(train_events)
    cluster_of = dict(zip(clustered_users["visitorid"], clustered_users["cluster"]))
    fallback_cluster = int(clustered_users["cluster"].value_counts().idxmax())

    def unseen(user_id, items):
        seen = seen_by_user.get(user_id, set())
        return [i for i in items if i not in seen][:k]

    def popularity_fn(user_id):
        return unseen(user_id, popular)

    def cluster_fn(user_id):
        return unseen(user_id, cluster_top_items.get(int(cluster_of.get(user_id, fallback_cluster)), []))

    def memory_fn(user_id):
        recs = memory_model.recommend(user_to_idx[user_id], n=k)
        return [idx_to_item[i] for i, _ in recs]

    def als_fn(user_id):
        recs = als_model.recommend(user_to_idx[user_id], n=k)
        return [idx_to_item[i] for i, _ in recs]

    def make_hybrid(min_history):
        def hybrid_fn(user_id):
            # O híbrido de produção não filtra itens vistos no fallback; aqui filtramos para
            # que todas as estratégias sejam comparadas sob as mesmas regras.
            items, strategy = hybrid_recommend(
                user_id, interaction_matrix, als_model, clustered_users, cluster_top_items, idx_to_item,
                n=k, min_history=min_history,
            )
            if strategy == "cluster_popularity_fallback":
                return unseen(user_id, cluster_top_items.get(int(cluster_of.get(user_id, fallback_cluster)), []))
            return items

        return hybrid_fn

    strategies = {
        "popularidade_global": popularity_fn,
        "popularidade_por_cluster": cluster_fn,
        "cf_memory_based": memory_fn,
        "cf_als": als_fn,
        f"hibrido_limiar_{LEGACY_HYBRID_THRESHOLD}": make_hybrid(LEGACY_HYBRID_THRESHOLD),
        f"hibrido_limiar_{HYBRID_MIN_HISTORY}": make_hybrid(HYBRID_MIN_HISTORY),
    }

    segments = split_users_by_history(relevant_all, interaction_matrix)

    rng = np.random.default_rng(seed)
    rows = []
    for segment_name, users in segments.items():
        users = sorted(users)
        if not users:
            continue
        sample = rng.choice(users, size=min(n_users_per_segment, len(users)), replace=False)
        relevant = {int(u): relevant_all[int(u)] for u in sample}
        for strategy_name, fn in strategies.items():
            timed = _timed(fn)
            metrics = evaluate_recommender(timed, relevant, k=k, n_catalog_items=n_catalog)
            metrics["latency_ms"] = round(1000 * timed.stats["total"] / max(timed.stats["calls"], 1), 2)
            rows.append({"segmento": segment_name, "estrategia": strategy_name, **metrics})
            logger.info("%-16s %-26s %s", segment_name, strategy_name, metrics)
    return pd.DataFrame(rows)
