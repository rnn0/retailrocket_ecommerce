"""
Fase CRISP-DM: Evaluation.
Métricas de ranking (para os recomendadores) e de clustering (para o K-Means).
"""
import numpy as np
import pandas as pd


def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    if k == 0:
        return 0.0
    top_k = recommended[:k]
    hits = len(set(top_k) & relevant)
    return hits / k


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    top_k = recommended[:k]
    hits = len(set(top_k) & relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    top_k = recommended[:k]
    dcg = sum(1.0 / np.log2(i + 2) for i, item in enumerate(top_k) if item in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def average_precision_at_k(recommended: list, relevant: set, k: int) -> float:
    """AP@K: média da precisão nas posições em que houve acerto (premia acertar cedo).
    O MAP é a média do AP@K sobre os usuários."""
    if not relevant:
        return 0.0
    hits, score = 0, 0.0
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            hits += 1
            score += hits / (i + 1)
    return score / min(len(relevant), k)


def evaluate_recommender(recommend_fn, relevant, k: int = 5, n_catalog_items: int | None = None) -> dict:
    """
    recommend_fn: callable(user_id) -> list[item_id] (ou None se não houver recomendação)
    relevant: dict {user_id: set(item_id)} OU DataFrame com colunas visitorid/itemid
              (itens "held-out" do split temporal)
    n_catalog_items: se informado, calcula a cobertura de catálogo (fração dos itens
              que apareceram em alguma recomendação) -> mede diversidade.
    """
    if isinstance(relevant, pd.DataFrame):
        relevant = relevant.groupby("visitorid")["itemid"].apply(set).to_dict()

    precisions, recalls, ndcgs, aps = [], [], [], []
    recommended_items = set()
    for user_id, relevant_items in relevant.items():
        recommended = recommend_fn(user_id)
        if recommended is None:
            continue
        recommended_items.update(recommended[:k])
        precisions.append(precision_at_k(recommended, relevant_items, k))
        recalls.append(recall_at_k(recommended, relevant_items, k))
        ndcgs.append(ndcg_at_k(recommended, relevant_items, k))
        aps.append(average_precision_at_k(recommended, relevant_items, k))

    def _mean(values):
        return round(float(np.mean(values)), 4) if values else 0.0

    result = {
        f"precision@{k}": _mean(precisions),
        f"recall@{k}": _mean(recalls),
        f"ndcg@{k}": _mean(ndcgs),
        f"map@{k}": _mean(aps),
        "n_users_evaluated": len(precisions),
    }
    if n_catalog_items:
        result["catalog_coverage"] = round(len(recommended_items) / n_catalog_items, 4)
    return result
