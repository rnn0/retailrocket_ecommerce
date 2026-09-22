"""
Fase CRISP-DM: Evaluation.
Métricas de ranking (para os recomendadores) e de clustering (para o K-Means).
"""
import numpy as np


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


def evaluate_recommender(recommend_fn, test_events, k: int = 5) -> dict:
    """
    recommend_fn: callable(user_id) -> list[item_id]
    test_events: DataFrame com colunas visitorid, itemid (eventos "held-out" do split temporal)
    """
    relevant_by_user = test_events.groupby("visitorid")["itemid"].apply(set).to_dict()

    precisions, recalls, ndcgs = [], [], []
    for user_id, relevant_items in relevant_by_user.items():
        recommended = recommend_fn(user_id)
        if recommended is None:
            continue
        precisions.append(precision_at_k(recommended, relevant_items, k))
        recalls.append(recall_at_k(recommended, relevant_items, k))
        ndcgs.append(ndcg_at_k(recommended, relevant_items, k))

    return {
        f"precision@{k}": round(float(np.mean(precisions)), 4) if precisions else 0.0,
        f"recall@{k}": round(float(np.mean(recalls)), 4) if recalls else 0.0,
        f"ndcg@{k}": round(float(np.mean(ndcgs)), 4) if ndcgs else 0.0,
        "n_users_evaluated": len(precisions),
    }
