"""
Estratégia híbrida: combina CF (ALS) com fallback por cluster.
Resolve cold-start parcial: usuário sem histórico suficiente recebe os itens
mais populares do cluster mais próximo; usuário com histórico usa CF.
"""
from src.config import HYBRID_MIN_HISTORY


def hybrid_recommend(
    user_id,
    interaction_matrix,
    als_model,
    clustered_users,
    cluster_top_items: dict,
    idx_to_item: dict,
    n: int = 5,
    min_history: int = HYBRID_MIN_HISTORY,
) -> tuple[list, str]:
    """Retorna (lista_de_item_ids, estratégia_usada).

    min_history: nº mínimo de itens distintos no histórico para usar CF (ver config.HYBRID_MIN_HISTORY)."""
    user_to_idx = interaction_matrix.user_to_idx

    if user_id in user_to_idx:
        user_idx = user_to_idx[user_id]
        n_interactions = interaction_matrix.matrix[user_idx].nnz
        if n_interactions >= min_history:
            recs = als_model.recommend(user_idx, n=n)
            item_ids = [idx_to_item[i] for i, _ in recs]
            return item_ids, "collaborative_filtering_als"

    # Cold-start: usuário novo ou com poucas interações -> fallback por cluster
    row = clustered_users[clustered_users["visitorid"] == user_id]
    if not row.empty:
        cluster_id = int(row.iloc[0]["cluster"])
    else:
        # Usuário totalmente novo, sem nenhuma interação: cluster mais "genérico" (maior)
        cluster_id = clustered_users["cluster"].value_counts().idxmax()

    item_ids = cluster_top_items.get(cluster_id, [])[:n]
    return item_ids, "cluster_popularity_fallback"
