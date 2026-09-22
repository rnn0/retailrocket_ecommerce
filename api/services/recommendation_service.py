"""
Camada de serviço: contém a lógica de negócio da recomendação,
desacoplada do framework web (facilita testes e reuso).
"""
from api.dependencies import ModelArtifacts
from src.models.hybrid import hybrid_recommend
from src.config import TOP_K_DEFAULT


def get_recommendations_for_user(user_id_raw: str, artifacts: ModelArtifacts, k: int = TOP_K_DEFAULT):
    # IDs do Retailrocket são inteiros; a API aceita string para não travar em outros formatos
    try:
        user_id = int(user_id_raw)
    except ValueError:
        user_id = user_id_raw

    item_ids, strategy = hybrid_recommend(
        user_id=user_id,
        interaction_matrix=artifacts.interaction_matrix,
        als_model=artifacts.als_model,
        clustered_users=artifacts.clustered_users,
        cluster_top_items=artifacts.cluster_top_items,
        idx_to_item=artifacts.interaction_matrix.idx_to_item,
        n=k,
    )
    return [str(i) for i in item_ids], strategy
