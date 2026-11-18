"""
Camada de serviço: contém a lógica de negócio da recomendação,
desacoplada do framework web (facilita testes e reuso).
"""
from dataclasses import dataclass

from api.dependencies import ModelArtifacts
from src.config import TOP_K_DEFAULT
from src.models.hybrid import hybrid_recommend

# Busca mais candidatos do que o pedido para poder descartar produtos fora de estoque
# (regra de negócio da API; a avaliação offline em src/evaluation não aplica esse filtro).
OVERFETCH_FACTOR = 4


@dataclass
class RecommendationResult:
    user_id: str
    strategy: str
    item_ids: list[str]
    products: list[dict]
    segment: dict | None


def parse_user_id(user_id_raw: str):
    """IDs do Retailrocket são inteiros; aceitamos string para não travar em outros formatos."""
    try:
        return int(user_id_raw)
    except ValueError:
        return user_id_raw


def prefer_available(candidates: list, artifacts: ModelArtifacts, k: int) -> list:
    """Mantém a ordem de relevância, dá preferência a produtos disponíveis e só completa com
    indisponíveis se não houver k candidatos disponíveis (para sempre respeitar o contrato de k itens)."""
    in_stock = [i for i in candidates if artifacts.is_available(i)]
    out_of_stock = [i for i in candidates if not artifacts.is_available(i)]
    return (in_stock + out_of_stock)[:k]


def get_recommendations_for_user(user_id_raw: str, artifacts: ModelArtifacts, k: int = TOP_K_DEFAULT) -> RecommendationResult:
    user_id = parse_user_id(user_id_raw)
    candidates, strategy = hybrid_recommend(
        user_id=user_id,
        interaction_matrix=artifacts.interaction_matrix,
        als_model=artifacts.als_model,
        clustered_users=artifacts.clustered_users,
        cluster_top_items=artifacts.cluster_top_items,
        idx_to_item=artifacts.interaction_matrix.idx_to_item,
        n=k * OVERFETCH_FACTOR,
    )
    item_ids = prefer_available(candidates, artifacts, k)
    cluster_id = artifacts.cluster_of.get(user_id)
    segment = None
    if cluster_id is not None:
        segment = {"cluster_id": int(cluster_id), "label": artifacts.cluster_labels.get(int(cluster_id), "")}

    return RecommendationResult(
        user_id=user_id_raw,
        strategy=strategy,
        item_ids=[str(i) for i in item_ids],
        products=[artifacts.describe_item(i, rank) for rank, i in enumerate(item_ids, start=1)],
        segment=segment,
    )
