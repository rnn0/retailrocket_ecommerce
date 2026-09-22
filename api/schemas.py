from pydantic import BaseModel, Field


class RecommendationResponse(BaseModel):
    user_id: str = Field(..., description="ID do usuário/visitante")
    strategy: str = Field(..., description="Estratégia usada: collaborative_filtering_als | cluster_popularity_fallback")
    recommendations: list[str] = Field(..., description="Lista de IDs de produtos recomendados, em ordem de relevância")


class HealthResponse(BaseModel):
    status: str
    n_users: int
    n_items: int
