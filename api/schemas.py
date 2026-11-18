from pydantic import BaseModel, ConfigDict, Field


class Product(BaseModel):
    rank: int = Field(..., description="Posição na lista (1 = mais relevante)")
    item_id: str
    name: str = Field(..., description='Nome de exibição, ex.: "Produto 461686"')
    category_id: int | None = None
    parent_category_id: int | None = None
    price: float | None = Field(None, description="Preço (já convertido da escala ofuscada do dataset)")
    available: bool | None = None


class Segment(BaseModel):
    cluster_id: int
    label: str


class RecommendationResponse(BaseModel):
    user_id: str = Field(..., description="ID do usuário/visitante")
    strategy: str = Field(..., description="collaborative_filtering_als | cluster_popularity_fallback")
    segment: Segment | None = Field(None, description="Segmento K-Means do usuário (se ele estiver na base de treino)")
    recommendations: list[str] = Field(..., description="IDs dos produtos recomendados, em ordem de relevância")
    products: list[Product] = Field(..., description="Os mesmos produtos, com detalhes de catálogo")


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_version: str
    n_users: int
    n_items: int


class DemoUsersResponse(BaseModel):
    com_historico: list[int]
    pouco_historico: list[int]
