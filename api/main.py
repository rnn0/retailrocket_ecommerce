"""
API REST do motor de recomendação.
Arquitetura em camadas: rota (main.py) -> service (services/recommendation_service.py)
-> modelo (src/models/*).
Rodar com: uvicorn api.main:app --reload --port 8000
A vitrine de demonstração (aplicação de e-commerce) fica em http://localhost:8000/
"""
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from api.dependencies import get_artifacts
from api.schemas import DemoUsersResponse, HealthResponse, RecommendationResponse
from api.services.recommendation_service import get_recommendations_for_user

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="E-commerce Recommendation API",
    description="Motor de recomendação personalizado (Filtragem Colaborativa ALS + K-Means) sobre o Retailrocket.",
    version="1.0.0",
)

MODELS_NOT_READY = "Modelos ainda não treinados. Rode `python -m src.pipeline` antes de subir a API."


def _artifacts_or_503():
    try:
        return get_artifacts()
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail=MODELS_NOT_READY) from None


@app.get("/", include_in_schema=False)
def storefront():
    """Aplicação de e-commerce de demonstração, que consome /recommendations/{user_id}."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", response_model=HealthResponse)
def health():
    artifacts = _artifacts_or_503()
    return HealthResponse(
        status="ok",
        model_version=artifacts.version,
        n_users=len(artifacts.interaction_matrix.user_to_idx),
        n_items=len(artifacts.interaction_matrix.item_to_idx),
    )


@app.get("/model", summary="Metadados do modelo em produção (versão, parâmetros, métricas)")
def model_info():
    return _artifacts_or_503().metadata


@app.get("/demo/users", response_model=DemoUsersResponse, summary="Usuários de exemplo para a demonstração")
def demo_users():
    return DemoUsersResponse(**_artifacts_or_503().metadata["demo_users"])


@app.get("/recommendations/{user_id}", response_model=RecommendationResponse)
def get_recommendations(
    user_id: str,
    k: int = Query(default=5, ge=1, le=50, description="Número de produtos a recomendar"),
):
    artifacts = _artifacts_or_503()
    result = get_recommendations_for_user(user_id, artifacts, k=k)

    if not result.item_ids:
        raise HTTPException(status_code=404, detail=f"Nenhuma recomendação disponível para user_id={user_id}")

    return RecommendationResponse(
        user_id=result.user_id,
        strategy=result.strategy,
        segment=result.segment,
        recommendations=result.item_ids,
        products=result.products,
    )
