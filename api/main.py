"""
API REST do motor de recomendação.
Arquitetura em camadas: rota (main.py) -> service (services/recommendation_service.py)
-> modelo (src/models/*).
Rodar com: uvicorn api.main:app --reload --port 8000
"""
from fastapi import FastAPI, HTTPException, Query

from api.dependencies import get_artifacts
from api.schemas import RecommendationResponse, HealthResponse
from api.services.recommendation_service import get_recommendations_for_user

app = FastAPI(
    title="E-commerce Recommendation API",
    description="Motor de recomendação personalizado (CF colaborativa + K-Means).",
    version="1.0.0",
)


@app.get("/health", response_model=HealthResponse)
def health():
    artifacts = get_artifacts()
    return HealthResponse(
        status="ok",
        n_users=len(artifacts.interaction_matrix.user_to_idx),
        n_items=len(artifacts.interaction_matrix.item_to_idx),
    )


@app.get("/recommendations/{user_id}", response_model=RecommendationResponse)
def get_recommendations(
    user_id: str,
    k: int = Query(default=5, ge=1, le=50, description="Número de produtos a recomendar"),
):
    try:
        artifacts = get_artifacts()
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Modelos ainda não treinados. Rode `python -m src.pipeline` antes de subir a API.",
        )

    recommendations, strategy = get_recommendations_for_user(user_id, artifacts, k=k)

    if not recommendations:
        raise HTTPException(status_code=404, detail=f"Nenhuma recomendação disponível para user_id={user_id}")

    return RecommendationResponse(user_id=user_id, strategy=strategy, recommendations=recommendations)
