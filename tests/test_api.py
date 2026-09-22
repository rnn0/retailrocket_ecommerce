"""
Testes de integração da API. Pré-requisito: `python -m src.pipeline` já ter rodado
ao menos uma vez, para existirem artefatos em models/ (o pipeline usa dados sintéticos
automaticamente se data/raw estiver vazio).
"""
import pytest
from fastapi.testclient import TestClient

from src.config import MODELS_DIR

pytestmark = pytest.mark.skipif(
    not (MODELS_DIR / "als_model.pkl").exists(),
    reason="Artefatos do modelo não encontrados. Rode `python -m src.pipeline` antes dos testes.",
)


@pytest.fixture(scope="module")
def client():
    from api.main import app

    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["n_users"] > 0
    assert body["n_items"] > 0


def test_recommendations_known_user_returns_k_items(client):
    response = client.get("/recommendations/1?k=5")
    assert response.status_code == 200
    body = response.json()
    assert body["user_id"] == "1"
    assert len(body["recommendations"]) <= 5
    assert body["strategy"] in ("collaborative_filtering_als", "cluster_popularity_fallback")


def test_recommendations_unknown_user_falls_back_to_cluster(client):
    response = client.get("/recommendations/999999999?k=5")
    assert response.status_code == 200
    body = response.json()
    assert body["strategy"] == "cluster_popularity_fallback"
    assert len(body["recommendations"]) > 0


def test_recommendations_respects_k_param(client):
    response = client.get("/recommendations/1?k=3")
    assert response.status_code == 200
    assert len(response.json()["recommendations"]) <= 3


def test_recommendations_invalid_k_is_rejected(client):
    response = client.get("/recommendations/1?k=0")
    assert response.status_code == 422
