"""Testes de integração da API sobre artefatos treinados em diretório temporário (ver conftest.py)."""
import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(trained_env, monkeypatch_module):
    import api.dependencies as deps

    monkeypatch_module.setattr(deps, "MODELS_DIR", trained_env["base"] / "models")
    deps.get_artifacts.cache_clear()
    import api.main as main

    importlib.reload(main)
    yield TestClient(main.app)
    deps.get_artifacts.cache_clear()


@pytest.fixture(scope="module")
def monkeypatch_module():
    with pytest.MonkeyPatch.context() as mp:
        yield mp


@pytest.fixture(scope="module")
def demo(client):
    return client.get("/demo/users").json()


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["n_users"] > 0 and body["n_items"] > 0
    assert body["model_version"]


def test_model_metadata(client):
    assert "params" in client.get("/model").json()


def test_known_user_gets_k_unique_products_with_details(client, demo):
    uid = demo["com_historico"][0]
    body = client.get(f"/recommendations/{uid}?k=5").json()
    assert len(body["recommendations"]) == 5
    assert len(set(body["recommendations"])) == 5
    assert body["strategy"] == "collaborative_filtering_als"
    assert body["segment"]["label"]
    assert body["products"][0]["name"].startswith("Produto ")
    assert [p["item_id"] for p in body["products"]] == body["recommendations"]


def test_unknown_user_gets_fallback(client):
    body = client.get("/recommendations/999999999").json()
    assert body["strategy"] == "cluster_popularity_fallback"
    assert body["segment"] is None
    assert len(body["recommendations"]) == 5


def test_non_numeric_user_does_not_crash(client):
    assert client.get("/recommendations/abc").status_code == 200


@pytest.mark.parametrize("k", [0, 51, -1])
def test_k_validation(client, k):
    assert client.get(f"/recommendations/1?k={k}").status_code == 422


def test_storefront_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Recomendado para você" in r.text
