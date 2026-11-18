"""Smoke test: o pipeline inteiro roda (dados sintéticos) e gera todos os artefatos esperados."""
import json


def test_pipeline_generates_artifacts(trained_env):
    models = trained_env["base"] / "models"
    reports = trained_env["base"] / "reports"
    for name in ["als_model.pkl", "interaction_matrix.pkl", "clustered_users.pkl", "cluster_top_items.pkl", "metadata.json"]:
        assert (models / name).exists(), name
    assert (models / "registry.jsonl").exists()
    assert (reports / "figures").is_dir()


def test_metadata_has_versioning_and_metrics(trained_env):
    meta = json.loads((trained_env["base"] / "models" / "metadata.json").read_text())
    for key in ["version", "params", "clustering", "evaluation", "demo_users"]:
        assert key in meta
    assert meta["demo_users"]["com_historico"]
