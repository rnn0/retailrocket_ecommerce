"""
Carrega os artefatos do modelo (treinados por src/pipeline.py) uma única vez,
na subida da API (evita recarregar a cada request).
"""
import joblib
from functools import lru_cache

from src.config import MODELS_DIR


class ModelArtifacts:
    def __init__(self):
        self.interaction_matrix = joblib.load(MODELS_DIR / "interaction_matrix.pkl")
        self.als_model = joblib.load(MODELS_DIR / "als_model.pkl")
        self.clustered_users = joblib.load(MODELS_DIR / "clustered_users.pkl")
        self.cluster_top_items = joblib.load(MODELS_DIR / "cluster_top_items.pkl")


@lru_cache(maxsize=1)
def get_artifacts() -> ModelArtifacts:
    return ModelArtifacts()
