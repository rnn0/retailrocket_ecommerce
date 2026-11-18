"""
Carrega os artefatos do modelo (treinados por src/pipeline.py) uma única vez,
na subida da API (evita recarregar a cada request).
"""
import json
import math
from functools import lru_cache

import joblib

from src.config import MODELS_DIR


def _clean(value):
    """NaN do pandas -> None (JSON não aceita NaN)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return value


class ModelArtifacts:
    def __init__(self):
        self.interaction_matrix = joblib.load(MODELS_DIR / "interaction_matrix.pkl")
        self.als_model = joblib.load(MODELS_DIR / "als_model.pkl")
        self.clustered_users = joblib.load(MODELS_DIR / "clustered_users.pkl")
        self.cluster_top_items = joblib.load(MODELS_DIR / "cluster_top_items.pkl")
        self.metadata = json.loads((MODELS_DIR / "metadata.json").read_text())

        catalog_path = MODELS_DIR / "item_catalog.pkl"
        catalog = joblib.load(catalog_path) if catalog_path.exists() else None
        self.catalog = {} if catalog is None or catalog.empty else catalog.set_index("itemid").to_dict("index")
        self.cluster_of = dict(zip(self.clustered_users["visitorid"], self.clustered_users["cluster"], strict=False))
        self.cluster_labels = {int(k): v for k, v in self.metadata["clustering"]["cluster_labels"].items()}

    @property
    def version(self) -> str:
        return self.metadata["version"]

    def is_available(self, item_id) -> bool:
        """Itens sem informação de disponibilidade no catálogo são tratados como disponíveis."""
        available = _clean(self.catalog.get(item_id, {}).get("available"))
        return True if available is None else bool(available)

    def describe_item(self, item_id, rank: int) -> dict:
        row = self.catalog.get(item_id, {})
        category = _clean(row.get("categoryid"))
        parent = _clean(row.get("parent_categoryid"))
        available = _clean(row.get("available"))
        price = _clean(row.get("price"))
        return {
            "rank": rank,
            "item_id": str(item_id),
            "name": f"Produto {item_id}",
            "category_id": int(category) if category is not None else None,
            "parent_category_id": int(parent) if parent is not None else None,
            "price": round(float(price), 2) if price is not None else None,
            "available": bool(available) if available is not None else None,
        }


@lru_cache(maxsize=1)
def get_artifacts() -> ModelArtifacts:
    return ModelArtifacts()
