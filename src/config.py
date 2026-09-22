"""Configurações centrais do projeto."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"

for d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Pesos de cada tipo de evento na matriz usuário-item (dados implícitos)
EVENT_WEIGHTS = {
    "view": 1.0,
    "addtocart": 3.0,
    "transaction": 5.0,
}

# Mínimo de interações para um usuário/item entrar no treino (reduz sparsity/cold-start extremo)
MIN_USER_INTERACTIONS = 3
MIN_ITEM_INTERACTIONS = 3

RANDOM_STATE = 42
TOP_K_DEFAULT = 5
N_CLUSTERS_DEFAULT = 5
