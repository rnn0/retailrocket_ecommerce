"""Configurações centrais do projeto.

Os diretórios podem ser sobrescritos por variáveis de ambiente (usado pelos testes
e pelo Docker para isolar dados/artefatos sem mexer no código):
  RECSYS_DATA_DIR    -> pasta que contém raw/ e processed/   (padrão: ./data)
  RECSYS_MODELS_DIR  -> pasta de artefatos de modelo          (padrão: ./models)
  RECSYS_REPORTS_DIR -> pasta de relatórios e figuras         (padrão: ./reports)
"""
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("RECSYS_DATA_DIR", ROOT_DIR / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = Path(os.environ.get("RECSYS_MODELS_DIR", ROOT_DIR / "models"))
REPORTS_DIR = Path(os.environ.get("RECSYS_REPORTS_DIR", ROOT_DIR / "reports"))
FIGURES_DIR = REPORTS_DIR / "figures"

for d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Pesos de cada tipo de evento na matriz usuário-item (dados implícitos).
# O Retailrocket não tem avaliações explícitas (ratings): o peso de cada evento
# funciona como uma "avaliação implícita" (view < addtocart < transaction).
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

# Quantidade de usuários amostrados por segmento na comparação de estratégias.
EVAL_USERS_DEFAULT = 1000

# Histórico mínimo (nº de itens distintos) para o híbrido usar Filtragem Colaborativa em vez do
# fallback por cluster. Valor definido a partir da comparação em reports/strategy_comparison.csv:
# com limiar=3 o fallback tinha nDCG@5 menor que o ALS para usuários de pouco histórico.
HYBRID_MIN_HISTORY = 1
