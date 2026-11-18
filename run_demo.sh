#!/usr/bin/env bash
# Roda a API + vitrine (Linux/macOS/Git Bash). Requer Python 3.12 e a pasta models/ treinada.
set -e
cd "$(dirname "$0")"
PY=$(command -v python3.12 || command -v python)
[ -f models/metadata.json ] || { echo "models/ vazia: use o pacote com modelos ou rode python -m src.pipeline"; exit 1; }
if [ ! -d .venv ]; then "$PY" -m venv .venv; . .venv/bin/activate 2>/dev/null || . .venv/Scripts/activate; pip install -r requirements.txt; else . .venv/bin/activate 2>/dev/null || . .venv/Scripts/activate; fi
echo "Abra http://localhost:8000/"
uvicorn api.main:app --port 8000
