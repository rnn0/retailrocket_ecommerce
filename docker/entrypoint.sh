#!/bin/sh
set -e
# Treina o modelo na primeira subida (precisa de data/raw/*.csv montado em /app/data/raw).
if [ ! -f models/metadata.json ]; then
  echo "[entrypoint] Modelos não encontrados: executando o pipeline de treino..."
  python -m src.pipeline
fi
exec uvicorn api.main:app --host 0.0.0.0 --port 8000
