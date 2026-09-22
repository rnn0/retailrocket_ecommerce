FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Treina o pipeline (usa dados sintéticos automaticamente se data/raw estiver vazio)
# e sobe a API. Em produção, prefira rodar o pipeline uma vez e persistir /app/models
# como volume, em vez de retreinar a cada build.
RUN python -m src.pipeline

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
