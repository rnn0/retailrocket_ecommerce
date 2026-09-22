# Sistema de Recomendação para E-commerce (Retailrocket)

Motor de recomendação personalizado com Filtragem Colaborativa (memory-based + ALS) e
segmentação de usuários via K-Means, exposto como API REST (FastAPI).

## 1. Como obter os dados reais

O dataset **Retailrocket** está no Kaggle:
https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset

1. Baixe `events.csv`, `item_properties_part1.csv`, `item_properties_part2.csv` e `category_tree.csv`.
2. Coloque todos em `data/raw/`.
3. Rode o pipeline (abaixo). Ele detecta os arquivos reais automaticamente.

## 2. Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Ou via Docker:
```bash
docker compose up --build
```

## 3. Rodar o pipeline (gera dados de exemplo se não achar os reais em data/raw)

```bash
python -m src.pipeline
```

Isso executa, em ordem (etapas do CRISP-DM: Data Prep -> Modeling -> Evaluation):
1. Ingestão/geração dos dados
2. Limpeza e construção da matriz usuário-item
3. Feature engineering (RFM) para clustering
4. Treino: CF memory-based, CF via ALS (implicit), K-Means
5. Avaliação: Precision@K, Recall@K, NDCG@K, Silhouette Score
6. Serialização dos artefatos em `models/`

## 4. Subir a API

```bash
uvicorn api.main:app --reload --port 8000
```

Endpoint principal:
```
GET /recommendations/{user_id}?k=5
```

Resposta:
```json
{
  "user_id": "123",
  "strategy": "hybrid",
  "recommendations": ["item_45", "item_12", "item_88", "item_3", "item_71"]
}
```
## 5. Testes

```bash
pytest -v
```

## 6. Estrutura do projeto

```
src/data/            -> ingestão, geração de dados sintéticos, pré-processamento
src/features/        -> feature engineering (RFM, matriz esparsa)
src/models/          -> CF memory-based, CF via ALS, K-Means, estratégia híbrida
src/evaluation/       -> métricas de ranking e de clustering
src/pipeline.py       -> orquestrador ponta a ponta
api/                  -> API FastAPI (camadas: rotas -> service -> modelo)
tests/                -> testes automatizados (pytest)
docs/CRISP-DM.md      -> documentação do processo seguindo o CRISP-DM
```

## 7. Metodologia (CRISP-DM)

Ver `docs/CRISP-DM.md` para o detalhamento completo de cada fase (entendimento do
negócio, dos dados, preparação, modelagem, avaliação e deployment).
