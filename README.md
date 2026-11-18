# Sistema de Recomendação para E-commerce (Retailrocket)

Projeto da disciplina **Projeto de Mineração de Dados** (item 6 — *Sistema de Recomendação para E-commerce*).
Motor de recomendação personalizado: **Filtragem Colaborativa** (memory-based + ALS) +
**K-Means** (segmentação de consumidores), exposto como **API REST** (`GET /recommendations/{user_id}`)
e integrado a uma **vitrine de e-commerce** de demonstração (`/`).

Autor: Renan Henrique Pootz da Rocha

## Resultados (dados reais: 2,76 M eventos, 164 mil usuários, 91 mil produtos)

| Estratégia (nDCG@5, split temporal, itens novos) | Com histórico | Pouco histórico |
|---|---|---|
| Popularidade global (baseline) | 0,0029 | 0,0044 |
| Popularidade por cluster K-Means | 0,0029 | 0,0032 |
| CF memory-based (cosseno) | 0,0141 | 0,0211 |
| **CF ALS (BM25, 128 fatores)** | **0,0182** | 0,0195 |
| Híbrido (ALS + fallback por cluster, limiar 1) | **0,0182** | 0,0195 |

- ALS ≈ **6×** a popularidade para usuários com histórico; ~4 ms por recomendação.
- K-Means: **K=2** por silhouette (0,433) — "Alto engajamento" (34,5 mil usuários) e "Baixo engajamento" (129 mil).
- Duas correções guiadas pela avaliação: limiar do híbrido 3→1 (nDCG de 0,0195 caindo p/ 0,0032 com limiar 3) e `log1p` no K-Means (evita cluster de outliers).
- Os valores são baixos em termos absolutos por natureza do problema (catálogo de 91 mil itens, feedback implícito, avaliação com itens **novos** para o usuário), por isso a comparação relevante é contra os baselines.

Tabelas e gráficos completos: `reports/` e `reports/figures/`.

## Como rodar

### 1. Dados
Baixe do [Kaggle (Retailrocket)](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset) `events.csv`,
`item_properties_part1.csv`, `item_properties_part2.csv` e `category_tree.csv` e coloque em `data/raw/`.
Sem os arquivos, o pipeline gera um dataset sintético no mesmo schema (usado nos testes).

### 2. Ambiente
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: use Git Bash
pip install -r requirements-dev.txt                    # Python 3.12
```

### 3. Pipeline (≈2 min nos dados reais)
```bash
python -m src.pipeline                # ingestão → sanitização → features → modelos → avaliação → artefatos
python -m src.experiments.tune_als    # (opcional) grade de hiperparâmetros do ALS → reports/als_tuning.csv
```

### 4. API + vitrine
```bash
uvicorn api.main:app --port 8000
# vitrine:  http://localhost:8000/      Swagger: http://localhost:8000/docs
curl http://localhost:8000/recommendations/616529?k=5
```

### 5. Testes e lint
```bash
make test      # pytest (46 testes, inclui pipeline ponta a ponta em pastas temporárias)
make lint      # ruff
```

### 6. Docker
```bash
docker compose up --build   # treina na primeira subida (data/raw montado) e serve a API
```
> A imagem Docker não pôde ser construída no ambiente de desenvolvimento deste trabalho (sem Docker disponível);
> o build é verificado pelo job `docker` do GitHub Actions.

## Contrato da API

`GET /recommendations/{user_id}?k=5` →
```json
{
  "user_id": "616529",
  "strategy": "collaborative_filtering_als",
  "segment": {"cluster_id": 1, "label": "Alto engajamento"},
  "recommendations": ["429907", "396491", "347576", "67045", "133398"],
  "products": [{"rank": 1, "item_id": "429907", "name": "Produto 429907", "category_id": 1, "price": 12.3, "available": true}]
}
```
Outros endpoints: `/health`, `/model` (versão, parâmetros, métricas), `/demo/users`, `/` (vitrine).

## Critérios de avaliação → onde está

| # | Critério | Evidência no repositório |
|---|---|---|
| 1 | Problema e CRISP-DM | `docs/CRISP-DM.md` |
| 2 | Aquisição, qualidade e preparação | `src/data/` (ingestão, sanitização), `notebooks/Projeto_Recomendacao_Ecommerce_Completo.ipynb`, `reports/sanitization_report.csv` |
| 3 | Mineração e modelagem | `src/models/`, `src/experiments/tune_als.py`, `notebooks/Projeto_Recomendacao_Ecommerce_Completo.ipynb` |
| 4 | Avaliação e interpretação | `src/evaluation/`, `reports/strategy_comparison.csv`, `reports/figures/` |
| 5 | Engenharia do pipeline | `src/pipeline.py` (etapas, logs, CLI, versionamento em `models/metadata.json` e `registry.jsonl`) |
| 6 | Arquitetura e software | `api/` (rotas → serviço → modelo), `docs/ARCHITECTURE.md` |
| 7 | Reprodutibilidade, testes, DevOps | `requirements*.txt` (versões fixas), `tests/`, `Dockerfile`, `docker-compose.yml`, `Makefile`, `.github/workflows/ci.yml` |
| 8 | Apresentação e demo | `docs/apresentacao.pptx`, `docs/DEMO.md`, vitrine em `/` |

## Limitações
- Sem ratings explícitos: feedback implícito com pesos (view=1, addtocart=3, transaction=5).
- Avaliação apenas offline; não há teste A/B.
- K-Means resultou em apenas 2 segmentos (melhor silhouette); K maiores têm silhouette menor.
- Usuário completamente novo recebe os itens mais populares do maior segmento.
- Nomes de produto são "Produto {id}" (o dataset é anonimizado).
