# Roteiro de demonstração (5 minutos)

| Tempo | O que mostrar | Critério |
|---|---|---|
| 0:00–0:45 | Problema, objetivo, fases CRISP-DM (slides 2) | 1 |
| 0:45–1:45 | Dados reais, problemas de qualidade e sanitização (slides 3–4; `reports/sanitization_report.csv`) | 2 |
| 1:45–3:00 | Modelagem: tuning do ALS, K-Means e o aprendizado do silhouette (slides 5–6) | 3 |
| 3:00–3:45 | Avaliação vs. baselines e as 2 correções (slide 7) | 4 |
| 3:45–4:30 | Vitrine em `http://localhost:8000/`: usuário 616529 (ALS, "Alto engajamento"), 982059 (pouco histórico) e 999999999 (usuário novo → fallback) | 6 |
| 4:30–5:00 | Pipeline em 2 min, 46 testes, CI, versionamento (slides 8–10) | 5, 7, 8 |

Antes: `python -m src.pipeline && uvicorn api.main:app --port 8000`.
