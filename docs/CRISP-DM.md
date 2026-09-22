# Metodologia CRISP-DM — Sistema de Recomendação para E-commerce

## 1. Business Understanding (Entendimento do Negócio)

**Problema de negócio**: um marketplace com milhares de produtos precisa determinar,
para cada usuário, quais produtos têm maior probabilidade de gerar engajamento
(clique/visualização) e conversão (compra), aumentando ticket médio e retenção.

**Objetivo de mineração de dados**: construir um sistema que, a partir do histórico de
interações (cliques, visualizações, compras, avaliações), gere uma lista ranqueada de
produtos por usuário, resolvendo dois cenários:
- **Usuário com histórico**: recomendação personalizada via Filtragem Colaborativa.
- **Usuário novo/cold-start**: recomendação por segmento (cluster) via K-Means, usando
  os produtos mais populares do grupo comportamental mais próximo.

**Critério de sucesso**: precisão e recall das recomendações no top-K (avaliados contra
um conjunto de teste temporal), e uma API funcional que devolve as recomendações em
tempo real.

## 2. Data Understanding (Entendimento dos Dados)

**Fonte**: dataset Retailrocket (Kaggle) — `events.csv`, `item_properties_part1/2.csv`,
`category_tree.csv`. Schema de `events.csv`: `timestamp, visitorid, event, itemid,
transactionid`, com `event ∈ {view, addtocart, transaction}`.

**Análise exploratória realizada** (`src/data/ingestion.py::data_quality_report`):
- Volume de linhas, usuários e itens únicos.
- Valores nulos por coluna (esperado: `transactionid` nulo fora de eventos de compra).
- Duplicatas.
- Distribuição dos tipos de evento (funil view → addtocart → transaction).
- Esparsidade da matriz usuário-item (tipicamente >95% em datasets de e-commerce reais).

**Limitação identificada**: o Retailrocket não possui avaliações explícitas (ratings).
Por isso, o projeto trata os dados como **feedback implícito**, atribuindo pesos por
tipo de evento (ver `EVENT_WEIGHTS` em `src/config.py`), abordagem padrão na literatura
de recomendação para dados de clique/compra.

## 3. Data Preparation (Preparação dos Dados)

Implementado em `src/data/preprocessing.py` e `src/features/build_features.py`:

1. **Limpeza**: remoção de nulos essenciais e duplicatas; filtragem de usuários/itens
   com menos de `MIN_USER_INTERACTIONS`/`MIN_ITEM_INTERACTIONS` interações, para reduzir
   o cold-start extremo que degrada a Filtragem Colaborativa.
2. **Split temporal treino/teste**: eventos mais antigos viram treino, os mais recentes
   viram teste — evita vazamento de informação futura, mais realista que split aleatório
   para sistemas de recomendação.
3. **Matriz usuário-item esparsa**: agregação dos pesos de evento por par
   `(usuário, item)`, representada como `scipy.sparse.csr_matrix` (necessário dado o
   nível de esparsidade dos dados).
4. **Features RFM para clustering**: por usuário, calcula-se recência (dias desde a
   última interação), frequência, número de itens únicos vistos, score de engajamento
   (soma ponderada de eventos) e taxa de conversão (compras/visualizações).

## 4. Modeling (Modelagem)

Implementado em `src/models/`:

- **Filtragem Colaborativa memory-based** (`collaborative_filtering.py::MemoryBasedCF`):
  similaridade de cosseno usuário-usuário sobre a matriz esparsa; recomenda itens mais
  consumidos pelos vizinhos mais próximos.
- **Filtragem Colaborativa model-based** (`ALSCollaborativeFiltering`): Matrix
  Factorization via ALS (biblioteca `implicit`), adequada para dados implícitos e mais
  escalável que a abordagem memory-based.
- **K-Means** (`clustering.py`): segmentação de usuários a partir das features RFM,
  com padronização (StandardScaler) e escolha do K via Silhouette Score
  (`choose_k_by_silhouette`).
- **Estratégia híbrida** (`hybrid.py`): usuários com histórico suficiente recebem
  recomendações do modelo ALS; usuários novos ou com poucas interações recebem os itens
  mais populares do cluster mais próximo (fallback de cold-start).

## 5. Evaluation (Avaliação)

Implementado em `src/evaluation/metrics.py`:

- **Métricas de ranking**: Precision@K, Recall@K e NDCG@K, calculadas comparando as
  recomendações geradas a partir do treino contra as interações reais do conjunto de
  teste (split temporal).
- **Métricas de clustering**: Silhouette Score, usado tanto para escolher o K quanto
  para validar a qualidade final da segmentação.
- **Interpretação dos clusters**: perfil médio de cada cluster (`cluster_profile`)
  permite nomear segmentos de negócio, por exemplo: cluster de alta frequência e alta
  conversão (usuários engajados) vs. cluster de alta recência e baixa conversão
  (usuários em risco de churn).

## 6. Deployment (Implantação)

- Artefatos (matriz de interação, modelo ALS, modelo K-Means, scaler, tabela de
  clusters, itens populares por cluster) são serializados com `joblib` em `models/`.
- API REST em FastAPI (`api/`) expõe `GET /recommendations/{user_id}`, carregando os
  artefatos uma única vez na subida (`api/dependencies.py`).
- Empacotamento via Docker (`Dockerfile`, `docker-compose.yml`) para reprodutibilidade
  em qualquer ambiente.
- Pipeline de CI (`.github/workflows/ci.yml`) treina o modelo e roda a suíte de testes
  a cada push, garantindo que mudanças no código não quebrem o sistema.
