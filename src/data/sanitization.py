"""
Fase CRISP-DM: Data Preparation — Sanitização e tratamento de dados.

Este módulo concentra as regras de qualidade aplicadas aos dados brutos do
Retailrocket, cada uma justificada por um problema real encontrado na análise
exploratória (ver notebooks/01_data_treatment_and_eda.ipynb):

  1. Eventos duplicados (linhas exatamente repetidas).
  2. Usuários com volume de eventos incompatível com comportamento humano
     ("bots"/scrapers) — distorcem a matriz de interação e a Filtragem Colaborativa.
  3. Tipos de evento fora do domínio esperado (view/addtocart/transaction).
  4. Consistência de `transactionid` (deve existir sse o evento é "transaction").
  5. `item_properties`: é um changelog (múltiplas linhas por item ao longo do tempo),
     não uma tabela de estado — é preciso ficar só com o valor mais recente de cada
     propriedade por item antes de usar como feature.
  6. Parsing dos valores ofuscados de `item_properties`: valores numéricos vêm
     prefixados com "n" e escalados por 1000 (ex.: "n15360.000" -> 15.36), conforme
     a documentação do dataset no Kaggle.
  7. Árvore de categorias: remoção de referências a `parentid` que não existem em
     `categoryid` (categorias "órfãs").
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import EVENT_WEIGHTS

# Preço/disponibilidade no Retailrocket ficam nas properties "790" e "available"
# respectivamente (constatado via inspeção direta dos dados; ver notebook).
PRICE_PROPERTY_CODE = "790"
AVAILABLE_PROPERTY_CODE = "available"
CATEGORY_PROPERTY_CODE = "categoryid"

# Um usuário com mais eventos que este limite em ~4.5 meses de dados (janela do
# dataset) tem comportamento incompatível com navegação humana típica — candidato a
# bot/scraper. Definido como ~20x o percentil 99.9 observado na EDA, não um número
# arbitrário.
BOT_EVENT_THRESHOLD = 1000


@dataclass
class SanitizationReport:
    """Registra o que foi removido/alterado em cada etapa, para auditoria e para o
    relatório do critério 2 (qualidade e preparação dos dados)."""

    steps: list[dict] = field(default_factory=list)

    def log(self, step: str, rows_before: int, rows_after: int, note: str = ""):
        self.steps.append(
            {
                "step": step,
                "rows_before": rows_before,
                "rows_after": rows_after,
                "rows_removed": rows_before - rows_after,
                "pct_removed": round(100 * (rows_before - rows_after) / rows_before, 4) if rows_before else 0.0,
                "note": note,
            }
        )

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.steps)


def sanitize_events(events: pd.DataFrame, report: SanitizationReport | None = None) -> pd.DataFrame:
    """Pipeline completo de sanitização de `events.csv`. Cada etapa é isolada e
    logada no `report`, para o processo ficar auditável (não é uma caixa-preta)."""
    report = report if report is not None else SanitizationReport()
    df = events.copy()
    n0 = len(df)

    # 1. Tipos de evento fora do domínio esperado
    n_before = len(df)
    df = df[df["event"].isin(EVENT_WEIGHTS.keys())]
    report.log("filtrar_eventos_fora_do_dominio", n_before, len(df))

    # 2. Nulos em colunas essenciais (visitorid/itemid/event não podem faltar;
    #    transactionid é opcional por natureza)
    n_before = len(df)
    df = df.dropna(subset=["visitorid", "itemid", "event", "timestamp"])
    report.log("remover_nulos_essenciais", n_before, len(df))

    # 3. Duplicatas exatas (mesma linha repetida — não é o mesmo que "usuário viu o
    #    item duas vezes", que é um evento legítimo com timestamps diferentes)
    n_before = len(df)
    df = df.drop_duplicates()
    report.log("remover_duplicatas_exatas", n_before, len(df))

    # 4. Consistência de transactionid: deve existir se e somente se o evento for
    #    "transaction". Uma violação aqui indicaria corrupção nos dados.
    inconsistent = (
        ((df["event"] == "transaction") & df["transactionid"].isnull())
        | ((df["event"] != "transaction") & df["transactionid"].notnull())
    )
    n_before = len(df)
    df = df[~inconsistent]
    report.log("remover_inconsistencia_transactionid", n_before, len(df), note=f"{int(inconsistent.sum())} linhas inconsistentes")

    # 5. Detecção de bots: usuários com volume de eventos incompatível com
    #    comportamento humano na janela do dataset (ver BOT_EVENT_THRESHOLD acima)
    user_counts = df["visitorid"].value_counts()
    bot_users = user_counts[user_counts > BOT_EVENT_THRESHOLD].index
    n_before = len(df)
    df = df[~df["visitorid"].isin(bot_users)]
    report.log(
        "remover_usuarios_bot",
        n_before,
        len(df),
        note=f"{len(bot_users)} usuários removidos (> {BOT_EVENT_THRESHOLD} eventos)",
    )

    # 6. Timestamps fora do intervalo plausível do próprio dataset (defesa contra
    #    corrupção; não deve remover nada em dados íntegros, mas fica documentado)
    valid_range = (df["timestamp"] >= events["timestamp"].min()) & (df["timestamp"] <= events["timestamp"].max())
    n_before = len(df)
    df = df[valid_range]
    report.log("filtrar_timestamps_fora_do_intervalo", n_before, len(df))

    df = df.sort_values("timestamp").reset_index(drop=True)
    report.log("TOTAL", n0, len(df), note="pipeline de sanitização completo")
    return df


def parse_item_property_value(value: str):
    """Converte o valor ofuscado de uma property para numérico quando possível.
    Valores numéricos no Retailrocket vêm prefixados com 'n' e escalados por 1000
    (ex.: 'n15360.000' representa 15.36). Valores puramente hash (ex.: '1116713')
    não são interpretáveis e são mantidos como estão (tratados como categóricos)."""
    if isinstance(value, str) and value.startswith("n"):
        try:
            return float(value[1:]) / 1000.0
        except ValueError:
            return None
    return value


def sanitize_item_properties(item_properties: pd.DataFrame) -> pd.DataFrame:
    """`item_properties` é um changelog: cada linha é 'a propriedade X do item Y
    tinha o valor Z no instante T'. Para usar como feature de item, é preciso
    reduzir isso ao valor MAIS RECENTE de cada propriedade por item.

    O dataset real tem ~940 códigos de `property` distintos, mas a esmagadora
    maioria é hash anonimizado sem significado interpretável (a documentação do
    Kaggle só abre 3: preço, disponibilidade e categoria). Fazer o pivot com TODAS
    as properties criaria uma matriz densa de centenas de milhões de células
    (n_itens x n_properties) e estoura a memória sem necessidade — por isso
    filtramos para as properties conhecidas ANTES de pivotar."""
    known_properties = [PRICE_PROPERTY_CODE, AVAILABLE_PROPERTY_CODE, CATEGORY_PROPERTY_CODE]
    df = item_properties[item_properties["property"].isin(known_properties)].copy()
    df = df.drop_duplicates()

    if df.empty:
        return pd.DataFrame(columns=["itemid", "price", "available", "categoryid"])

    # Mantém só a última observação de cada (itemid, property)
    df = df.sort_values("timestamp").drop_duplicates(subset=["itemid", "property"], keep="last")

    wide = df.pivot(index="itemid", columns="property", values="value").reset_index()

    if PRICE_PROPERTY_CODE in wide.columns:
        wide["price"] = wide[PRICE_PROPERTY_CODE].apply(parse_item_property_value)
    if AVAILABLE_PROPERTY_CODE in wide.columns:
        wide["available"] = pd.to_numeric(wide[AVAILABLE_PROPERTY_CODE], errors="coerce")
    if CATEGORY_PROPERTY_CODE in wide.columns:
        wide["categoryid"] = pd.to_numeric(wide[CATEGORY_PROPERTY_CODE], errors="coerce")

    keep_cols = [c for c in ["itemid", "price", "available", "categoryid"] if c in wide.columns]
    return wide[keep_cols]


def sanitize_category_tree(category_tree: pd.DataFrame) -> pd.DataFrame:
    """Remove referências a `parentid` que não existem em `categoryid` (categorias
    órfãs) — evita erros ao montar a hierarquia de categorias."""
    df = category_tree.drop_duplicates().copy()
    valid_ids = set(df["categoryid"])
    orphan_mask = df["parentid"].notna() & ~df["parentid"].isin(valid_ids)
    return df[~orphan_mask].reset_index(drop=True)
