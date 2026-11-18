"""
Fase CRISP-DM: Data Understanding.
Carrega events.csv e item_properties (reais, se existirem em data/raw; senão gera sintéticos).
"""
import glob

import pandas as pd

from src.config import RAW_DIR
from src.data.make_sample_dataset import main as generate_sample_data


def load_events() -> pd.DataFrame:
    path = RAW_DIR / "events.csv"
    if not path.exists():
        print("[ingestion] events.csv não encontrado em data/raw -> gerando dataset sintético de exemplo.")
        generate_sample_data()
    df = pd.read_csv(path)
    df["event"] = df["event"].str.lower()
    return df


# Properties interpretáveis do Retailrocket: preço ("790"), disponibilidade e categoria.
# As outras ~940 são hashes anônimos (ver src/data/sanitization.py).
KNOWN_PROPERTIES = {"790", "available", "categoryid"}


def load_item_properties(chunksize: int = 2_000_000) -> pd.DataFrame:
    """Lê os item_properties em blocos e já descarta as properties sem significado.

    Os dois arquivos somam ~900 MB (~20M de linhas): ler tudo de uma vez custa
    gigabytes de RAM sem necessidade, já que só 3 properties são interpretáveis."""
    files = sorted(glob.glob(str(RAW_DIR / "item_properties_part*.csv")))
    if not files:
        raise FileNotFoundError(
            "Nenhum item_properties_part*.csv encontrado em data/raw. "
            "Rode load_events() primeiro ou baixe os dados reais do Kaggle."
        )
    parts = []
    for f in files:
        for chunk in pd.read_csv(f, chunksize=chunksize, dtype={"property": str, "value": str}):
            parts.append(chunk[chunk["property"].isin(KNOWN_PROPERTIES)])
    return pd.concat(parts, ignore_index=True)


def load_category_tree() -> pd.DataFrame:
    path = RAW_DIR / "category_tree.csv"
    if not path.exists():
        return pd.DataFrame(columns=["categoryid", "parentid"])
    return pd.read_csv(path)


def data_quality_report(events: pd.DataFrame) -> dict:
    """Relatório simples de qualidade dos dados (Fase Data Understanding)."""
    report = {
        "n_rows": len(events),
        "n_unique_users": events["visitorid"].nunique(),
        "n_unique_items": events["itemid"].nunique(),
        "n_nulls_per_col": events.isnull().sum().to_dict(),
        "n_duplicates": int(events.duplicated().sum()),
        "event_distribution": events["event"].value_counts().to_dict(),
        "sparsity_pct": round(
            100
            * (
                1
                - len(events.drop_duplicates(["visitorid", "itemid"]))
                / (events["visitorid"].nunique() * events["itemid"].nunique())
            ),
            4,
        ),
    }
    return report


if __name__ == "__main__":
    events = load_events()
    import json

    print(json.dumps(data_quality_report(events), indent=2, ensure_ascii=False))
