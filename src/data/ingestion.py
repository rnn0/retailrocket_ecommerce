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


def load_item_properties() -> pd.DataFrame:
    files = glob.glob(str(RAW_DIR / "item_properties_part*.csv"))
    if not files:
        raise FileNotFoundError(
            "Nenhum item_properties_part*.csv encontrado em data/raw. "
            "Rode load_events() primeiro ou baixe os dados reais do Kaggle."
        )
    dfs = [pd.read_csv(f) for f in files]
    return pd.concat(dfs, ignore_index=True)


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
