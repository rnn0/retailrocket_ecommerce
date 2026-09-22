"""
Gera um dataset sintético no MESMO SCHEMA do Retailrocket
(events.csv: timestamp, visitorid, event, itemid, transactionid),
para que o pipeline inteiro possa ser validado antes de plugar os dados reais.

Schema real do Retailrocket (Kaggle):
  events.csv -> timestamp, visitorid, event, itemid, transactionid
  event in {"view", "addtocart", "transaction"}
"""
import numpy as np
import pandas as pd
from src.config import RAW_DIR, RANDOM_STATE


def generate_sample_events(
    n_users: int = 500,
    n_items: int = 300,
    n_events: int = 15_000,
    seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Popularidade enviesada (Zipf) -> mais realista que uniforme
    item_popularity = rng.zipf(a=1.6, size=n_items).astype(float)
    item_probs = item_popularity / item_popularity.sum()

    visitor_ids = rng.integers(1, n_users + 1, size=n_events)
    item_ids = rng.choice(np.arange(1, n_items + 1), size=n_events, p=item_probs)

    # view (70%) -> addtocart (20%) -> transaction (10%), funil realista
    event_types = rng.choice(
        ["view", "addtocart", "transaction"], size=n_events, p=[0.70, 0.20, 0.10]
    )

    base_ts = 1_600_000_000_000  # ms epoch, arbitrário
    timestamps = base_ts + rng.integers(0, 60 * 24 * 60 * 60 * 1000, size=n_events)
    timestamps.sort()

    transaction_id = np.where(
        event_types == "transaction",
        rng.integers(1, n_events, size=n_events),
        np.nan,
    )

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "visitorid": visitor_ids,
            "event": event_types,
            "itemid": item_ids,
            "transactionid": transaction_id,
        }
    )
    return df.sort_values("timestamp").reset_index(drop=True)


def generate_sample_item_properties(n_items: int = 300, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    categories = rng.integers(1, 25, size=n_items)
    prices = rng.integers(500, 500000, size=n_items)  # em "centavos", como no Retailrocket
    rows = []
    for item_id, cat, price in zip(range(1, n_items + 1), categories, prices):
        rows.append({"itemid": item_id, "property": "categoryid", "value": str(cat), "timestamp": 1_600_000_000_000})
        rows.append({"itemid": item_id, "property": "price", "value": str(price), "timestamp": 1_600_000_000_000})
    return pd.DataFrame(rows)


def main():
    events = generate_sample_events()
    props = generate_sample_item_properties()
    events.to_csv(RAW_DIR / "events.csv", index=False)
    props.to_csv(RAW_DIR / "item_properties_part1.csv", index=False)
    print(f"[make_sample_dataset] events.csv: {len(events)} linhas -> {RAW_DIR / 'events.csv'}")
    print(f"[make_sample_dataset] item_properties_part1.csv: {len(props)} linhas")


if __name__ == "__main__":
    main()
