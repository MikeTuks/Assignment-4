"""Assignment 2 data checks (notebook cells 1-15); prints the facts quoted in the report."""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
SEED = 0
NOMINAL = ["proto", "service", "state"]
CLASSES = (pd.read_csv(HERE / "attack_category_map.csv")
             .sort_values("Mapping")["Attack Category Name"].tolist())


def load():
    return pd.read_csv(HERE / "networkTraffic.csv", na_values=["?"])


def numeric(df):
    return df.drop(columns=["id", "attack_cat"] + NOMINAL)


def correlated_pairs(num, thresh=0.95):
    corr = num.corr().abs()
    pairs = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool)).stack()
    return pairs[pairs > thresh].sort_values(ascending=False)


def drop_corr(pairs):
    """Second feature of each correlated pair."""
    return sorted({b for _, b in pairs.index})


def main():
    df = load()
    print("shape", df.shape)

    miss = df.isna().mean().mul(100).round(2)
    print("\nmissing %\n", miss[miss > 0].to_string())

    counts = df["attack_cat"].value_counts().sort_index()
    print("\nimbalance ratio", round(counts.max() / counts.min(), 1))
    print(pd.DataFrame({"class": CLASSES, "n": counts.values,
                        "pct": (counts / len(df) * 100).round(3).values}).to_string())

    print("\nlevels\n", df[NOMINAL + ["id"]].nunique().to_string())

    num = numeric(df)
    print("\nwidest ranges\n", num.agg(["min", "max", "std"]).T
          .sort_values("max", ascending=False).head(10).to_string())

    pairs = correlated_pairs(num)
    print("\ncorrelated pairs |r| > 0.95\n", pairs.to_string())
    print("\ndropped for k-NN:", drop_corr(pairs))

    out = (num - num.mean()).abs() > 3 * num.std()
    print("\n% beyond 3 sd\n", out.mean().mul(100).round(2)
          .sort_values(ascending=False).head(10).to_string())
    print(f"instances with any feature >3 sd: {out.any(axis=1).mean() * 100:.2f}%")
    print(f"sload range: {num['sload'].max():.3e}")

    # report Table IV: service absent by class, and which protocols carry a service
    by_class = (df.assign(c=df["attack_cat"].map(dict(enumerate(CLASSES))))
                  .groupby("c", observed=True)["service"]
                  .apply(lambda s: s.isna().mean() * 100)
                  .sort_values().round(1))
    print("\nservice absent %\n", by_class.to_string())
    has_svc = df.groupby("proto", observed=True)["service"].apply(lambda s: s.notna().any())
    print(f"service present: {df['service'].notna().mean() * 100:.2f}%")
    print(f"protocols with a service: {list(has_svc[has_svc].index)}")
    print(f"protocols with none:      {(~has_svc).sum()} of {len(has_svc)}")

    # one-hot min_frequency: levels kept and resulting dimensions per threshold
    rows = []
    for t in (0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
        d = {"thresh %": t}
        for col in NOMINAL:
            vc = df[col].value_counts(dropna=False)
            kept = (vc / len(df) * 100 >= t).sum()
            d[col] = kept + (1 if kept < len(vc) else 0)   # + infrequent column
        d["total"] = sum(d[c] for c in NOMINAL)
        rows.append(d)
    print("\none-hot dimensions\n", pd.DataFrame(rows).set_index("thresh %").to_string())


if __name__ == "__main__":
    main()
