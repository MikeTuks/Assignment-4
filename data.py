"""Data, folds and scores shared by every script."""

import sys
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent / "assignment2_old_code"))
from a2_data_checks import (CLASSES, NOMINAL, correlated_pairs, drop_corr,  # noqa: E402
                            load, numeric)

K = len(CLASSES)
N_OUTER, N_INNER = 5, 3
METRICS = ["macro_f1", "balanced_accuracy", "precision_macro", "recall_macro", "accuracy"]


def ranks(X):
    """Rank of each value among its column's distinct values (the tree sorts on these)."""
    R = np.column_stack([np.unique(c, return_inverse=True)[1] for c in X.T]).astype(np.int32)
    assert R.max() < 2**18, "tree.py's radix sort handles ranks below 2**18"
    return R


@lru_cache(maxsize=1)
def dataset():
    """X holds all 42 features (nominal ones as integer codes, used as labels only).
    The k-NN arrays drop the correlated features (|r| > 0.95)."""
    df = load()
    feats = df.drop(columns=["id", "attack_cat"])
    for c in NOMINAL:
        # A missing service becomes its own code.
        feats[c] = pd.factorize(feats[c], use_na_sentinel=False)[0]
    is_nom = feats.columns.isin(NOMINAL)
    n_levels = np.where(is_nom, feats.max().to_numpy() + 1, 0).astype(np.int64)
    dropped = drop_corr(correlated_pairs(numeric(df)))
    knn_num = [c for c in feats.columns if c not in NOMINAL and c not in dropped]
    X = feats.to_numpy(np.float64)
    return SimpleNamespace(
        X=X, R=ranks(X), y=df["attack_cat"].to_numpy(np.int64),
        is_nom=is_nom, n_levels=n_levels,
        knn_num=feats[knn_num].to_numpy(np.float64),
        knn_nom=feats[NOMINAL].to_numpy(np.int64),
        knn_levels=n_levels[is_nom])


def outer_folds(y, seed):
    """Seed 0 gives the assignment 2 folds."""
    skf = StratifiedKFold(N_OUTER, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y))


def inner_folds(y, train, seed):
    """N_INNER folds inside an outer training set, as row indices of the full data."""
    skf = StratifiedKFold(N_INNER, shuffle=True, random_state=seed)
    return [(train[a], train[b]) for a, b in skf.split(np.zeros(len(train)), y[train])]


def class_weights(y, mode):
    """'balanced' gives n / (K * n_c); anything else gives ones."""
    if mode != "balanced":
        return np.ones(K)
    n_c = np.bincount(y, minlength=K)
    return np.where(n_c > 0, len(y) / (K * np.maximum(n_c, 1)), 0.0)


def confusion(y, pred):
    return np.bincount(y * K + pred, minlength=K * K).reshape(K, K)


def per_class(cm):
    """Precision, recall and F1 per class; 0 where undefined."""
    tp = np.diag(cm).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.nan_to_num(tp / cm.sum(0))
        r = np.nan_to_num(tp / cm.sum(1))
        f = np.nan_to_num(2 * p * r / (p + r))
    return p, r, f


def scores(y, pred):
    cm = confusion(y, pred)
    p, r, f = per_class(cm)
    return {"macro_f1": f.mean(), "balanced_accuracy": r[cm.sum(1) > 0].mean(),
            "precision_macro": p.mean(), "recall_macro": r.mean(),
            "accuracy": np.trace(cm) / cm.sum()}


def macro_f1(y, pred):
    return per_class(confusion(y, pred))[2].mean()
