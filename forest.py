"""Bagging and random forest. Bagging is the forest with mtry = all features.

Trees are not stored: each one adds its probabilities to the evaluation rows
and its own out-of-bag (OOB) rows, then is dropped. 300 stored unlimited-depth
trees would not fit in 9 GB.
"""

import numpy as np

from data import K, class_weights, macro_f1
from tree import CRIT, fit, predict


def n_features(spec, d):
    """'sqrt', 'all' or a fraction of d, e.g. '0.3'."""
    if spec == "sqrt":
        return int(np.sqrt(d))
    if spec == "all":
        return d
    return max(1, int(float(spec) * d))


def bootstrap(rng, n):
    """Positions drawn with replacement, and the positions never drawn (OOB)."""
    pick = rng.integers(0, n, n)
    seen = np.zeros(n, bool)
    seen[pick] = True
    return pick, np.flatnonzero(~seen)


def forest(D, train, evals, n_trees, mtry, min_leaf, cw, seed, crit="gini",
           max_depth=-1, boot=True, record=()):
    """Mean class probabilities on D.X[evals], and {t: OOB macro-F1} for t in record.
    cw is 'none', 'balanced' or 'balanced_subsample' (weights per bootstrap).
    One tree without bootstrap and mtry='all' is the single tree."""
    rng = np.random.default_rng(seed)
    n, y_tr = train.size, D.y[train]
    m = n_features(mtry, D.X.shape[1])
    w = class_weights(y_tr, cw)
    p_eval = np.zeros((evals.size, K))
    p_oob = np.zeros((n, K))
    curve = {}
    for t in range(1, n_trees + 1):
        pick, oob = bootstrap(rng, n) if boot else (np.arange(n), np.arange(0))
        rows = train[pick]
        if cw == "balanced_subsample":
            w = class_weights(D.y[rows], "balanced")
        tree = fit(D.X, D.R, D.y, rows, w, D.is_nom, D.n_levels, K, CRIT[crit], max_depth,
                   min_leaf, m, rng.integers(2**31 - 1))
        if evals.size:
            p_eval += predict(tree, D.X, evals, D.is_nom)
        if oob.size:
            p_oob[oob] += predict(tree, D.X, train[oob], D.is_nom)
        if t in record:
            has = p_oob.sum(1) > 0
            curve[t] = macro_f1(y_tr[has], p_oob[has].argmax(1))
    return p_eval / n_trees, curve
