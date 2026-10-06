"""Recomputes every number quoted in the report from results/."""

import json
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import stats

from data import CLASSES, K, METRICS, confusion, dataset, per_class
from run import OOB, OOB_RECORD, OUTER, STACKS, chosen_n_trees, forest_inner, pred_path

MODELS = ["knn", "tree", "bagging", "forest", "stack"]
VARIANTS = [m for m in STACKS if m != "stack"]      # stacks with one base model removed
REMOVAL = [*STACKS, "soft_vote"]
ALPHA = 0.05
# (a, b, question): two-sided test of a - b = 0. "removal" tests are exploratory.
TESTS = [("bagging", "tree", "Q1"), ("forest", "bagging", "Q2"), ("stack", "forest", "Q4"),
         ("stack", "stack_no_knn", "Q5"), ("stack", "stack_no_tree", "removal"),
         ("stack", "stack_no_forest", "removal"), ("stack", "soft_vote", "Q6")]


def load():
    """Outer rows, with a guard against missing or mixed runs (assignment 3 lesson)."""
    o = pd.read_csv(OUTER)
    repeats = o.repeat.nunique()
    assert sorted(o.repeat.unique()) == list(range(repeats)), "a repeat is missing"
    J = 5 * repeats
    n_oob = len(pd.read_csv(OOB))
    assert n_oob == J * len(OOB_RECORD), f"oob: {n_oob} rows, expected {J * len(OOB_RECORD)}"
    counts = o.model.value_counts()
    for m in MODELS + VARIANTS + ["soft_vote"]:
        assert counts.get(m, 0) == J, f"{m}: {counts.get(m, 0)} rows, expected {J}"
    ens = o[~o.model.isin(["knn", "tree"])]
    assert ens.n_trees.nunique() == 1, "ensembles ran with different n_trees"
    return o.sort_values(["repeat", "outer"])


def by_fold(o, m, metric="macro_f1"):
    return o[o.model == m][metric].to_numpy()


def majority():
    """Accuracy of always predicting the most common class."""
    y = dataset().y
    return np.bincount(y).max() / len(y)


def nadeau_bengio(d, n_test, n_train):
    """Corrected resampled t-test: variance inflated by 1/J + n_test/n_train."""
    J = len(d)
    t = d.mean() / np.sqrt((1 / J + n_test / n_train) * d.var(ddof=1))
    return t, 2 * stats.t.sf(abs(t), J - 1)


def holm(p):
    p = np.asarray(p)
    order = np.argsort(p)
    adj = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    out = np.empty_like(adj)
    out[order] = np.minimum(adj, 1)
    return out


@lru_cache(maxsize=None)
def pooled(m, repeat=0):
    """True labels and predictions over all outer test folds of one repeat."""
    y = dataset().y
    parts = [np.load(pred_path(m, repeat, o)) for o in range(5)]
    rows = np.concatenate([p["rows"] for p in parts])
    return y[rows], np.concatenate([p["proba"].argmax(1) for p in parts])


@lru_cache(maxsize=None)
def cm(m):
    return confusion(*pooled(m))


def main():
    o = load()
    pd.set_option("display.width", 200)

    print("== Selected settings per fold ==")
    for m in MODELS + VARIANTS:
        print(m)
        for r, f, p in o[o.model == m][["repeat", "outer", "params"]].itertuples(index=False):
            print(f"   r{r} o{f}: {p}")

    curve = pd.read_csv(OOB).groupby("n_trees").oob_f1.agg(["mean", "std"])
    print(f"\n== OOB curve: n_trees = {chosen_n_trees()} ==\n{curve.round(4)}")

    print("\n== Overall (mean ± sd over folds) ==")
    tab = o.groupby("model")[METRICS].agg(["mean", "std"])
    print(tab.loc[MODELS].round(4).to_string())
    print(f"majority-class accuracy baseline: {majority():.4f}")

    print("\n== Friedman (macro-F1) ==")
    f1 = np.column_stack([by_fold(o, m) for m in MODELS])
    chi2, p = stats.friedmanchisquare(*f1.T)
    ranks = stats.rankdata(-f1, axis=1).mean(0)
    print(f"chi2 = {chi2:.2f}, df = {len(MODELS) - 1}, p = {p:.3g}; mean ranks "
          + ", ".join(f"{m} {r:.1f}" for m, r in zip(MODELS, ranks)))

    n_test, n_train = o.n_test.mean(), o.n_train.mean()
    diffs = [by_fold(o, a) - by_fold(o, b) for a, b, _ in TESTS]
    res = [nadeau_bengio(d, n_test, n_train) for d in diffs]
    print("\n== Nadeau-Bengio, Holm-corrected (macro-F1) ==")
    # Holm families: the five question tests; the two exploratory removal tests.
    fam = np.array([q != "removal" for _, _, q in TESTS])
    ph_all = np.empty(len(TESTS))
    for f in (fam, ~fam):
        ph_all[f] = holm([r[1] for r, k in zip(res, f) if k])
    for (a, b, q), d, (t, p), ph in zip(TESTS, diffs, res, ph_all):
        print(f"{q:8} {a} - {b}: diff {d.mean():+.4f}, {(d > 0).sum()}/{len(d)} folds, "
              f"t={t:.2f}, p={p:.4g}, p_holm={ph:.4g}, {'significant' if ph < ALPHA else 'not significant'}")

    print("\n== Q3: class weighting ==")
    for m in ("bagging", "forest"):
        picks = o[o.model == m].params.map(lambda s: json.loads(s)["cw"]).value_counts()
        print(f"{m} selected: {picks.to_dict()}")
    fi = forest_inner()
    mean = fi.groupby(["mtry", "min_leaf", "cw"]).macro_f1.mean().unstack("cw")
    print((mean.sub(mean["none"], axis=0)).round(4).to_string())
    # Per outer fold: best weighted minus best unweighted mean inner score. Inner folds
    # hold out 1/3, so the correction uses n_test/n_train = 1/2.
    best = fi.groupby(["repeat", "outer", fi.cw != "none", "cfg"]).macro_f1.mean() \
        .groupby(level=[0, 1, 2]).max().unstack()
    d = (best[True] - best[False]).to_numpy()
    t, p = nadeau_bengio(d, 1, 2)
    print(f"forest, best weighted - best unweighted inner macro-F1: diff {d.mean():+.4f}, "
          f"{(d > 0).sum()}/{len(d)} folds, t={t:.2f}, p={p:.4g}, "
          f"{'significant' if p < ALPHA else 'not significant'} (uncorrected for the other tests)")

    print("\n== Removal test ==")
    print(o.groupby("model")[["macro_f1", "balanced_accuracy"]].agg(["mean", "std"])
          .loc[REMOVAL].round(4).to_string())

    print("\n== Per-class F1 (pooled outer folds, repeat 0) ==")
    pcf = pd.DataFrame({m: per_class(cm(m))[2] for m in MODELS},
                       index=CLASSES)
    print(pcf.round(3).to_string())

    print("\n== DoS vs Exploits (pooled, repeat 0) ==")
    dos, exp = CLASSES.index("DoS"), CLASSES.index("Exploits")
    for m in MODELS:
        c = cm(m)
        print(f"{m:8} DoS->Exploits {c[dos, exp]:6d} of {c[dos].sum()}, "
              f"Exploits->DoS {c[exp, dos]:6d} of {c[exp].sum()}")

    print("\n== Base-model disagreement per class (Q5, repeat 0) ==")
    y = pooled("knn")[0]
    P = np.column_stack([pooled(m)[1] for m in ("knn", "tree", "forest")])
    split = (P != P[:, :1]).any(1)
    print(pd.Series([split[y == c].mean() for c in range(K)], index=CLASSES)
          .round(3).to_string())
    details(o, diffs, fi)


def paired(o, a, b, metric):
    d = by_fold(o, a, metric) - by_fold(o, b, metric)
    return f"{a} - {b} {metric}: {d.mean():+.4f}, higher in {(d > 0).sum()}/{len(d)} folds"


def by_cw(o, m, weighted):
    """Mean outer scores of the folds that did / did not select class weights."""
    om = o[o.model == m]
    cw = om.params.map(lambda s: json.loads(s)["cw"] != "none")
    for flag, g in om.groupby(cw.to_numpy()):
        print(f"{m} {weighted if flag else 'unweighted'} ({len(g)} folds): "
              + ", ".join(f"{k} {g[k].mean():.4f}" for k in METRICS[:3]))


def details(o, diffs, fi):
    """Supporting numbers quoted in the Results text."""
    print("\n== OOB gaps ==")
    oob = pd.read_csv(OOB)
    curve = oob.groupby("n_trees").oob_f1.mean()
    nt, top = chosen_n_trees(), max(OOB_RECORD)
    print(f"mean gap to {top} trees: {nt} trees {curve[top] - curve[nt]:.4f}, "
          f"100 trees {curve[top] - curve[100]:.4f}")
    w = oob.pivot_table(index=["repeat", "outer"], columns="n_trees", values="oob_f1")
    print(f"largest single-fold gap {nt} vs {top}: {(w[top] - w[nt]).max():.4f}")

    print("\n== Forest inner grid (mean inner macro-F1) ==")
    grid = fi.groupby(["cw", "min_leaf", "mtry"]).macro_f1.mean()
    print(grid.unstack("mtry").round(4).to_string())
    best_w, best_u = grid.drop("none").max(), grid["none"].max()
    print(f"best weighted {best_w:.4f} ({grid.drop('none').idxmax()}), "
          f"best unweighted {best_u:.4f}, gap {best_u - best_w:.4f}")

    print("\n== Outer folds with and without class weights ==")
    by_cw(o, "forest", "weighted")
    by_cw(o, "stack_no_knn", "balanced")

    print("\n== Repeat 0 (the assignment 2 folds) ==")
    print(o[o.repeat == 0].groupby("model").macro_f1.mean().loc[MODELS].round(4).to_string())

    print("\n== Effect sizes (mean / sd of the fold differences) ==")
    for (a, b, q), d in zip(TESTS, diffs):
        print(f"{q:8} {a} - {b}: sd {d.std(ddof=1):.4f}, effect {d.mean() / d.std(ddof=1):+.2f}")

    print("\n== Paired differences on the other measures ==")
    for a, b, m in [("bagging", "tree", "accuracy"), ("bagging", "tree", "balanced_accuracy"),
                    ("stack", "forest", "balanced_accuracy"), ("stack", "forest", "precision_macro"),
                    ("stack", "forest", "accuracy"), ("stack", "soft_vote", "precision_macro"),
                    ("stack", "soft_vote", "balanced_accuracy")]:
        print(paired(o, a, b, m))
    acc = o.groupby("model").accuracy.mean().loc[MODELS]
    print(f"smallest margin over the majority baseline: {acc.min() - majority():.4f} ({acc.idxmin()})")

    print("\n== Meta-learner settings selected ==")
    for m in STACKS:
        print(m, o[o.model == m].params.value_counts().to_dict())

    print("\n== Per class (pooled outer folds, repeat 0) ==")
    y = pooled("knn")[0]
    print("instances:", dict(zip(CLASSES, np.bincount(y, minlength=K))))
    prf = {m: per_class(cm(m)) for m in MODELS}
    for i, name in enumerate(("precision", "recall")):
        print(name)
        print(pd.DataFrame({m: v[i] for m, v in prf.items()}, index=CLASSES).round(3).to_string())
    c = cm("knn")
    w, e = CLASSES.index("Worms"), CLASSES.index("Exploits")
    print(f"knn Worms: {c[w, w]} correct of {c[w].sum()}, {c[w, e]} to Exploits")
    dos = CLASSES.index("DoS")
    both = {m: cm(m)[dos, e] + cm(m)[e, dos] for m in MODELS}
    print("DoS<->Exploits both directions:", both)
    print(f"stack - tree DoS F1: {prf['stack'][2][dos] - prf['tree'][2][dos]:+.3f}; "
          f"largest stack - tree gain on any class: {(prf['stack'][2] - prf['tree'][2]).max():+.3f}")


if __name__ == "__main__":
    main()
