"""Nested-CV runs. Rows append to results/*.csv; a rerun skips tasks already written.

  pilot   time one k-NN, tree and two 100-tree forest inner tasks; record memory
  oob     OOB macro-F1 against the number of trees (50-1,000)
  base    k-NN and tree (the revised assignment 2 models)
  forest  bagging and random forest; needs oob
  fixedtree  untuned Gini tree, no depth limit, on the outer folds (Q1 control)
  stack   stack, its removal variants and soft voting; needs base and forest
base and forest run every inner task first, then one outer task per fold.
"""

import argparse
import csv
import os

# One thread per worker (parallelism is across tasks). Must be set before
# numpy and numba are imported.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import resource  # noqa: E402
import time  # noqa: E402
import zlib  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from functools import lru_cache  # noqa: E402
from itertools import product  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data import (K, METRICS, N_INNER, N_OUTER, dataset, inner_folds, macro_f1,  # noqa: E402
                  outer_folds, scores)
from forest import forest  # noqa: E402
from knn import neighbours, vote  # noqa: E402
from stack import fit_softmax, oof, predict_softmax  # noqa: E402

RESULTS = Path(__file__).parent / "results"
PREDS = RESULTS / "preds"
INNER, OUTER, OOB = (RESULTS / f for f in ("inner.csv", "outer.csv", "oob.csv"))

GRIDS = {
    "tree": [dict(crit=c, max_depth=d, min_leaf=m, cw=w) for c, d, m, w in
             product(("entropy", "gini"), (-1, 10, 15, 20, 25, 30), (1, 2, 3, 5, 10, 20),
                     ("none", "balanced"))],
    "knn": [dict(k=k, weighting=w, cw=c) for k, w, c in
            product((1, 5, 11, 25), ("uniform", "distance"), ("none", "balanced"))],
    "forest": [dict(mtry=f, min_leaf=m, cw=w) for f, m, w in
               product(("sqrt", "0.3", "0.6", "all"), (1, 5),
                       ("none", "balanced", "balanced_subsample"))],
}
# model: (grid it is tuned on, which settings of that grid it may select).
# Bagging and the forest share one grid but select disjoint settings.
MODELS = {"knn": ("knn", lambda p: True), "tree": ("tree", lambda p: True),
          "tree_fixed": ("tree", lambda p: True),
          "bagging": ("forest", lambda p: p["mtry"] == "all"),
          "forest": ("forest", lambda p: p["mtry"] != "all")}
STAGES = {"base": ("knn", "tree"), "forest": ("bagging", "forest")}
META_GRID = [dict(C=C, cw=w) for C, w in product((0.01, 0.1, 1, 10, 100, 1000),
                                                  ("none", "balanced"))]
OOB_PARAMS = dict(mtry="0.3", min_leaf=1, cw="none")     # mid-grid
OOB_RECORD = (50, *range(100, 1001, 100))
BASE = ("knn", "tree", "forest")
TREE_FIXED = dict(crit="gini", max_depth=-1, min_leaf=1, cw="none")
STACKS = {"stack": (0, 1, 2), "stack_no_knn": (1, 2), "stack_no_tree": (0, 2),
          "stack_no_forest": (0, 1)}

FIELDS = {
    INNER: ["model", "repeat", "outer", "inner", "cfg", "params", "macro_f1",
            "n_train", "n_test", "n_trees", "seconds"],
    OUTER: ["model", "repeat", "outer", "params", *METRICS, "n_train", "n_test",
            "n_trees", "seconds"],
    OOB: ["repeat", "outer", "n_trees", "oob_f1", "n_train", "seconds"],
}
KEY = {INNER: ("model", "repeat", "outer", "inner", "cfg"),
       OUTER: ("model", "repeat", "outer"), OOB: ("repeat", "outer", "n_trees")}


def task(kind, model, r, o, **kw):
    return {"kind": kind, "model": model, "repeat": r, "outer": o, "inner": 0,
            "cfgs": (), "n_trees": 0, "params": None, **kw}


def row_keys(t):
    """Keys of every row the task writes; it is done when all are present."""
    r, o = t["repeat"], t["outer"]
    keys = {"inner": [(t["model"], r, o, t["inner"], c) for c in t["cfgs"]],
            "outer": [(t["model"], r, o)],
            "stack": [(m, r, o) for m in (*STACKS, "soft_vote")],
            "oob": [(r, o, n) for n in OOB_RECORD]}[t["kind"]]
    return {tuple(map(str, k)) for k in keys}


def seed_of(t):
    return zlib.crc32(repr((t["model"], t["repeat"], t["outer"], t["inner"])).encode())


def pred_path(model, repeat, outer):
    return PREDS / f"{model}_r{repeat}_o{outer}.npz"


@lru_cache(maxsize=None)
def split(repeat, outer):
    """Outer (train, test) and the inner folds of the outer training set."""
    train, test = outer_folds(dataset().y, repeat)[outer]
    return train, test, inner_folds(dataset().y, train, repeat)


def predict_many(grid, params, train, test, seed, n_trees=0):
    """Class probabilities on test for each settings dict in params."""
    D = dataset()
    if grid == "knn":          # every k and weighting from one neighbour list
        idx, dist = neighbours(D, train, test, max(p["k"] for p in params))
        return [vote(idx, dist, D.y[train], **p) for p in params]
    if grid == "tree":
        return [forest(D, train, test, 1, "all", p["min_leaf"], p["cw"], seed,
                       crit=p["crit"], max_depth=p["max_depth"], boot=False)[0]
                for p in params]
    return [forest(D, train, test, n_trees, p["mtry"], p["min_leaf"], p["cw"], seed)[0]
            for p in params]


def outer_row(t, model, params, test, P, n_train, t0):
    np.savez_compressed(pred_path(model, t["repeat"], t["outer"]),
                        proba=P.astype(np.float32), rows=test)
    return {"model": model, "repeat": t["repeat"], "outer": t["outer"],
            "params": json.dumps(params), **scores(dataset().y[test], P.argmax(1)),
            "n_train": n_train, "n_test": len(test), "n_trees": t["n_trees"],
            "seconds": round(time.time() - t0, 1)}


def run_inner(t):
    y, t0 = dataset().y, time.time()
    tr, te = split(t["repeat"], t["outer"])[2][t["inner"]]
    grid = GRIDS[t["model"]]
    P = predict_many(t["model"], [grid[c] for c in t["cfgs"]], tr, te, seed_of(t),
                     t["n_trees"])
    return [{"model": t["model"], "repeat": t["repeat"], "outer": t["outer"],
             "inner": t["inner"], "cfg": c, "params": json.dumps(grid[c]),
             "macro_f1": macro_f1(y[te], p.argmax(1)), "n_train": len(tr),
             "n_test": len(te), "n_trees": t["n_trees"],
             "seconds": round(time.time() - t0, 1)} for c, p in zip(t["cfgs"], P)]


def run_outer(t):
    t0 = time.time()
    train, test, _ = split(t["repeat"], t["outer"])
    P = predict_many(MODELS[t["model"]][0], [t["params"]], train, test, seed_of(t),
                     t["n_trees"])[0]
    return [outer_row(t, t["model"], t["params"], test, P, len(train), t0)]


def run_stack(t):
    """Out-of-fold base probabilities (3 inner folds) train the meta-learner; the
    outer test inputs are the saved outer predictions of the base models."""
    t0, y = time.time(), dataset().y
    train, test, inner = split(t["repeat"], t["outer"])
    pos = np.empty(len(y), np.int64)
    pos[train] = np.arange(len(train))
    base = [lambda a, b, m=m: predict_many(MODELS[m][0], [t["params"][m]], a, b,
                                           seed_of(t), t["n_trees"])[0] for m in BASE]
    # Z does not depend on the meta settings: cache it so a killed run resumes here.
    zpath = pred_path("oof", t["repeat"], t["outer"])
    if zpath.exists():
        Z = np.load(zpath)["Z"]
    else:
        Z = oof(base, inner, len(train), pos)
        tmp = zpath.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, Z=Z)
        os.replace(tmp, zpath)      # atomic: a kill mid-save leaves no half file
    Zt = np.hstack([np.load(pred_path(m, t["repeat"], t["outer"]))["proba"] for m in BASE])

    rows = []
    for name, keep in STACKS.items():
        cols = np.concatenate([np.arange(b * K, (b + 1) * K) for b in keep])
        Zc = Z[:, cols]
        cv = [(Zc[pos[a]], y[a], Zc[pos[b]], y[b]) for a, b in inner]
        best = max(META_GRID, key=lambda p: np.mean(
            [macro_f1(yb, predict_softmax(fit_softmax(Za, ya, **p), Zb).argmax(1))
             for Za, ya, Zb, yb in cv]))
        P = predict_softmax(fit_softmax(Zc, y[train], **best), Zt[:, cols])
        rows.append(outer_row(t, name, best, test, P, len(train), t0))
    soft = Zt.reshape(len(test), len(BASE), K).mean(1)
    rows.append(outer_row(t, "soft_vote", {}, test, soft, len(train), t0))
    return rows


def run_oob(t):
    t0 = time.time()
    train = split(t["repeat"], t["outer"])[0]
    _, curve = forest(dataset(), train, np.arange(0), max(OOB_RECORD), **OOB_PARAMS,
                      seed=seed_of(t), record=OOB_RECORD)
    return [{"repeat": t["repeat"], "outer": t["outer"], "n_trees": n, "oob_f1": f,
             "n_train": len(train), "seconds": round(time.time() - t0, 1)}
            for n, f in curve.items()]


RUNNERS = {"inner": run_inner, "outer": run_outer, "stack": run_stack, "oob": run_oob}


def run_pilot(t):
    """One real inner task in a fresh process, so ru_maxrss is its own peak."""
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    t0 = time.time()
    run_inner(t)
    return {"model": t["model"], "cfgs": len(t["cfgs"]), "n_trees": t["n_trees"],
            "seconds": round(time.time() - t0, 1), "start_mb": round(rss0),
            "peak_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)}


def run(tasks, path, workers):
    """Run the tasks not yet in path and append their rows."""
    done = set()
    if path.exists():
        with open(path) as f:
            done = {tuple(r[k] for k in KEY[path]) for r in csv.DictReader(f)}
    todo = [t for t in tasks if not row_keys(t) <= done]
    print(f"{path.name}: {len(tasks)} tasks, {len(tasks) - len(todo)} done, "
          f"{len(todo)} to run on {workers} workers", flush=True)
    if not todo:
        return
    t0, failed = time.time(), []
    new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as f, ProcessPoolExecutor(workers) as pool:
        w = csv.DictWriter(f, fieldnames=FIELDS[path])
        if new:
            w.writeheader()
            f.flush()
        futs = {pool.submit(RUNNERS[t["kind"]], t): t for t in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            try:       # one failure must not discard the other workers' results
                w.writerows(fut.result())
                f.flush()
            except Exception as e:
                failed.append(futs[fut])
                print(f"  FAILED {futs[fut]['kind']} {futs[fut]['model']} "
                      f"r{futs[fut]['repeat']} o{futs[fut]['outer']}: {e!r}", flush=True)
            el = time.time() - t0
            print(f"  {i}/{len(todo)}  {el / 60:.1f} min, "
                  f"eta {(len(todo) - i) * el / i / 60:.1f} min", flush=True)
    if failed:
        raise RuntimeError(f"{len(failed)} tasks failed; rerun the stage to retry them")


def forest_inner():
    """Forest inner rows with the JSON params expanded into one column per setting."""
    df = pd.read_csv(INNER).query("model == 'forest'")
    return df.join(pd.json_normalize(df.params.map(json.loads).tolist()).set_index(df.index))


def select(model, repeat, outer):
    """Settings with the best mean inner macro-F1. Asserts every inner run is present."""
    grid_name, allowed = MODELS[model]
    inner = pd.read_csv(INNER)
    s = inner[(inner.model == grid_name) & (inner.repeat == repeat) & (inner.outer == outer)]
    g = s.groupby("cfg").macro_f1.agg(["mean", "size"])
    grid = GRIDS[grid_name]
    assert len(g) == len(grid) and (g["size"] == N_INNER).all(), f"incomplete inner runs: {model}"
    keep = [c for c in g.index if allowed(grid[c])]
    return grid[g.loc[keep, "mean"].idxmax()]


def chosen_n_trees():
    """Smallest recorded tree count whose mean OOB macro-F1 is within 0.001 of 1,000 trees'."""
    curve = pd.read_csv(OOB).groupby("n_trees").oob_f1.mean()
    return int(curve.index[curve >= curve[max(OOB_RECORD)] - 0.001].min())


def selected_params(model, repeat, outer):
    o = pd.read_csv(OUTER)
    row = o[(o.model == model) & (o.repeat == repeat) & (o.outer == outer)]
    assert len(row) == 1, f"run stage base/forest first: {model} r{repeat} o{outer}"
    return json.loads(row.params.iloc[0])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", required=True, choices=("pilot", "oob", *STAGES, "fixedtree", "stack"))
    p.add_argument("--repeats", type=int, default=1, help="repeat r uses seed r")
    p.add_argument("--workers", type=int, default=12)
    a = p.parse_args()
    PREDS.mkdir(parents=True, exist_ok=True)
    dataset()       # load before the pool: forked workers share it
    ro = list(product(range(a.repeats), range(N_OUTER)))

    if a.stage == "pilot":
        rows = []
        for t in (task("inner", "knn", 0, 0, cfgs=tuple(range(len(GRIDS["knn"])))),
                  task("inner", "tree", 0, 0, cfgs=(0,)),
                  task("inner", "forest", 0, 0, cfgs=(0,), n_trees=100),      # sqrt
                  task("inner", "forest", 0, 0, n_trees=100, cfgs=(        # all
                      [p["mtry"] for p in GRIDS["forest"]].index("all"),))):
            with ProcessPoolExecutor(1) as pool:
                rows.append(pool.submit(run_pilot, t).result())
            print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(RESULTS / "pilot.csv", index=False)
    elif a.stage == "oob":
        run([task("oob", "forest", r, o) for r, o in ro], OOB, a.workers)
    elif a.stage in STAGES:
        nt = chosen_n_trees() if a.stage == "forest" else 0
        print(f"n_trees = {nt}")
        inner = []
        for g in dict.fromkeys(MODELS[m][0] for m in STAGES[a.stage]):
            n = len(GRIDS[g])
            # k-NN scores its whole grid from one neighbour list, so one task per fold.
            batches = [tuple(range(n))] if g == "knn" else [(c,) for c in range(n)]
            inner += [task("inner", g, r, o, inner=i, cfgs=b, n_trees=nt)
                      for r, o in ro for i in range(N_INNER) for b in batches]
        run(inner, INNER, a.workers)
        run([task("outer", m, r, o, params=select(m, r, o), n_trees=nt)
             for m in STAGES[a.stage] for r, o in ro], OUTER, a.workers)
    elif a.stage == "fixedtree":
        run([task("outer", "tree_fixed", r, o, params=TREE_FIXED) for r, o in ro],
            OUTER, a.workers)
    else:
        nt = chosen_n_trees()
        run([task("stack", "stack", r, o, n_trees=nt,
                  params={m: selected_params(m, r, o) for m in BASE}) for r, o in ro],
            OUTER, a.workers)


if __name__ == "__main__":
    main()
