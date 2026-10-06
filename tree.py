"""Classification tree in Numba, grown depth-first with an explicit stack.

Numeric features: threshold splits, sorted per node. Nominal features (integer
codes): one level against the rest. A tree is five flat arrays; feature -1
marks a leaf.
"""

import numpy as np
from numba import njit

CRIT = {"entropy": 0, "gini": 1}


@njit(cache=True)
def _children(total, left, wl, W, crit):
    """wl * I(left) + (W - wl) * I(total - left), without allocating the right counts."""
    a = b = 0.0
    wr = W - wl
    for c in range(total.size):
        lc, rc = left[c], total[c] - left[c]
        if crit == 0:
            if lc > 0.0:
                a -= lc * np.log2(lc / wl)
            if rc > 0.0:
                b -= rc * np.log2(rc / wr)
        else:
            a += lc * lc
            b += rc * rc
    if crit == 0:
        return a + b
    return (wl - a / wl if wl > 0.0 else 0.0) + (wr - b / wr if wr > 0.0 else 0.0)


@njit(cache=True)
def _radix_pass(key, src, dst, m, shift, count):
    """Stable counting sort of src[:m] into dst[:m] by 9 bits of key, from shift."""
    count[:] = 0
    for i in range(m):
        count[((key[src[i]] >> shift) & 511) + 1] += 1
    for b in range(1, 512):                     # count[b] becomes bucket b's start
        count[b] += count[b - 1]
    for i in range(m):
        b = (key[src[i]] >> shift) & 511
        dst[count[b]] = src[i]
        count[b] += 1


@njit(cache=True)
def _push(stack, top, node, start, end, depth):
    stack[top, 0], stack[top, 1], stack[top, 2], stack[top, 3] = node, start, end, depth


@njit(cache=True)
def fit(X, R, y, rows, cw, is_nom, n_levels, n_classes, crit, max_depth, min_leaf,
        mtry, seed):
    """Grow a tree on X[rows]; rows may repeat (bootstrap). R holds X's per-column
    ranks (data.ranks). cw is the weight per class, max_depth < 0 means unlimited,
    mtry features are tried per split."""
    np.random.seed(seed)
    n, d = rows.size, X.shape[1]
    cap = 2 * n
    feat = np.full(cap, -1, np.int64)
    thr = np.zeros(cap)
    left = np.zeros(cap, np.int64)
    right = np.zeros(cap, np.int64)
    value = np.zeros((cap, n_classes))

    idx = rows.copy()
    feats = np.arange(d)
    vals = np.empty(n)
    keys = np.empty(n, np.int32)                # ranks of vals, for the radix sort
    order = np.empty(n, np.int64)
    tmp = np.empty(n, np.int64)
    count = np.empty(513, np.int64)
    total = np.empty(n_classes)
    lc = np.empty(n_classes)
    n_lv = max(n_levels.max(), 1)
    lvl = np.empty((n_lv, n_classes))
    nlv = np.empty(n_lv, np.int64)

    stack = np.empty((cap, 4), np.int64)        # node, start, end, depth
    _push(stack, 0, 0, 0, n, 0)
    top, n_nodes = 1, 1
    while top > 0:
        top -= 1
        node, start, end, depth = stack[top, 0], stack[top, 1], stack[top, 2], stack[top, 3]
        m = end - start
        total[:] = 0.0
        for i in range(start, end):
            c = y[idx[i]]
            total[c] += cw[c]
        W = total.sum()
        value[node] = total / W
        parent = _children(total, total, W, W, crit) / W
        if parent <= 1e-12 or m < 2 * min_leaf or (0 <= max_depth <= depth):
            continue

        best, best_f, best_t = np.inf, -1, 0.0
        for j in range(mtry):
            r = j + np.random.randint(0, d - j)  # partial Fisher-Yates draw
            feats[j], feats[r] = feats[r], feats[j]
            f = feats[j]
            if is_nom[f]:
                L = n_levels[f]
                lvl[:L] = 0.0
                nlv[:L] = 0
                for i in range(start, end):
                    v, c = int(X[idx[i], f]), y[idx[i]]
                    lvl[v, c] += cw[c]
                    nlv[v] += 1
                for v in range(L):
                    if nlv[v] < min_leaf or m - nlv[v] < min_leaf:
                        continue
                    s = _children(total, lvl[v], lvl[v].sum(), W, crit)
                    if s < best:
                        best, best_f, best_t = s, f, v
            else:
                for i in range(m):
                    vals[i] = X[idx[start + i], f]
                    keys[i] = R[idx[start + i], f]
                    order[i] = i
                _radix_pass(keys, order, tmp, m, 0, count)  # two 9-bit passes sort
                _radix_pass(keys, tmp, order, m, 9, count)  # ranks < 2**18
                lc[:] = 0.0
                for t in range(m - min_leaf):
                    c = y[idx[start + order[t]]]
                    lc[c] += cw[c]
                    a, b = vals[order[t]], vals[order[t + 1]]
                    if t + 1 < min_leaf or a == b:
                        continue
                    # Left weight from the class counts, not a running sum, so equal
                    # splits score identically whatever order tied rows were visited.
                    s = _children(total, lc, lc.sum(), W, crit)
                    if s < best:
                        mid = a + (b - a) / 2
                        best, best_f, best_t = s, f, mid if mid < b else a
        # ponytail: no valid split among the mtry drawn features makes a leaf
        # (scikit-learn keeps drawing); add a redraw if forests underfit.
        if best_f < 0:
            continue

        p = start
        for i in range(start, end):
            x = X[idx[i], best_f]
            if (x == best_t) if is_nom[best_f] else (x <= best_t):
                idx[i], idx[p] = idx[p], idx[i]
                p += 1
        feat[node], thr[node] = best_f, best_t
        left[node], right[node] = n_nodes, n_nodes + 1
        _push(stack, top, n_nodes + 1, p, end, depth + 1)
        _push(stack, top + 1, n_nodes, start, p, depth + 1)
        top += 2
        n_nodes += 2
    return feat[:n_nodes], thr[:n_nodes], left[:n_nodes], right[:n_nodes], value[:n_nodes]


@njit(cache=True)
def predict(tree, X, rows, is_nom):
    """Class probabilities for X[rows]. An unseen nominal level goes right."""
    feat, thr, left, right, value = tree
    out = np.empty((rows.size, value.shape[1]))
    for i in range(rows.size):
        node = 0
        while feat[node] >= 0:
            f = feat[node]
            x = X[rows[i], f]
            go_left = (x == thr[node]) if is_nom[f] else (x <= thr[node])
            node = left[node] if go_left else right[node]
        out[i] = value[node]
    return out
