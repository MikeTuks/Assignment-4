"""k-NN with the heterogeneous value difference metric, HVDM (Wilson & Martinez 1997).

Numeric features: |x - y| / 4 sigma. Nominal features: the normalised VDM,
read from a per-feature table of squared differences in P(class | level).
The data has no missing values once a missing service is its own level, so
the HVDM missing-value rule (distance 1) never applies and is left out.
"""

import numpy as np
from numba import njit, prange

from data import K, class_weights


def vdm_tables(nom, y, n_levels):
    """D[a, u, v] = sum_c (P(c | a=u) - P(c | a=v))^2. Unseen levels get P = 0."""
    L = n_levels.max()
    D = np.zeros((nom.shape[1], L, L))
    for a in range(nom.shape[1]):
        cnt = np.zeros((L, K))
        np.add.at(cnt, (nom[:, a], y), 1)
        P = cnt / np.maximum(cnt.sum(1, keepdims=True), 1)
        D[a] = ((P[:, None, :] - P[None, :, :]) ** 2).sum(2)
    return D


@njit(cache=True)
def hvdm2(za, na, zb, nb, D, bound=np.inf):
    """Squared HVDM between two rows (numeric part already divided by 4 sigma).
    Stops early, returning a partial sum >= bound, once the sum reaches bound."""
    s = 0.0
    for f in range(za.size):
        t = za[f] - zb[f]
        s += t * t
        if s >= bound:                            # terms are >= 0: can only grow
            return s
    for a in range(na.size):
        s += D[a, na[a], nb[a]]
    return s


@njit(parallel=True, cache=True)
def _neighbours(Ztr, Ntr, Zq, Nq, D, k):
    nq = Zq.shape[0]
    idx = np.empty((nq, k), np.int64)
    dist = np.empty((nq, k))
    for q in prange(nq):
        bd = np.full(k, np.inf)
        bi = np.zeros(k, np.int64)
        for j in range(Ztr.shape[0]):
            s = hvdm2(Zq[q], Nq[q], Ztr[j], Ntr[j], D, bd[k - 1])
            if s < bd[k - 1]:                     # insert into the sorted top k
                p = k - 1
                while p > 0 and bd[p - 1] > s:
                    bd[p], bi[p] = bd[p - 1], bi[p - 1]
                    p -= 1
                bd[p], bi[p] = s, j
        idx[q], dist[q] = bi, np.sqrt(bd)
    return idx, dist


def neighbours(D, train, query, k=25):
    """The k nearest training rows (positions in train) of each query row, by HVDM."""
    num, nom = D.knn_num[train], D.knn_nom[train]
    sigma = num.std(0)
    scale = 4 * np.where(sigma > 0, sigma, 1)
    num /= scale
    tab = vdm_tables(nom, D.y[train], D.knn_levels)
    return _neighbours(num, nom, D.knn_num[query] / scale, D.knn_nom[query], tab, k)


def vote(idx, dist, y_train, k, weighting, cw):
    """Class probabilities from the first k neighbours. weighting is 'uniform' or
    'distance'; cw 'balanced' multiplies each vote by the inverse class frequency."""
    lab = y_train[idx[:, :k]]
    d = dist[:, :k]
    if weighting == "distance":
        zero = d == 0
        with np.errstate(divide="ignore"):
            # An exact match takes the whole vote, as in scikit-learn.
            w = np.where(zero.any(1, keepdims=True), zero, 1 / d)
    else:
        w = np.ones_like(d)
    w = w * class_weights(y_train, cw)[lab]
    P = np.zeros((len(lab), K))
    np.add.at(P, (np.arange(len(lab))[:, None], lab), w)
    return P / P.sum(1, keepdims=True)
