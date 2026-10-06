"""Stacking: softmax logistic regression on out-of-fold base-model probabilities."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_softmax, softmax

from data import K, class_weights


def fit_softmax(Z, y, C, cw="none"):
    """Minimise 0.5 ||W||^2 + C * sum_i w_i * cross-entropy_i (bias not penalised),
    the same objective as scikit-learn's LogisticRegression."""
    d = Z.shape[1]
    s = C * class_weights(y, cw)[y][:, None]
    Y = np.eye(K)[y]

    def f(theta):
        W, b = theta[:-K].reshape(d, K), theta[-K:]
        logp = log_softmax(Z @ W + b, axis=1)
        G = s * (np.exp(logp) - Y)
        loss = -(s * Y * logp).sum() + 0.5 * (W * W).sum()
        return loss, np.concatenate([(Z.T @ G + W).ravel(), G.sum(0)])

    theta = minimize(f, np.zeros(d * K + K), jac=True, method="L-BFGS-B").x
    return theta[:-K].reshape(d, K), theta[-K:]


def predict_softmax(model, Z):
    W, b = model
    return softmax(Z @ W + b, axis=1)


def oof(base, folds, n_rows, row_pos):
    """Out-of-fold probabilities: base is a list of fit_predict(train, test)
    callables, folds the (train, test) row pairs; row_pos maps a data row to its
    position in the n_rows output. Returns (n_rows, K * len(base))."""
    Z = np.zeros((n_rows, K * len(base)))
    for tr, te in folds:
        for b, fp in enumerate(base):
            Z[row_pos[te], b * K:(b + 1) * K] = fp(tr, te)
    return Z
