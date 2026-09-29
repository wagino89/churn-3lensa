"""Evaluation metrics for the three lenses.

Lens A (statistical): ROC-AUC, PR-AUC, F1, MCC, Brier score.
Lens B (profit): EMPC and the expected optimal fraction of customers targeted,
plus precision at a fixed campaign capacity.
"""
import numpy as np
from scipy import stats
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             matthews_corrcoef, brier_score_loss)


def empc(y_true, scores, clv=200.0, d=10.0, f=1.0, alpha=6.0, beta=14.0, n_gamma=200):
    """Expected Maximum Profit measure for Customer churn (Verbraken et al., 2013).

    Profit per customer when the top fraction of scores is targeted by a retention campaign:
        P(t; g) = CLV*(g*(1-delta) - phi)*pi0*F0(t) - CLV*(delta + phi)*pi1*F1(t)
    with pi0 = share of churners, F0 = TPR, F1 = FPR, delta = d/CLV, phi = f/CLV,
    and g (probability that a targeted churner accepts the offer) ~ Beta(alpha, beta).
    EMPC = E_g[ max_t P(t; g) ], computed by midpoint quadrature over Beta quantiles.
    The maximum over the empirical ROC points equals the maximum over the convex
    hull because the objective is linear in (TPR, FPR).

    Returns (empc, expected_targeted_fraction).
    """
    y = np.asarray(y_true).astype(bool)
    s = np.asarray(scores, dtype=float)
    n_pos, n_neg = y.sum(), (~y).sum()
    if n_pos == 0 or n_neg == 0:
        return np.nan, np.nan
    pi0, pi1 = n_pos / len(y), n_neg / len(y)
    order = np.argsort(-s, kind="mergesort")
    s_sorted, y_sorted = s[order], y[order]
    tp, fp = np.cumsum(y_sorted), np.cumsum(~y_sorted)
    # only keep boundaries between distinct scores (tie handling)
    last_of_group = np.r_[np.diff(s_sorted) != 0, True]
    tpr = np.r_[0.0, tp[last_of_group] / n_pos]
    fpr = np.r_[0.0, fp[last_of_group] / n_neg]
    delta, phi = d / clv, f / clv
    q = (np.arange(n_gamma) + 0.5) / n_gamma
    gammas = stats.beta.ppf(q, alpha, beta)
    profit = clv * ((gammas[:, None] * (1 - delta) - phi) * pi0 * tpr[None, :]
                    - (delta + phi) * pi1 * fpr[None, :])
    best_idx = profit.argmax(axis=1)
    frac = pi0 * tpr + pi1 * fpr
    return float(profit.max(axis=1).mean()), float(frac[best_idx].mean())


def precision_at_fraction(y_true, scores, fraction=0.10):
    y = np.asarray(y_true).astype(int)
    k = max(1, int(round(fraction * len(y))))
    top = np.argsort(-np.asarray(scores), kind="mergesort")[:k]
    return float(y[top].mean())


def best_f1_threshold(y_true, scores):
    """Threshold that maximizes F1 (used by the 'threshold' technique on validation data)."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    cand = np.unique(np.quantile(s, np.linspace(0.01, 0.99, 99)))
    f1s = [f1_score(y, (s >= t).astype(int), zero_division=0) for t in cand]
    return float(cand[int(np.argmax(f1s))])


def evaluate(y_true, scores, threshold=0.5, empc_params=None, top_fraction=0.10):
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    pred = (s >= threshold).astype(int)
    out = {
        "roc_auc": roc_auc_score(y, s),
        "pr_auc": average_precision_score(y, s),
        "f1": f1_score(y, pred, zero_division=0),
        "mcc": matthews_corrcoef(y, pred) if pred.min() != pred.max() else 0.0,
        "brier": brier_score_loss(y, np.clip(s, 0, 1)),
        "prec_at_top": precision_at_fraction(y, s, top_fraction),
        "threshold": threshold,
    }
    out["empc"], out["empc_frac"] = empc(y, s, **(empc_params or {}))
    return out


METRICS = ["roc_auc", "pr_auc", "f1", "mcc", "brier", "prec_at_top", "empc"]
LENS = {"roc_auc": "A", "pr_auc": "A", "f1": "A", "mcc": "A", "brier": "A",
        "prec_at_top": "B", "empc": "B"}
HIGHER_IS_BETTER = {m: (m != "brier") for m in METRICS}
