"""Unit tests for the components the paper's conclusions depend on.

    pip install pytest && pytest -q
"""
import numpy as np
import pandas as pd
import pytest
from scipy import integrate, stats

from src import config
from src.core import run_unit, stable_seed
from src.experiment_temporal import subsample_panel
from src.metrics import best_f1_threshold, empc, evaluate
from src.resampling import DirichletExtSMOTE, EasyEnsemble, get_sampler
from src.models import make_model


def _toy(n=600, rate=0.15, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < rate).astype(int)
    X = pd.DataFrame({"a": rng.normal(y * 1.2, 1.0), "b": rng.normal(-y * 0.8, 1.0),
                      "c": rng.choice(["x", "y", "z"], n)})
    return X, y


def test_empc_perfect_classifier_matches_analytical_value():
    """For a perfect ranking, EMPC = CLV * pi0 * E[max(gamma(1-delta) - phi, 0)]."""
    y = np.r_[np.ones(200), np.zeros(800)]
    s = np.r_[np.ones(200), np.zeros(800)]
    p = config.EMPC_PARAMS
    delta, phi = p["d"] / p["clv"], p["f"] / p["clv"]
    g = lambda x: max(x * (1 - delta) - phi, 0) * stats.beta.pdf(x, p["alpha"], p["beta"])
    analytical = p["clv"] * 0.2 * integrate.quad(g, 0, 1)[0]
    # midpoint quadrature over 200 Beta quantiles: relative error below 0.02%
    assert empc(y, s, **p)[0] == pytest.approx(analytical, rel=2e-4)


def test_empc_of_uninformative_scores_equals_target_all_or_nobody():
    """Constant scores only allow targeting nobody or everybody."""
    y = np.r_[np.ones(200), np.zeros(800)]
    p = config.EMPC_PARAMS
    delta, phi = p["d"] / p["clv"], p["f"] / p["clv"]
    g = lambda x: max((x * (1 - delta) - phi) * 0.2 - (delta + phi) * 0.8, 0) * stats.beta.pdf(x, p["alpha"], p["beta"])
    analytical = p["clv"] * integrate.quad(g, 0, 1)[0]
    assert empc(y, np.zeros(1000), **p)[0] == pytest.approx(analytical, rel=2e-3)


def test_empc_increases_with_ranking_quality():
    rng = np.random.default_rng(1)
    y = (rng.random(4000) < 0.2).astype(int)
    weak, strong = y + rng.normal(0, 2.0, len(y)), y + rng.normal(0, 0.5, len(y))
    assert empc(y, strong, **config.EMPC_PARAMS)[0] > empc(y, weak, **config.EMPC_PARAMS)[0]


def test_threshold_tuning_returns_a_score_quantile():
    rng = np.random.default_rng(2)
    y = (rng.random(500) < 0.2).astype(int)
    s = y * 0.3 + rng.random(500) * 0.7
    t = best_f1_threshold(y, s)
    assert s.min() <= t <= s.max()


def test_test_fold_is_never_resampled():
    """Leakage-safe protocol: the test fold keeps its original size and churn rate."""
    X, y = _toy()
    tr, te = np.arange(0, 450), np.arange(450, 600)
    for tech in ["none", "smote@1.0", "ros@1.0", "rus@1.0", "dirichlet@1.0", "easy_ensemble", "threshold"]:
        res = run_unit(X.iloc[tr], y[tr], X.iloc[te], y[te], tech, "lr", seed=3)
        assert res["n_test"] == len(te)
        assert res["test_churn_rate"] == pytest.approx(y[te].mean())
    res = run_unit(X.iloc[tr], y[tr], X.iloc[te], y[te], "smote@1.0", "lr", seed=3)
    n_neg = int((y[tr] == 0).sum())
    assert res["n_train_fit"] == 2 * n_neg        # training data balanced to ratio 1.0


def test_dirichlet_extsmote_counts_and_convexity():
    rng = np.random.default_rng(4)
    X = np.vstack([rng.normal(0, 1, (300, 3)), rng.normal(3, 1, (40, 3))])
    y = np.r_[np.zeros(300, int), np.ones(40, int)]
    Xr, yr = DirichletExtSMOTE(ratio=1.0, k=5, m=1.0, random_state=0).fit_resample(X, y)
    assert (yr == 1).sum() == (yr == 0).sum() == 300
    synth = Xr[len(X):]
    lo, hi = X[y == 1].min(axis=0), X[y == 1].max(axis=0)
    assert np.all(synth >= lo - 1e-9) and np.all(synth <= hi + 1e-9)   # convex combinations of minority points


def test_smote_half_ratio():
    X, y = _toy(1000, 0.1)
    Xn = X[["a", "b"]].to_numpy()
    _, yr = get_sampler("smote", 0.5, 5, random_state=0).fit_resample(Xn, y)
    assert (yr == 1).sum() == round(0.5 * (y == 0).sum())


def test_easy_ensemble_probabilities():
    X, y = _toy()
    Xn = X[["a", "b"]].to_numpy()
    m = EasyEnsemble(make_model("lr"), n_estimators=5, random_state=0).fit(Xn, y)
    p = m.predict_proba(Xn)
    assert p.shape == (len(y), 2) and np.allclose(p.sum(axis=1), 1)


def test_subsample_panel_gives_exact_rate_per_snapshot():
    rng = np.random.default_rng(5)
    panel = pd.DataFrame({"cutoff": np.repeat(pd.date_range("2011-01-01", periods=4, freq="MS"), 500),
                          "churn": (rng.random(2000) < 0.5).astype(int), "x": rng.random(2000)})
    for rate in (0.2, 0.1):
        sub = subsample_panel(panel, rate)
        per = sub.groupby("cutoff").churn.mean()
        assert np.all(np.abs(per - rate) < 0.002)
        assert (sub.churn == 0).sum() == (panel.churn == 0).sum()   # non-churners are all kept
    assert subsample_panel(panel, "natural") is panel


def test_seeds_are_deterministic():
    assert stable_seed("telco", "natural", 0, 1, "smote@1.0", "lr") == \
        stable_seed("telco", "natural", 0, 1, "smote@1.0", "lr")
    assert stable_seed("a", 1) != stable_seed("a", 2)


def test_evaluate_returns_all_metrics():
    rng = np.random.default_rng(6)
    y = (rng.random(300) < 0.2).astype(int)
    out = evaluate(y, rng.random(300), 0.5, config.EMPC_PARAMS)
    assert {"roc_auc", "pr_auc", "f1", "mcc", "brier", "prec_at_top", "empc", "empc_frac"} <= set(out)
