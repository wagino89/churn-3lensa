"""Class-imbalance handling techniques.

When imbalanced-learn is installed, ROS/RUS/SMOTE/ADASYN/SMOTE-ENN use the
official imbalanced-learn implementations (required for reported results).
Otherwise ROS/RUS/SMOTE fall back to the implementations below and
ADASYN/SMOTE-ENN are skipped. Dirichlet ExtSMOTE and EasyEnsemble always use
the implementations in this file.

All samplers are called ONLY on the training data inside a fold, after the
preprocessing has been fitted on the same training data (leakage-safe protocol).
"""
import warnings
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.neighbors import NearestNeighbors

try:
    import imblearn.over_sampling as _ios
    import imblearn.under_sampling as _ius
    import imblearn.combine as _icomb
    HAS_IMBLEARN = True
except ImportError:  # pragma: no cover
    HAS_IMBLEARN = False


class NotAvailable(Exception):
    pass


def _n_target(y, ratio):
    n_min, n_maj = int((y == 1).sum()), int((y == 0).sum())
    return n_min, n_maj, int(round(ratio * n_maj))


class RandomOver:
    def __init__(self, ratio=1.0, random_state=None):
        self.ratio, self.rng = ratio, np.random.default_rng(random_state)

    def fit_resample(self, X, y):
        n_min, _, target = _n_target(y, self.ratio)
        extra = max(0, target - n_min)
        idx = self.rng.choice(np.where(y == 1)[0], size=extra, replace=True)
        return np.vstack([X, X[idx]]), np.r_[y, y[idx]]


class RandomUnder:
    def __init__(self, ratio=1.0, random_state=None):
        self.ratio, self.rng = ratio, np.random.default_rng(random_state)

    def fit_resample(self, X, y):
        n_min = int((y == 1).sum())
        keep_maj = min(int((y == 0).sum()), int(round(n_min / self.ratio)))
        maj = self.rng.choice(np.where(y == 0)[0], size=keep_maj, replace=False)
        idx = np.r_[np.where(y == 1)[0], maj]
        return X[idx], y[idx]


class SimpleSMOTE:
    """Classical SMOTE (Chawla et al., 2002): linear interpolation towards one random neighbour (fallback)."""

    def __init__(self, ratio=1.0, k=5, random_state=None):
        self.ratio, self.k, self.rng = ratio, k, np.random.default_rng(random_state)

    def fit_resample(self, X, y):
        n_min, _, target = _n_target(y, self.ratio)
        n_new = max(0, target - n_min)
        Xm = X[y == 1]
        if n_new == 0 or len(Xm) < 2:
            return X, y
        k = min(self.k, len(Xm) - 1)
        nn = NearestNeighbors(n_neighbors=k + 1).fit(Xm)
        neigh = nn.kneighbors(Xm, return_distance=False)[:, 1:]
        base = self.rng.integers(0, len(Xm), n_new)
        pick = neigh[base, self.rng.integers(0, k, n_new)]
        gap = self.rng.random((n_new, 1))
        synth = Xm[base] + gap * (Xm[pick] - Xm[base])
        return np.vstack([X, synth]), np.r_[y, np.ones(n_new, dtype=y.dtype)]


class DirichletExtSMOTE:
    """Dirichlet ExtSMOTE, inverse-distance variant (Matharaarachchi et al., 2024).

    For every synthetic sample: pick a random minority instance r, take its k
    minority neighbours, compute the distance d_j of each neighbour to the
    coordinate-wise median of the minority class, set alpha_j = m / d_j, draw
    weights w ~ Dirichlet(alpha), and return x_new = sum_j w_j * x_j. Neighbours
    close to the centre receive larger weights, so abnormal minority instances
    have less influence.
    Implementation note: r itself is not counted as a neighbour.
    """

    def __init__(self, ratio=1.0, k=5, m=1.0, random_state=None):
        self.ratio, self.k, self.m = ratio, k, m
        self.rng = np.random.default_rng(random_state)

    def fit_resample(self, X, y):
        n_min, _, target = _n_target(y, self.ratio)
        n_new = max(0, target - n_min)
        Xm = X[y == 1]
        if n_new == 0 or len(Xm) < 2:
            return X, y
        k = min(self.k, len(Xm) - 1)
        mu = np.median(Xm, axis=0)
        dist_to_mu = np.linalg.norm(Xm - mu, axis=1)
        nn = NearestNeighbors(n_neighbors=k + 1).fit(Xm)
        neigh = nn.kneighbors(Xm, return_distance=False)[:, 1:]
        base = self.rng.integers(0, len(Xm), n_new)
        nb = neigh[base]                                   # (n_new, k)
        alpha = self.m / np.maximum(dist_to_mu[nb], 1e-8)  # (n_new, k)
        g = self.rng.gamma(alpha)                          # Dirichlet via Gamma
        w = g / np.maximum(g.sum(axis=1, keepdims=True), 1e-300)
        synth = np.einsum("nk,nkp->np", w, Xm[nb])
        return np.vstack([X, synth]), np.r_[y, np.ones(n_new, dtype=y.dtype)]


class EasyEnsemble(BaseEstimator, ClassifierMixin):
    """Average of B base models, each trained on a balanced subset (all churners
    plus an equally sized random subset of non-churners). The base model is the
    classifier under test, so that the comparison is fair across classifiers."""

    def __init__(self, base, n_estimators=10, random_state=None):
        self.base, self.n_estimators, self.random_state = base, n_estimators, random_state

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        self.models_ = []
        for _ in range(self.n_estimators):
            sub = np.r_[pos, rng.choice(neg, size=min(len(pos), len(neg)), replace=False)]
            self.models_.append(clone(self.base).fit(X[sub], y[sub]))
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.models_], axis=0)


def get_sampler(name, ratio=1.0, k=5, m=1.0, random_state=None):
    """Return an object with .fit_resample(X, y). Raise NotAvailable if impossible."""
    if name == "ros":
        return (_ios.RandomOverSampler(sampling_strategy=ratio, random_state=random_state)
                if HAS_IMBLEARN else RandomOver(ratio, random_state))
    if name == "rus":
        return (_ius.RandomUnderSampler(sampling_strategy=ratio, random_state=random_state)
                if HAS_IMBLEARN else RandomUnder(ratio, random_state))
    if name in ("smote", "smote_leaky"):
        return (_ios.SMOTE(sampling_strategy=ratio, k_neighbors=k, random_state=random_state)
                if HAS_IMBLEARN else SimpleSMOTE(ratio, k, random_state))
    if name == "adasyn":
        if not HAS_IMBLEARN:
            raise NotAvailable("ADASYN requires imbalanced-learn")
        return _ios.ADASYN(sampling_strategy=ratio, n_neighbors=k, random_state=random_state)
    if name == "smote_enn":
        if not HAS_IMBLEARN:
            raise NotAvailable("SMOTE-ENN requires imbalanced-learn")
        return _icomb.SMOTEENN(sampling_strategy=ratio, random_state=random_state)
    if name == "dirichlet":
        return DirichletExtSMOTE(ratio, k, m, random_state)
    raise ValueError(f"Unknown resampling technique: {name}")


def parse_technique(tech):
    """'smote@0.5' -> ('smote', 0.5); 'none' -> ('none', None)."""
    if "@" in tech:
        name, r = tech.split("@")
        return name, float(r)
    return tech, None


SAMPLER_NAMES = {"ros", "rus", "smote", "adasyn", "smote_enn", "dirichlet", "smote_leaky"}

if not HAS_IMBLEARN:
    warnings.warn("imbalanced-learn is not installed: using fallback SMOTE/ROS/RUS and "
                  "skipping ADASYN & SMOTE-ENN. Install imbalanced-learn for final results.")
