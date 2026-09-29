"""Classifiers and preprocessing.

Hyperparameters are fixed (not tuned) so that differences between imbalance
techniques are not confounded with tuning. The values are reported in the
Methods section; tuning can be added as a sensitivity analysis.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:  # pragma: no cover
    HAS_XGB = False
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:  # pragma: no cover
    HAS_LGBM = False


def make_preprocessor(X):
    """Median imputation + standardization (numeric), mode imputation + one-hot (categorical)."""
    cat = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    num = [c for c in X.columns if c not in cat]
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat),
    ], sparse_threshold=0.0)


def available_models(names):
    ok = []
    for n in names:
        if n == "xgb" and not HAS_XGB:
            print("[info] xgboost not installed -> model 'xgb' skipped")
            continue
        if n == "lgbm" and not HAS_LGBM:
            print("[info] lightgbm not installed -> model 'lgbm' skipped")
            continue
        ok.append(n)
    return ok


def make_model(name, balanced=False, pos_weight=1.0, seed=0):
    cw = "balanced" if balanced else None
    if name == "lr":
        return LogisticRegression(max_iter=3000, C=1.0, class_weight=cw)
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight=cw,
                                      n_jobs=1, random_state=seed)
    if name == "hgb":
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                              early_stopping=False, class_weight=cw,
                                              random_state=seed)
    if name == "xgb":
        return XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=5, subsample=0.8,
                             colsample_bytree=0.8, tree_method="hist", eval_metric="logloss",
                             scale_pos_weight=pos_weight if balanced else 1.0,
                             n_jobs=1, random_state=seed, verbosity=0)
    if name == "lgbm":
        return LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.8,
                              subsample_freq=1, colsample_bytree=0.8, class_weight=cw,
                              n_jobs=1, random_state=seed, verbose=-1)
    raise ValueError(name)
