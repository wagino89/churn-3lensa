"""One experimental unit: (training data, test data, technique, model) -> metrics.

Order of operations inside every fold (leakage-safe protocol):
 1. preprocessing is fitted ONLY on the training part and then applied to the test part
 2. resampling is applied ONLY to the preprocessed training data
 3. the model is fitted and scores the original test data (never resampled)
"""
import time
import zlib

import numpy as np
from sklearn.model_selection import train_test_split

from . import config
from .metrics import evaluate, best_f1_threshold
from .models import make_model, make_preprocessor
from .resampling import EasyEnsemble, get_sampler, parse_technique, SAMPLER_NAMES


def stable_seed(*parts):
    """Deterministic seed derived from the experimental factors (independent of run order)."""
    return zlib.crc32("|".join(map(str, parts)).encode()) % (2**31 - 1)


def fit_predict(Xtr, ytr, Xte, technique, model_name, seed):
    """Xtr/Xte are already preprocessed arrays. Returns (scores, threshold, n_train)."""
    name, ratio = parse_technique(technique)
    pos_weight = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    threshold = 0.5

    if name in SAMPLER_NAMES and name != "smote_leaky":
        sampler = get_sampler(name, ratio, config.K_NEIGHBORS, config.DIRICHLET_M, seed)
        Xtr, ytr = sampler.fit_resample(Xtr, ytr)
        model = make_model(model_name, seed=seed)
    elif name == "class_weight":
        model = make_model(model_name, balanced=True, pos_weight=pos_weight, seed=seed)
    elif name == "easy_ensemble":
        model = EasyEnsemble(make_model(model_name, seed=seed), config.EASY_ENSEMBLE_N, seed)
    elif name in ("none", "smote_leaky", "threshold"):
        model = make_model(model_name, seed=seed)
    else:
        raise ValueError(technique)

    if name == "threshold":
        # choose the F1-optimal threshold on a 20% validation split, then refit on the full fold
        Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=0.2, stratify=ytr, random_state=seed)
        threshold = best_f1_threshold(yv, make_model(model_name, seed=seed).fit(Xa, ya)
                                      .predict_proba(Xv)[:, 1])
    model.fit(Xtr, ytr)
    return model.predict_proba(Xte)[:, 1], threshold, len(ytr)


def run_unit(Xtr_df, ytr, Xte_df, yte, technique, model_name, seed, empc_params=None):
    t0 = time.time()
    pre = make_preprocessor(Xtr_df).fit(Xtr_df)
    Xtr, Xte = pre.transform(Xtr_df), pre.transform(Xte_df)
    scores, thr, n_train = fit_predict(Xtr, ytr, Xte, technique, model_name, seed)
    res = evaluate(yte, scores, thr, empc_params or config.EMPC_PARAMS, config.TOP_FRACTION)
    res.update({"n_train_fit": n_train, "n_test": len(yte), "test_churn_rate": float(np.mean(yte)),
                "seconds": round(time.time() - t0, 3)})
    return res
