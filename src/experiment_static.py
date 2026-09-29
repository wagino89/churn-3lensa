"""Static experiment: Lens A (statistical) + Lens B (profit); answers RQ1, RQ3, RQ4.

Examples:
    python -m src.experiment_static --quick
    python -m src.experiment_static --datasets telco bank --n-jobs 8
"""
import argparse
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold

from . import config
from .core import run_unit, stable_seed
from .datasets import load_static, subsample_to_rate
from .metrics import evaluate
from .models import available_models, make_model, make_preprocessor
from .resampling import NotAvailable, get_sampler, parse_technique


def _task(X, y, tr, te, tech, model, seed, meta):
    try:
        res = run_unit(X.iloc[tr], y[tr], X.iloc[te], y[te], tech, model, seed)
    except NotAvailable as e:
        return {**meta, "technique": tech, "model": model, "status": f"skipped: {e}"}
    except Exception as e:  # record the error, do not stop the whole experiment
        return {**meta, "technique": tech, "model": model, "status": f"error: {type(e).__name__}: {e}"}
    return {**meta, "technique": tech, "model": model, "status": "ok", **res}


def _leaky_unit(Xres, yres, tr, te, model, seed, meta):
    """One fold of the DELIBERATELY WRONG condition (SMOTE before splitting)."""
    t0 = time.time()
    try:
        m = make_model(model, seed=seed).fit(Xres[tr], yres[tr])
        res = evaluate(yres[te], m.predict_proba(Xres[te])[:, 1], 0.5, config.EMPC_PARAMS, config.TOP_FRACTION)
    except Exception as e:
        return {**meta, "status": f"error: {type(e).__name__}: {e}"}
    res.update({"n_train_fit": len(tr), "n_test": len(te), "test_churn_rate": float(yres[te].mean()),
                "seconds": round(time.time() - t0, 3)})
    return {**meta, "status": "ok", **res}


def _leaky_jobs(X, y, tech, models, repeats, folds, meta_base):
    """DELIBERATELY WRONG: preprocessing + SMOTE on the WHOLE dataset, then CV.
    Test folds contain synthetic samples -> measures the inflation caused by leakage (RQ3)."""
    _, ratio = parse_technique(tech)
    jobs = []
    Xall = make_preprocessor(X).fit_transform(X)
    for r in range(repeats):
        seed = stable_seed(meta_base["dataset"], meta_base["ir_level"], r, tech)
        Xres, yres = get_sampler("smote_leaky", ratio, config.K_NEIGHBORS, random_state=seed).fit_resample(Xall, y)
        skf = StratifiedKFold(folds, shuffle=True, random_state=stable_seed("cv", r))
        for k, (tr, te) in enumerate(skf.split(Xres, yres)):
            for mdl in models:
                meta = {**meta_base, "repeat": r, "fold": k, "technique": tech, "model": mdl}
                jobs.append(delayed(_leaky_unit)(Xres, yres, tr, te, mdl, seed, meta))
    return jobs


def _combine(parts_dir, pattern, out_path):
    files = sorted(parts_dir.glob(pattern))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def run(datasets, techniques, models, cv, ir_levels, n_jobs, out_path, data_dir=None, deadline=None):
    """Results are checkpointed per (dataset, imbalance level) in results/parts/. Existing
    parts are skipped, so the experiment can be spread over several sessions. If `deadline`
    (epoch seconds) is given, a new block is not started when it is expected to exceed it."""
    models = available_models(models)
    parts = out_path.parent / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    last_dur, stop = 0.0, False
    for ds in datasets:
        if stop:
            break
        try:
            X0, y0, meta = load_static(ds, data_dir)
        except FileNotFoundError as e:
            print(f"[skip] {e}")
            continue
        pd.DataFrame([meta]).to_csv(parts / f"meta_{ds}.csv", index=False)
        print(f"\n=== {ds}: n={meta['n']}, churn={meta['churn_rate']:.3f}, features={meta['n_features']}")
        for ir in ir_levels:
            if ir != "natural" and ir >= y0.mean():
                continue
            part = parts / f"{out_path.stem}__{ds}__{ir}.csv"
            if part.exists():
                print(f"  ir={ir}: already exists ({part.name}) -> skipped")
                continue
            if deadline and time.time() + 1.3 * last_dur > deadline:
                print(f"  [time] block {ds}/{ir} not started to stay within the session limit. Resume in the next session.")
                stop = True
                break
            X, y = subsample_to_rate(X0, y0, ir, stable_seed(ds, ir))
            base = {"dataset": ds, "ir_level": str(ir), "churn_rate": round(float(y.mean()), 4), "n": len(y)}
            jobs = []
            for r in range(cv["repeats"]):
                skf = StratifiedKFold(cv["folds"], shuffle=True, random_state=stable_seed("cv", r))
                for k, (tr, te) in enumerate(skf.split(X, y)):
                    for tech in techniques:
                        if tech.startswith("smote_leaky"):
                            continue
                        for mdl in models:
                            seed = stable_seed(ds, ir, r, k, tech, mdl)
                            jobs.append(delayed(_task)(X, y, tr, te, tech, mdl, seed,
                                                       {**base, "repeat": r, "fold": k}))
            for tech in [t for t in techniques if t.startswith("smote_leaky")]:
                jobs += _leaky_jobs(X, y, tech, models, cv["repeats"], cv["folds"], base)
            t0 = time.time()
            rows = Parallel(n_jobs=n_jobs, verbose=0)(jobs)
            pd.DataFrame(rows).to_csv(part, index=False)
            last_dur = time.time() - t0
            n_bad = sum(r.get("status") != "ok" for r in rows)
            print(f"  ir={ir}: {len(jobs)} units in {last_dur / 60:.1f} min ({n_bad} skipped/error) -> {part.name}")
    df = _combine(parts, f"{out_path.stem}__*.csv", out_path)
    metas = sorted(parts.glob("meta_*.csv"))
    if metas:
        pd.concat([pd.read_csv(f) for f in metas]).to_csv(out_path.with_name("datasets_summary.csv"), index=False)
    if len(df):
        bad = df[df["status"] != "ok"]
        print(f"\nCombined: {out_path} ({len(df)} rows; {len(bad)} skipped/error)")
        if len(bad):
            print(bad["status"].str.slice(0, 90).value_counts().head(10).to_string())
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=list(config.STATIC_DATASETS))
    p.add_argument("--techniques", nargs="+")
    p.add_argument("--models", nargs="+")
    p.add_argument("--quick", action="store_true", help="quick test mode (3 folds, 2 models)")
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    q = a.quick
    out = config.RESULTS_DIR / ("static_quick.csv" if q else "static_full.csv") if a.out is None else \
        __import__("pathlib").Path(a.out)
    run(a.datasets,
        a.techniques or (config.TECHNIQUES_QUICK if q else config.TECHNIQUES_FULL),
        a.models or (config.MODELS_QUICK if q else config.MODELS_FULL),
        config.CV_QUICK if q else config.CV_FULL,
        config.IR_LEVELS_QUICK if q else config.IR_LEVELS_FULL,
        a.n_jobs, out, a.data_dir)


if __name__ == "__main__":
    main()
