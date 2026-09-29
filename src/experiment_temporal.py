"""Temporal experiment: Lens C (out-of-time) versus a random split; answers RQ2.

Rolling-origin protocol: for every test cutoff c_test, the training data are all
snapshots with cutoff c such that c + horizon <= c_test (their labels are fully
observed before the test date). The "random" comparator uses the SAME rows but
splits them at random (stratified, identical test size) -- common practice in the
literature -- so the difference between the two measures the optimism of random splitting.

Churners are subsampled within every snapshot to the levels in config.RETAIL["ir_levels"]
(the natural 90-day churn rate of the panel is ~51%).

Example:
    python -m src.experiment_temporal --quick
"""
import argparse

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

from . import config
from .core import run_unit, stable_seed
from .datasets import build_retail_panel
from .experiment_static import _task
from .models import available_models

FEATURE_EXCLUDE = {"customer", "churn", "cutoff"}


def subsample_panel(panel, rate):
    """Remove churners at random WITHIN EACH snapshot so that the churn rate equals `rate` (deterministic)."""
    if rate == "natural":
        return panel
    keep = []
    for c, g in panel.groupby("cutoff"):
        pos, neg = g.index[g["churn"] == 1], g.index[g["churn"] == 0]
        n_pos = min(len(pos), int(round(rate * len(neg) / (1 - rate))))
        rng = np.random.default_rng(stable_seed("retail_ir", str(pd.Timestamp(c).date()), rate))
        keep.append(neg)
        keep.append(pd.Index(rng.choice(pos, size=n_pos, replace=False)))
    idx = keep[0].append(keep[1:]).sort_values()
    return panel.loc[idx].reset_index(drop=True)


def run(techniques, models, n_jobs, out_path, data_dir=None, max_folds=None, panel=None, deadline=None,
        ir_levels=None):
    """One rolling-origin pass for every imbalance level in `ir_levels`."""
    panel0 = build_retail_panel(data_dir) if panel is None else panel
    ir_levels = ir_levels or config.RETAIL.get("ir_levels", ["natural"])
    frames = []
    for ir in ir_levels:
        print(f"\n=== temporal lens, churn level: {ir}")
        frames.append(_run_level(techniques, models, n_jobs, out_path, subsample_panel(panel0, ir), ir,
                                 max_folds, deadline))
    parts = out_path.parent / "parts"
    files = sorted(parts.glob(f"{out_path.stem}_fold*.csv")) + sorted(parts.glob(f"{out_path.stem}__*_fold*.csv"))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Combined temporal results: {out_path} ({len(df)} rows)")
    return df


def _run_level(techniques, models, n_jobs, out_path, panel, ir, max_folds, deadline):
    models = available_models(models)
    cfg = config.RETAIL
    suffix = "" if ir == "natural" else f"__{ir}"
    panel_path = out_path.with_name(f"retail_panel_summary{suffix}.csv")
    (panel.groupby("cutoff").agg(n=("churn", "size"), churn_rate=("churn", "mean"))
     .to_csv(panel_path))
    feats = [c for c in panel.columns if c not in FEATURE_EXCLUDE]
    cutoffs = sorted(panel["cutoff"].unique())
    H = pd.Timedelta(days=cfg["horizon_days"])
    test_cutoffs = [c for c in cutoffs
                    if sum(pd.Timestamp(x) + H <= pd.Timestamp(c) for x in cutoffs) >= cfg["min_train_cutoffs"]]
    if max_folds:
        test_cutoffs = test_cutoffs[-max_folds:]
    print(f"Panel: {len(panel)} rows, {len(cutoffs)} cutoffs; {len(test_cutoffs)} rolling-origin folds")
    import time
    parts = out_path.parent / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    tag = out_path.stem + suffix  # the "natural" level keeps the original part names
    last_dur = 0.0
    for k, c_test in enumerate(test_cutoffs):
        part = parts / f"{tag}_fold{k:02d}.csv"
        if part.exists():
            print(f"  fold {k}: already exists -> skipped")
            continue
        if deadline and time.time() + 1.3 * last_dur > deadline:
            print(f"  [time] fold {k} not started to stay within the session limit. Resume in the next session.")
            break
        jobs = []
        c_test = pd.Timestamp(c_test)
        tr_mask = panel["cutoff"].map(lambda c: pd.Timestamp(c) + H <= c_test).to_numpy()
        te_mask = (panel["cutoff"] == c_test).to_numpy()
        sub = panel[tr_mask | te_mask].reset_index(drop=True)
        X, y = sub[feats], sub["churn"].to_numpy()
        tr_t, te_t = np.where(tr_mask[tr_mask | te_mask])[0], np.where(te_mask[tr_mask | te_mask])[0]
        # random comparator: same rows, same test size as the temporal fold
        tr_r, te_r = train_test_split(np.arange(len(y)), test_size=len(te_t), stratify=y,
                                      random_state=stable_seed("rand", k))
        for lens, tr, te in (("temporal", tr_t, te_t), ("random", tr_r, te_r)):
            for tech in techniques:
                if tech.startswith("smote_leaky"):
                    continue
                for mdl in models:
                    seed = stable_seed("retail", k, tech, mdl)
                    meta = {"dataset": "online_retail_ii", "lens_split": lens, "fold": k,
                            "test_cutoff": str(c_test.date()), "ir_level": str(ir)}
                    jobs.append(delayed(_task)(X, y, tr, te, tech, mdl, seed, meta))
        t0 = time.time()
        rows = Parallel(n_jobs=n_jobs)(jobs)
        pd.DataFrame(rows).to_csv(part, index=False)
        last_dur = time.time() - t0
        print(f"  fold {k} (test {c_test.date()}): {len(jobs)} units in {last_dur / 60:.1f} min")
    files = sorted(parts.glob(f"{tag}_fold*.csv"))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--techniques", nargs="+")
    p.add_argument("--models", nargs="+")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    q = a.quick
    from pathlib import Path
    out = Path(a.out) if a.out else config.RESULTS_DIR / ("temporal_quick.csv" if q else "temporal_full.csv")
    run(a.techniques or (config.TECHNIQUES_QUICK if q else config.TECHNIQUES_FULL),
        a.models or (config.MODELS_QUICK if q else config.MODELS_FULL),
        a.n_jobs, out, a.data_dir, max_folds=3 if q else None)


if __name__ == "__main__":
    main()
