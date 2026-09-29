"""Reproduce every number, table and figure reported in the paper from the raw results.

    python -m src.paper                       # reads results/raw, writes results/paper
    python -m src.paper --raw results/raw --out results/paper

Inputs (results/raw):
    static_full.csv                    30,000 model fits on the five static datasets
    temporal_full.csv                  2,640 model fits on the Online Retail II panel
    datasets_summary.csv               realized size, churn rate and features per dataset
    retail_panel_summary*.csv          rows and churn rate per monthly snapshot and churn level
Outputs (results/paper):
    numbers.json                       every number quoted in the text of the paper
    tables/Table{1,3,4,5,6}.csv        the tables of the paper; tables/tables.md for reading
    tables/wilcoxon_static.csv         all Wilcoxon tests (dataset x level x classifier x technique x metric)
    tables/wilcoxon_temporal.csv       strategy vs baseline under each split (Holm within level, split, metric)
    tables/leakage.csv, tables/temporal_rank_agreement.csv
    figures/Figure{1..4}.{png,eps}     grayscale figures, 600 dpi
"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import config
from .analysis import (PRETTY, deltas_vs_baseline, figures_journal, friedman_ranks, holm, lens_agreement,
                       leakage_inflation, load, wilcoxon_table)

warnings.filterwarnings("ignore")
LEVELS = ["natural", "0.1", "0.05", "0.02"]
T_LEVELS = ["natural", "0.2", "0.1"]
TECH_ORDER = ["class_weight", "threshold", "ros@1.0", "rus@1.0", "smote@1.0", "smote@0.5", "adasyn@1.0",
              "smote_enn@1.0", "dirichlet@1.0", "easy_ensemble"]
DS_LABEL = {"telco": "IBM Telco", "bank": "Bank churn", "bankchurners": "Credit card",
            "iranian": "Iranian churn", "telecom_bigml": "Telecom (BigML)"}


def _paired(df, t1, t2, metric, key=("dataset", "ir_level", "model", "repeat", "fold")):
    a = df[df.technique == t1].set_index(list(key))[metric]
    b = df[df.technique == t2].set_index(list(key))[metric]
    x = (a - b).dropna()
    return dict(med=float(x.median()), win=float((x > 0).mean()), p=float(stats.wilcoxon(x).pvalue), n=len(x))


def temporal_vs_baseline(td):
    """Each strategy vs the baseline over the 40 (test month x classifier) pairs, per level and split.
    Holm correction across strategies within (level, split, metric)."""
    rows = []
    for (ir, lens), g in td.groupby(["ir_level", "lens_split"]):
        base = g[g.technique == "none"].set_index(["fold", "model"])
        for tech, gg in g[g.technique != "none"].groupby("technique"):
            x = gg.set_index(["fold", "model"])
            for m in ["roc_auc", "pr_auc", "f1", "mcc", "empc"]:
                d = (x[m] - base[m]).dropna()
                p = stats.wilcoxon(d).pvalue if (d != 0).any() else 1.0
                rows.append(dict(ir_level=ir, lens_split=lens, technique=tech, metric=m, median_delta=d.median(),
                                 win_rate=(d > 0).mean(), n_pairs=len(d), p=p))
    t = pd.DataFrame(rows)
    t["p_holm"] = np.nan
    for _, idx in t.groupby(["ir_level", "lens_split", "metric"]).groups.items():
        t.loc[idx, "p_holm"] = holm(t.loc[idx, "p"])
    return t


FIG_W = 3.3  # single-column width of the IJC layout (inches); figures are drawn at print size


def figure_protocol(fig_dir):
    """Figure 1: leakage-safe protocol and the three lenses. Drawn at print size, all text 10 pt."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                         "font.size": 10, "mathtext.fontset": "stix"})
    W, H = FIG_W, 4.35                      # inches; one data unit = one inch
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    LW = 0.8

    def box(x0, y0, w, h, text, fill="white", bold=False, italic=False):
        ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=0.05",
                                    fc=fill, ec="black", lw=LW))
        ax.text(x0 + w / 2, y0 + h / 2, text, ha="center", va="center", fontsize=10, linespacing=1.1,
                fontweight="bold" if bold else "normal", style="italic" if italic else "normal")
        return dict(l=x0, r=x0 + w, b=y0, t=y0 + h, cx=x0 + w / 2, cy=y0 + h / 2)

    def arrow(x, y1, y2):
        ax.annotate("", xy=(x, y2), xytext=(x, y1),
                    arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.18", lw=LW, color="black",
                                    shrinkA=0, shrinkB=0))

    M = 0.06                                # outer margin
    G = 0.17                                # vertical gap between rows (arrow length)
    data = box(M, 3.93, W - 2 * M, 0.36, "5 churn datasets  +  1 transaction log", fill="0.88", bold=True)
    split = box(M, data["b"] - G - 0.36, W - 2 * M, 0.36, "Split: 5×5 stratified CV  |  rolling origin")

    # training-fold frame (left) and test fold (right)
    fx0, fw = M, 2.14
    ft = split["b"] - G
    fb = ft - 1.62
    ax.add_patch(FancyBboxPatch((fx0, fb), fw, ft - fb, boxstyle="round,pad=0,rounding_size=0.06",
                                fc="0.96", ec="0.35", lw=LW, ls=(0, (4, 2))))
    ax.text(fx0 + 0.13, (ft + fb) / 2, "training fold only", rotation=90, ha="center", va="center",
            fontsize=10, style="italic", color="0.2")
    bx0, bw, bh = fx0 + 0.27, fw - 0.37, 0.40
    gap = (ft - fb - 3 * bh) / 4
    b1 = box(bx0, ft - gap - bh, bw, bh, "Fit preprocessing\n(impute, scale, encode)")
    b2 = box(bx0, b1["b"] - gap - bh, bw, bh, "Resample\n(12 strategies)")
    b3 = box(bx0, b2["b"] - gap - bh, bw, bh, "Fit classifier\n(5 models)")
    tx0 = fx0 + fw + 0.12
    test = box(tx0, fb, W - M - tx0, ft - fb, "Test fold\n(never\nresampled)")

    score = box(M, fb - G - 0.36, W - 2 * M, 0.36, "Score the test fold")
    lw_ = (W - 2 * M - 2 * 0.08) / 3
    lens = [box(M + k * (lw_ + 0.08), score["b"] - G - 0.46, lw_, 0.46, t, fill="0.88")
            for k, t in enumerate(["Lens A\nstatistical", "Lens B\nprofit (EMPC)", "Lens C\nout-of-time"])]

    arrow(data["cx"], data["b"], split["t"])
    arrow(b1["cx"], split["b"], ft)
    arrow(test["cx"], split["b"], test["t"])
    arrow(b1["cx"], ft, b1["t"]) if gap > 0.12 else None
    arrow(b1["cx"], b1["b"], b2["t"])
    arrow(b2["cx"], b2["b"], b3["t"])
    arrow(b3["cx"], b3["b"], score["t"])
    arrow(test["cx"], test["b"], score["t"])
    for L in lens:
        arrow(L["cx"], score["b"], L["t"])
    ax.text(W / 2, lens[0]["b"] - 0.16, "Leaky control (RQ3): SMOTE before the split",
            ha="center", va="center", fontsize=10, style="italic")
    ax.set_ylim(lens[0]["b"] - 0.30, H)   # trim empty space below the note
    fig.set_size_inches(W, H - (lens[0]["b"] - 0.30))
    fig.savefig(fig_dir / "Figure1.png", dpi=600)
    fig.savefig(fig_dir / "Figure1.eps")
    plt.close(fig)


def figure_temporal(td, fig_dir):
    """Figure 3: optimism of random splitting by classifier and churn level."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                         "font.size": 10})
    k = ["ir_level", "fold", "technique", "model"]
    t = td[td.lens_split == "temporal"].set_index(k)
    r = td[td.lens_split == "random"].set_index(k)
    models = ["lr", "rf", "hgb", "xgb", "lgbm"]
    ml = {"lr": "LR", "rf": "RF", "hgb": "HGB", "xgb": "XGB", "lgbm": "LGBM"}
    nat = td[(td.lens_split == "temporal") & (td.ir_level == "natural")].groupby("fold").test_churn_rate.mean().mean()
    levels = [("natural", f"natural (~{100 * nat:.0f}%)", "0.25", ""), ("0.2", "20%", "0.6", "///"),
              ("0.1", "10%", "white", "")]
    fig, axes = plt.subplots(2, 1, figsize=(FIG_W, 4.2), sharex=True)
    for ax, met in zip(axes, ["roc_auc", "pr_auc"]):
        for j, (ir, lab, fc, hatch) in enumerate(levels):
            d = (r[met] - t[met]).xs(ir, level="ir_level")
            vals = [float(d.xs(mo, level="model").mean()) for mo in models]
            ax.bar(np.arange(5) + (j - 1) * 0.26, vals, 0.26, color=fc, edgecolor="black", linewidth=0.6,
                   hatch=hatch, label=lab)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.grid(axis="y", color="0.85", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.set_ylabel("Optimism, Δ" + ("ROC-AUC" if met == "roc_auc" else "PR-AUC"))
    axes[-1].set_xticks(range(5), [ml[m] for m in models])
    axes[-1].set_xlabel("Classifier")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, title="Churn rate", title_fontsize=10, fontsize=10, frameon=False, ncol=3, loc="upper center",
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    for ext in ("png", "eps"):
        fig.savefig(fig_dir / f"Figure3.{ext}", dpi=600)
    plt.close(fig)


def fmt(v, d=3, sign=True):
    s = f"{v:+.{d}f}" if sign else f"{v:.{d}f}"
    return f"{0:.{d}f}" if sign and float(s) == 0 else s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=str(config.RESULTS_DIR / "raw"))
    ap.add_argument("--out", default=str(config.RESULTS_DIR / "paper"))
    a = ap.parse_args()
    raw, out = Path(a.raw), Path(a.out)
    tab, fig = out / "tables", out / "figures"
    tab.mkdir(parents=True, exist_ok=True)
    fig.mkdir(parents=True, exist_ok=True)

    df = load(raw / "static_full.csv")
    td_all = pd.read_csv(raw / "temporal_full.csv")
    td_all["ir_level"] = td_all["ir_level"].astype(str)
    td = td_all[td_all.status == "ok"].copy()
    ds = pd.read_csv(raw / "datasets_summary.csv")
    N = {}
    N["n_static"] = int(len(pd.read_csv(raw / "static_full.csv", usecols=["status"])))
    N["n_temporal"] = int(len(td_all))
    N["n_temporal_err"] = int((td_all.status != "ok").sum())
    N["temporal_errors"] = td_all[td_all.status != "ok"].groupby(["ir_level", "technique"]).size() \
        .rename("n").reset_index().to_dict("records")
    N["datasets"] = ds.set_index("dataset")[["n", "churn_rate", "n_features"]].to_dict("index")
    ps = pd.read_csv(raw / "retail_panel_summary.csv")
    N["ret_rows"] = int(ps.n.sum())
    N["ret_rate"] = float(np.average(ps.churn_rate, weights=ps.n))
    N["ret_rate_min"], N["ret_rate_max"] = float(ps.churn_rate.min()), float(ps.churn_rate.max())
    for r in ("0.2", "0.1"):
        N[f"ret_rows_{r}"] = int(pd.read_csv(raw / f"retail_panel_summary__{r}.csv").n.sum())

    # ---------------------------------------------------------------- RQ1
    nat = df[df.ir_level == "natural"]
    b = nat[nat.technique == "none"].groupby(["dataset", "model"]).roc_auc.mean()
    N["base_auc_best"] = b.groupby("dataset").max().round(3).to_dict()
    N["base_auc_best_model"] = b.groupby("dataset").idxmax().map(lambda x: x[1]).to_dict()
    w = wilcoxon_table(deltas_vs_baseline(df))
    w["ir_level"] = w["ir_level"].astype(str)
    w.to_csv(tab / "wilcoxon_static.csv", index=False)
    wn = w[w.ir_level == "natural"]
    N["n_blocks"] = int(wn.groupby(["dataset", "model"]).ngroups)
    t3 = wn.groupby(["technique", "metric"]).agg(md=("median_delta", "median"), win=("win_rate", "mean"))
    N["t3"] = {f"{a_}|{m}": [float(r.md), float(r.win)] for (a_, m), r in t3.iterrows()}
    v = wn.groupby(["technique", "metric", "verdict"]).size().unstack(fill_value=0)
    N["verdict"] = {f"{a_}|{m}": {k: int(x) for k, x in r.items()} for (a_, m), r in v.iterrows()}
    ag = lens_agreement(nat)
    N["tau"] = {c: float(ag[c].median()) for c in ag.columns if c.startswith("tau")}
    N["best_change"] = float(ag.best_changes_f1_vs_empc.mean())
    N["best_change_by_ds"] = ag.groupby("dataset").best_changes_f1_vs_empc.mean().to_dict()
    fr = friedman_ranks(nat)
    fr.to_csv(tab / "friedman_natural.csv", index=False)
    N["friedman"] = {m: {"p": float(g.friedman_p.iloc[0]), "top": list(g.technique[:3]),
                         "bottom": list(g.technique[-2:])} for m, g in fr.groupby("metric")}
    N["smote_vs_thr_f1"] = _paired(nat, "smote@1.0", "threshold", "f1")
    N["f1_mean"] = {t: float(nat[nat.technique == t].f1.mean())
                    for t in ["none", "threshold", "smote@1.0", "ros@1.0", "class_weight"]}
    N["dir_vs_smote"] = {m: _paired(df, "dirichlet@1.0", "smote@1.0", m) for m in ["f1", "roc_auc", "pr_auc", "empc"]}
    ee = wn[wn.technique == "easy_ensemble"]
    N["ee_by_ds"] = ee.pivot_table(index="dataset", columns="metric", values="median_delta") \
        [["roc_auc", "f1", "empc"]].round(4).to_dict("index")
    N["brier_worse"] = {t: int(((wn.technique == t) & (wn.metric == "brier") & (wn.verdict == "worse")).sum())
                        for t in sorted(wn.technique.unique())}

    # ---------------------------------------------------------------- RQ3
    lk = leakage_inflation(df)
    lk.to_csv(tab / "leakage.csv", index=False)
    g = lk.groupby(["dataset", "ir_level"]).mean(numeric_only=True)
    cols = ["roc_auc_correct", "roc_auc_leaky", "roc_auc_inflation", "f1_correct", "f1_leaky", "f1_inflation"]
    N["leak"] = {f"{d}|{i}": {c: float(r[c]) for c in cols} for (d, i), r in g.iterrows()}
    N["leak_nat_mean"] = {c: float(lk[lk.ir_level == "natural"][c].mean()) for c in ["roc_auc_inflation", "f1_inflation"]}
    N["leak_002_mean"] = {c: float(lk[lk.ir_level == "0.02"][c].mean()) for c in ["roc_auc_inflation", "f1_inflation"]}

    # ---------------------------------------------------------------- RQ4
    mod = w.groupby(["technique", "metric", "ir_level"]).median_delta.median().unstack()[LEVELS]
    N["mod"] = {f"{a_}|{m}": r.round(4).to_dict() for (a_, m), r in mod.iterrows()}
    better = w[w.verdict == "better"].groupby(["technique", "metric", "ir_level"]).size()
    tot = w.groupby(["technique", "metric", "ir_level"]).size()
    N["mod_better"] = {f"{a_}|{m}|{i}": f"{int(better.get((a_, m, i), 0))}/{int(n)}" for (a_, m, i), n in tot.items()}

    # ---------------------------------------------------------------- RQ2
    k = ["ir_level", "fold", "technique", "model"]
    t = td[td.lens_split == "temporal"].set_index(k)
    r = td[td.lens_split == "random"].set_index(k)
    N["opt"] = {}
    for ir in T_LEVELS:
        for m in ["roc_auc", "pr_auc", "f1", "mcc", "empc", "brier", "prec_at_top"]:
            x = (r[m] - t[m]).xs(ir, level="ir_level").dropna()
            N["opt"][f"{ir}|{m}"] = dict(mean=float(x.mean()), med=float(x.median()), share=float((x > 0).mean()),
                                         p=float(stats.wilcoxon(x).pvalue), n=len(x))
    mm = td.groupby(["ir_level", "model", "lens_split"]).roc_auc.mean().unstack()
    N["opt_model_auc"] = {f"{i}|{m}": [float(q.temporal), float(q.random)] for (i, m), q in mm.iterrows()}
    rows = []
    for (ir, mdl), g in td.groupby(["ir_level", "model"]):
        for m in ["roc_auc", "pr_auc", "f1", "empc"]:
            s = g.groupby(["lens_split", "technique"])[m].mean().unstack(0).dropna()
            rows.append(dict(ir_level=ir, model=mdl, metric=m, tau=stats.kendalltau(s.temporal, s.random).statistic,
                             same_best=bool(s.temporal.idxmax() == s.random.idxmax()),
                             best_temporal=s.temporal.idxmax(), best_random=s.random.idxmax()))
    kk = pd.DataFrame(rows)
    kk.to_csv(tab / "temporal_rank_agreement.csv", index=False)
    sub = kk[kk.ir_level != "natural"]
    N["rank_tr_sub"] = {m: dict(tau=float(g.tau.mean()), same=int(g.same_best.sum()), n=len(g))
                        for m, g in sub.groupby("metric")}
    tw = temporal_vs_baseline(td)
    tw.to_csv(tab / "wilcoxon_temporal.csv", index=False)
    N["tdelta"] = {f"{x.ir_level}|{x.lens_split}|{x.technique}|{x.metric}": [float(x.median_delta), float(x.win_rate),
                                                                            float(x.p), float(x.p_holm)]
                   for x in tw.itertuples()}
    oot = tw[(tw.lens_split == "temporal") & (tw.metric == "empc") & (tw.ir_level != "natural")]
    N["oot_empc_min_p_holm"] = oot.sort_values("p_holm").iloc[0][["ir_level", "technique", "median_delta", "p",
                                                                  "p_holm"]].to_dict()
    N["oot_empc_n_sig"] = int((oot.p_holm < 0.05).sum())
    tt = td[td.ir_level != "natural"]
    q = tt.groupby(["technique", "lens_split"])[["pr_auc", "empc"]].mean().unstack()
    N["t5"] = {tech: {f"{m}|{l}": float(q.loc[tech, (m, l)]) for m in ["pr_auc", "empc"]
                      for l in ["temporal", "random"]} for tech in q.index}

    json.dump(N, open(out / "numbers.json", "w"), indent=1, default=str)

    # ---------------------------------------------------------------- tables
    T = {}
    T["Table1"] = pd.DataFrame(
        [[DS_LABEL[d], f"{N['datasets'][d]['n']:,}", f"{100 * N['datasets'][d]['churn_rate']:.1f}%",
          N["datasets"][d]["n_features"]] for d in DS_LABEL]
        + [["Online Retail II (snapshots)", f"{N['ret_rows']:,}", f"{100 * N['ret_rate']:.1f}%", 15]],
        columns=["Dataset", "Rows", "Churn rate", "Features"])
    T["Table3"] = pd.DataFrame(
        [[PRETTY[x]] + [f"{fmt(N['t3'][f'{x}|{m}'][0])} ({100 * N['t3'][f'{x}|{m}'][1]:.0f})"
                        for m in ["roc_auc", "pr_auc", "f1", "empc"]] for x in TECH_ORDER],
        columns=["Strategy", "ROC-AUC", "PR-AUC", "F1", "EMPC"])
    vc = lambda x, m: f"{N['verdict'][f'{x}|{m}'].get('better', 0)} / {N['verdict'][f'{x}|{m}'].get('worse', 0)}"
    T["Table4"] = pd.DataFrame([[PRETTY[x]] + [vc(x, m) for m in ["roc_auc", "f1", "empc", "brier"]]
                                for x in TECH_ORDER], columns=["Strategy", "ROC-AUC", "F1", "EMPC", "Brier"])
    T["Table5"] = pd.DataFrame(
        [[PRETTY[x], f"{N['t5'][x]['pr_auc|temporal']:.3f}", f"{N['t5'][x]['pr_auc|random']:.3f}",
          f"{N['t5'][x]['empc|temporal']:.2f}", f"{N['t5'][x]['empc|random']:.2f}"] for x in ["none"] + TECH_ORDER],
        columns=["Strategy", "PR-AUC OOT", "PR-AUC random", "EMPC OOT", "EMPC random"])
    T["Table6"] = pd.DataFrame(
        [[DS_LABEL[d], f"{N['leak'][f'{d}|natural']['roc_auc_correct']:.3f}",
          f"{N['leak'][f'{d}|natural']['roc_auc_leaky']:.3f}", fmt(N['leak'][f'{d}|natural']['roc_auc_inflation']),
          fmt(N['leak'][f'{d}|natural']['f1_inflation'])] for d in DS_LABEL],
        columns=["Dataset", "Correct", "Leaky", "ΔROC-AUC", "ΔF1"])
    captions = {
        "Table1": "Datasets used in the study",
        "Table3": "Median difference to the baseline (win rate, %) at the natural churn rate",
        "Table4": f"Settings (of {N['n_blocks']}) in which a strategy was significantly better / worse than the "
                  "baseline (Wilcoxon, Holm-adjusted p < 0.05)",
        "Table5": "PR-AUC and EMPC under out-of-time (OOT) and random splits, retail panel, mean over the 20% and "
                  "10% churn levels, eight test months and five classifiers",
        "Table6": "ROC-AUC of correct and leaky SMOTE pipelines and inflation of ROC-AUC and F1 (natural churn rate)",
    }
    md = ["# Tables of the paper (generated by `python -m src.paper`)\n",
          "Table 2 (strategies and settings) is descriptive; see `src/config.py`.\n"]
    for name, t_ in T.items():
        t_.to_csv(tab / f"{name}.csv", index=False)
        md += [f"\n## {name.replace('Table', 'Table ')}. {captions[name]}\n",
               "| " + " | ".join(t_.columns) + " |", "|" + "---|" * len(t_.columns)]
        md += ["| " + " | ".join(map(str, row)) + " |" for row in t_.values]
    (tab / "tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # ---------------------------------------------------------------- figures
    figure_protocol(fig)
    figures_journal(w, fig)
    for src_, dst in (("fig_j2_winrate", "Figure2"), ("fig_j3_imbalance", "Figure4")):
        for ext in ("png", "eps"):
            (fig / f"{src_}.{ext}").replace(fig / f"{dst}.{ext}")
    figure_temporal(td, fig)
    print(f"numbers : {out / 'numbers.json'}\ntables  : {tab}\nfigures : {fig}")


if __name__ == "__main__":
    main()
