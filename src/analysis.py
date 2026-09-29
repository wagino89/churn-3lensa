"""Statistical analysis, tables and exploratory figures for RQ1-RQ4.

Example:
    python -m src.analysis --static results/raw/static_full.csv --temporal results/raw/temporal_full.csv --out results/analysis
Output: <out>/tables/*.csv and <out>/figures/*.png
The exact numbers, tables and figures of the paper (natural churn rate unless stated
otherwise) are produced by src/paper.py.
"""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import config
from .metrics import METRICS, HIGHER_IS_BETTER

KEY = ["dataset", "ir_level", "model"]
PAIR = KEY + ["repeat", "fold"]
EXCLUDE_FROM_RANKING = {"smote_leaky@1.0"}

# categorical palette + text/surface tokens for the exploratory (colour) figures
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
     "magenta": "#e87ba4", "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e",
     "grid": "#e4e3df", "mid": "#f0efec"}
SERIES = [C["blue"], C["orange"], C["aqua"], C["yellow"], C["magenta"]]
MARKERS = ["o", "s", "^", "D", "v"]


def holm(pvals):
    p = np.asarray(pvals, float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    idx = np.argsort(p[ok])
    m = ok.sum()
    adj = np.maximum.accumulate((m - np.arange(m)) * p[ok][idx])
    tmp = np.empty(m)
    tmp[idx] = np.minimum(adj, 1.0)
    out[ok] = tmp
    return out


def load(path):
    df = pd.read_csv(path)
    if "status" in df:
        df = df[df["status"] == "ok"].copy()
    df["ir_level"] = df["ir_level"].astype(str)
    return df


# ------------------------------------------------------------------ RQ1 ----
def deltas_vs_baseline(df, baseline="none"):
    """Paired differences (technique - baseline) on the same fold."""
    honest = df[~df["technique"].isin(EXCLUDE_FROM_RANKING)]
    base = honest[honest["technique"] == baseline].set_index(PAIR)[METRICS]
    rows = []
    for tech, g in honest[honest["technique"] != baseline].groupby("technique"):
        d = g.set_index(PAIR)[METRICS] - base.reindex(g.set_index(PAIR).index)
        d["technique"] = tech
        rows.append(d.reset_index())
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def wilcoxon_table(deltas):
    rows = []
    for (ds, ir, mdl, tech), g in deltas.groupby(KEY + ["technique"]):
        for m in METRICS:
            x = g[m].dropna().to_numpy()
            if not HIGHER_IS_BETTER[m]:
                x = -x
            p = stats.wilcoxon(x).pvalue if len(x) >= 5 and np.any(x != 0) else np.nan
            rows.append({"dataset": ds, "ir_level": ir, "model": mdl, "technique": tech, "metric": m,
                         "median_delta": float(np.median(x)) if len(x) else np.nan,
                         "win_rate": float(np.mean(x > 0)) if len(x) else np.nan,
                         "n_pairs": len(x), "p": p})
    t = pd.DataFrame(rows)
    t["p_holm"] = np.nan
    for _, idx in t.groupby(KEY + ["metric"]).groups.items():
        t.loc[idx, "p_holm"] = holm(t.loc[idx, "p"])
    t["verdict"] = np.select([(t["p_holm"] < 0.05) & (t["median_delta"] > 0),
                              (t["p_holm"] < 0.05) & (t["median_delta"] < 0)],
                             ["better", "worse"], "no difference")
    return t


def lens_agreement(df):
    """Kendall tau between the technique rankings produced by different metrics."""
    honest = df[~df["technique"].isin(EXCLUDE_FROM_RANKING)]
    means = honest.groupby(KEY + ["technique"])[METRICS].mean().reset_index()
    pairs = [("roc_auc", "empc"), ("pr_auc", "empc"), ("f1", "empc"), ("mcc", "empc"), ("roc_auc", "f1")]
    rows = []
    for key, g in means.groupby(KEY):
        if len(g) < 3:
            continue
        rec = dict(zip(KEY, key))
        for a, b in pairs:
            rec[f"tau_{a}_vs_{b}"] = stats.kendalltau(g[a], g[b]).statistic
        for m in ("roc_auc", "f1", "empc"):
            rec[f"best_by_{m}"] = g.loc[g[m].idxmax(), "technique"]
        rec["best_changes_f1_vs_empc"] = rec["best_by_f1"] != rec["best_by_empc"]
        rows.append(rec)
    return pd.DataFrame(rows)


def friedman_ranks(df):
    honest = df[~df["technique"].isin(EXCLUDE_FROM_RANKING)]
    means = honest.groupby(KEY + ["technique"])[METRICS].mean().reset_index()
    out = []
    for m in METRICS:
        piv = means.pivot_table(index=KEY, columns="technique", values=m).dropna()
        if piv.shape[0] < 2 or piv.shape[1] < 3:
            continue
        ranks = (-piv if HIGHER_IS_BETTER[m] else piv).rank(axis=1)
        chi, p = stats.friedmanchisquare(*[piv[c] for c in piv.columns])
        r = ranks.mean().sort_values()
        out.append(pd.DataFrame({"metric": m, "technique": r.index, "avg_rank": r.values,
                                 "friedman_chi2": chi, "friedman_p": p, "n_blocks": len(piv)}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ------------------------------------------------------------------ RQ3 ----
def leakage_inflation(df):
    sub = df[df["technique"].isin(["smote@1.0", "smote_leaky@1.0"])]
    if sub["technique"].nunique() < 2:
        return pd.DataFrame()
    m = sub.groupby(KEY + ["technique"])[["roc_auc", "pr_auc", "f1", "mcc"]].mean().unstack("technique")
    out = pd.DataFrame(index=m.index)
    for met in ["roc_auc", "pr_auc", "f1", "mcc"]:
        out[f"{met}_correct"] = m[(met, "smote@1.0")]
        out[f"{met}_leaky"] = m[(met, "smote_leaky@1.0")]
        out[f"{met}_inflation"] = out[f"{met}_leaky"] - out[f"{met}_correct"]
    return out.reset_index()


# ------------------------------------------------------------------ RQ4 ----
def ir_moderator(wtab):
    t = wtab.groupby(["dataset", "ir_level", "technique", "metric"])["median_delta"].median().reset_index()
    return t.pivot_table(index=["dataset", "technique", "metric"], columns="ir_level",
                         values="median_delta").reset_index()


# ------------------------------------------------------------------ RQ2 ----
def temporal_optimism(tdf):
    if "ir_level" not in tdf:
        tdf = tdf.assign(ir_level="natural")
    tdf = tdf.assign(ir_level=tdf["ir_level"].astype(str))
    m = tdf.groupby(["lens_split", "ir_level", "technique", "model"])[METRICS].mean()
    if not {"random", "temporal"} <= set(m.index.get_level_values(0)):
        return pd.DataFrame()
    diff = (m.loc["random"] - m.loc["temporal"]).add_suffix("_optimism")
    both = m.loc["temporal"].add_suffix("_temporal").join(m.loc["random"].add_suffix("_random")).join(diff)
    return both.reset_index()


def temporal_deltas(tdf):
    out = []
    for lens, g in tdf.groupby("lens_split"):
        g = g.copy()
        g["repeat"] = 0
        w = wilcoxon_table(deltas_vs_baseline(g))
        w["lens_split"] = lens
        out.append(w)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ------------------------------------------------------- Exploratory figures ----
def _style(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor(C["surface"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C["grid"])
    ax.tick_params(colors=C["ink2"], labelsize=9)
    ax.grid(axis="y", color=C["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, loc="left", fontsize=11, color=C["ink"], fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel, color=C["ink2"], fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=C["ink2"], fontsize=9)


def figures(wtab, agree, leak, tdelta, topt, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    fig_dir.mkdir(parents=True, exist_ok=True)
    if not len(wtab):  # temporal results only (no static experiment)
        wtab = pd.DataFrame(columns=["ir_level", "technique", "metric", "win_rate", "median_delta"])
    plt.rcParams.update({"figure.facecolor": C["surface"], "savefig.facecolor": C["surface"],
                         "font.size": 9})
    labels = {"roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "f1": "F1", "mcc": "MCC", "brier": "Brier",
              "prec_at_top": "Prec@10%", "empc": "EMPC"}

    # F1: win rate of each technique vs the baseline per metric (natural rate) -- RQ1
    nat = wtab[wtab["ir_level"] == "natural"]
    if len(nat):
        piv = nat.groupby(["technique", "metric"])["win_rate"].mean().unstack()[METRICS]
        cmap = LinearSegmentedColormap.from_list("div", [C["orange"], C["mid"], C["blue"]])
        fig, ax = plt.subplots(figsize=(8, 0.45 * len(piv) + 1.6))
        im = ax.imshow(piv.values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(METRICS)), [labels[m] for m in METRICS])
        ax.set_yticks(range(len(piv)), piv.index)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=8, color=C["ink"])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title("How often does a technique beat the baseline 'none'?\n"
                     "(share of folds; >50% blue = wins, <50% orange = loses)",
                     loc="left", fontsize=10, color=C["ink"])
        cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cb.ax.tick_params(labelsize=8, colors=C["ink2"])
        fig.tight_layout()
        fig.savefig(fig_dir / "fig1_winrate_by_metric.png", dpi=200)
        plt.close(fig)

    # F2: agreement of rankings between lenses (Kendall tau)
    cols = [c for c in agree.columns if c.startswith("tau_")] if len(agree) else []
    if cols:
        fig, ax = plt.subplots(figsize=(7, 3.4))
        rng = np.random.default_rng(0)
        for i, c in enumerate(cols):
            v = agree[c].dropna().to_numpy()
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=22, color=C["blue"], alpha=0.55,
                       edgecolor=C["surface"], linewidth=0.8)
            ax.hlines(np.median(v), i - 0.25, i + 0.25, color=C["ink"], linewidth=2)
            ax.text(i, np.median(v) + 0.07, f"median {np.median(v):.2f}", ha="center", va="bottom", fontsize=8,
                    color=C["ink"], bbox=dict(boxstyle="round,pad=0.15", fc=C["surface"], ec="none", alpha=0.9))
        ax.axhline(0, color=C["ink2"], linewidth=0.8)
        ax.set_xticks(range(len(cols)), [" vs\n".join(labels[x] for x in c.replace("tau_", "").split("_vs_")) for c in cols])
        ax.set_ylim(-1.05, 1.05)
        _style(ax, "Agreement of technique rankings between metrics (Kendall τ)", None, "τ (1 = identical ranking)")
        fig.tight_layout()
        fig.savefig(fig_dir / "fig2_lens_agreement.png", dpi=200)
        plt.close(fig)

    # F3: leakage inflation
    if len(leak):
        g = leak[leak["ir_level"] == "natural"].groupby("dataset")[["roc_auc_correct", "roc_auc_leaky"]].mean()
        fig, ax = plt.subplots(figsize=(7, 0.5 * len(g) + 1.4))
        y = np.arange(len(g))
        ax.hlines(y, g["roc_auc_correct"], g["roc_auc_leaky"], color=C["grid"], linewidth=2)
        ax.scatter(g["roc_auc_correct"], y, s=64, color=C["blue"], marker="o", label="SMOTE inside the fold (correct)", zorder=3)
        ax.scatter(g["roc_auc_leaky"], y, s=64, color=C["orange"], marker="s", label="SMOTE before splitting (leaky)", zorder=3)
        for yi, (b, l) in enumerate(zip(g["roc_auc_correct"], g["roc_auc_leaky"])):
            ax.text(max(b, l) + 0.005, yi, f"+{l - b:.3f}", va="center", fontsize=8, color=C["ink2"])
        ax.set_yticks(y, g.index)
        ax.grid(axis="x", color=C["grid"])
        _style(ax, "ROC-AUC inflation caused by resampling leakage", "ROC-AUC (mean over models and folds)")
        ax.grid(axis="y", visible=False)
        ax.set_ylim(-0.6, len(g) - 0.4)
        ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.38), ncol=2)
        fig.tight_layout()
        fig.savefig(fig_dir / "fig3_leakage_inflation.png", dpi=200)
        plt.close(fig)

    # F4: imbalance moderator (small multiples, one axis per panel)
    order = ["natural", "0.1", "0.05", "0.02"]
    show = ["smote@1.0", "dirichlet@1.0", "class_weight", "easy_ensemble"]
    sub = wtab[wtab["technique"].isin(show) & wtab["ir_level"].isin(order)]
    if sub["ir_level"].nunique() > 1:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        for ax, met in zip(axes, ["pr_auc", "empc"]):
            s = sub[sub["metric"] == met].groupby(["technique", "ir_level"])["median_delta"].median().unstack()
            s = s[[c for c in order if c in s.columns]]
            techs = [t for t in show if t in s.index]
            for i, tech in enumerate(techs):
                ax.plot(range(s.shape[1]), s.loc[tech], color=SERIES[i], marker=MARKERS[i], linewidth=2,
                        markersize=7, label=tech)
            # end-of-line labels, nudged apart to avoid overlap
            ends = s.loc[techs].iloc[:, -1].dropna().sort_values()
            lo, hi = ax.get_ylim()
            gap, pos, last = 0.07 * (hi - lo), {}, -np.inf
            for tech, v in ends.items():
                last = max(v, last + gap)
                pos[tech] = last
            for tech, v in pos.items():
                ax.text(s.shape[1] - 1 + 0.12, v, tech, fontsize=8, color=C["ink2"], va="center")
            ax.axhline(0, color=C["ink2"], linewidth=0.8)
            ax.set_xticks(range(s.shape[1]), ["natural" if c == "natural" else f"{float(c):.0%}" for c in s.columns])
            ax.set_xlim(-0.2, s.shape[1] + 0.6)
            _style(ax, f"Δ{labels[met]} vs baseline", "churn rate", "median difference")
        axes[0].legend(frameon=False, fontsize=8, loc="best")
        fig.suptitle("Does the benefit of a technique depend on the degree of imbalance?", x=0.01, ha="left",
                     fontsize=11, fontweight="bold", color=C["ink"])
        fig.tight_layout()
        fig.savefig(fig_dir / "fig4_imbalance_moderator.png", dpi=200)
        plt.close(fig)

    # F5: optimism of random vs temporal splits
    if len(topt):
        g = topt.groupby("technique")[["roc_auc_temporal", "roc_auc_random"]].mean().sort_values("roc_auc_temporal")
        fig, ax = plt.subplots(figsize=(7, 0.45 * len(g) + 1.4))
        y = np.arange(len(g))
        ax.hlines(y, g["roc_auc_temporal"], g["roc_auc_random"], color=C["grid"], linewidth=2)
        ax.scatter(g["roc_auc_temporal"], y, s=64, color=C["blue"], marker="o", label="out-of-time (temporal)", zorder=3)
        ax.scatter(g["roc_auc_random"], y, s=64, color=C["orange"], marker="s", label="random split", zorder=3)
        ax.set_yticks(y, g.index)
        _style(ax, "Online Retail II: random split vs out-of-time", "ROC-AUC (mean over models and folds)")
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=C["grid"])
        ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
        fig.tight_layout()
        fig.savefig(fig_dir / "fig5_temporal_vs_random.png", dpi=200)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--static", default=str(config.RESULTS_DIR / "static_full.csv"))
    p.add_argument("--temporal", default=str(config.RESULTS_DIR / "temporal_full.csv"))
    p.add_argument("--out", default=str(config.RESULTS_DIR))
    a = p.parse_args()
    out = Path(a.out)
    tdir = out / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    wtab = agree = leak = tdelta = topt = pd.DataFrame()

    if Path(a.static).exists():
        df = load(a.static)
        df.groupby(KEY + ["technique"])[METRICS + ["empc_frac"]].agg(["mean", "std"]).round(4) \
            .to_csv(tdir / "T1_static_summary.csv")
        wtab = wilcoxon_table(deltas_vs_baseline(df))
        wtab.to_csv(tdir / "T2_wilcoxon_vs_baseline.csv", index=False)
        agree = lens_agreement(df)
        agree.to_csv(tdir / "T3_lens_agreement.csv", index=False)
        friedman_ranks(df).to_csv(tdir / "T3b_friedman_rank.csv", index=False)
        leak = leakage_inflation(df)
        leak.to_csv(tdir / "T4_leakage_inflation.csv", index=False)
        ir_moderator(wtab).to_csv(tdir / "T5_imbalance_moderator.csv", index=False)
        print("RQ1  share of blocks (all churn levels pooled) in which the best technique by F1 != best by EMPC:",
              f"{agree['best_changes_f1_vs_empc'].mean():.0%}" if len(agree) else "-")
        if len(leak):
            print("RQ3  mean ROC-AUC inflation caused by leakage (all churn levels pooled):", f"{leak['roc_auc_inflation'].mean():+.3f}")
    else:
        print(f"[info] {a.static} not found -> static analysis skipped")

    if Path(a.temporal).exists():
        tdf = load(a.temporal)
        topt = temporal_optimism(tdf)
        topt.to_csv(tdir / "T6_random_split_optimism.csv", index=False)
        tdelta = temporal_deltas(tdf)
        tdelta.to_csv(tdir / "T7_wilcoxon_temporal.csv", index=False)
        if len(topt):
            print("RQ2  mean ROC-AUC optimism of random vs temporal split (all churn levels pooled):",
                  f"{topt['roc_auc_optimism'].mean():+.3f}")
    else:
        print(f"[info] {a.temporal} not found -> temporal analysis skipped")

    figures(wtab, agree, leak, tdelta, topt, out / "figures")
    figures_journal(wtab, out / "figures")
    print(f"Tables: {tdir}\nFigures: {out / 'figures'}")


# journal figure sizes (inches), matching the single-column width of the IJC template
JOURNAL_SIZES = {"fig_j2_winrate": (3.3, 3.4), "fig_j3_imbalance": (3.3, 4.1)}
PRETTY = {"none": "Baseline", "class_weight": "Class weighting", "threshold": "Threshold tuning",
          "ros@1.0": "Random oversampling", "rus@1.0": "Random undersampling", "smote@1.0": "SMOTE (1.0)",
          "smote@0.5": "SMOTE (0.5)", "adasyn@1.0": "ADASYN", "smote_enn@1.0": "SMOTE-ENN",
          "dirichlet@1.0": "Dirichlet ExtSMOTE", "easy_ensemble": "EasyEnsemble"}


def figures_journal(wtab, fig_dir):
    """Grayscale figures for the International Journal of Computing (PNG 600 dpi + EPS).
    Series are distinguished by marker and line style, not colour (Figures 2 and 4 of the paper)."""
    if not len(wtab):
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                         "font.size": 10, "axes.linewidth": 0.6, "figure.facecolor": "white",
                         "savefig.facecolor": "white"})
    lab = {"roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "f1": "F1", "mcc": "MCC", "brier": "Brier",
           "prec_at_top": "P@10%", "empc": "EMPC"}

    nat = wtab[wtab["ir_level"] == "natural"]
    if len(nat):
        piv = nat.groupby(["technique", "metric"])["win_rate"].mean().unstack()[METRICS]
        fig, ax = plt.subplots(figsize=JOURNAL_SIZES["fig_j2_winrate"])
        ax.imshow(piv.values, cmap="Greys", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(METRICS)), [lab[m] for m in METRICS], rotation=45, ha="right")
        ax.set_yticks(range(len(piv)), [PRETTY.get(t, t) for t in piv.index])
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{100 * v:.0f}", ha="center", va="center", fontsize=10,
                            color="white" if v > 0.55 else "black")
        ax.set_xlabel("Folds beating the baseline (%)")
        fig.tight_layout()
        for ext in ("png", "eps"):
            fig.savefig(fig_dir / f"fig_j2_winrate.{ext}", dpi=600)
        plt.close(fig)

    order = ["natural", "0.1", "0.05", "0.02"]
    show = ["smote@1.0", "dirichlet@1.0", "class_weight", "easy_ensemble"]
    styles = [("o", "-"), ("s", "--"), ("^", "-."), ("D", ":")]
    sub = wtab[wtab["technique"].isin(show) & wtab["ir_level"].isin(order)]
    if sub["ir_level"].nunique() > 1:
        fig, axes = plt.subplots(2, 1, figsize=JOURNAL_SIZES["fig_j3_imbalance"], sharex=True)
        for ax, met in zip(axes, ["pr_auc", "empc"]):
            s = sub[sub["metric"] == met].groupby(["technique", "ir_level"])["median_delta"].median().unstack()
            s = s[[c for c in order if c in s.columns]]
            for (mk, ls), tech in zip(styles, [t for t in show if t in s.index]):
                ax.plot(range(s.shape[1]), s.loc[tech], color="black", marker=mk, linestyle=ls,
                        linewidth=1.2, markersize=5, markerfacecolor="white", label=PRETTY.get(tech, tech))
            ax.axhline(0, color="0.5", linewidth=0.6)
            ax.set_ylabel(f"Median Δ{lab[met]}")
            ax.set_xticks(range(s.shape[1]), ["natural" if c == "natural" else f"{float(c):.0%}" for c in s.columns])
            ax.grid(axis="y", color="0.85", linewidth=0.5)
        axes[-1].set_xlabel("Churn rate after subsampling")
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, fontsize=10, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.0))
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        for ext in ("png", "eps"):
            fig.savefig(fig_dir / f"fig_j3_imbalance.{ext}", dpi=600)
        plt.close(fig)


if __name__ == "__main__":
    main()
