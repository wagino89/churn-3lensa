# churn-3lensa: a leakage-safe, profit-aware and out-of-time benchmark of imbalance handling for churn prediction

Code, raw results and reproduction scripts for the paper

> **Resampling in Churn Prediction: A Leakage-Safe, Profit-Aware and Out-of-Time Evaluation.**
> Wagino, Arafat, Nur Alamsyah. Submitted to the *International Journal of Computing*, 2026.

The study evaluates twelve class-imbalance strategies (baseline, class weighting, threshold tuning, random
over/undersampling, SMOTE at two ratios, ADASYN, SMOTE-ENN, Dirichlet ExtSMOTE, EasyEnsemble, and a deliberately
leaky SMOTE control) with five classifiers (logistic regression, random forest, histogram gradient boosting,
XGBoost, LightGBM) on five public churn datasets and one retail transaction log — **32,640 model fits** in total —
through three evaluation lenses:

| Lens | Question | Metrics |
|---|---|---|
| A. Statistical | How well does the model separate churners? | ROC-AUC, PR-AUC, F1, MCC, Brier score |
| B. Profit | What is a retention campaign worth? | EMPC (Verbraken et al., 2013), precision in the top 10% |
| C. Out-of-time | Does the score hold on future months? | Lens A and B metrics, rolling-origin vs random split |

## Main findings (all numbers are reproduced by `python -m src.paper`)

| RQ | Finding |
|---|---|
| RQ1 | Resampling raises F1, but **threshold tuning matches it** (mean F1 0.731 vs 0.726 for SMOTE, 0.696 baseline). Oversampling rarely improves EMPC, and the best strategy by F1 differs from the best by EMPC in **76%** of the 25 dataset–classifier settings. |
| RQ2 | Random splits overstate PR-AUC by **0.016–0.027** relative to out-of-time evaluation, and the strategy ranked best under a random split was best on future months in only 1 of 10 cases (PR-AUC) and 0 of 10 (EMPC). |
| RQ3 | Applying SMOTE before splitting inflates ROC-AUC by +0.05 and F1 by +0.18 on average; at 2% churn on IBM Telco, F1 rises from 0.07 to 0.95. |
| RQ4 | No strategy becomes more useful as churn gets rarer; EasyEnsemble deteriorates. |

## Repository layout

```
src/                      experiment package (the exact code used for the paper)
  config.py               every design choice: datasets, techniques, classifiers, CV, EMPC parameters
  datasets.py             loaders for the static datasets; monthly snapshots of Online Retail II
  resampling.py           samplers, Dirichlet ExtSMOTE, EasyEnsemble
  models.py               preprocessing and classifiers (fixed hyperparameters)
  metrics.py              ROC-AUC, PR-AUC, F1, MCC, Brier, precision@10%, EMPC
  core.py                 one leakage-safe experimental unit (fit preprocessing -> resample -> fit -> score)
  experiment_static.py    Lens A + B, imbalance moderator and leaky control (RQ1, RQ3, RQ4)
  experiment_temporal.py  Lens C: rolling-origin vs random split (RQ2)
  analysis.py             Wilcoxon/Holm, Friedman, Kendall tau, leakage and optimism tables
  paper.py                regenerates every number, table and figure of the paper
notebooks/
  churn_3lensa_kaggle.ipynb   self-contained notebook used to run the experiment on Kaggle (CPU)
results/
  raw/                    raw per-fold results of the paper (one row per model fit)
  paper/                  numbers.json, Tables 1, 3–6 (CSV + tables.md), Figures 1–4 (PNG 600 dpi + EPS)
  environment/            library versions, hardware and run time of each Kaggle run
data/                     put the downloaded datasets in data/raw/ (see data/README.md)
docs/EXPERIMENT_LOG.md    how and in which order the runs were performed
tests/                    unit tests and mock-data generator
tools/                    notebook generator
```

## 1. Installation

Python 3.10–3.12. The paper was run with Python 3.12 and the pinned versions in `requirements.txt`.

```bash
git clone https://github.com/wagino89/churn-3lensa.git
cd churn-3lensa
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Verify the numbers of the paper (about one minute, no data download)

```bash
python -m src.paper            # reads results/raw, writes results/paper
```

`results/paper/numbers.json` contains every number quoted in the text, `results/paper/tables/tables.md` the
tables, and `results/paper/figures/` the figures. Where each item comes from:

| Paper item | Source in `results/paper/` | Computed in |
|---|---|---|
| Table 1 (datasets) | `tables/Table1.csv` | `src/datasets.py`, `src/paper.py` |
| Table 2 (strategies) | descriptive | `src/config.py`, `src/resampling.py`, `src/core.py` |
| Table 3 (median Δ and win rate) | `tables/Table3.csv` | `analysis.deltas_vs_baseline`, `analysis.wilcoxon_table` |
| Table 4 (significant settings) | `tables/Table4.csv`, `tables/wilcoxon_static.csv` | `analysis.wilcoxon_table` (Holm) |
| Table 5 (out-of-time vs random) | `tables/Table5.csv`, `tables/wilcoxon_temporal.csv` | `paper.temporal_vs_baseline` |
| Table 6 (leakage) | `tables/Table6.csv`, `tables/leakage.csv` | `analysis.leakage_inflation` |
| Figure 1 (protocol) | `figures/Figure1.*` | `paper.figure_protocol` |
| Figure 2 (win rates) | `figures/Figure2.*` | `analysis.figures_journal` |
| Figure 3 (optimism) | `figures/Figure3.*` | `paper.figure_temporal` |
| Figure 4 (imbalance moderator) | `figures/Figure4.*` | `analysis.figures_journal` |
| Kendall tau, Friedman, paired tests in the text | `numbers.json` | `src/paper.py` |

The raw files contain one row per model fit with the factors (`dataset`, `ir_level`, `repeat`/`fold` or
`lens_split`/`test_cutoff`, `technique`, `model`), a `status` column, and all metrics.

## 3. Run the tests and a smoke test (a few minutes)

```bash
pip install pytest && pytest -q                      # unit tests: EMPC, protocol, samplers, subsampling
bash tests/smoke_test.sh                             # whole pipeline on mock data
```

Numbers obtained on mock data are meaningless; the smoke test only checks that everything runs.

## 4. Reproduce the experiment from scratch

1. Download the six datasets as described in [`data/README.md`](data/README.md) and put them in `data/raw/`.
2. Run the full experiment (about 8 CPU-hours on 4 cores; results are checkpointed and resumable):

```bash
bash run_all.sh                     # writes results/static_full.csv, results/temporal_full.csv, ...
python -m src.paper --raw results --out results/paper_rerun
```

Alternatively, run `notebooks/churn_3lensa_kaggle.ipynb` on Kaggle in three parts (A, B, C), as was done for the
paper; the notebook contains the same `src/` package and packs the results into a zip file.

**Determinism.** All random seeds are derived from the experimental factors (`core.stable_seed`), so a rerun with
the pinned library versions reproduces the raw results. Other versions of scikit-learn, imbalanced-learn, XGBoost
or LightGBM can change individual scores slightly.

## Protocol in brief

* **Leakage-safe:** inside every fold, preprocessing (median/mode imputation, scaling, one-hot encoding) is fitted
  on the training part only; resampling is applied only to the preprocessed training data; test folds are never
  resampled. The leaky control (`smote_leaky@1.0`) applies preprocessing and SMOTE to the whole dataset first.
* **Static datasets:** 5 × 5 repeated stratified cross-validation; churners subsampled to 10%, 5% and 2%
  (moderator, RQ4).
* **Online Retail II:** monthly customer snapshots from June 2010 to September 2011; churn = no purchase in the
  next 90 days among customers active in the previous 180 days; rolling-origin evaluation with eight test months;
  the natural churn rate (~51%) is subsampled within every month to 20% and 10%, and the random comparator uses
  the same rows and test size.
* **Statistics:** Wilcoxon signed-rank tests with Holm correction, Friedman test, Kendall tau.
* **EMPC:** CLV = 200, incentive d = 10, contact cost f = 1, acceptance probability γ ~ Beta(6, 14).

## Data

The datasets are **not redistributed** in this repository; they are publicly available from Kaggle and the UCI
Machine Learning Repository under their own terms (links in `data/README.md`).

## License and citation

Code: MIT License (see `LICENSE`). Results in `results/`: CC BY 4.0. If you use this repository, please cite the
paper (see `CITATION.cff`).
