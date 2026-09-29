# Experiment log

This file records how the results in `results/raw/` were produced, including a design change made during the
study, so that readers can judge the analysis path.

## Runs

All runs used `notebooks/churn_3lensa_kaggle.ipynb` on Kaggle (CPU, 4 cores, Python 3.12.13, numpy 2.0.2,
pandas 2.3.3, scipy 1.16.3, scikit-learn 1.6.1, imbalanced-learn 0.14.1, XGBoost 3.2.0, LightGBM 4.6.0).
Details per run are in `results/environment/`.

| Run | Content | Model fits | Failed fits | Wall time |
|---|---|---|---|---|
| A | IBM Telco, Bank churn; natural, 10%, 5%, 2% churn | 12,000 | 0 | 2.18 h |
| B | Credit card, Iranian, Telecom BigML; natural, 10%, 5%, 2% churn | 18,000 | 0 | 3.60 h |
| C1 | Online Retail II, natural churn rate | 880 | 160 | 1.16 h |
| C2 | Online Retail II, churn subsampled to 20% and 10% | 1,760 | 0 | 1.29 h |

`results/raw/static_full.csv` concatenates A and B; `results/raw/temporal_full.csv` concatenates C1 and C2.

## Design change in the temporal lens

The temporal lens was first run on the unsubsampled retail panel (run C1). Inspection of that run showed that the 90-day churn
rate of the panel is about 51% (38–62% per month), so churners are not a minority and imbalance handling is not
meaningful there. In months where churners are the majority, ADASYN and SMOTE at ratio 0.5 cannot operate; these
160 fits are recorded with `status = error` in `temporal_full.csv` and excluded from all analyses.

The temporal lens was therefore redesigned: churners are randomly removed within every
monthly snapshot so that the churn rate is 20% or 10% up to rounding to whole customers (`experiment_temporal.subsample_panel`, seed fixed
per month and level), and the same subsampled rows feed both the temporal and the random split (run C2). The
paper reports the subsampled levels as the main temporal analysis and the natural panel as a sensitivity analysis.
The static experiment (runs A and B) was not changed.

## Code changes after the runs (no effect on the reported results)

* Comments, log messages and derived column labels were translated into English (e.g. `lebih baik` → `better`,
  `roc_auc_bocor` → `roc_auc_leaky`). A test on mock data confirmed that the raw outputs of the translated code are
  identical to those of the code used on Kaggle, and `python -m src.paper` reproduces the numbers of the paper
  exactly.
* `datasets.load_retail_transactions` now reads its cache with `float_precision="round_trip"`. Without this, the
  cached transaction table differed from the freshly built one in the last digit of some amounts, so a second
  local run could give slightly different temporal results than the first. Every Kaggle run built the table from
  the source file (fresh session), i.e. used the path that is unchanged.
* `src/paper.py`, `tests/test_core.py` and this documentation were added.
