"""Configuration of the three-lens churn experiment.

Every design choice that affects the results is collected here so that it can
be reported verbatim in the Methods section and changed for sensitivity analyses.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "results"
SEED = 2026

# ---------------------------------------------------------------------------
# Static datasets (Lens A = statistical metrics, Lens B = profit metrics).
# 'file' may be a glob pattern. Column names are matched case-insensitively and
# repeated spaces are collapsed, so small variations in column names still work.
# ---------------------------------------------------------------------------
STATIC_DATASETS = {
    "telco": {
        "file": "WA_Fn-UseC_-Telco-Customer-Churn*.csv",
        "source": "Kaggle: blastchar/telco-customer-churn (IBM)",
        "target": "Churn", "positive": "Yes",
        "drop": ["customerID"],
        "to_numeric": ["TotalCharges"],
    },
    "bank": {
        "file": "Churn_Modelling*.csv",
        "source": "Kaggle: shrutimechlearn/churn-modelling",
        "target": "Exited", "positive": 1,
        "drop": ["RowNumber", "CustomerId", "Surname"],
    },
    "bankchurners": {
        "file": "BankChurners*.csv",
        "source": "Kaggle: sakshigoyal7/credit-card-customers",
        "target": "Attrition_Flag", "positive": "Attrited Customer",
        "drop": ["CLIENTNUM"],
        # The last two columns of this dataset are outputs of a naive Bayes
        # classifier computed from the label -> they MUST be removed (target leakage).
        "drop_prefix": ["Naive_Bayes_Classifier"],
    },
    "iranian": {
        "file": "Customer Churn*.csv",
        "source": "UCI ML Repository id 563 (Iranian Churn)",
        "target": "Churn", "positive": 1,
        "drop": [],
    },
    "telecom_bigml": {
        "file": "bigml*.csv",
        "source": "Kaggle: becksddf/churn-in-telecoms-dataset",
        "target": "churn", "positive": True,
        "drop": ["phone number"],
    },
}

# ---------------------------------------------------------------------------
# Temporal dataset (Lens C = out-of-time validation)
# ---------------------------------------------------------------------------
RETAIL = {
    "file": "online_retail_II*",           # .xlsx from UCI or a converted .csv
    "source": "UCI ML Repository id 502 (Online Retail II)",
    "horizon_days": 90,                     # churn = no purchase in the next 90 days
    "active_window_days": 180,              # population = purchased in the previous 180 days
    "first_cutoff": "2010-06-01",
    "last_cutoff": "2011-09-01",            # last_cutoff + horizon <= end of data (2011-12-09)
    "min_train_cutoffs": 6,                 # rolling origin starts after >= 6 training snapshots
    # The 90-day churn rate of this panel is ~51% (not imbalanced). To make imbalance
    # handling meaningful, churners are subsampled WITHIN EACH snapshot to the target
    # rate; the same rows feed both the temporal and the random split.
    "ir_levels": ["natural", 0.20, 0.10],
}

# ---------------------------------------------------------------------------
# EMPC parameters (Expected Maximum Profit measure for Customer churn).
# Defaults follow Verbraken et al. (2013) / the R package 'EMP':
# CLV = 200, incentive cost d = 10, contact cost f = 1,
# probability that a targeted churner accepts the offer gamma ~ Beta(6, 14).
# ---------------------------------------------------------------------------
EMPC_PARAMS = {"clv": 200.0, "d": 10.0, "f": 1.0, "alpha": 6.0, "beta": 14.0}

# ---------------------------------------------------------------------------
# Experimental factors.
# Technique format: "name" or "name@ratio" (ratio = minority/majority after
# resampling; 1.0 = fully balanced, 0.5 = half).
# ---------------------------------------------------------------------------
TECHNIQUES_FULL = [
    "none",            # baseline without imbalance handling
    "class_weight",    # algorithm level (cost-sensitive)
    "threshold",       # algorithm level: F1-optimal threshold on a validation split
    "ros@1.0",         # random oversampling
    "rus@1.0",         # random undersampling
    "smote@1.0",
    "smote@0.5",       # non-balanced ratio (cf. Pandey et al., 2021)
    "adasyn@1.0",      # requires imbalanced-learn
    "smote_enn@1.0",   # requires imbalanced-learn
    "dirichlet@1.0",   # Dirichlet ExtSMOTE (Matharaarachchi et al., 2024)
    "easy_ensemble",   # ensemble level
    "smote_leaky@1.0", # DELIBERATELY WRONG: SMOTE before splitting -> measures leakage inflation
]
TECHNIQUES_QUICK = ["none", "class_weight", "threshold", "smote@1.0", "dirichlet@1.0",
                    "easy_ensemble", "smote_leaky@1.0"]

MODELS_FULL = ["lr", "rf", "hgb", "xgb", "lgbm"]  # xgb/lgbm are skipped if not installed
MODELS_QUICK = ["lr", "hgb"]

CV_FULL = {"repeats": 5, "folds": 5}   # 25 paired scores per condition -> Wilcoxon test
CV_QUICK = {"repeats": 1, "folds": 3}

# Moderator: the churn rate is made more extreme by subsampling churners.
# "natural" = as is. Levels above a dataset's natural rate are skipped.
IR_LEVELS_FULL = ["natural", 0.10, 0.05, 0.02]
IR_LEVELS_QUICK = ["natural"]

DIRICHLET_M = 1.0   # concentration multiplier m
K_NEIGHBORS = 5     # k for SMOTE/ADASYN/Dirichlet (as in Matharaarachchi et al.)
EASY_ENSEMBLE_N = 10
TOP_FRACTION = 0.10 # campaign capacity for the precision@10% metric
