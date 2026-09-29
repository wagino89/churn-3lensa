"""Generate notebooks/churn_3lensa_kaggle.ipynb from the sources in src/.

    python tools/make_kaggle_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
FILES = ["__init__.py", "config.py", "metrics.py", "resampling.py", "models.py", "datasets.py",
         "core.py", "experiment_static.py", "experiment_temporal.py", "analysis.py"]

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n").splitlines(True)})


md(r"""
# Three lenses: does oversampling really help churn prediction?

This notebook runs the complete experiment of the paper on Kaggle (CPU) and packs the raw results into
**one zip file**. It writes the package in `src/` of the repository to disk and runs it unchanged.

## Steps on Kaggle

1. **Settings → Accelerator: None (CPU).** No GPU is used.
2. **Add Input → Datasets** and add these four datasets (type the owner name in the search box):
   - `blastchar/telco-customer-churn`
   - `shrutimechlearn/churn-modelling`
   - `sakshigoyal7/credit-card-customers`
   - `becksddf/churn-in-telecoms-dataset`
3. **Iranian Churn and Online Retail II** (both from UCI): either set **Settings → Internet: On**
   (the notebook downloads them from UCI) or add a Kaggle copy of each. Files are recognised by their
   **column names**, so file and folder names do not matter.
4. **Quick test (recommended):** set `PART = "ALL"` and `QUICK = True` and click **Run All** (10–20 min).
   Then set `QUICK = False`.
5. Choose **PART** in the settings cell. A Kaggle session is limited to 12 hours, so the experiment is split:

   | PART | Content | Approximate time (4 CPU cores) |
   |---|---|---|
   | `"A"` | IBM Telco + Bank churn | ~2.2 h (paper run) |
   | `"B"` | Credit card + Iranian + Telecom BigML | ~3.6 h (paper run) |
   | `"C"` | Online Retail II (temporal lens; natural, 20% and 10% churn) | ~2.5 h (paper runs) |

6. **Save Version → Save & Run All (Commit)**, then download `results_churn3lensa_<PART>.zip` from the **Output** tab.

**If a session times out:** results are checkpointed per block. Create a new version and add the previous
version via **Add Input → Notebook Output**; finished blocks are skipped automatically.
""")

code(r"""
# ================= SETTINGS =================
PART = "A"            # "A", "B", "C" or "ALL"
QUICK = False         # True = quick test (3 folds, 2 models). Final results: False
HOURS_LIMIT = 11.3    # stop starting new blocks after this many hours (Kaggle limit: 12 h)
N_JOBS = -1           # -1 = all cores
TEMPORAL_LEVELS = ["natural", 0.20, 0.10]   # churn levels of the temporal lens (PART "C")
# ============================================
import os, sys, time, json, shutil, platform, subprocess, glob, zipfile, warnings
# Hide cosmetic warnings (e.g. LightGBM "X does not have valid feature names").
# This does not change any computation; it also applies to the parallel workers.
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
T_START = time.time()
DEADLINE = T_START + HOURS_LIMIT * 3600
WORK = os.environ.get("KAGGLE_WORK", "/kaggle/working")
INPUT_ROOT = os.environ.get("KAGGLE_INPUT", "/kaggle/input")
PKG = os.path.join(WORK, "churn3lensa")
os.makedirs(os.path.join(PKG, "src"), exist_ok=True)
os.makedirs(os.path.join(PKG, "data", "raw"), exist_ok=True)
os.makedirs(os.path.join(PKG, "results"), exist_ok=True)
print("PART:", PART, "| QUICK:", QUICK, "| working folder:", PKG)
""")

code(r"""
# Make sure the libraries are available (usually pre-installed on Kaggle)
need = {"imblearn": "imbalanced-learn", "xgboost": "xgboost", "lightgbm": "lightgbm", "openpyxl": "openpyxl"}
for mod, pipname in need.items():
    try:
        __import__(mod)
    except ImportError:
        print("installing", pipname, "...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pipname], check=False)
import numpy, pandas, scipy, sklearn, matplotlib, joblib
versions = {"python": platform.python_version(), "numpy": numpy.__version__, "pandas": pandas.__version__,
            "scipy": scipy.__version__, "scikit-learn": sklearn.__version__, "matplotlib": matplotlib.__version__}
for mod in ("imblearn", "xgboost", "lightgbm"):
    try:
        versions[mod] = __import__(mod).__version__
    except ImportError:
        versions[mod] = "NOT INSTALLED"
versions["cpu_count"] = os.cpu_count()
versions["platform"] = platform.platform()
print(json.dumps(versions, indent=1))
""")

md("## Experiment code\nThe following cells write the package `src/` of the repository to disk, unchanged. No need to edit them.")

code(r'''
def write_src(name, text):
    with open(os.path.join(PKG, "src", name), "w", encoding="utf-8") as f:
        f.write(text)
''')

for fname in FILES:
    text = (SRC / fname).read_text(encoding="utf-8")
    assert "'''" not in text, fname
    code("write_src(%r, r'''%s''')" % (fname, text))

code(r"""
if PKG not in sys.path:
    sys.path.insert(0, PKG)
for m in [k for k in list(sys.modules) if k == "src" or k.startswith("src.")]:
    del sys.modules[m]
from src import config, experiment_static, experiment_temporal, analysis
from src.datasets import load_static
print("package ready:", config.ROOT)
""")

md("## Locating the datasets\nFiles are recognised by their **column names** and copied to `data/raw/` under standard names.")

code(r"""
import pandas as pd
RAW = str(config.DATA_DIR)
SIGNATURES = {   # dataset -> (required columns, standard file name)
    "telco":         ({"customerID", "Contract", "Churn"}, "WA_Fn-UseC_-Telco-Customer-Churn.csv"),
    "bank":          ({"Exited", "Geography", "CreditScore"}, "Churn_Modelling.csv"),
    "bankchurners":  ({"Attrition_Flag", "CLIENTNUM"}, "BankChurners.csv"),
    "iranian":       ({"Complains", "Churn", "Seconds of Use"}, "Customer Churn.csv"),
    "telecom_bigml": ({"customer service calls", "churn"}, "bigml_telecom.csv"),
}
def norm_cols(cols):
    return {" ".join(str(c).split()) for c in cols}

found = {}
for path in glob.glob(os.path.join(INPUT_ROOT, "**", "*.csv"), recursive=True):
    if "/parts/" in path:
        continue
    try:
        cols = norm_cols(pd.read_csv(path, nrows=2, encoding_errors="replace").columns)
    except Exception:
        continue
    for ds, (sig, _) in SIGNATURES.items():
        if ds not in found and sig <= cols:
            found[ds] = path
    if "retail" not in found and {"StockCode", "Quantity"} <= cols and ("InvoiceDate" in cols):
        found["retail"] = path
if "retail" not in found:
    xl = [p for p in glob.glob(os.path.join(INPUT_ROOT, "**", "*.xlsx"), recursive=True)
          if "retail" in os.path.basename(p).lower()]
    if xl:
        found["retail"] = xl[0]

def try_uci(url, want):
    # Download the zip from UCI (requires Settings -> Internet: On)
    import urllib.request, io
    try:
        data = urllib.request.urlopen(url, timeout=120).read()
        zdir = os.path.join(WORK, "uci_" + want)
        zipfile.ZipFile(io.BytesIO(data)).extractall(zdir)
        for p in glob.glob(os.path.join(zdir, "**", "*"), recursive=True):
            if p.lower().endswith((".csv", ".xlsx")):
                return p
    except Exception as e:
        print(f"  UCI download failed ({type(e).__name__}: {e})")
    return None

if "iranian" not in found and PART in ("B", "ALL"):
    print("Iranian Churn not in the inputs -> trying to download it from UCI ...")
    p = try_uci("https://archive.ics.uci.edu/static/public/563/iranian+churn+dataset.zip", "iranian")
    if p: found["iranian"] = p
if "retail" not in found and PART in ("C", "ALL"):
    print("Online Retail II not in the inputs -> trying to download it from UCI (~45 MB) ...")
    p = try_uci("https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip", "retail")
    if p: found["retail"] = p

for ds, (sig, canon) in SIGNATURES.items():
    if ds in found:
        shutil.copy(found[ds], os.path.join(RAW, canon))
if "retail" in found:
    ext = os.path.splitext(found["retail"])[1]
    shutil.copy(found["retail"], os.path.join(RAW, "online_retail_II" + ext))
print("\nDatasets found:")
for k in list(SIGNATURES) + ["retail"]:
    print(f"  {k:14s} {'OK  ' + found[k] if k in found else 'MISSING'}")
""")

code(r"""
# Check the static datasets needed by this part
PART_DS = {"A": ["telco", "bank"], "B": ["bankchurners", "iranian", "telecom_bigml"], "C": [],
           "ALL": list(config.STATIC_DATASETS)}[PART]
missing = [ds for ds in PART_DS if ds not in found]
if PART in ("C", "ALL") and "retail" not in found:
    missing.append("retail")
if missing:
    raise SystemExit(f"Missing datasets: {missing}. See steps 2-3 in the first cell.")
for ds in PART_DS:
    X, y, meta = load_static(ds)
    print(f"{ds:14s} n={meta['n']:>6}  churn={meta['churn_rate']:.3f}  features={meta['n_features']}  dropped={meta['dropped']}")
""")

code(r"""
# Resume from a previous version (if its output was added as an input)
parts_dir = os.path.join(str(config.RESULTS_DIR), "parts")
os.makedirs(parts_dir, exist_ok=True)
n_prev = 0
for p in glob.glob(os.path.join(INPUT_ROOT, "**", "parts", "*.csv"), recursive=True):
    dst = os.path.join(parts_dir, os.path.basename(p))
    if not os.path.exists(dst):
        shutil.copy(p, dst); n_prev += 1
print("result parts reused from previous versions:", n_prev)
""")

md("## Running the experiment")

code(r"""
from pathlib import Path
q = QUICK
tag = "quick" if q else "full"
if PART_DS:
    experiment_static.run(
        PART_DS,
        config.TECHNIQUES_QUICK if q else config.TECHNIQUES_FULL,
        config.MODELS_QUICK if q else config.MODELS_FULL,
        config.CV_QUICK if q else config.CV_FULL,
        config.IR_LEVELS_QUICK if q else config.IR_LEVELS_FULL,
        N_JOBS, config.RESULTS_DIR / f"static_{tag}.csv", deadline=DEADLINE)
print(f"elapsed: {(time.time() - T_START) / 3600:.2f} h")
""")

code(r"""
if PART in ("C", "ALL"):
    experiment_temporal.run(
        config.TECHNIQUES_QUICK if q else config.TECHNIQUES_FULL,
        config.MODELS_QUICK if q else config.MODELS_FULL,
        N_JOBS, config.RESULTS_DIR / f"temporal_{tag}.csv",
        max_folds=3 if q else None, deadline=DEADLINE, ir_levels=TEMPORAL_LEVELS)
print(f"elapsed: {(time.time() - T_START) / 3600:.2f} h")
""")

md("## Quick analysis and packing the results")

code(r"""
res = config.RESULTS_DIR
sys.argv = ["analysis", "--static", str(res / f"static_{tag}.csv"),
            "--temporal", str(res / f"temporal_{tag}.csv"), "--out", str(res / "analysis")]
try:
    analysis.main()
except Exception as e:
    print("analysis skipped:", type(e).__name__, e)
""")

code(r"""
versions["part"], versions["quick"] = PART, QUICK
versions["hours_elapsed"] = round((time.time() - T_START) / 3600, 2)
versions["dataset_sources"] = found
with open(res / "environment.json", "w") as f:
    json.dump(versions, f, indent=1, default=str)

summary = []
for f in sorted(glob.glob(str(res / "*.csv"))):
    try:
        d = pd.read_csv(f)
        if "status" in d:
            summary.append((os.path.basename(f), len(d), int((d["status"] != "ok").sum())))
    except Exception:
        pass
print("Result files (rows, skipped/error):")
for s in summary:
    print("  ", s)

out_zip = os.path.join(WORK, f"results_churn3lensa_{PART}{'_quick' if QUICK else ''}.zip")
with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
    for p in glob.glob(str(res / "**" / "*"), recursive=True):
        if os.path.isfile(p):
            z.write(p, os.path.relpath(p, str(res)))
print(f"\nDONE. Download this file from the Output tab:\n  {out_zip}  ({os.path.getsize(out_zip) / 1e6:.1f} MB)")
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = ROOT / "notebooks" / "churn_3lensa_kaggle.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("notebook:", out, "| sel:", len(cells))
