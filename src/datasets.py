"""Loaders for the static datasets and the monthly snapshot builder for Online Retail II."""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


def _norm(name):
    return re.sub(r"\s+", " ", str(name)).strip().lower()


def _find_file(pattern, data_dir):
    hits = sorted(Path(data_dir).glob(pattern))
    if not hits:
        raise FileNotFoundError(
            f"File '{pattern}' not found in {data_dir}. See data/README.md for download instructions.")
    return hits[0]


def _is_positive(series, positive):
    """Compare labels with the positive value tolerantly (Yes/1/True/'True')."""
    s = series.astype(str).str.strip().str.lower()
    return (s == str(positive).strip().lower()).astype(int).to_numpy()


def load_static(name, data_dir=None):
    """Return (X: feature DataFrame, y: 0/1 array, meta: dict)."""
    cfg = config.STATIC_DATASETS[name]
    path = _find_file(cfg["file"], data_dir or config.DATA_DIR)
    df = pd.read_csv(path)
    colmap = {_norm(c): c for c in df.columns}
    target = colmap.get(_norm(cfg["target"]))
    if target is None:
        raise KeyError(f"Target column '{cfg['target']}' not found in {path.name}. "
                       f"Available columns: {list(df.columns)}")
    y = _is_positive(df[target], cfg["positive"])
    drop = {target} | {colmap[_norm(c)] for c in cfg.get("drop", []) if _norm(c) in colmap}
    for pre in cfg.get("drop_prefix", []):
        drop |= {c for c in df.columns if str(c).startswith(pre)}
    X = df.drop(columns=list(drop))
    for c in cfg.get("to_numeric", []):
        if _norm(c) in colmap and colmap[_norm(c)] in X:
            X[colmap[_norm(c)]] = pd.to_numeric(X[colmap[_norm(c)]], errors="coerce")
    # text columns with very high cardinality (e.g. identifiers missed above) are dropped
    for c in list(X.columns):
        if not pd.api.types.is_numeric_dtype(X[c]) and X[c].nunique() > 0.5 * len(X):
            print(f"[info] {name}: column '{c}' dropped (high cardinality, probably an identifier)")
            X = X.drop(columns=c)
    X.columns = [str(c) for c in X.columns]
    meta = {"dataset": name, "file": path.name, "n": len(y), "churn_rate": float(y.mean()),
            "n_features": X.shape[1], "dropped": sorted(map(str, drop - {target}))}
    return X, y, meta


def subsample_to_rate(X, y, rate, seed):
    """Imbalance moderator: remove churners at random so that their share equals `rate`."""
    if rate == "natural" or rate >= y.mean():
        return X, y
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    n_pos = int(round(rate * len(neg) / (1 - rate)))
    keep = np.r_[rng.choice(pos, size=max(n_pos, 10), replace=False), neg]
    keep.sort()
    return X.iloc[keep].reset_index(drop=True), y[keep]


# ---------------------------------------------------------------------------
# Online Retail II -> monthly customer snapshots with a forward-looking churn label
# ---------------------------------------------------------------------------
_RETAIL_ALIASES = {
    "invoice": ["invoice", "invoiceno"], "stockcode": ["stockcode"],
    "quantity": ["quantity"], "invoicedate": ["invoicedate"],
    "price": ["price", "unitprice"], "customer": ["customer id", "customerid"],
    "country": ["country"],
}


def load_retail_transactions(data_dir=None, cache=True):
    data_dir = Path(data_dir or config.DATA_DIR)
    cache_path = data_dir / "online_retail_II_clean.csv.gz"
    if cache and cache_path.exists():
        # round_trip: read floats exactly as written, so the cached and the freshly built
        # transaction tables are identical (the default parser can differ in the last digit)
        return pd.read_csv(cache_path, parse_dates=["invoicedate"], float_precision="round_trip")
    path = _find_file(config.RETAIL["file"], data_dir)
    if path.suffix.lower() in (".xlsx", ".xls"):
        print("[info] reading xlsx (1M+ rows, 2-5 minutes; the result is cached)...")
        df = pd.concat(pd.read_excel(path, sheet_name=None).values(), ignore_index=True)
    else:
        df = pd.read_csv(path, encoding_errors="replace")
    colmap = {_norm(c): c for c in df.columns}
    ren = {}
    for std, alts in _RETAIL_ALIASES.items():
        for a in alts:
            if a in colmap:
                ren[colmap[a]] = std
                break
    df = df.rename(columns=ren)[list(_RETAIL_ALIASES)]
    df = df.dropna(subset=["customer"])
    df["customer"] = df["customer"].astype(int)
    df["invoicedate"] = pd.to_datetime(df["invoicedate"])
    df["invoice"] = df["invoice"].astype(str)
    df["is_return"] = df["invoice"].str.startswith("C") | (df["quantity"] < 0)
    df["amount"] = df["quantity"] * df["price"]
    df = df.drop_duplicates()
    if cache:
        df.to_csv(cache_path, index=False, compression="gzip")
    return df


def build_snapshot(tx, cutoff, horizon_days, active_days):
    """Customer features from transactions BEFORE the cutoff; label from [cutoff, cutoff + H)."""
    cutoff = pd.Timestamp(cutoff)
    past = tx[tx["invoicedate"] < cutoff]
    buys = past[~past["is_return"]]
    last_buy = buys.groupby("customer")["invoicedate"].max()
    active = last_buy[last_buy >= cutoff - pd.Timedelta(days=active_days)].index
    if len(active) == 0:
        return pd.DataFrame()
    b = buys[buys["customer"].isin(active)]
    r = past[past["is_return"] & past["customer"].isin(active)]
    days = lambda s: (cutoff - s).dt.days

    def win(df, d, col="invoicedate"):
        return df[df[col] >= cutoff - pd.Timedelta(days=d)]

    inv = b.groupby(["customer", "invoice"]).agg(t=("invoicedate", "min"), v=("amount", "sum")).reset_index()
    g = inv.groupby("customer")
    feat = pd.DataFrame(index=pd.Index(active, name="customer"))
    feat["recency_days"] = days(g["t"].max())
    feat["tenure_days"] = days(g["t"].min())
    feat["n_invoices_total"] = g.size()
    for d in (30, 90, 365):
        w = win(inv, d, "t").groupby("customer")
        feat[f"n_invoices_{d}d"] = w.size()
        feat[f"spend_{d}d"] = w["v"].sum()
    feat["avg_basket_value"] = g["v"].mean()
    gaps = inv.sort_values("t").groupby("customer")["t"].diff().dt.days
    feat["mean_days_between"] = gaps.groupby(inv.sort_values("t")["customer"]).mean()
    feat["n_products_365d"] = win(b, 365).groupby("customer")["stockcode"].nunique()
    feat["n_returns_365d"] = win(r, 365).groupby("customer")["invoice"].nunique()
    feat["return_value_365d"] = -win(r, 365).groupby("customer")["amount"].sum()
    feat["trend_90_vs_365"] = feat["n_invoices_90d"] / (feat["n_invoices_365d"] / 4.0)
    feat["country_uk"] = (b.groupby("customer")["country"].agg(lambda s: s.mode().iat[0])
                          == "United Kingdom").astype(int)
    feat = feat.fillna(0.0)
    future = tx[(tx["invoicedate"] >= cutoff)
                & (tx["invoicedate"] < cutoff + pd.Timedelta(days=horizon_days))
                & (~tx["is_return"])]
    feat["churn"] = (~feat.index.isin(future["customer"].unique())).astype(int)
    feat["cutoff"] = cutoff
    return feat.reset_index()


def build_retail_panel(data_dir=None):
    cfg = config.RETAIL
    tx = load_retail_transactions(data_dir)
    cutoffs = pd.date_range(cfg["first_cutoff"], cfg["last_cutoff"], freq="MS")
    end = tx["invoicedate"].max()
    assert cutoffs[-1] + pd.Timedelta(days=cfg["horizon_days"]) <= end + pd.Timedelta(days=1), \
        "last_cutoff + horizon exceeds the end of the data -> last labels would be incomplete"
    panel = pd.concat([build_snapshot(tx, c, cfg["horizon_days"], cfg["active_window_days"])
                       for c in cutoffs], ignore_index=True)
    return panel
