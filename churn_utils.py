"""Shared helpers for the churn project (imported by the notebooks)."""
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

warnings.filterwarnings("ignore", category=DeprecationWarning)  # harmless pandas/NumPy notice

WINDOW_DAYS = 90                         # churn = no purchase in this many days after the cutoff
TRAIN_CUTOFF = pd.Timestamp("2011-06-10")
TEST_CUTOFF = pd.Timestamp("2011-09-09")
ACTIVE_DAYS = 180                        # "active" = bought within this many days before the cutoff
assert TRAIN_CUTOFF + pd.Timedelta(days=WINDOW_DAYS) <= TEST_CUTOFF, "label windows overlap"


def build_features(sales, returns, cutoff, window_days=WINDOW_DAYS):
    """One row per customer, using ONLY data before `cutoff`.

    The churn label is 1 if the customer made no purchase in the `window_days`
    days after the cutoff. Index = customer_id.
    """
    cutoff = pd.Timestamp(cutoff)
    label_end = cutoff + pd.Timedelta(days=window_days)
    hist = sales[sales["invoice_date"] < cutoff]
    future = sales[(sales["invoice_date"] >= cutoff) & (sales["invoice_date"] < label_end)]
    hist_ret = returns[returns["invoice_date"] < cutoff]
    assert hist["invoice_date"].max() < cutoff, "future data leaked into features"

    # One row per invoice, then one row per customer
    inv = (hist.groupby(["customer_id", "invoice"])
               .agg(inv_date=("invoice_date", "min"), inv_value=("revenue", "sum"),
                    inv_items=("quantity", "sum"))
               .reset_index().sort_values(["customer_id", "inv_date"]))
    g = inv.groupby("customer_id")
    feat = pd.DataFrame({
        "first_purchase": g["inv_date"].min(), "last_purchase": g["inv_date"].max(),
        "frequency": g["invoice"].nunique(), "monetary": g["inv_value"].sum(),
        "total_items": g["inv_items"].sum()})

    # Recency, frequency, monetary and basic behaviour
    feat["recency_days"] = (cutoff - feat["last_purchase"]).dt.days
    feat["tenure_days"] = (cutoff - feat["first_purchase"]).dt.days
    feat["avg_order_value"] = feat["monetary"] / feat["frequency"]
    feat["avg_items_per_order"] = feat["total_items"] / feat["frequency"]
    feat["n_unique_products"] = hist.groupby("customer_id")["stock_code"].nunique()

    # Average gap between orders (NaN for one-time buyers; the models handle it)
    inv["gap_days"] = inv.groupby("customer_id")["inv_date"].diff().dt.total_seconds() / 86400
    feat["avg_gap_days"] = inv.groupby("customer_id")["gap_days"].mean()

    # Recent activity windows
    for d in (30, 60, 90):
        recent = inv[inv["inv_date"] >= cutoff - pd.Timedelta(days=d)]
        feat[f"orders_last_{d}d"] = recent.groupby("customer_id")["invoice"].nunique()
        feat[f"spend_last_{d}d"] = recent.groupby("customer_id")["inv_value"].sum()
    recent_cols = [c for c in feat.columns if "_last_" in c]
    feat[recent_cols] = feat[recent_cols].fillna(0)
    feat["spend_share_last_90d"] = feat["spend_last_90d"] / feat["monetary"]

    # Returns (only before the cutoff)
    r = hist_ret.groupby("customer_id").agg(return_lines=("invoice", "size"),
                                            return_value=("revenue", "sum"))
    feat["return_lines"] = r["return_lines"]
    feat["return_value"] = -r["return_value"]
    feat[["return_lines", "return_value"]] = feat[["return_lines", "return_value"]].fillna(0)
    feat["return_rate"] = feat["return_value"] / feat["monetary"]

    # Country and bulk-buyer flag
    country = hist.sort_values("invoice_date").groupby("customer_id")["country"].last()
    feat["is_uk"] = (country.reindex(feat.index) == "United Kingdom").astype(int)
    feat["has_bulk_order"] = hist.groupby("customer_id")["is_outlier"].any().astype(int)

    # Label
    active_later = set(future["customer_id"].unique())
    feat["churn"] = (~feat.index.isin(active_later)).astype(int)
    return feat.drop(columns=["first_purchase", "last_purchase"])


def top_decile_lift(y, p):
    """Churn rate among the 10% highest-scored customers divided by the overall churn rate."""
    y, p = np.asarray(y), np.asarray(p)
    k = max(1, int(0.1 * len(y)))
    return y[np.argsort(-p)[:k]].mean() / y.mean()


def metrics_with_ci(y, p, n_boot=300, seed=42):
    """ROC-AUC, PR-AUC and top-decile lift, each with a bootstrap 95% interval (as text)."""
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    resamples = [rng.integers(0, len(y), len(y)) for _ in range(n_boot)]
    fns = {"ROC-AUC": roc_auc_score, "PR-AUC": average_precision_score,
           "Top-decile lift": top_decile_lift}
    out = {}
    for name, fn in fns.items():
        boots = [fn(y[i], p[i]) for i in resamples if y[i].min() != y[i].max()]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out[name] = f"{fn(y, p):.3f} ({lo:.3f}-{hi:.3f})"
    return out


def prior_shift(p, to_rate, from_rate=None):
    """Shift probabilities (on the log-odds scale) so their average equals `to_rate`.

    Churn rates change between periods (and class weights distort probabilities), so the
    raw average is often off. `from_rate` is unused and kept only for compatibility.
    """
    from scipy.optimize import brentq
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p))
    sigmoid = lambda shift: 1 / (1 + np.exp(-(z + shift)))
    shift = brentq(lambda s: sigmoid(s).mean() - to_rate, -10, 10)
    return sigmoid(shift)
