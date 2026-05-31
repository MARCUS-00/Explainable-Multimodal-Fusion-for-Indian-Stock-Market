"""
xai/shap_explain.py
===================
SHAP-based explainability for the XGBoost model.

Outputs
-------
global_importance()   : top-N features by mean |SHAP| across test set
explain_prediction()  : per-row SHAP values + human-readable bullets
plot_summary()        : beeswarm summary plot (saved to file)
plot_waterfall()      : waterfall for a single prediction
"""

import os

import numpy as np
import pandas as pd
from typing import Optional

from models.xgboost.predict import load_xgb, build_x

try:
    import shap
    _SHAP_OK = True
except ImportError:
    _SHAP_OK = False

SHAP_REQUIRED_ERROR = "shap package required"


def _get_explainer(payload):
    base = payload.get("base_model")
    if base is None:
        raise RuntimeError("base_model missing from payload — re-train")
    return shap.TreeExplainer(base)


def global_importance(df_test: pd.DataFrame, payload=None, top_n: int = 20) -> pd.DataFrame:
    """
    Compute mean absolute SHAP value per feature across all test rows.
    Returns DataFrame sorted by importance descending.
    """
    if not _SHAP_OK:
        raise ImportError(SHAP_REQUIRED_ERROR)
    if payload is None:
        payload = load_xgb()

    X = build_x(df_test, payload)
    explainer   = _get_explainer(payload)
    shap_values = explainer.shap_values(X)   # list of arrays or (n_samples, n_features, n_classes) array

    # Aggregate: mean |SHAP| across classes and samples
    if isinstance(shap_values, list):
        # Old SHAP: list of per-class arrays
        mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    elif shap_values.ndim == 3:
        # (n_samples, n_features, n_classes)
        mean_abs = np.abs(shap_values).mean(axis=0).mean(axis=-1)
    else:
        # New SHAP binary: (n_samples, n_features)
        mean_abs = np.abs(shap_values).mean(axis=0)
        
    result   = pd.DataFrame({
        "feature":    X.columns.tolist(),
        "mean_shap":  mean_abs,
    }).sort_values("mean_shap", ascending=False).head(top_n).reset_index(drop=True)

    print(f"\n{'='*55}")
    print(f"  GLOBAL FEATURE IMPORTANCE (mean |SHAP|, top {top_n})")
    print(f"{'='*55}")
    for _, row in result.iterrows():
        bar = "█" * int(row["mean_shap"] / result["mean_shap"].max() * 30)
        print(f"  {row['feature']:<35s} {row['mean_shap']:.4f}  {bar}")
    print(f"{'='*55}\n")
    return result


def explain_prediction(row: pd.Series, payload=None, class_idx: int = 1) -> dict:
    """
    Compute SHAP values for a single row.
    class_idx: 0=DOWN, 1=UP (default UP)
    Returns dict with feature→shap_value mapping, sorted by |shap|.
    """
    if not _SHAP_OK:
        raise ImportError(SHAP_REQUIRED_ERROR)
    if payload is None:
        payload = load_xgb()

    X         = build_x(row.to_frame().T, payload)
    explainer = _get_explainer(payload)
    shap_vals = explainer.shap_values(X)

    # shap >= 0.40 returns ndarray (n_samples, n_features) for binary XGB.
    # Older versions return a list of two arrays [DOWN_vals, UP_vals].
    # Handle all three cases safely.
    if isinstance(shap_vals, list):
        # Old SHAP: list[class_idx] → (n_samples, n_features)
        class_idx = min(class_idx, len(shap_vals) - 1)
        sv = shap_vals[class_idx][0]
    elif shap_vals.ndim == 3:
        # (n_samples, n_features, n_classes)
        sv = shap_vals[0, :, class_idx]
    else:
        # New SHAP binary: (n_samples, n_features) — positive = pushes toward UP
        sv = shap_vals[0]

    result = dict(sorted(
        zip(X.columns.tolist(), sv.tolist()),
        key=lambda x: abs(x[1]), reverse=True
    ))
    return result


# Friendly names shown in the app instead of raw feature names
_FRIENDLY_NAMES = {
    "days_since_fundamental":   "Financials are outdated",
    "has_fundamental":          "Financial data available",
    "market_return_20d":        "Market trend (20 days)",
    "rolling_volatility_20d":   "Market volatility",
    "relative_strength_vs_market": "Stock vs market performance",
    "nifty_trend_20d":          "Overall market direction",
    "bullish_regime":           "Market in bullish phase",
    "bearish_regime":           "Market in bearish phase",
    "nifty_ret_5d":             "Market return this week",
    "ret_lag_1d":               "Stock return yesterday",
    "ret_lag_3d":               "Stock return (3 days)",
    "ret_lag_5d":               "Stock return this week",
    "rsi_norm":                 "Momentum (overbought/oversold)",
    "macd_hist_norm":           "Trend strength",
    "price_to_ema20":           "Price vs 20-day average",
    "price_to_ema50":           "Price vs 50-day average",
    "ema_cross":                "Short-term trend crossing long-term",
    "atr_ratio":                "Daily price swing size",
    "bb_pct":                   "Price position in range",
    "hist_vol_20d":             "Historical volatility (20d)",
    "volatility_5d":            "Recent volatility (5d)",
    "volatility_10d":           "Recent volatility (10d)",
    "vol_spike":                "Unusual trading volume today",
    "vol_breakout":             "Price broke recent high",
    "obv_change":               "Volume trend",
    "price_pos_20d":            "Price vs 20-day average",
    "momentum_5d":              "5-day price momentum",
    "momentum_10d":             "10-day price momentum",
    "momentum_strength":        "Momentum consistency",
    "pct_from_52h":             "Distance from 52-week high",
    "mom_z_5":                  "Momentum vs history (5d)",
    "mom_z_10":                 "Momentum vs history (10d)",
    "vol_z":                    "Volatility vs history",
    "cs_rank_ret":              "Return rank vs all stocks",
    "cs_rank_mom":              "Momentum rank vs all stocks",
    "consec_dir":               "Consecutive up/down days",
    "sector_ret_1d":            "Sector return yesterday",
    "sector_ret_5d":            "Sector return this week",
    "sector_rel_momentum":      "Stock vs sector momentum",
    "sector_encoded":           "Sector",
    "PE_Ratio":                 "Price-to-Earnings ratio",
    "ROE":                      "Return on equity",
    "Revenue_Growth":           "Revenue growth",
    "Profit_Growth":            "Profit growth",
    "news_score":               "News sentiment",
    "news_pos":                 "Positive news intensity",
    "news_neg":                 "Negative news intensity",
    "news_decay":               "Recent news sentiment",
    "news_sentiment_mom":       "Sentiment trend (improving/worsening)",
    "news_count_1d":            "News articles today",
    "news_count_3d":            "News articles (3 days)",
    "days_since_news":          "Days since last news",
    "has_news":                 "Recent news available",
    "is_rbi":                   "RBI policy event",
    "is_gdp":                   "GDP data release",
    "is_cpi":                   "Inflation data release",
    "is_budget":                "Budget event",
    "is_earnings":              "Earnings announcement",
    "is_dividend":              "Dividend announcement",
    "ret_vs_nifty_1d":          "Stock vs market yesterday",
    "ret_vs_nifty_5d":          "Stock vs market this week",
    "sector_ret_loo":           "Sector peers performance",
}

def _friendly(feat: str, shap_val: float) -> str:
    """Convert a raw feature + SHAP value into a plain-English sentence."""
    name = _FRIENDLY_NAMES.get(feat, feat.replace("_", " ").title())
    impact = "pushing price UP" if shap_val > 0 else "pulling price DOWN"
    strength = abs(shap_val)

    # Strength label
    if strength > 0.3:
        strength_word = "strongly"
    elif strength > 0.1:
        strength_word = "moderately"
    else:
        strength_word = "slightly"

    emoji = "📈" if shap_val > 0 else "📉"
    return f"{emoji} {name} is {strength_word} {impact}"


def build_explanation_bullets(row: pd.Series, payload=None, max_bullets: int = 5) -> list:
    """Friendly plain-English explanation bullets for the top SHAP drivers."""
    try:
        shap_map = explain_prediction(row, payload, class_idx=1)
    except Exception:
        return ["Explanation unavailable"]

    bullets = []
    for feat, shap_val in list(shap_map.items())[:max_bullets]:
        bullets.append(_friendly(feat, shap_val))
    return bullets


def plot_summary(df_test: pd.DataFrame, payload=None, save_path: Optional[str] = None):
    """Beeswarm summary plot. Requires matplotlib."""
    if not _SHAP_OK:
        raise ImportError(SHAP_REQUIRED_ERROR)
    if payload is None:
        payload = load_xgb()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    X         = build_x(df_test, payload)
    explainer = _get_explainer(payload)
    shap_vals = explainer.shap_values(X)

    # Show SHAP for UP class
    plt.figure(figsize=(10, 8))
    if isinstance(shap_vals, list):
        up_shap = shap_vals[1]
    elif shap_vals.ndim == 3:
        up_shap = shap_vals[..., 1]
    else:
        # New SHAP binary: (n_samples, n_features) — already UP direction
        up_shap = shap_vals
        
    shap.summary_plot(up_shap, X, show=False, max_display=20)
    plt.title("SHAP Summary — P(UP) drivers")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[SHAP] Summary plot saved → {save_path}")
    else:
        plt.show()
    plt.close()


def plot_waterfall(row: pd.Series, payload=None, save_path: Optional[str] = None):
    """Waterfall plot for a single prediction."""
    if not _SHAP_OK:
        raise ImportError(SHAP_REQUIRED_ERROR)
    if payload is None:
        payload = load_xgb()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    X         = build_x(row.to_frame().T, payload)
    explainer = _get_explainer(payload)
    ev        = explainer(X)

    shap.plots.waterfall(ev[0, :, 1], show=False)   # class_idx=1 → UP
    plt.title(f"SHAP Waterfall — {row.get('Stock', 'Stock')} P(UP)")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[SHAP] Waterfall saved → {save_path}")
    else:
        plt.show()
    plt.close()


def plot_calibration(results_df: pd.DataFrame, save_path: str):
    """
    Calibration plot: bins prob_up into 10 equal-width buckets and plots
    mean predicted probability vs actual UP rate per bucket.
    Saves to `save_path`.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if "prob_up" not in results_df.columns or "label" not in results_df.columns:
        raise KeyError("results_df must contain 'prob_up' and 'label' columns")

    df = results_df.copy()
    df = df.dropna(subset=["prob_up", "label"]) 
    df["prob_up"] = pd.to_numeric(df["prob_up"], errors="coerce")
    df = df.dropna(subset=["prob_up"]) 
    df["bin"] = pd.cut(df["prob_up"], bins=10)
    grp = df.groupby("bin").agg(mean_pred=("prob_up", "mean"),
                                  actual_up=("label", lambda x: (x==1).mean()),
                                  n=("label", "size")).reset_index()

    plt.figure(figsize=(6, 6))
    plt.plot(grp["mean_pred"], grp["actual_up"], marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    for _, r in grp.iterrows():
        plt.text(r["mean_pred"], r["actual_up"], str(int(r["n"])), fontsize=8,
                 ha="center", va="bottom")
    plt.xlabel("Mean predicted P(UP)")
    plt.ylabel("Observed UP rate")
    plt.title("Calibration plot")
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[SHAP] Calibration plot saved → {save_path}")