"""
features/sentiment_decay.py
============================
Per-stock optimal news sentiment decay constants learned from training data.

The standard approach uses a global decay half-life (NEWS_DECAY_K=5 days).
This module replaces that with a stock-personalised constant learned by
maximising |correlation(decayed_sentiment, forward_return)| on the training
set only - no lookahead.

Novelty
-------
Per-stock decay personalisation for Nifty 50 equities. Commodity / metals
stocks forget news in ~2 days; consumer staples carry sentiment for 2+ weeks.
Learning k from data rather than fixing it globally is the key contribution.

No-lookahead guarantees
-----------------------
* Learning uses TRAIN data only (Date <= TRAIN_END).
* Learned constants are then applied to all rows (train + val + test).
* No forward information enters the constant-learning step.

Usage
-----
    from features.sentiment_decay import learn_all_decay_constants, apply_personalised_decay

    decay_map = learn_all_decay_constants(tech_df, finbert_df)
    df["news_decay"] = apply_personalised_decay(df, decay_map)

Standalone: python features/sentiment_decay.py [--force]
"""

import argparse
import json
import logging
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from config.settings import (
    DECAY_CONSTANTS_PATH, DECAY_K_CANDIDATES, DEFAULT_DECAY_K,
    FINBERT_CSV, LABEL_HORIZON, STOCKS, TECHNICAL_CSV, TRAIN_END,
)

log = logging.getLogger(__name__)

_MIN_NEWS_ROWS = 15
_MIN_LABEL_ROWS = 40


def _apply_decay(last_score: pd.Series, days_since: pd.Series, k: float) -> pd.Series:
    k = max(k, 1e-6)
    return last_score.fillna(0.0) * np.exp(-days_since.fillna(999.0) / k)


def _build_stock_series(stock: str,
                        tech_df: pd.DataFrame,
                        news_df: pd.DataFrame) -> pd.DataFrame:
    """Build aligned (fwd_ret, last_score, days_since) series for one stock."""
    s_tech = (tech_df[tech_df["Stock"] == stock]
              .sort_values("Date")
              .query("Date <= @TRAIN_END")
              .copy())

    if len(s_tech) < _MIN_LABEL_ROWS:
        return pd.DataFrame()

    s_tech["fwd_ret"] = s_tech["Close"].shift(-LABEL_HORIZON) / s_tech["Close"] - 1
    s_tech = s_tech.dropna(subset=["fwd_ret", "Close"])
    if len(s_tech) < _MIN_LABEL_ROWS:
        return pd.DataFrame()

    s_news = (news_df[news_df["Stock"] == stock]
              .sort_values("Date")
              .query("Date <= @TRAIN_END")
              .copy())

    if "finbert_pos" in s_news.columns and "finbert_neg" in s_news.columns:
        s_news["news_score"] = s_news["finbert_pos"] - s_news["finbert_neg"]
    elif "news_score" not in s_news.columns:
        return pd.DataFrame()

    s_news = s_news[["Date", "news_score"]].dropna()
    if len(s_news) < _MIN_NEWS_ROWS:
        return pd.DataFrame()

    merged = (s_tech[["Date", "fwd_ret"]]
              .merge(s_news[["Date", "news_score"]], on="Date", how="left")
              .sort_values("Date")
              .reset_index(drop=True))

    # Treat 0 as "no news" before ffill so zeros don't overwrite real sentiment
    merged["last_score"] = merged["news_score"].replace(0, np.nan).ffill().fillna(0.0)
    last_news_date = merged["Date"].where(
        merged["news_score"].replace(0, np.nan).notna()
    ).ffill()
    merged["days_since"] = (
        (merged["Date"] - last_news_date).dt.days.clip(lower=0).fillna(999)
    )
    return merged[["Date", "fwd_ret", "last_score", "days_since"]]


def learn_decay_constant(stock: str,
                          tech_df: pd.DataFrame,
                          news_df: pd.DataFrame) -> float:
    """Find the k in DECAY_K_CANDIDATES that maximises signed corr(decayed, fwd_ret).

    We prefer positive correlations; if no candidate yields a positive
    Pearson r then fall back to DEFAULT_DECAY_K to avoid picking a spurious
    negative association.
    """
    series = _build_stock_series(stock, tech_df, news_df)
    if series.empty:
        return DEFAULT_DECAY_K

    best_k, best_r = DEFAULT_DECAY_K, float("-inf")
    for k in DECAY_K_CANDIDATES:
        decayed = _apply_decay(series["last_score"], series["days_since"], k)
        mask = series["last_score"].notna() & (decayed != 0)
        if mask.sum() < _MIN_NEWS_ROWS:
            continue
        try:
            r, _ = pearsonr(decayed[mask], series.loc[mask, "fwd_ret"])
        except Exception:
            continue
        if np.isfinite(r) and r > best_r:
            best_r = r
            best_k = k

    # Require a positive correlation to accept the learned k, otherwise fallback.
    if best_r <= 0 or not np.isfinite(best_r):
        return DEFAULT_DECAY_K
    return best_k


def learn_all_decay_constants(tech_df: pd.DataFrame,
                               news_df: pd.DataFrame,
                               force_relearn: bool = False) -> dict:
    """Learn and cache per-stock decay constants."""
    if not force_relearn and os.path.exists(DECAY_CONSTANTS_PATH):
        with open(DECAY_CONSTANTS_PATH) as fh:
            constants = json.load(fh)
        log.info("Loaded cached decay constants (%d stocks).", len(constants))
        return constants

    log.info("Learning per-stock decay constants (%d stocks)...", len(STOCKS))

    # Normalise tickers consistently
    for df_ in (tech_df, news_df):
        df_["Stock"] = (df_["Stock"].astype(str)
                        .str.replace(r"\.(NS|BO)$", "", regex=True).str.strip())
        df_["Date"] = pd.to_datetime(df_["Date"])

    constants = {stock: learn_decay_constant(stock, tech_df, news_df) for stock in STOCKS}

    os.makedirs(os.path.dirname(DECAY_CONSTANTS_PATH), exist_ok=True)
    with open(DECAY_CONSTANTS_PATH, "w") as fh:
        json.dump(constants, fh, indent=2)
    log.info("Saved decay constants -> %s", DECAY_CONSTANTS_PATH)
    _print_summary(constants)
    return constants


def apply_personalised_decay(df: pd.DataFrame, decay_constants: dict) -> pd.Series:
    """
    Apply per-stock decay constants.
    Requires columns: Stock, news_score, days_since_news.
    Treats news_score=0 as "no news" before forward-filling.
    """
    result = pd.Series(0.0, index=df.index, dtype=float)

    if "_last_news_score" in df.columns:
        last_score_col = "_last_news_score"
    else:
        _last = df.groupby("Stock")["news_score"].transform(
            lambda x: x.replace(0, np.nan).ffill().fillna(0.0)
        )
        df = df.copy()
        df["__tmp_last_score"] = _last
        last_score_col = "__tmp_last_score"

    for stock in df["Stock"].unique():
        mask = df["Stock"] == stock
        k = decay_constants.get(stock, DEFAULT_DECAY_K)
        result[mask] = _apply_decay(
            df.loc[mask, last_score_col],
            df.loc[mask, "days_since_news"],
            k=k,
        ).values
    return result


def _print_summary(constants: dict) -> None:
    interp = {
        1: "very fast  (1-day memory)",
        2: "fast       (2-day memory)",
        3: "moderate   (3-day memory)",
        5: "default    (5-day memory)",
        7: "slow       (1-week memory)",
        10: "slow      (10-day memory)",
        14: "very slow  (2-week memory)",
    }
    print("\n" + "=" * 58)
    print("  PER-STOCK SENTIMENT DECAY CONSTANTS (learned from data)")
    print("=" * 58)
    for stock, k in sorted(constants.items(), key=lambda x: (x[1], x[0])):
        print(f"  {stock:<14} {k:>4}   {interp.get(k, f'{k}-day memory')}")
    values = list(constants.values())
    print(f"\n  Mean k : {np.mean(values):.1f}  Min : {min(values)}  Max : {max(values)}")
    print("=" * 58 + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="  [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(
        description="Learn per-stock sentiment decay constants and cache them."
    )
    parser.add_argument("--force", action="store_true", help="Force re-learning even if cache exists")
    args = parser.parse_args()

    if not os.path.exists(TECHNICAL_CSV):
        raise SystemExit(f"[ERROR] {TECHNICAL_CSV} not found.")
    if not os.path.exists(FINBERT_CSV):
        raise SystemExit(f"[ERROR] {FINBERT_CSV} not found.")

    tech = pd.read_csv(TECHNICAL_CSV, parse_dates=["Date"])
    news = pd.read_csv(FINBERT_CSV, parse_dates=["Date"])
    learn_all_decay_constants(tech, news, force_relearn=args.force)
