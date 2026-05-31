"""
features/build_features.py
===========================
Single-file feature engineering with ZERO lookahead bias.

Rules enforced
--------------
* Sort by (Stock, Date) before every group operation.
* `ret_lag_*` and `sector_ret_*` features are shifted by ≥1 day.
* `momentum_*`, `pct_from_52h`, `cs_rank_mom`, and the z-scored alpha
  features use Close_T as the prediction anchor (valid for end-of-day
  prediction targeting Close_{T+LABEL_HORIZON}).
* Raw ATR / EMA / OBV never used directly - only normalised derivatives.
* No bfill anywhere.
* Sector LOO return excludes self to prevent contamination.
* Forward-return label computed last and stripped from the feature matrix.
* Nifty features shifted(1) - no same-day leakage.
* Fundamentals merged with merge_asof(direction='backward') only.
"""

import logging
import os

import numpy as np
import pandas as pd
import yfinance as yf

from config.settings import (
    DATE_END, DATE_START,
    EVENTS_CSV, FINBERT_CSV, FUNDAMENTAL_CSV,
    LABEL_HORIZON,
    MERGED_CSV, NIFTY_CACHE,
    SECTOR_MAP, SECTOR_TO_CODE,
    TECHNICAL_CSV,
)

log = logging.getLogger(__name__)

# --- Constants ----------------------------------------------------------------
# NOTE: The global threshold constant from settings.py is NOT used for label
# generation. The actual labels use a per-stock adaptive threshold:
#     thresh = (stock_vol * 0.5).clip(0.005, 0.025)
# See build_label() below (search for `_fwd_ret`). The settings.py value
# is retained for reference only; do not reintroduce a global threshold
# without updating build_label() in lockstep.
_NEWS_DECAY_K = 5    # global half-life for news decay (days); per-stock values override this
_EPS = 1e-8

_TICKER_ALIAS = {
    "M_M": "M&M", "M-M": "M&M", "MM": "M&M",
    "BAJAJ_AUTO": "BAJAJ-AUTO", "BAJAJAUTO": "BAJAJ-AUTO",
    "HDFC BANK": "HDFCBANK", "HDFC-BANK": "HDFCBANK",
}


def _norm_ticker(series: pd.Series) -> pd.Series:
    """Normalise ticker strings: strip .NS/.BO suffix and apply known aliases."""
    return (
        series.astype(str)
              .str.replace(r"\.(NS|BO)$", "", regex=True)
              .str.strip()
              .map(lambda t: _TICKER_ALIAS.get(t, t))
    )


# --- Nifty index data ---------------------------------------------------------

def _load_nifty() -> pd.DataFrame:
    """Download Nifty 50; fall back to on-disk cache. Returns daily frame with
    shifted returns to prevent same-day leakage."""
    raw = pd.DataFrame()
    try:
        raw = yf.download("^NSEI", start=DATE_START, end=DATE_END,
                          auto_adjust=True, progress=False)
    except Exception as exc:
        log.warning("yfinance ^NSEI download failed: %s", exc)

    if raw is None or raw.empty:
        if os.path.exists(NIFTY_CACHE):
            log.warning("Using cached Nifty data from %s", NIFTY_CACHE)
            return pd.read_csv(NIFTY_CACHE, parse_dates=["Date"])
        raise RuntimeError("No Nifty data available and no cache found.")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    nifty = raw[["Close"]].copy()
    nifty.index = pd.to_datetime(nifty.index)
    nifty = nifty.sort_index()
    # shift(1) prevents same-day leakage into features
    nifty["nifty_ret_1d"] = nifty["Close"].pct_change(1).shift(1)
    nifty["nifty_ret_5d"] = nifty["Close"].pct_change(5).shift(1)
    nifty = (nifty.drop(columns=["Close"])
                  .reset_index()
                  .rename(columns={"index": "Date"}))
    nifty.columns.name = None
    nifty["Date"] = pd.to_datetime(nifty["Date"])

    os.makedirs(os.path.dirname(NIFTY_CACHE), exist_ok=True)
    nifty.to_csv(NIFTY_CACHE, index=False)
    return nifty


# --- Technical features -------------------------------------------------------

def build_technical(tech: pd.DataFrame) -> pd.DataFrame:
    """
    Compute normalised / lagged technical features.

    Input : technical.csv columns (Close, RSI, MACD, MACD_signal,
            ATR, OBV, EMA_20, Volume, Return_1d).
    Output: same rows plus engineered feature columns.

    Raw ATR / EMA / OBV are NOT in the returned feature list; only their
    normalised derivatives appear in XGBOOST_FEATURES.
    """
    df = tech.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Stock"] = _norm_ticker(df["Stock"])
    df = df.sort_values(["Stock", "Date"]).reset_index(drop=True)
    g = df.groupby("Stock", group_keys=False)

    # Lagged returns (all shift >= 1 - no lookahead)
    df["ret_lag_1d"] = g["Return_1d"].transform(lambda x: x.shift(1))
    df["ret_lag_3d"] = g["Close"].transform(lambda x: x.pct_change(3).shift(1))
    df["ret_lag_5d"] = g["Close"].transform(lambda x: x.pct_change(5).shift(1))

    # Trend: normalised distance from exponential moving averages
    ema20 = g["Close"].transform(lambda x: x.ewm(span=20, adjust=False).mean())
    ema50 = g["Close"].transform(lambda x: x.ewm(span=50, adjust=False).mean())
    df["price_to_ema20"] = df["Close"] / (ema20 + _EPS) - 1
    df["price_to_ema50"] = df["Close"] / (ema50 + _EPS) - 1
    df["ema_cross"] = (ema20 > ema50).astype(int)

    # Momentum
    df["rsi_norm"] = df["RSI"] / 100.0
    df["macd_hist_norm"] = (df["MACD"] - df["MACD_signal"]) / (df["Close"].abs() + _EPS)
    df["momentum_5d"] = g["Close"].transform(lambda x: x.pct_change(5))
    df["momentum_10d"] = g["Close"].transform(lambda x: x.pct_change(10))
    df["momentum_diff"] = df["momentum_5d"] - df["momentum_10d"]
    df["momentum_strength"] = g["momentum_5d"].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )

    # Volatility: ATR scaled by price (dimensionless), Bollinger %, rolling std
    df["atr_ratio"] = df["ATR"] / (df["Close"].abs() + _EPS)
    bb_mid = g["Close"].transform(lambda x: x.rolling(20, min_periods=5).mean())
    bb_std = g["Close"].transform(lambda x: x.rolling(20, min_periods=5).std())
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    df["bb_pct"] = (df["Close"] - bb_lower) / (bb_upper - bb_lower + _EPS)
    df["volatility_5d"] = g["Return_1d"].transform(lambda x: x.rolling(5, min_periods=2).std())
    df["volatility_10d"] = g["Return_1d"].transform(lambda x: x.rolling(10, min_periods=3).std())
    df["hist_vol_20d"] = g["Return_1d"].transform(
        lambda x: x.rolling(20, min_periods=5).std()
    )

    # Volume
    df["obv_change"] = g["OBV"].transform(lambda x: x.pct_change(5))
    vol_ma20 = g["Volume"].transform(lambda x: x.rolling(20, min_periods=1).mean())
    df["vol_spike"] = (df["Volume"] > vol_ma20 * 1.5).astype(int)
    close_max20 = g["Close"].transform(lambda x: x.rolling(20, min_periods=1).max())
    close_ma20 = g["Close"].transform(lambda x: x.rolling(20, min_periods=1).mean())
    # shift(1) on prior max prevents same-day leakage
    df["vol_breakout"] = (df["Close"] > close_max20.shift(1)).astype(int)
    df["price_pos_20d"] = df["Close"] / (close_ma20 + _EPS)

    return df


# --- Market / Sector features -------------------------------------------------

def build_market_sector(df: pd.DataFrame, nifty: pd.DataFrame) -> pd.DataFrame:
    """Attach Nifty-relative and sector LOO return features."""
    df = df.merge(nifty[["Date", "nifty_ret_1d", "nifty_ret_5d"]],
                  on="Date", how="left")
    # Alpha vs market - uses lagged return on left (already shifted in _load_nifty)
    df["ret_vs_nifty_1d"] = df["ret_lag_1d"] - df["nifty_ret_1d"]
    df["ret_vs_nifty_5d"] = df["ret_lag_5d"] - df["nifty_ret_5d"]

    # Sector assignment
    df["Sector"] = df["Stock"].map(SECTOR_MAP).fillna("Unknown")

    # Sector LOO (leave-one-out) return: exclude self to avoid contamination
    day_sector = (
        df.groupby(["Date", "Sector"], as_index=False)
          .agg(_sum=("ret_lag_1d", "sum"), _cnt=("ret_lag_1d", "count"))
    )
    df = df.merge(day_sector, on=["Date", "Sector"], how="left")
    df["sector_ret_loo"] = (
        (df["_sum"] - df["ret_lag_1d"]) / (df["_cnt"] - 1).clip(lower=1)
    )
    df.drop(columns=["_sum", "_cnt"], inplace=True)

    # Sector returns shifted 1 day for lag features
    sector_raw = (
        df.groupby(["Date", "Sector"])["Return_1d"]
          .mean().rename("_sect_raw").reset_index()
    )
    sector_raw = sector_raw.sort_values(["Sector", "Date"])
    sector_raw["sector_ret_1d"] = (
        sector_raw.groupby("Sector")["_sect_raw"]
                  .transform(lambda x: x.shift(1))
    )
    sector_raw["sector_ret_5d"] = (
        sector_raw.groupby("Sector")["_sect_raw"]
                  .transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum())
    )
    sector_raw.drop(columns=["_sect_raw"], inplace=True)
    df = df.merge(sector_raw, on=["Date", "Sector"], how="left")

    df["sector_ret_1d"] = df["sector_ret_1d"].fillna(0.0)
    df["sector_ret_5d"] = df["sector_ret_5d"].fillna(0.0)
    df["sector_rel_momentum"] = df["momentum_10d"] - df["sector_ret_5d"]
    df["sector_encoded"] = df["Sector"].map(SECTOR_TO_CODE).fillna(-1).astype(int)
    return df


# --- Regime features ----------------------------------------------------------

def build_regime_features(df: pd.DataFrame, nifty: pd.DataFrame) -> pd.DataFrame:
    """
    Market-regime features derived from Nifty index data.
    All values use shift >= 1 - no same-day leakage.

    Features added
    --------------
    market_return_20d        Nifty 20-day return (shifted 1 day)
    nifty_trend_20d          sign of market_return_20d (+1 / -1 / 0)
    rolling_volatility_20d   20-day std of Nifty daily returns (shifted 1)
    relative_strength_vs_market  stock 20d return minus Nifty 20d return
    bullish_regime           1 when trend > 0 and vol <= 1.2 x median
    bearish_regime           1 when trend < 0 and vol >= 0.8 x median
    """
    if os.path.exists(NIFTY_CACHE):
        nc = pd.read_csv(NIFTY_CACHE, parse_dates=["Date"])
    else:
        nc = nifty.copy()

    nc = nc.sort_values("Date").reset_index(drop=True)

    # Derive market-level features from already-shifted daily returns present
    # in the Nifty cache (preferred). This avoids re-downloading or relying on
    # an unshifted Close column which could reintroduce same-day leakage.
    ret_col = "nifty_ret_1d" if "nifty_ret_1d" in nc.columns else None
    if ret_col:
        nc["market_return_20d"] = nc[ret_col].rolling(20, min_periods=5).sum().shift(1)
        nc["rolling_volatility_20d"] = nc[ret_col].rolling(20, min_periods=5).std().shift(1)
    else:
        nc["market_return_20d"] = 0.0
        nc["rolling_volatility_20d"] = 0.0

    nc["nifty_trend_20d"] = np.sign(nc["market_return_20d"]).fillna(0).astype(int)

    # FIX B1: compute volatility median on TRAIN ROWS ONLY (no test-set leakage).
    # The threshold defines bullish_regime / bearish_regime labels; if computed on
    # the full timeline, future test-set volatility leaks into training labels.
    from config.settings import TRAIN_END
    train_mask = nc["Date"] <= pd.to_datetime(TRAIN_END)
    vol_median = nc.loc[train_mask, "rolling_volatility_20d"].median()
    if not np.isfinite(vol_median):
        log.warning("vol_median NaN on train slice — falling back to global median")
        vol_median = nc["rolling_volatility_20d"].median()
    nc["bullish_regime"] = (
        (nc["nifty_trend_20d"] > 0) &
        (nc["rolling_volatility_20d"] <= vol_median * 1.2)
    ).astype(int)
    nc["bearish_regime"] = (
        (nc["nifty_trend_20d"] < 0) &
        (nc["rolling_volatility_20d"] >= vol_median * 0.8)
    ).astype(int)

    regime_cols = ["Date", "market_return_20d", "nifty_trend_20d",
                   "rolling_volatility_20d", "bullish_regime", "bearish_regime"]
    nc_merge = nc[[c for c in regime_cols if c in nc.columns]].drop_duplicates("Date")

    df = df.merge(nc_merge, on="Date", how="left")

    stock_ret_20d = df.groupby("Stock")["Close"].transform(
        lambda x: x.pct_change(20).shift(1)
    )
    df["relative_strength_vs_market"] = stock_ret_20d - df["market_return_20d"].fillna(0)

    for col in ["market_return_20d", "nifty_trend_20d", "rolling_volatility_20d",
                "bullish_regime", "bearish_regime", "relative_strength_vs_market"]:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = df[col].fillna(0.0)

    log.info("Regime features added.")
    return df


# --- Fundamentals -------------------------------------------------------------

def build_fundamentals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge fundamental data using merge_asof(direction='backward').
    No bfill. No zero-fill. Sector-year median imputation as fallback.
    """
    fund_cols = ["PE_Ratio", "ROE", "Revenue_Growth", "Profit_Growth"]

    if not os.path.exists(FUNDAMENTAL_CSV):
        log.warning("Fundamental CSV missing - skipping")
        for col in fund_cols:
            df[col] = np.nan
        df["has_fundamental"] = 0
        df["days_since_fundamental"] = np.nan
        return df

    fund = pd.read_csv(FUNDAMENTAL_CSV)
    fund["Stock"] = _norm_ticker(fund["Stock"])
    fund["fundamental_date"] = pd.to_datetime(
        fund["Year"].astype(int).astype(str) + "-12-31"
    )
    fund = (fund.sort_values(["Stock", "fundamental_date"])
                .drop_duplicates(subset=["Stock", "fundamental_date"], keep="last"))

    df = df.sort_values("Date").reset_index(drop=True)
    fund_sorted = fund[["Stock", "fundamental_date"] + fund_cols].sort_values(
        "fundamental_date"
    ).reset_index(drop=True)
    # R4: assert sort precondition for merge_asof
    assert df["Date"].is_monotonic_increasing, "Left frame not sorted by Date"
    assert fund_sorted["fundamental_date"].is_monotonic_increasing, "Right frame not sorted"
    merged = pd.merge_asof(
        df, fund_sorted,
        left_on="Date", right_on="fundamental_date",
        by="Stock", direction="backward",
    )
    merged = merged.sort_values(["Stock", "Date"]).reset_index(drop=True)

    # Forward-fill within stock (propagates last known fundamental forward)
    for col in fund_cols:
        merged[col] = merged.groupby("Stock")[col].transform(lambda x: x.ffill())
    merged["fundamental_date"] = merged.groupby("Stock")["fundamental_date"].transform(
        lambda x: x.ffill()
    )

    merged["has_fundamental"] = merged[fund_cols].notna().all(axis=1).astype(int)
    merged["days_since_fundamental"] = (
        (merged["Date"] - merged["fundamental_date"]).dt.days
    )

    # FIX B3: sector-year / year / global median imputation uses TRAIN ROWS ONLY.
    from config.settings import TRAIN_END
    _train_end = pd.to_datetime(TRAIN_END)
    merged["_year"] = merged["Date"].dt.year
    train_source = merged[merged["Date"] <= _train_end].copy()
    for col in fund_cols:
        if not merged[col].isna().any():
            continue
        train_col = train_source[train_source[col].notna()].copy()
        sec_year_med = train_col.groupby(["Sector", "_year"])[col].median()
        year_med = train_col.groupby("_year")[col].median()
        global_med = train_col[col].median()

        missing = merged[col].isna()
        if missing.any():
            merged.loc[missing, col] = [
                sec_year_med.get((sector, year), np.nan)
                for sector, year in zip(merged.loc[missing, "Sector"], merged.loc[missing, "_year"])
            ]

        missing = merged[col].isna()
        if missing.any():
            merged.loc[missing, col] = merged.loc[missing, "_year"].map(year_med)

        missing = merged[col].isna()
        if missing.any():
            merged.loc[missing, col] = global_med if np.isfinite(global_med) else 0.0
    merged.drop(columns=["_year"], inplace=True)

    log.info("Fundamentals merged | has_fundamental=1: %d rows",
             int(merged["has_fundamental"].sum()))
    return merged


# --- News sentiment -----------------------------------------------------------

def build_news(df: pd.DataFrame) -> pd.DataFrame:
    """
    Load FinBERT sentiment scores and compute:
      news_score        finbert_pos - finbert_neg (latest per stockxday)
      news_pos / neg    positive / negative intensity separately
      news_count_1d/3d  article counts
      news_decay        decayed sentiment score (per-stock k if available)
      news_sentiment_mom  decay minus 5-day rolling average (trend)
      days_since_news
      has_news

    Merged with left-join + ffill - no future news used.

    Design note: separate pos / neg features let the model weight a
    negative-news spike (e.g. earnings miss) differently from a low
    positive score - qualitatively different market signals.
    """
    new_cols = ["news_score", "news_pos", "news_neg", "news_decay",
                "news_count_1d", "news_count_3d", "news_sentiment_mom",
                "days_since_news", "has_news"]

    if not os.path.exists(FINBERT_CSV):
        log.warning("finbert_scores.csv not found - news features set to 0")
        for col in new_cols:
            df[col] = 0.0
        return df

    fb = pd.read_csv(FINBERT_CSV, parse_dates=["Date"])
    fb["Stock"] = _norm_ticker(fb["Stock"])

    fb_agg = (
        fb.groupby(["Date", "Stock"], as_index=False)
          .agg(
              finbert_pos=("finbert_pos", "mean"),
              finbert_neg=("finbert_neg", "mean"),
              news_count_1d=("finbert_pos", "count"),
          )
    )
    fb_agg["news_score"] = fb_agg["finbert_pos"] - fb_agg["finbert_neg"]
    fb_agg = fb_agg.rename(columns={"finbert_pos": "news_pos", "finbert_neg": "news_neg"})
    fb_agg = (fb_agg[["Date", "Stock", "news_score", "news_pos", "news_neg", "news_count_1d"]]
                    .sort_values(["Stock", "Date"]))

    df = df.sort_values(["Stock", "Date"])
    df = df.merge(fb_agg, on=["Date", "Stock"], how="left")

    df["news_count_1d"] = df["news_count_1d"].fillna(0).astype(int)
    df["news_count_3d"] = (
        df.groupby("Stock")["news_count_1d"]
          .transform(lambda x: x.rolling(3, min_periods=1).sum())
    )

    # Forward-fill last known sentiment; treat 0 as "no news" before ffill
    df["_last_news_score"] = df.groupby("Stock")["news_score"].transform(
        lambda x: x.replace(0, np.nan).ffill().fillna(0.0)
    )
    df["_news_date"] = df.groupby("Stock")["Date"].transform(
        lambda x: x.where(df["news_score"].notna()).ffill()
    )
    df["days_since_news"] = (df["Date"] - df["_news_date"]).dt.days.clip(lower=0)

    # Per-stock sentiment decay (falls back to global k if constants not cached)
    applied_personalised = False
    try:
        from features.sentiment_decay import apply_personalised_decay, DECAY_CONSTANTS_PATH
        import json
        if os.path.exists(DECAY_CONSTANTS_PATH):
            with open(DECAY_CONSTANTS_PATH) as fh:
                decay_constants = json.load(fh)
            df["news_decay"] = apply_personalised_decay(df, decay_constants)
            applied_personalised = True
            log.info("Personalised sentiment decay applied (%d stocks).",
                     len(decay_constants))
    except Exception as exc:
        log.debug("Personalised decay unavailable (%s); using global k.", exc)

    if not applied_personalised:
        df["news_decay"] = (
            df["_last_news_score"].fillna(0)
            * np.exp(-df["days_since_news"] / _NEWS_DECAY_K)
        )
        log.info("Global sentiment decay applied (k=%d). "
                 "Run features/sentiment_decay.py to enable per-stock decay.", _NEWS_DECAY_K)

    # Sentiment momentum: current decayed score minus 5-day rolling average
    rolling_avg = df.groupby("Stock")["news_decay"].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )
    df["news_sentiment_mom"] = df["news_decay"] - rolling_avg

    df["has_news"] = df["news_score"].notna().astype(int)
    df["news_score"] = df["news_score"].fillna(0.0)
    df["news_pos"] = df["news_pos"].fillna(0.0)
    df["news_neg"] = df["news_neg"].fillna(0.0)
    df.drop(columns=["_last_news_score", "_news_date"], inplace=True)

    log.info("News merged | has_news=%d (%.1f%%) | mean_count_1d=%.2f",
             int(df["has_news"].sum()),
             df["has_news"].mean() * 100,
             df["news_count_1d"].mean())
    return df


# --- Events -------------------------------------------------------------------

def build_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build clean, separated event flags.

    Macro flags (date-level, broadcast to all stocks):
        is_rbi, is_gdp, is_cpi, is_budget

    Corporate flags (per stock):
        is_earnings, is_dividend
    """
    event_cols = ["is_rbi", "is_gdp", "is_cpi", "is_budget", "is_pmi", "is_earnings", "is_dividend"]
    for col in event_cols:
        df[col] = 0

    if not os.path.exists(EVENTS_CSV):
        log.warning("Events CSV missing - event features set to 0")
        return df

    ev = pd.read_csv(EVENTS_CSV)
    ev.columns = ev.columns.str.strip().str.lower()

    date_col = next((c for c in ev.columns if "date" in c), None)
    sym_col = next((c for c in ev.columns if c in ("symbol", "stock", "ticker")), None)

    if date_col is None or sym_col is None:
        log.warning("Events CSV missing date or symbol column (found: %s) - skipping",
                    list(ev.columns))
        return df

    ev["Date"] = pd.to_datetime(ev[date_col], errors="coerce")
    ev = ev[ev["Date"].notna()].copy()
    ev["Stock"] = _norm_ticker(ev[sym_col])
    ev["event_category"] = ev["event_category"].astype(str).str.strip().str.upper()
    ev["event_name"] = ev["event_name"].fillna("NONE").astype(str).str.strip().str.upper()

    # Drop synthetic calendar-rule dates that lack verified timestamps
    if "is_estimated" in ev.columns:
        n_est = int((ev["is_estimated"] == 1).sum())
        if n_est:
            log.info("Dropping %d estimated (is_estimated==1) event rows", n_est)
            ev = ev[ev["is_estimated"] != 1].copy()

    # Macro events: one flag per date, merged to all stocks
    macro_mask = (
        ev["event_category"].str.contains("MACRO|GOVT", na=False) |
        ev["event_name"].str.contains("RBI|GDP|CPI|BUDGET|PMI", na=False)
    )
    macro = ev[macro_mask].copy()
    macro_by_date = (
        macro.groupby("Date")["event_name"]
             .apply(lambda x: "|".join(x.unique()))
             .reset_index()
    )
    for flag, keyword in [("is_rbi", "RBI"), ("is_gdp", "GDP"),
                          ("is_cpi", "CPI"), ("is_budget", "BUDGET"),
                          ("is_pmi", "PMI")]:
        macro_by_date[flag] = (
            macro_by_date["event_name"].str.contains(keyword, na=False).astype(int)
        )
    macro_by_date = macro_by_date[["Date", "is_rbi", "is_gdp", "is_cpi", "is_budget", "is_pmi"]]
    df = df.merge(macro_by_date, on="Date", how="left", suffixes=("", "_ev"))
    for flag in ["is_rbi", "is_gdp", "is_cpi", "is_budget", "is_pmi"]:
        ev_col = flag + "_ev"
        if ev_col in df.columns:
            df[flag] = df[ev_col].fillna(0).astype(int)
            df.drop(columns=[ev_col], inplace=True)
        else:
            df[flag] = df[flag].fillna(0).astype(int)

    # Corporate events: per stock+date
    corp = ev[~macro_mask].copy()
    corp["is_earnings"] = corp["event_name"].str.contains("EARNINGS", na=False).astype(int)
    corp["is_dividend"] = corp["event_name"].str.contains("DIVIDEND", na=False).astype(int)
    corp_agg = (
        corp.groupby(["Date", "Stock"], as_index=False)
            .agg(is_earnings=("is_earnings", "max"),
                 is_dividend=("is_dividend", "max"))
    )
    df = df.merge(corp_agg, on=["Date", "Stock"], how="left", suffixes=("", "_c"))
    for flag in ["is_earnings", "is_dividend"]:
        corp_col = flag + "_c"
        if corp_col in df.columns:
            df[flag] = df[corp_col].fillna(0).astype(int)
            df.drop(columns=[corp_col], inplace=True)
        else:
            df[flag] = df[flag].fillna(0).astype(int)

    for col in event_cols:
        df[col] = df[col].fillna(0).astype(int)

    log.info("Events merged | RBI=%d GDP=%d CPI=%d BUDGET=%d PMI=%d EARNINGS=%d DIVIDEND=%d",
             df["is_rbi"].sum(), df["is_gdp"].sum(), df["is_cpi"].sum(),
             df["is_budget"].sum(), df["is_pmi"].sum(),
             df["is_earnings"].sum(), df["is_dividend"].sum())
    return df



# --- Interaction features -----------------------------------------------------

def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineered interaction terms from existing (already-lagged) columns.
    Inf values replaced with 0.
    """
    eps = _EPS
    df["bb_rsi_combo"] = df["bb_pct"] * df["rsi_norm"]
    df["vol_adj_mom"] = df["momentum_5d"] / (df["volatility_5d"].abs() + eps)
    df["rsi_x_mom"] = df["rsi_norm"] * df["momentum_5d"]
    df["mom_vol_ratio"] = df["momentum_10d"] / (df["volatility_10d"].abs() + eps)
    for col in ["bb_rsi_combo", "vol_adj_mom", "rsi_x_mom", "mom_vol_ratio"]:
        df[col] = df[col].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    log.info("Interaction features added.")
    return df


# --- Alpha features -----------------------------------------------------------

def build_alpha_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-sectional and time-series alpha signals.
    All inputs are already lagged - no new lookahead introduced.
    """
    df = df.sort_values(["Stock", "Date"]).reset_index(drop=True)
    g = df.groupby("Stock", group_keys=False)
    eps = _EPS

    def _z_score(series: pd.Series, window: int = 60) -> pd.Series:
        mu = series.rolling(window, min_periods=10).mean()
        std = series.rolling(window, min_periods=10).std()
        return (series - mu) / (std + eps)

    df["mom_z_5"] = g["momentum_5d"].transform(_z_score)
    df["mom_z_10"] = g["momentum_10d"].transform(_z_score)
    df["vol_z"] = g["atr_ratio"].transform(_z_score)

    df["pct_from_52h"] = g["Close"].transform(
        lambda x: x / (x.rolling(252, min_periods=50).max() + eps) - 1
    )
    df["cs_rank_ret"] = df.groupby("Date")["ret_lag_1d"].rank(pct=True)
    df["cs_rank_mom"] = df.groupby("Date")["momentum_5d"].rank(pct=True)
    df["consec_dir"] = g["ret_lag_1d"].transform(
        lambda x: pd.Series(np.sign(x), index=x.index).rolling(5, min_periods=1).sum()
    )

    log.info("Alpha features built.")
    return df


# --- Label --------------------------------------------------------------------

def build_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binary label: UP=1, DOWN=-1. Ambiguous rows dropped.

    Threshold: adaptive per-stock 20-day volatility x 0.5,
    clipped to [0.5%, 2.5%].
    Horizon: LABEL_HORIZON trading days forward.
    """
    df = df.sort_values(["Stock", "Date"])
    stock_vol = df.groupby("Stock")["Close"].transform(
        lambda x: x.pct_change().rolling(20, min_periods=5).std()
    )
    thresh = (stock_vol * 0.5).clip(0.005, 0.025)

    df["_fwd_close"] = df.groupby("Stock")["Close"].shift(-LABEL_HORIZON)
    df["_fwd_ret"] = (df["_fwd_close"] - df["Close"]) / df["Close"].replace(0, np.nan)
    df["label"] = np.where(
        df["_fwd_ret"] > thresh, 1,
        np.where(df["_fwd_ret"] < -thresh, -1, np.nan)
    )
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)
    df.drop(columns=["_fwd_close", "_fwd_ret"], inplace=True)

    counts = df["label"].value_counts().sort_index()
    n = len(df)
    log.info("Labels (binary, horizon=%dd): total=%d  DOWN=%d(%.1f%%)  UP=%d(%.1f%%)",
             LABEL_HORIZON, n,
             counts.get(-1, 0), counts.get(-1, 0) / n * 100,
             counts.get(1, 0), counts.get(1, 0) / n * 100)
    return df


# --- Master pipeline ----------------------------------------------------------

def run() -> pd.DataFrame:
    """Run the full feature-engineering pipeline end to end."""
    print("\n" + "=" * 60)
    print("  BUILD FEATURES  (zero lookahead)")
    print("=" * 60)

    if not os.path.exists(TECHNICAL_CSV):
        raise FileNotFoundError(f"Missing: {TECHNICAL_CSV}. Run data_collection/build_technical.py first.")

    tech = pd.read_csv(TECHNICAL_CSV)
    tech["Date"] = pd.to_datetime(tech["Date"])
    tech["Stock"] = _norm_ticker(tech["Stock"])
    tech = (tech[(tech["Date"] >= DATE_START) & (tech["Date"] <= DATE_END)]
                .sort_values(["Stock", "Date"])
                .drop_duplicates(subset=["Stock", "Date"], keep="last")
                .reset_index(drop=True))
    log.info("Technical loaded: %s", tech.shape)

    # Recompute MACD signal for any rows where it is NaN (warm-up period)
    for stock, sub in tech.groupby("Stock"):
        if sub["MACD_signal"].isna().any():
            idx = sub.index
            ema12 = sub["Close"].ewm(span=12, adjust=False).mean()
            ema26 = sub["Close"].ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            tech.loc[idx, "MACD"] = macd.values
            tech.loc[idx, "MACD_signal"] = macd.ewm(span=9, adjust=False).mean().values

    df = build_technical(tech)
    nifty = _load_nifty()
    df = build_market_sector(df, nifty)
    df = build_regime_features(df, nifty)
    df = build_fundamentals(df)
    df = build_news(df)
    df = build_events(df)
    df = add_interaction_features(df)
    df = build_alpha_features(df)
    df = build_label(df)

    # Drop non-feature columns that were needed internally
    df.drop(columns=[c for c in ("Direction", "Sector", "Year") if c in df.columns],
            inplace=True)

    # Part 2 cleanup: remove columns that are not consumed downstream.
    # Close and Return_1d are kept because prediction/* and evaluation/backtest.py use them.
    drop_before_save = [
        "Open", "High", "Low", "Volume", "EMA_20", "RSI", "MACD",
        "MACD_signal", "ATR", "OBV", "nifty_ret_1d", "fundamental_date",
    ]
    df.drop(columns=[c for c in drop_before_save if c in df.columns], inplace=True)

    # Sanitise
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    fund_set = {"PE_Ratio", "ROE", "Revenue_Growth", "Profit_Growth"}
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    df[[c for c in num_cols if c not in fund_set]] = (
        df[[c for c in num_cols if c not in fund_set]].fillna(0)
    )
    df[list(fund_set)] = df[list(fund_set)].fillna(0)
    obj_cols = df.select_dtypes(include=["object", "string"]).columns
    df[obj_cols] = df[obj_cols].fillna("")

    os.makedirs(os.path.dirname(MERGED_CSV), exist_ok=True)
    df.to_csv(MERGED_CSV, index=False)

    print(f"\n[OK] Saved: {MERGED_CSV}  shape={df.shape}")
    print(f"     Stocks     : {df['Stock'].nunique()}")
    print(f"     Date range : {df['Date'].min()} -> {df['Date'].max()}")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="  [%(levelname)s] %(message)s")
    run()
