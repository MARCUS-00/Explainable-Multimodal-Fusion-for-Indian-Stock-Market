"""
evaluation/backtest.py
======================
Equal-weight portfolio backtest with transaction costs.
"""
import os
import sys

import numpy as np
import pandas as pd

TRANSACTION_COST = 0.001  # 0.1% one-way; applied once per position change


def load_predictions() -> pd.DataFrame | None:
    """Load the best available test-predictions CSV and ensure Return_1d is present.

    Prefers ensemble_results.csv; falls back to xgboost_results.csv.
    If Return_1d is absent, merges it from technical.csv (test split only).
    Returns None (with a printed skip message) if no predictions file exists.
    """
    from config.settings import (
        ENSEMBLE_RESULTS_PATH, TECHNICAL_CSV, TEST_START, XGB_RESULTS_PATH,
    )

    path = ENSEMBLE_RESULTS_PATH if os.path.exists(ENSEMBLE_RESULTS_PATH) else XGB_RESULTS_PATH
    if not os.path.exists(path):
        print(f"  [SKIP] {path} not found")
        return None

    preds = pd.read_csv(path, parse_dates=["Date"])

    if "Return_1d" not in preds.columns:
        if not os.path.exists(TECHNICAL_CSV):
            raise FileNotFoundError("Return_1d missing and technical.csv not found")
        tech = (
            pd.read_csv(TECHNICAL_CSV, parse_dates=["Date"])
            .query("Date >= @TEST_START")
            .sort_values(["Stock", "Date"])
        )
        preds = preds.merge(tech[["Date", "Stock", "Return_1d"]],
                            on=["Date", "Stock"], how="left")

    return preds


def _next_day_return(df: pd.DataFrame, ret_col: str) -> pd.Series:
    """Return on T+1 for a position taken at close-of-T.
    Shift the per-stock return column by -1.
    """
    return df.groupby("Stock")[ret_col].shift(-1).fillna(0.0)


def run_backtest(test_df: pd.DataFrame,
                 pred_col: str = "Predicted",
                 label_col: str = "label",
                 ret_col: str = "Return_1d",
                 label_horizon: int = 1,
                 min_confidence: float = 0.0) -> dict:
    """Run equal-weight backtest and return summary statistics.

    Parameters mirror earlier pipeline conventions. Returns a dict with
    strategy_return, bh_return, sharpe, max_drawdown, win_rate, coverage,
    and daily (DataFrame of per-day strat/bh returns and cumulative curves).
    """
    df = test_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Stock", "Date"]).reset_index(drop=True)

    _to_pos = {1: 1.0, -1: -1.0, 0: 0.0, "UP": 1.0, "DOWN": -1.0}
    df["signal"] = df[pred_col].map(_to_pos).fillna(0.0)

    coverage_before = int((df["signal"] != 0).sum())
    if min_confidence > 0.0:
        if "Confidence" in df.columns:
            conf = df["Confidence"].fillna(0.0)
        elif "prob_up" in df.columns and "prob_down" in df.columns:
            conf = df[["prob_up", "prob_down"]].max(axis=1).fillna(0.0)
        else:
            import logging
            logging.getLogger(__name__).warning(
                "min_confidence set but no Confidence/prob columns found — gate disabled."
            )
            conf = pd.Series(1.0, index=df.index)
        df.loc[conf < min_confidence, "signal"] = 0.0

    if ret_col not in df.columns:
        raise KeyError(f"Return column '{ret_col}' not found in dataframe")

    h = max(int(label_horizon), 1)

    def _expand_array(sig_arr: np.ndarray) -> np.ndarray:
        pos = np.zeros_like(sig_arr, dtype=float)
        countdown = 0
        cur_pos = 0.0
        for i, s in enumerate(sig_arr):
            if s != 0.0:
                cur_pos = s
                countdown = h
            if countdown > 0:
                pos[i] = cur_pos
                countdown -= 1
            else:
                cur_pos = 0.0
        return pos

    position = np.zeros(len(df), dtype=float)
    for _, idx in df.groupby("Stock").groups.items():
        idx = np.asarray(idx)
        sig_g = df.loc[idx, "signal"].to_numpy()
        position[idx] = _expand_array(sig_g)
    df["position"] = position

    coverage_after = int((df["position"] != 0).sum())
    coverage_pct = coverage_after / max(len(df), 1) * 100

    df["next_ret"] = _next_day_return(df, ret_col)
    df["prev_pos"] = df.groupby("Stock")["position"].shift(1).fillna(0.0)
    df["trade"] = (df["position"] != df["prev_pos"]).astype(float)
    df["tc"] = df["trade"] * TRANSACTION_COST

    df["gross_ret"] = df["position"] * df["next_ret"]
    df["strat_ret"] = df["gross_ret"] - df["tc"]
    df["bh_ret"] = df[ret_col].fillna(0.0)

    daily = df.groupby("Date")[['strat_ret', 'bh_ret']].mean()
    daily['strat_cum'] = (1 + daily['strat_ret']).cumprod()
    daily['bh_cum'] = (1 + daily['bh_ret']).cumprod()

    if daily.empty:
        return {"strategy_return": 0.0, "bh_return": 0.0, "sharpe": 0.0,
                "max_drawdown": 0.0, "win_rate": 0.0, "daily": daily}

    strat_ret = float(daily['strat_cum'].iloc[-1] - 1)
    bh_ret = float(daily['bh_cum'].iloc[-1] - 1)

    std = float(daily['strat_ret'].std())
    sharpe = float(daily['strat_ret'].mean() / std * np.sqrt(252)) if std > 0 else 0.0

    cummax = daily['strat_cum'].cummax()
    max_dd = float(((daily['strat_cum'] - cummax) / cummax.replace(0, np.nan)).min())
    if np.isnan(max_dd):
        max_dd = 0.0

    entries = df[(df["trade"] == 1.0) & (df["position"] == 1.0)].copy()
    if len(entries) > 0:
        cum_h = pd.Series(0.0, index=df.index)
        for _, idx in df.groupby("Stock").groups.items():
            idx = np.asarray(idx)
            r = df.loc[idx, "next_ret"].fillna(0.0).to_numpy()
            log_r = np.log1p(r)
            # FIX: forward-looking h-day realized return at entry index i.
            # Reverse, trailing-roll, reverse back ⇒ leading window.
            roll = (pd.Series(log_r[::-1])
                      .rolling(h, min_periods=1).sum()
                      .to_numpy()[::-1])
            cum_h.loc[idx] = np.expm1(roll)
        entries["realized_h"] = cum_h.loc[entries.index]
        win_rate = float((entries["realized_h"] > 0).mean())
    else:
        win_rate = 0.0

    print(f"\n{'-'*50}")
    print("  BACKTEST RESULTS")
    print(f"{'-'*50}")
    print(f"  Label Horizon    : {label_horizon} day(s)")
    print(f"  Min Confidence   : {min_confidence:.2f} ({'ON' if min_confidence > 0 else 'OFF'})")
    print(f"  Coverage         : {coverage_after}/{coverage_before} positions ({coverage_pct:.1f}%)")
    print(f"  Strategy Return  : {strat_ret * 100:.2f}%")
    print(f"  Benchmark (B&H)  : {bh_ret * 100:.2f}%")
    print(f"  Alpha            : {(strat_ret - bh_ret) * 100:.2f}%")
    print(f"  Sharpe Ratio     : {sharpe:.3f}")
    print(f"  Max Drawdown     : {max_dd * 100:.2f}%")
    print(f"  Win Rate (UP)    : {win_rate * 100:.1f}%")
    print(f"  UP Entries       : {len(entries)}")
    print(f"  DOWN Positions   : {int((df['position'] == -1).sum())}")
    print(f"  Transaction Cost : {TRANSACTION_COST*100:.1f}% per position change (entry + exit each)")
    print(f"{'-'*50}\n")

    return {
        "strategy_return": strat_ret,
        "bh_return": bh_ret,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "coverage": coverage_pct,
        "daily": daily,
    }


if __name__ == "__main__":
    # Ensure project root is on sys.path when the script is invoked directly
    # (Python puts the script's directory first, not the CWD).
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

    from config.settings import LABEL_HORIZON

    preds = load_predictions()
    if preds is None:
        sys.exit(0)

    run_backtest(preds, pred_col="Predicted", ret_col="Return_1d",
                 label_horizon=LABEL_HORIZON)
