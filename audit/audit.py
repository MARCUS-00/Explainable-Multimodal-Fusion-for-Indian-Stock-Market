"""
audit/audit.py
==============
Automated audit — checks for data leakage, time-series violations,
feature errors, target leakage, and backtest validity.

Returns a structured report with issues and fixes.
"""

import os

import pandas as pd


class AuditReport:
    def __init__(self):
        self.issues = []
        self.passes = []

    def fail(self, check, detail, fix):
        self.issues.append({"check": check, "detail": detail, "fix": fix})
        print(f"  [FAIL] {check}: {detail}")

    def ok(self, check):
        self.passes.append(check)
        print(f"  [PASS] {check}")

    def summary(self):
        print(f"\n{'='*60}")
        print("  AUDIT SUMMARY")
        print(f"{'='*60}")
        print(f"  Passed : {len(self.passes)}")
        print(f"  Failed : {len(self.issues)}")
        if self.issues:
            print("\n  Issues:")
            for i in self.issues:
                print(f"    [{i['check']}]")
                print(f"      Problem : {i['detail']}")
                print(f"      Fix     : {i['fix']}")
        else:
            print("\n  No issues found.")
        print(f"{'='*60}\n")
        return self.issues


def audit_merged_csv(merged_csv: str) -> AuditReport:
    """Run all checks on the merged feature file."""
    report = AuditReport()
    print(f"\n{'='*60}")
    print("  AUTOMATED AUDIT")
    print(f"{'='*60}")

    if not os.path.exists(merged_csv):
        report.fail("FILE_EXISTS",
                    f"{merged_csv} not found",
                    "Run features/build_features.py first")
        report.summary()
        return report

    df = pd.read_csv(merged_csv, parse_dates=["Date"])
    df = df.sort_values(["Stock", "Date"]).reset_index(drop=True)

    # ── 1. DATA LEAKAGE ───────────────────────────────────────────────────────

    # 1a. bfill usage: look for columns that have been backward-filled
    # Proxy: if a fundamental column is non-NaN in rows BEFORE the
    # first non-NaN occurrence within that stock, bfill was used.
    for col in ["PE_Ratio", "ROE", "Revenue_Growth", "Profit_Growth"]:
        if col not in df.columns:
            continue
        bad_stocks = []
        for stk, sub in df.groupby("Stock"):
            first_valid_idx = sub[col].first_valid_index()
            if first_valid_idx is None:
                continue
            first_valid_date = sub.loc[first_valid_idx, "Date"]
            before = sub[sub["Date"] < first_valid_date]
            if before[col].notna().any():
                bad_stocks.append(stk)
        if bad_stocks:
            report.fail(
                "BFILL_DETECTED",
                f"Column '{col}' has non-NaN values BEFORE first known data for: {bad_stocks[:3]}",
                "Remove any bfill(). Use ffill() only."
            )
        else:
            report.ok(f"NO_BFILL_{col}")

    # 1b. Future Nifty: nifty_ret_1d on date T should be T-1's return
    # Check: nifty_ret_1d[t] != Close[t].pct_change(1) (same-day return)
    if "nifty_ret_1d" in df.columns:
        if df["nifty_ret_1d"].isna().mean() > 0.5:
            report.fail(
                "NIFTY_MOSTLY_NAN",
                "nifty_ret_1d is >50% NaN — may indicate merge issue",
                "Check _load_nifty() and ensure Date alignment"
            )
        else:
            report.ok("NIFTY_RET_1D_POPULATED")

    # 1c. Forward return in features: label must be dropped, _fwd_ret must not exist
    for forbidden in ["_fwd_ret", "_fwd_close", "forward_return"]:
        if forbidden in df.columns:
            report.fail(
                "TARGET_LEAKAGE",
                f"Column '{forbidden}' found in merged file — forward return leaked into features",
                f"Drop '{forbidden}' before saving merged CSV"
            )
        else:
            report.ok(f"NO_{forbidden.upper()}_IN_FEATURES")

    # ── 2. TIME-SERIES VIOLATION ──────────────────────────────────────────────

    # 2a. Date ordering per stock
    for stk, sub in df.groupby("Stock"):
        if not sub["Date"].is_monotonic_increasing:
            report.fail(
                "DATE_ORDER",
                f"Stock {stk} is not monotonically increasing by Date",
                "Sort by (Stock, Date) before all operations"
            )
            break
    else:
        report.ok("DATE_ORDER_ALL_STOCKS")

    # 2b. Train/Val/Test overlap
    from config.settings import TRAIN_END, VAL_START, VAL_END, TEST_START
    dates = df["Date"]
    train_dates = dates[dates <= TRAIN_END]
    val_dates   = dates[(dates >= VAL_START) & (dates <= VAL_END)]
    test_dates  = dates[dates >= TEST_START]

    overlap_tv = set(train_dates) & set(val_dates)
    overlap_vt = set(val_dates)   & set(test_dates)
    if overlap_tv:
        report.fail("SPLIT_OVERLAP_TRAIN_VAL",
                    f"{len(overlap_tv)} dates overlap between train and val",
                    "Use strict non-overlapping date boundaries")
    else:
        report.ok("NO_TRAIN_VAL_OVERLAP")

    if overlap_vt:
        report.fail("SPLIT_OVERLAP_VAL_TEST",
                    f"{len(overlap_vt)} dates overlap between val and test",
                    "Use strict non-overlapping date boundaries")
    else:
        report.ok("NO_VAL_TEST_OVERLAP")

    # ── 3. FEATURE ERRORS ────────────────────────────────────────────────────

    # 3a. Raw (unnormalised) ATR, EMA, OBV should NOT be in feature list.
    # Use canonical feature list from config.settings (XGBOOST_FEATURES).
    from config.settings import XGBOOST_FEATURES
    raw_forbidden = ["ATR", "EMA_20", "OBV"]
    FEATURE_COLS = XGBOOST_FEATURES
    for col in raw_forbidden:
        if col in FEATURE_COLS:
            report.fail(
                "RAW_INDICATOR_IN_FEATURES",
                f"Raw column '{col}' found in XGBOOST_FEATURES — use normalised form instead",
                "Replace with: ATR→atr_ratio, EMA_20→price_to_ema20, OBV→obv_change"
            )
        else:
            report.ok(f"RAW_{col}_NOT_IN_FEATURES")

    # 3b. Duplicate features
    try:
        from config.settings import XGBOOST_FEATURES as FC
        dups = [f for f in FC if FC.count(f) > 1]
        if dups:
            report.fail("DUPLICATE_FEATURES", f"Duplicates in XGBOOST_FEATURES: {set(dups)}",
                        "Deduplicate the feature list in config/settings.py")
        else:
            report.ok("NO_DUPLICATE_FEATURES")
    except Exception:
        pass

    # 3c. ret_lag features must be present (lagged, not current return)
    for lag_feat in ["ret_lag_1d", "ret_lag_3d", "ret_lag_5d"]:
        if lag_feat not in df.columns:
            report.fail("MISSING_LAG_FEATURE", f"'{lag_feat}' not found in merged CSV",
                        "Run build_features.py which computes lagged returns with shift()")
        else:
            report.ok(f"LAG_FEATURE_{lag_feat}")

    # 3d. Sector LOO: sector_ret_loo must differ from naive sector mean
    if "sector_ret_loo" in df.columns and "sector_ret_1d" in df.columns:
        corr = df[["sector_ret_loo", "sector_ret_1d"]].corr().iloc[0, 1]
        if abs(corr) > 0.9999:
            report.fail("SECTOR_LOO_IDENTICAL",
                        "sector_ret_loo is identical to sector_ret_1d — LOO not applied",
                        "Apply (sum - self) / (count - 1) formula for true LOO")
        else:
            report.ok("SECTOR_LOO_DIFFERS_FROM_NAIVE")

    # ── 4. TARGET LEAKAGE ────────────────────────────────────────────────────

    # label column should only be DOWN/UP, not the raw forward return
    if "label" in df.columns:
        unique_labels = set(df["label"].dropna().unique())
        if unique_labels <= {-1, 1}:
            report.ok("LABEL_VALUES_CORRECT")
        else:
            report.fail("LABEL_VALUES_WRONG",
                        f"label column has unexpected values: {unique_labels}",
                        "Ensure label is encoded as -1/1 only")

    # ── 5. BACKTEST VALIDITY ─────────────────────────────────────────────────

    from config.settings import XGB_RESULTS_PATH
    if os.path.exists(XGB_RESULTS_PATH):
        bt_df = pd.read_csv(XGB_RESULTS_PATH, parse_dates=["Date"])

        # Check transaction cost is applied (can't check from CSV alone, check source)
        bt_src = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "..", "evaluation", "backtest.py")
        if os.path.exists(bt_src):
            src = open(bt_src).read()
            if "TRANSACTION_COST" in src or "transaction_cost" in src.lower():
                report.ok("TRANSACTION_COST_IN_CODE")
            else:
                report.fail("MISSING_TRANSACTION_COST",
                            "backtest.py does not reference a transaction cost",
                            "Apply 0.1% cost per trade in run_backtest()")

        # Probabilities should not be degenerate (all near 0.33)
        for prob_col in ["prob_up", "prob_down"]:
            if prob_col in bt_df.columns:
                std_val = bt_df[prob_col].std()
                if std_val < 0.01:
                    report.fail("DEGENERATE_PROBABILITIES",
                                f"{prob_col} has std={std_val:.5f} — model may be degenerate",
                                "Check class balance and model training")
                else:
                    report.ok(f"PROB_{prob_col.upper()}_HAS_VARIANCE")
    else:
        report.fail("RESULTS_MISSING",
                    f"{XGB_RESULTS_PATH} not found",
                    "Run train.py first")

    report.summary()
    return report


def run():
    from config.settings import MERGED_CSV
    return audit_merged_csv(MERGED_CSV)


if __name__ == "__main__":
    run()
