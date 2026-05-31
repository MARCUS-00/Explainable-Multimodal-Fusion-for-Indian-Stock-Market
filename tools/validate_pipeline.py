"""
tools/validate_pipeline.py
===========================
Complete pipeline health-check. Validates config, all CSV files,
stock coverage, data quality, and model files.

Run from project root:
    python tools/validate_pipeline.py

Prints PASS / WARN / FAIL for every check.
Exits with code 1 if any CRITICAL check fails.
"""

import os
import importlib.util
import sys
import traceback
from pathlib import Path

import pandas as pd

SEP = "=" * 70
PASS_ = "PASS "
FAIL_ = "FAIL "
WARN_ = "WARN "

_failures: list[str] = []
_warnings: list[str] = []


def _load_run_pipeline_all_steps() -> list[tuple[str, object]]:
    repo_root = Path(__file__).resolve().parents[1]
    run_pipeline_path = repo_root / "run_pipeline.py"
    spec = importlib.util.spec_from_file_location("run_pipeline", run_pipeline_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load run_pipeline.py from {run_pipeline_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ALL_STEPS


def _ok(label: str) -> None:
    print(f"  {PASS_} {label}")


def _fail(label: str, reason: str, critical: bool = True) -> None:
    tag = FAIL_ if critical else WARN_
    print(f"  {tag} {label}: {reason}")
    (_failures if critical else _warnings).append(f"{label}: {reason}")


def _section(title: str) -> None:
    print(f"\n{SEP}\n  {title}\n{SEP}")


def _load_csv(path: str, label: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        _fail(f"{label} exists", f"Not found: {path}")
        return None
    try:
        df = pd.read_csv(path)
        if df.empty:
            _fail(f"{label} not empty", "File is empty")
            return None
        return df
    except Exception as exc:
        _fail(f"{label} readable", str(exc))
        return None


def _check_stocks(df: pd.DataFrame, stock_col: str,
                  expected: list[str], label: str,
                  min_rows: int = 100) -> None:
    found   = set(df[stock_col].astype(str).str.strip().unique())
    missing = [s for s in expected if s not in found]
    per_stock = df.groupby(stock_col).size()
    low = per_stock[per_stock < min_rows].index.tolist()

    if missing:
        _fail(f"{label} stock coverage", f"Missing: {missing}")
    else:
        _ok(f"{label}: all {len(expected)} stocks present")

    if low:
        _fail(f"{label} row counts",
              f"Stocks with <{min_rows} rows: {low}", critical=False)
    else:
        _ok(f"{label}: all stocks have >= {min_rows} rows")


# ── Section 1: Config ─────────────────────────────────────────────────────────

def check_config() -> None:
    _section("1. CONFIG VALIDATION")
    try:
        from config.settings import (
            DATE_END, FINBERT_CSV, LABEL_HORIZON,
            NIFTY_CACHE, PREDICTION_THRESHOLD, STOCKS,
            TEST_START, TRAIN_END, VAL_END, VAL_START,
            WATCHLIST_MIN_CONFIDENCE, XGBOOST_FEATURES,
        )

        # Date range covers current year
        if DATE_END >= "2026-01-01":
            _ok(f"DATE_END={DATE_END}")
        else:
            _fail("DATE_END", f"{DATE_END} is before 2026 — data will be stale")

        # Non-overlapping splits
        if TRAIN_END < VAL_START:
            _ok(f"Train/val split: {TRAIN_END} / {VAL_START}")
        else:
            _fail("Train/val split",
                  f"TRAIN_END {TRAIN_END} >= VAL_START {VAL_START}")
        if VAL_END < TEST_START:
            _ok(f"Val/test split: {VAL_END} / {TEST_START}")
        else:
            _fail("Val/test split",
                  f"VAL_END {VAL_END} >= TEST_START {TEST_START}")

        # Label horizon
        if LABEL_HORIZON == 10:
            _ok(f"LABEL_HORIZON={LABEL_HORIZON}")
        else:
            _fail("LABEL_HORIZON",
                  f"{LABEL_HORIZON} — optimal value is 10",
                  critical=False)

        # Thresholds
        _ok(f"PREDICTION_THRESHOLD={PREDICTION_THRESHOLD}")
        _ok(f"WATCHLIST_MIN_CONFIDENCE={WATCHLIST_MIN_CONFIDENCE}")

        # Centralised paths
        for name, val in [("FINBERT_CSV", FINBERT_CSV),
                           ("NIFTY_CACHE", NIFTY_CACHE)]:
            if val:
                _ok(f"{name} = {os.path.basename(val)}")
            else:
                _fail(name, "empty path in settings")

        # Feature list includes interaction features
        required_feats = [
            "bb_rsi_combo", "vol_adj_mom", "rsi_x_mom", "mom_vol_ratio",
        ]
        missing_feats = [f for f in required_feats if f not in XGBOOST_FEATURES]
        if missing_feats:
            _fail("XGBOOST_FEATURES", f"Missing: {missing_feats}")
        else:
            _ok(f"XGBOOST_FEATURES: all {len(XGBOOST_FEATURES)} features defined")

        # Stock count
        if len(STOCKS) == 40:
            _ok(f"STOCKS count = {len(STOCKS)}")
        else:
            _fail("STOCKS count", f"{len(STOCKS)} — expected 40", critical=False)

    except Exception as exc:
        _fail("config import", str(exc))
        traceback.print_exc()


# ── Section 2: Module imports ──────────────────────────────────────────────────

def check_imports() -> None:
    _section("2. MODULE IMPORT VALIDATION")

    modules = [
        ("config.settings",                  "settings"),
        ("data_collection.build_technical",  "build_technical"),
        ("data_collection.build_fundamental","build_fundamental"),
        ("data_collection.build_news",       "build_news"),
        ("data_collection.build_events",     "build_events"),
        ("features.build_features",          "build_features"),
        ("features.sentiment_decay",         "sentiment_decay"),
        ("models.xgboost.train",             "xgboost.train"),
        ("models.xgboost.predict",           "xgboost.predict"),
        ("models.lstm.model",                "lstm.model"),
        ("models.lstm.dataset",              "lstm.dataset"),
        ("models.lstm.train",                "lstm.train"),
        ("models.ensemble.ensemble",         "ensemble"),
        ("prediction.single_stock",          "single_stock"),
        ("prediction.watchlist",             "watchlist"),
        ("prediction.adaptive_gate",         "adaptive_gate"),
        ("evaluation.metrics",               "metrics"),
        ("evaluation.backtest",              "backtest"),
        ("evaluation.ablation_novelty",      "ablation_novelty"),
        ("xai.shap_explain",                 "shap_explain"),
    ]

    for module_path, label in modules:
        try:
            __import__(module_path)
            _ok(f"import {label}")
        except Exception as exc:
            _fail(f"import {label}", str(exc)[:120])

    # Verify no dead files remain
    dead_files = [
        "config/nifty50_tickers.py",
        "prediction/recommendation.py",
        "evaluation/results/lgbm_results.csv",
    ]
    for path in dead_files:
        if os.path.exists(path):
            _fail("dead file present", path)
        else:
            _ok(f"dead file removed: {path}")

    # Verify no sys.path.insert hacks remain
    hack_files = []
    for root, _, files in os.walk("."):
        if "__pycache__" in root:
            continue
        # Skip virtualenv and build directories (environment-specific)
        skip_roots = (".venv", "venv", "env", "build", "site-packages")
        if any(s in root for s in skip_roots):
            continue
        for f in files:
            if not f.endswith(".py"):
                continue
            try:
                fp = os.path.join(root, f)
                # Allow known benign uses in app/app.py and this validator itself
                rel = os.path.normpath(fp).replace("\\", "/")
                if rel.endswith(("app/app.py", "tools/validate_pipeline.py")):
                    continue
                content = open(fp).read()
                if "sys.path.insert" in content:
                    hack_files.append(fp)
            except Exception:
                pass
    if hack_files:
        _fail("sys.path.insert hacks", f"Still present in: {hack_files}")
    else:
        _ok("No sys.path.insert hacks in any module")

    # Pipeline step registry
    try:
        all_steps = _load_run_pipeline_all_steps()
        names = [s[0] for s in all_steps]
        expected_steps = [
            "build_technical", "build_fundamental", "build_news",
            "build_events", "finbert", "learn_decay",
            "merge_features", "train_xgb", "train_lstm",
            "ensemble", "evaluate", "backtest", "explain",
            "ablation", "decay_chart",
        ]
        missing_steps = [s for s in expected_steps if s not in names]
        if missing_steps:
            _fail("pipeline steps", f"Missing: {missing_steps}")
        else:
            _ok(f"run_pipeline ALL_STEPS: all {len(expected_steps)} steps present")
    except Exception as exc:
        _fail("run_pipeline import", str(exc))


# ── Section 3: CSV file checks ────────────────────────────────────────────────

def check_technical_csv() -> None:
    from config.settings import STOCKS
    _section("3a. TECHNICAL CSV")

    df = _load_csv("data/technical/technical.csv", "technical.csv")
    if df is None:
        return

    _ok(f"technical.csv loaded: {df.shape[0]:,} rows, {df.shape[1]} cols")

    required = ["Date", "Stock", "Open", "High", "Low", "Close",
                "Volume", "EMA_20", "RSI", "MACD", "ATR", "OBV", "Return_1d"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        _fail("technical columns", f"Missing: {missing}")
    else:
        _ok(f"technical columns: all {len(required)} required columns present")

    _check_stocks(df, "Stock", STOCKS, "technical", min_rows=200)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if df["Date"].max().year < 2025:
        _fail("technical recency",
              f"Latest date {df['Date'].max().date()} is before 2025",
              critical=False)
    else:
        _ok(f"technical data is recent (latest: {df['Date'].max().date()})")

    dupes = df.duplicated(subset=["Stock", "Date"]).sum()
    if dupes > 0:
        _fail("technical duplicates", f"{dupes} duplicate (Stock, Date) rows")
    else:
        _ok("technical: no duplicate (Stock, Date) rows")


def check_fundamental_csv() -> None:
    from config.settings import STOCKS
    _section("3b. FUNDAMENTAL CSV")

    df = _load_csv("data/fundamental/fundamental.csv", "fundamental.csv")
    if df is None:
        return

    _ok(f"fundamental.csv loaded: {df.shape[0]} rows")

    required = ["Stock", "PE_Ratio", "ROE", "Revenue_Growth", "Profit_Growth"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        _fail("fundamental columns", f"Missing: {missing}")
    else:
        _ok("fundamental columns: all required columns present")

    _check_stocks(df, "Stock", STOCKS, "fundamental", min_rows=1)


def check_news_csv() -> None:
    from config.settings import STOCKS
    _section("3c. NEWS + FINBERT CSV")

    df = _load_csv("data/news/news.csv", "news.csv")
    if df is not None:
        _ok(f"news.csv loaded: {df.shape[0]:,} rows")
        stocks_found = set(df["Stock"].unique()) if "Stock" in df.columns else set()
        missing = [s for s in STOCKS if s not in stocks_found]
        if missing:
            _fail("news stock coverage", f"Missing: {missing}", critical=False)
        else:
            _ok(f"news: all {len(STOCKS)} stocks have articles")

    fb = _load_csv("data/news/finbert_scores.csv", "finbert_scores.csv")
    if fb is not None:
        _ok(f"finbert_scores.csv loaded: {fb.shape[0]:,} rows")
        required = ["Date", "Stock", "finbert_pos", "finbert_neg"]
        missing  = [c for c in required if c not in fb.columns]
        if missing:
            _fail("finbert columns", f"Missing: {missing}")
        else:
            _ok("finbert_scores.csv has all required columns")

        if "finbert_pos" in fb.columns and fb["finbert_pos"].std() < 0.01:
            _fail("finbert values",
                  "finbert_pos std near 0 — FinBERT may not have run",
                  critical=False)


def check_merged_csv() -> None:
    from config.settings import STOCKS, TEST_START, XGBOOST_FEATURES
    _section("3d. MERGED CSV")

    df = _load_csv("data/merged/merged_final.csv", "merged_final.csv")
    if df is None:
        return

    _ok(f"merged_final.csv loaded: {df.shape[0]:,} rows, {df.shape[1]} cols")
    _check_stocks(df, "Stock", STOCKS, "merged", min_rows=200)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if df["Date"].max().year < 2025:
        _fail("merged recency",
              f"Latest date {df['Date'].max().date()} — needs rebuild",
              critical=False)
    else:
        _ok(f"merged data is recent (latest: {df['Date'].max().date()})")

    if "label" in df.columns:
        up_pct = (df["label"] == 1).mean()  * 100
        dn_pct = (df["label"] == -1).mean() * 100
        _ok(f"labels: UP={up_pct:.1f}%  DOWN={dn_pct:.1f}%")
        if abs(up_pct - dn_pct) > 20:
            _fail("label balance",
                  f"Imbalanced: UP={up_pct:.1f}% DOWN={dn_pct:.1f}%",
                  critical=False)
    else:
        _fail("merged label column", "No 'label' column")

    missing_feats = [f for f in XGBOOST_FEATURES if f not in df.columns]
    if missing_feats:
        _fail("merged features",
              f"Missing {len(missing_feats)} features: {missing_feats[:10]}")
    else:
        _ok(f"merged features: all {len(XGBOOST_FEATURES)} present")

    test_rows = (df["Date"] >= TEST_START).sum()
    if test_rows < 1000:
        _fail("merged test size", f"Only {test_rows} test rows (need >= 1000)")
    else:
        _ok(f"merged test set: {test_rows:,} rows from {TEST_START}")

    dupes = df.duplicated(subset=["Stock", "Date"]).sum()
    if dupes > 0:
        _fail("merged duplicates", f"{dupes} duplicate (Stock, Date) rows")
    else:
        _ok("merged: no duplicate (Stock, Date) rows")


# ── Section 4: Model file checks ─────────────────────────────────────────────

def check_model_files() -> None:
    _section("4. MODEL FILE VALIDATION")

    # XGBoost
    xgb_path = "models/xgboost/saved/xgb_model.pkl"
    if not os.path.exists(xgb_path):
        _fail("xgb_model.pkl", "Not found — run models/xgboost/train.py")
    else:
        try:
            from models.artifact_io import safe_load, IntegrityError
            try:
                payload = safe_load(xgb_path)
            except IntegrityError as ie:
                _fail("xgb_model.pkl integrity", str(ie))
                payload = None
            if payload is not None:
                for key in ("model", "feature_names", "train_medians"):
                    if key not in payload:
                        _fail("xgb payload", f"Missing key: {key}")
                        break
                else:
                    _ok(f"xgb_model.pkl: {len(payload['feature_names'])} features  "
                        f"trained={payload.get('date_trained', 'unknown')}")
        except Exception as exc:
            _fail("xgb_model.pkl load", str(exc))

    # LSTM
    lstm_path = "models/lstm/saved/lstm_model.pt"
    if not os.path.exists(lstm_path):
        _fail("lstm_model.pt", "Not found — run models/lstm/train.py")
    else:
        try:
            from models.artifact_io import safe_load
            ckpt = safe_load(lstm_path)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                _ok(f"lstm_model.pt: seq_len={ckpt.get('seq_len', '?')}  "
                    f"features={len(ckpt.get('feature_cols', []))}")
            else:
                _fail("lstm checkpoint",
                      f"Unexpected format. Keys: {list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}")
        except Exception as exc:
            _fail("lstm_model.pt load", str(exc))

    # Ensemble results
    ens_path = "evaluation/results/ensemble_results.csv"
    if not os.path.exists(ens_path):
        _fail("ensemble_results.csv",
              "Not found — run models/ensemble/ensemble.py", critical=False)
    else:
        n = len(pd.read_csv(ens_path))
        _ok(f"ensemble_results.csv: {n:,} rows")

    # Watchlist
    wl_path = "evaluation/results/watchlist_latest.csv"
    if not os.path.exists(wl_path):
        _fail("watchlist_latest.csv",
              "Not found — run prediction/watchlist.py", critical=False)
    else:
        df = pd.read_csv(wl_path)
        _ok(f"watchlist_latest.csv: {len(df)} stocks")


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary() -> int:
    print(f"\n{SEP}\n  VALIDATION SUMMARY\n{SEP}")
    if not _failures and not _warnings:
        print("  ALL CHECKS PASSED — pipeline is healthy")
    else:
        if _warnings:
            print(f"  {len(_warnings)} WARNING(S):")
            for w in _warnings:
                print(f"    {WARN_} {w}")
        if _failures:
            print(f"\n  {len(_failures)} CRITICAL FAILURE(S):")
            for f in _failures:
                print(f"    {FAIL_} {f}")
        else:
            print("\n  No critical failures — pipeline should run.")
    print(SEP)
    return len(_failures)


def main() -> None:
    print(f"\n{SEP}\n  PREDICTION PIPELINE — FULL HEALTH CHECK\n{SEP}")
    check_config()
    check_imports()
    check_technical_csv()
    check_fundamental_csv()
    check_news_csv()
    check_merged_csv()
    check_model_files()
    sys.exit(1 if print_summary() > 0 else 0)


if __name__ == "__main__":
    main()
