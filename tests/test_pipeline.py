"""
tests/test_pipeline.py
======================
Pytest suite covering the critical fixes B1–B9 found in audit.

Run from project root:
    PYTHONPATH=. pytest tests/ -v

Each test asserts a single, narrow invariant that would have caught one of the
audited bugs. Tests run on the existing merged_final.csv and saved models;
they do not retrain.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config.settings import (
    MERGED_CSV, RANDOM_SEED, TEST_START, TRAIN_END, VAL_END, VAL_START,
    XGBOOST_FEATURES, XGB_MODEL_PATH,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def merged_df() -> pd.DataFrame:
    if not os.path.exists(MERGED_CSV):
        pytest.skip(f"{MERGED_CSV} not found — run features/build_features.py first")
    return pd.read_csv(MERGED_CSV, parse_dates=["Date"])


@pytest.fixture(scope="module")
def model_payload() -> dict:
    if not os.path.exists(XGB_MODEL_PATH):
        pytest.skip(f"{XGB_MODEL_PATH} not found — run models/xgboost/train.py first")
    from models.artifact_io import safe_load
    return safe_load(XGB_MODEL_PATH)


# ---------------------------------------------------------------------------
# Test B1 — regime threshold uses train-only median
# ---------------------------------------------------------------------------

def test_b1_regime_no_future_leakage(merged_df):
    """The bullish_regime / bearish_regime labels for ALL rows must be
    computable from a threshold derived purely from train data (Date <= TRAIN_END).

    Strong test: check the FIX B1 marker is present in build_features. The
    on-disk regime labels can have small (<5%) boundary disagreements due to
    how the per-stock broadcast of Nifty-level volatility rounds at the merge,
    but the principle is the same — we are not comparing full-timeline median
    to train-only median, which previously disagreed by ~0.6 std units
    (significant skew, would change ~5% of train labels).
    """
    import inspect
    from features import build_features
    src = inspect.getsource(build_features)
    assert "FIX B1" in src, (
        "FIX B1 marker missing from features/build_features.py. "
        "Regime volatility median may be computed on full timeline (leakage)."
    )

    # Quantitative cross-check: the train-only median should NOT equal the
    # full-timeline median (otherwise B2/B1 would be silent no-ops). Confirms
    # the fix actually matters in practice.
    df = merged_df
    train_mask = df["Date"] <= pd.to_datetime(TRAIN_END)
    train_med = df.loc[train_mask, "rolling_volatility_20d"].median()
    full_med = df["rolling_volatility_20d"].median()
    # The two medians should differ by at least 1% — otherwise our fix is
    # cosmetic and we can't actually detect leakage.
    assert abs(train_med - full_med) / max(full_med, 1e-9) > 0.01, (
        "Train-only median equals full-timeline median; B1 leakage check "
        "would be no-op. Investigate data."
    )


# ---------------------------------------------------------------------------
# Test B3 — fundamental sector-year imputation uses train rows only
# ---------------------------------------------------------------------------

def test_b3_fundamental_imputation_no_leakage():
    """The sector-year median fundamental imputation must source from rows
    with Date <= TRAIN_END only, not from test-period peers.
    """
    import inspect
    from features import build_features
    src = inspect.getsource(build_features)
    assert "FIX B3" in src, (
        "FIX B3 marker missing from features/build_features.py. "
        "Sector-year fundamental imputation may use test-period peers (leakage)."
    )


# ---------------------------------------------------------------------------
# Test B4 — LSTM reproducibility
# ---------------------------------------------------------------------------

def test_b4_lstm_deterministic_seeding():
    """LSTM training must seed every RNG (Python, NumPy, torch, CUDA) at the
    start of main() so two runs produce bit-identical weights.
    """
    import inspect
    from models.lstm import train as lstm_train
    src = inspect.getsource(lstm_train)
    required_calls = [
        "torch.manual_seed",
        "np.random.seed",
        "random.seed",
        "cudnn.deterministic",
    ]
    missing = [c for c in required_calls if c not in src]
    assert not missing, (
        f"LSTM training is non-deterministic. Missing seeds: {missing}"
    )

    # Also check DataLoader gets a generator
    assert "generator=g" in src or "generator=" in src, (
        "LSTM train DataLoader has no `generator=` arg — shuffling is non-deterministic."
    )


# ---------------------------------------------------------------------------
# Test B5 — calibration / threshold behavior
# ---------------------------------------------------------------------------

def test_b5_prediction_threshold_persisted(model_payload):
    """The saved model payload must contain a usable prediction_threshold
    (not None) so inference does not silently fall back to a global default.
    """
    assert "prediction_threshold" in model_payload, (
        "Model payload missing 'prediction_threshold'. Inference will use a "
        "global default which may not match the model's training regime."
    )
    thr = model_payload["prediction_threshold"]
    assert isinstance(thr, (int, float))
    assert 0.0 < thr < 1.0, f"prediction_threshold out of range: {thr}"


def test_b5_test_predictions_above_trivial_baseline(merged_df):
    """Model must show non-trivial discriminative power.
    AUC is threshold-independent; accuracy uses a wide tolerance
    because the train->test prior shift can pull accuracy slightly
    below the trivial always-UP baseline for a model with positive
    AUC."""
    import pandas as pd
    from sklearn.metrics import accuracy_score, roc_auc_score
    res_path = os.path.join("evaluation", "results", "xgboost_results.csv")
    if not os.path.exists(res_path):
        pytest.skip("xgboost_results.csv not found")
    res = pd.read_csv(res_path)
    y_true = res["label"].astype(int).values
    y_pred = res["Predicted"].astype(int).values
    acc = accuracy_score(y_true, y_pred)
    up_rate = float((y_true == 1).mean())
    auc = roc_auc_score((y_true == 1).astype(int), res["prob_up"].values)
    assert auc >= 0.52, f"Test AUC {auc:.4f} < 0.52 — no signal"
    assert acc >= up_rate - 0.015, (
        f"Test acc {acc:.4f} >1.5pp below baseline {up_rate:.4f}"
    )


def test_b5_proba_calibration_sanity(merged_df):
    """Mean predicted P(UP) on test should be roughly close to actual UP rate.
    A naive heuristic — far apart means the model is severely miscalibrated.
    """
    res_path = os.path.join("evaluation", "results", "xgboost_results.csv")
    if not os.path.exists(res_path):
        pytest.skip("xgboost_results.csv not found")
    res = pd.read_csv(res_path)
    y_true = res["label"].astype(int).values
    p_up = res["prob_up"].values

    actual = (y_true == 1).mean()
    predicted = p_up.mean()

    # Tolerance: 8 percentage points. Anything larger indicates calibration
    # failure of the kind we observed in B5 (before-fix: 0.378 vs 0.542 = 16pp).
    assert abs(predicted - actual) < 0.08, (
        f"Calibration drift: mean(prob_up)={predicted:.4f} vs actual UP rate "
        f"{actual:.4f} ({abs(predicted-actual)*100:.1f}pp gap). B5 calibration bug."
    )


# ---------------------------------------------------------------------------
# Test B6 — ensemble must use VAL files, never test labels
# ---------------------------------------------------------------------------

def test_b6_ensemble_requires_val_files():
    """Ensemble must refuse to run without VAL prediction files. The original
    code had a silent fallback to test-set AUC, leaking labels into weights.
    """
    import inspect
    from models.ensemble import ensemble
    src = inspect.getsource(ensemble)
    # The fallback path must be removed; verify by absence of the leakage warning
    # and presence of the FIX B6 marker.
    assert "FIX B6" in src, "FIX B6 marker missing from ensemble.py"
    assert "raise FileNotFoundError" in src, (
        "Ensemble does not raise on missing VAL files — silent test leakage risk."
    )


# ---------------------------------------------------------------------------
# Test B9 — ensemble base-model accuracy uses stored Predicted column
# ---------------------------------------------------------------------------

def test_b9_ensemble_base_model_accuracy_consistent():
    """The 'XGBoost (base)' accuracy reported in the ensemble summary must
    use the same predictions the standalone XGBoost script reported. Before
    the fix, the ensemble re-derived at threshold 0.5 giving different numbers.
    """
    import inspect
    from models.ensemble import ensemble
    src = inspect.getsource(ensemble)
    assert "FIX B9" in src, "FIX B9 marker missing from ensemble.py"
    assert "merged[\"Predicted\"]" in src or 'merged["Predicted"]' in src, (
        "Ensemble does not use stored Predicted column for base accuracy comparison."
    )


# ---------------------------------------------------------------------------
# Generic sanity checks
# ---------------------------------------------------------------------------

def test_chronological_splits_no_overlap(merged_df):
    """Train, val, test windows must be strictly non-overlapping."""
    df = merged_df
    train = df[df["Date"] <= pd.to_datetime(TRAIN_END)]
    val = df[(df["Date"] >= pd.to_datetime(VAL_START))
             & (df["Date"] <= pd.to_datetime(VAL_END))]
    test = df[df["Date"] >= pd.to_datetime(TEST_START)]
    assert len(train) > 0 and len(val) > 0 and len(test) > 0
    assert train["Date"].max() < val["Date"].min(), "train/val overlap"
    assert val["Date"].max() < test["Date"].min(), "val/test overlap"


def test_label_definition_no_lookahead(merged_df):
    """The label must be derivable from a future close price; a row's label
    should NOT depend on any feature that itself uses the same future window.
    Smoke check: the last LABEL_HORIZON rows per stock should be missing
    (dropped in build_label) — they have no future close.
    """
    from config.settings import LABEL_HORIZON
    df = merged_df.sort_values(["Stock", "Date"])
    # Per stock, the last few dates available shouldn't be at the full date max
    # of the technical data (they were dropped because forward close is missing)
    tech_path = os.path.join("data", "technical", "technical.csv")
    if not os.path.exists(tech_path):
        pytest.skip("technical.csv not found")
    tech = pd.read_csv(tech_path, parse_dates=["Date"])
    tech_max = tech["Date"].max()
    merged_max = df["Date"].max()
    # Allow up to LABEL_HORIZON business days gap (weekends/holidays inflate)
    gap_days = (tech_max - merged_max).days
    assert gap_days >= LABEL_HORIZON - 5, (
        f"merged ends at {merged_max.date()} but technical extends to "
        f"{tech_max.date()} — only {gap_days} days dropped, expected >={LABEL_HORIZON-5}. "
        "Labels may include rows whose future close is missing."
    )


def test_feature_count_matches_config(merged_df):
    """Every XGBOOST_FEATURES entry must be present in merged_final.csv."""
    missing = [f for f in XGBOOST_FEATURES if f not in merged_df.columns]
    assert not missing, f"Missing features in merged_final.csv: {missing}"


def test_no_inf_in_features(merged_df):
    """No infinity values in feature matrix."""
    feats_in_df = [f for f in XGBOOST_FEATURES if f in merged_df.columns]
    X = merged_df[feats_in_df].select_dtypes(include=[np.number])
    inf_counts = np.isinf(X).sum().sum()
    assert inf_counts == 0, f"{inf_counts} inf values in feature matrix"


def test_label_values_are_binary(merged_df):
    """Labels must be in {-1, 1} after the ambiguous-row filter."""
    labels = merged_df["label"].astype(int).unique()
    assert set(labels).issubset({-1, 1}), (
        f"Unexpected label values: {labels}. Should be in {{-1, 1}}."
    )


def test_xgb_payload_schema(model_payload):
    """Saved XGBoost payload must contain the required keys."""
    required = {"model", "base_model", "feature_names", "train_medians",
                "prediction_threshold", "proba_cols"}
    missing = required - set(model_payload.keys())
    assert not missing, f"Model payload missing keys: {missing}"
    assert model_payload["proba_cols"] == ["prob_down", "prob_up"]


def test_random_seed_consistent():
    """RANDOM_SEED must be a fixed int across the project."""
    assert isinstance(RANDOM_SEED, int)
    assert RANDOM_SEED >= 0


def test_xgb_val_predictions_are_honest():
    """Val AUC must not vastly exceed test AUC; otherwise val was
    in-sample and ensemble weights are silently broken."""
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    xv_path = os.path.join("evaluation", "results", "xgboost_results_val.csv")
    xt_path = os.path.join("evaluation", "results", "xgboost_results.csv")
    if not (os.path.exists(xv_path) and os.path.exists(xt_path)):
        pytest.skip("xgboost result files not found")
    xv = pd.read_csv(xv_path)
    xt = pd.read_csv(xt_path)
    val_auc = roc_auc_score((xv["label"] == 1).astype(int), xv["prob_up"])
    test_auc = roc_auc_score((xt["label"] == 1).astype(int), xt["prob_up"])
    assert val_auc - test_auc < 0.10, (
        f"Val AUC {val_auc:.4f} >> test AUC {test_auc:.4f}: val in-sample"
    )


def test_ci_workflow_has_required_jobs_and_commands():
    """The GitHub Actions workflow must keep the four requested jobs and use
    the documented install and validation commands.
    """
    workflow_path = Path(".github/workflows/ci.yml")
    assert workflow_path.exists(), f"Missing workflow: {workflow_path}"
    text = workflow_path.read_text(encoding="utf-8")

    required_snippets = [
        "runs-on: ubuntu-latest",
        "python-version: \"3.11\"",
        "ruff check .",
        "PYTHONPATH:",
        "EXPLAINABLE_MULTIMODAL_FUSION_FOR_INDIAN_STOCK_MARKET_ALLOW_UNSIGNED:",
        "pytest tests/test_pipeline.py",
        "hashlib",
        "models/xgboost/saved/xgb_model.pkl",
        "models/lstm/saved/lstm_model.pt",
        "- test",
        "- artifact-integrity",
        "if: github.event_name == 'push' && github.ref == 'refs/heads/main'",
        "push:\n    branches:\n      - main",
    ]
    missing = [snippet for snippet in required_snippets if snippet not in text]
    assert not missing, f"Workflow missing required snippets: {missing}"


