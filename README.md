---
title: Explainable Multimodal Fusion for Indian Stock Market
sdk: docker
app_port: 8501
---

# Explainable Multimodal Fusion for Indian Stock Market — Nifty 50 Subset Direction Prediction

10-day forward direction prediction for 40 Nifty 50 large-cap stocks.
Label rule (from `features/build_features.py::build_label`): UP if forward return > threshold, DOWN if < −threshold, else dropped.
Threshold is per-stock adaptive: 20-day rolling close-price volatility × 0.5, clipped to [0.5%, 2.5%].
XGBoost + LSTM ensemble with FinBERT sentiment, per-stock learned sentiment decay, regime-adaptive gating, and SHAP explainability.

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-pytest-brightgreen)](tests/)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ed)](Dockerfile)

> **Honest framing.** This is a research scaffold, not a deployed trading signal. Ensemble test AUC is ~4pp above random. The value here is the engineering rigor (leakage prevention, deterministic training, artifact integrity, regression tests for every fixed bug) — not the predictions.

## Results (measured from shipped artifacts, not claimed)

| Model                                  | Rows       | Accuracy   | AUC        | Always-UP baseline |
| -------------------------------------- | ---------- | ---------- | ---------- | ------------------ |
| XGBoost base                           | 11,036     | 0.5364     | 0.5373     | 0.5417             |
| **Ensemble (default pipeline output)** | **10,476** | **0.5330** | **0.5388** | **0.5466**         |
| Walk-forward XGB (separate experiment) | 11,036     | 0.5574     | 0.5713     | 0.5417             |

Notes:

- Ensemble coverage starts 2025-01-21 (560 fewer rows than XGBoost base) because the LSTM needs a 15-day sequence warmup.
- Ensemble accuracy (0.5330) is below its own always-UP baseline (0.5466); AUC is the metric carrying any signal.
- Walk-forward XGB is a separate experiment — it is **not** part of the default `run_pipeline.py` step list and must be run manually via `models/xgboost/train_walkforward.py`.

| Split                                | Rows  | Accuracy | AUC    |
| ------------------------------------ | ----- | -------- | ------ |
| Val (2024 H2, honest, out-of-sample) | 4,436 | 0.5117   | 0.5280 |

- Ensemble test AUC bootstrap 95% CI (1000 resamples, seed=42): **[0.5282, 0.5496]**
- XGBoost base test macro-F1: 0.5244

### Ablation (feature-substitution at inference, single trained model)

| Config              | Accuracy | AUC    |
| ------------------- | -------- | ------ |
| A - Baseline        | 0.5472   | 0.5339 |
| B - Learned Decay   | 0.5464   | 0.5333 |
| C - Adaptive Gate   | 0.5445   | 0.5370 |
| D - Both (Proposed) | 0.5445   | 0.5371 |

## What is well-engineered here

- **No data leakage.** Time-based splits with no shuffling. Regime volatility median uses train rows only (`FIX B1`). Sector-year fundamental imputation uses train rows only (`FIX B3`). Each is locked by a named regression test.
- **Honest validation.** XGBoost val predictions come from a train-only model (`m_diag`), never from the train+val production refit. Without this, the val→test AUC gap was 40+pp and silently broke ensemble weighting.
- **Ensemble integrity.** AUC weights are derived from out-of-sample val files; `ensemble.py` refuses to run if `*_val.csv` files are missing (`FIX B6`).
- **Deterministic LSTM.** Every RNG seeded; CUDA workspace pinned; DataLoader generator fixed (`FIX B4`).
- **Artifact integrity.** Every `.pkl` and `.pt` has a SHA-256 sidecar verified at load time. `EXPLAINABLE_MULTIMODAL_FUSION_FOR_INDIAN_STOCK_MARKET_ALLOW_UNSIGNED=1` is an explicit opt-out.
- **Audit hooks.** `audit/audit.py` checks NaN/Inf, leakage, split overlap, and artifact integrity. `audit/failure_simulation.py` runs the model under data corruption to characterize failure modes.

## Architecture

```mermaid
flowchart LR
    A[Data Collection] --> B[Feature Engineering]
    B --> C1[XGBoost]
    B --> C2[LSTM]
    C1 --> D[AUC-Weighted Ensemble]
    C2 --> D
    D --> E[Adaptive Regime Gate]
    E --> F[Watchlist / Single-Stock]
    D --> G[Backtest & SHAP]
```

## Structure

```
Explainable Multimodal Fusion for Indian Stock Market/
├── config/settings.py         # single source of truth
├── data_collection/           # market, news, macro
├── features/                  # 68-feature pipeline + sentiment decay
├── models/
│   ├── xgboost/               # binary classifier + Optuna + walk-forward
│   ├── lstm/                  # PyTorch StockLSTM (deterministic)
│   ├── finbert/               # ProsusAI/finbert inference
│   ├── ensemble/              # AUC-weighted blend (honest val only)
│   └── artifact_io.py         # SHA-256 sidecar I/O
├── prediction/                # watchlist, single-stock, live, regime gate
├── evaluation/                # metrics, backtest (with TC), ablation
├── xai/                       # SHAP summary + waterfall
├── audit/                     # leakage detection, failure simulation
├── tests/                     # B1-B9 regression tests + invariants
└── app/                       # Streamlit UI on :8501
```

## Quickstart

```bash
pip install -e .
pip install -r requirements.txt

python run_pipeline.py                        # end-to-end (uses cached Optuna params)
python run_pipeline.py --steps train_xgb ensemble evaluate
python run_pipeline.py --tune                 # force fresh Optuna search
```

## Inference

```bash
python prediction/watchlist.py                # top-K UP picks, adaptive threshold
python prediction/single_stock.py INFY
streamlit run app/app.py                      # interactive UI
```

Long-only by default. Pass `direction="down"` or `"both"` for shorts.

## Evaluation

```bash
python evaluation/backtest.py                 # 0.1% TC, B&H benchmark, Sharpe, max DD
python evaluation/ablation_novelty.py
python xai/shap_explain.py
```

## Environment variables

| Variable                                                                 | Default          | Effect                                                                                                                          |
| ------------------------------------------------------------------------ | ---------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `EXPLAINABLE_MULTIMODAL_FUSION_FOR_INDIAN_STOCK_MARKET_PRODUCTION_MODEL` | `train_plus_val` | Which XGBoost stage is shipped as `payload["model"]`. Val predictions are always written from `m_diag` (train-only) regardless. |
| `EXPLAINABLE_MULTIMODAL_FUSION_FOR_INDIAN_STOCK_MARKET_FORCE_DECAY`      | (off)            | Force re-learning of per-stock sentiment decay constants.                                                                       |
| `EXPLAINABLE_MULTIMODAL_FUSION_FOR_INDIAN_STOCK_MARKET_ALLOW_UNSIGNED`   | (off)            | Load model artifacts without a valid SHA-256 sidecar. Use only on trusted machines.                                             |

## Known limitations

1. **Signal is weak.** Ensemble test AUC ~0.5388 is ~4pp above random. Documented, not hidden.
2. **Train→test prior drift.** Train UP rate 0.5868, ensemble test UP rate 0.5466. Mean predicted P(UP) tracks the train prior, which is the dominant residual error mode.
3. **Stock universe is 40, not full Nifty 50.** See `config/settings.py:STOCKS`.
4. **Ablation methodology.** Configs A-D evaluate feature substitution on one trained model, not per-config retraining. Reported AUC deltas are lower bounds.
5. **Pickle artifacts.** SHA-256 sidecars detect corruption, not tampering. Do not load `.pkl` files from untrusted sources.

## Testing

```bash
PYTHONPATH=. EXPLAINABLE_MULTIMODAL_FUSION_FOR_INDIAN_STOCK_MARKET_ALLOW_UNSIGNED=1 pytest tests/ -v
python tools/validate_pipeline.py
```

CI in `.github/workflows/ci.yml` runs lint, syntax check, pytest, skip-rate guard, artifact-integrity, and Docker build on every push.

## Docker

```bash
docker build -t explainable-multimodal-fusion:latest .
docker run --rm -p 8501:8501 explainable-multimodal-fusion:latest
```

## Disclaimer

Research and educational use only. Not financial advice. The author makes no claim that this model produces tradeable signals.
