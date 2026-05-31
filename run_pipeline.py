"""
run_pipeline.py
===============
Master orchestrator — runs all pipeline steps in order.

Steps
-----
1.  build_technical    data_collection/build_technical.py
2.  build_fundamental  data_collection/build_fundamental.py
3.  build_news         data_collection/build_news.py
4.  build_events       data_collection/build_events.py
5.  finbert            models/finbert/infer_news.py
6.  learn_decay        features/sentiment_decay.py
7.  merge_features     features/build_features.py
8.  train_xgb          models/xgboost/train.py
9.  train_lstm         models/lstm/train.py
10. ensemble           models/ensemble/ensemble.py
11. evaluate           evaluation/metrics.py
12. backtest           evaluation/backtest.py
13. explain            xai/shap_explain.py
14. ablation           evaluation/ablation_novelty.py
15. decay_chart        generates decay_constants_chart.png

Usage
-----
    python run_pipeline.py                              # run all steps (no tuning)
    python run_pipeline.py --tune                       # run all steps + Optuna search
    python run_pipeline.py --tune --quick               # Optuna quick mode (fewer trials)
    python run_pipeline.py --from-step train_xgb --tune # resume from xgb with tuning
    python run_pipeline.py --steps train_xgb ensemble evaluate --tune
"""

import argparse
import json
import os
import sys
import traceback


def _step(name: str, fn) -> None:
    print(f"\n{'#' * 60}")
    print(f"  STEP: {name}")
    print(f"{'#' * 60}")
    try:
        fn()
        print(f"  [OK] {name} completed")
    except Exception as exc:
        print(f"  [ERROR] {name} failed: {exc}")
        traceback.print_exc()
        raise


def step_build_technical():
    """Reads market OHLCV sources and writes data/technical/technical.csv."""
    from data_collection.build_technical import main
    main()


def step_build_fundamental():
    """Reads fundamentals sources and writes data/fundamental/fundamental.csv."""
    from data_collection.build_fundamental import main
    main()


def step_build_news():
    """Reads raw news feeds and writes data/news/news.csv."""
    from data_collection.build_news import main
    main()


def step_build_events():
    """Reads macro/event calendars and writes data/events/events.csv."""
    from data_collection.build_events import main
    main()


def step_finbert():
    """Reads data/news/news.csv and writes data/news/finbert_scores.csv."""
    from models.finbert.infer_news import main
    main()


def step_learn_decay():
    """Reads technical.csv and finbert_scores.csv and writes decay constants artefacts."""
    import pandas as pd
    from config.settings import FINBERT_CSV, TECHNICAL_CSV
    from features.sentiment_decay import learn_all_decay_constants

    if not os.path.exists(TECHNICAL_CSV):
        print("  [SKIP] technical.csv not found — run build_technical first")
        return
    if not os.path.exists(FINBERT_CSV):
        print("  [SKIP] finbert_scores.csv not found — run finbert step first")
        return

    tech = pd.read_csv(TECHNICAL_CSV, parse_dates=["Date"])
    news = pd.read_csv(FINBERT_CSV,   parse_dates=["Date"])
    _force = os.environ.get(
        "EXPLAINABLE_MULTIMODAL_FUSION_FOR_INDIAN_STOCK_MARKET_FORCE_DECAY",
        "",
    ).lower() in ("1", "true", "yes")
    learn_all_decay_constants(tech, news, force_relearn=_force)


def step_merge_features():
    """Reads all staged CSV inputs and writes data/merged/merged_final.csv."""
    from features.build_features import run
    run()


def step_train_xgb(tune: bool = False, quick: bool = False):
    """Reads merged_final.csv and writes the XGBoost model and prediction CSVs."""
    from models.xgboost.train import main
    main(tune=tune, quick=quick)


def step_train_lstm():
    """Reads merged_final.csv and writes the LSTM checkpoint artefacts."""
    from models.lstm.train import main
    main()


def step_ensemble():
    """AUC-weighted blend of XGBoost + LSTM predictions."""
    from models.ensemble.ensemble import main
    main()


def step_evaluate():
    """Reads XGBoost or ensemble results CSVs and writes evaluation metrics."""
    import pandas as pd
    from config.settings import ENSEMBLE_RESULTS_PATH, TEST_START, XGB_RESULTS_PATH
    from evaluation.metrics import evaluate

    path = ENSEMBLE_RESULTS_PATH if os.path.exists(ENSEMBLE_RESULTS_PATH) else XGB_RESULTS_PATH
    name = "Ensemble (Test Set)" if os.path.exists(ENSEMBLE_RESULTS_PATH) else "XGBoost (Test Set)"

    if not os.path.exists(path):
        print(f"  [SKIP] {path} not found — run training first")
        return

    res = pd.read_csv(path, parse_dates=["Date"])
    res = res[res["Date"] >= TEST_START]

    if "prob_down" not in res.columns or "prob_up" not in res.columns:
        raise KeyError(f"Expected prob_down/prob_up in {path}.")

    evaluate(
        y_true=res["label"].values,
        y_pred=res["Predicted"].values,
        y_proba=res[["prob_down", "prob_up"]].values,
        model_name=name,
    )


def step_backtest():
    """Reads prediction CSVs and returns backtest metrics from historical returns."""
    from config.settings import LABEL_HORIZON
    from evaluation.backtest import load_predictions, run_backtest

    preds = load_predictions()
    if preds is None:
        return

    run_backtest(preds, pred_col="Predicted", ret_col="Return_1d",
                 label_horizon=LABEL_HORIZON)


def step_explain():
    """Reads merged_final.csv and XGBoost artefacts and writes SHAP outputs."""
    import pandas as pd
    from config.settings import MERGED_CSV, TEST_START, XGB_RESULTS_PATH
    from models.xgboost.predict import load_xgb
    from xai.shap_explain import global_importance, plot_summary

    if not os.path.exists(XGB_RESULTS_PATH):
        print("  [SKIP] xgboost_results.csv not found")
        return

    merged  = pd.read_csv(MERGED_CSV, parse_dates=["Date"])
    test    = merged[merged["Date"] >= TEST_START].copy()
    payload = load_xgb()
    global_importance(test, payload, top_n=20)

    save_path = os.path.join("evaluation", "results", "shap_summary.png")
    try:
        plot_summary(test.head(2000), payload, save_path=save_path)
    except Exception as exc:
        print(f"  [WARNING] SHAP plot failed: {exc}")


def step_ablation():
    """Reads saved evaluation artefacts and writes ablation_results.csv."""
    from evaluation.ablation_novelty import run_ablation
    run_ablation()


def step_decay_chart():
    """Reads decay_constants.json and writes decay_constants_chart.png."""
    from config.settings import DECAY_CONSTANTS_PATH

    if not os.path.exists(DECAY_CONSTANTS_PATH):
        print(f"  [SKIP] {DECAY_CONSTANTS_PATH} not found — run learn_decay first")
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        with open(DECAY_CONSTANTS_PATH) as fh:
            dc = json.load(fh)

        sorted_dc = sorted(dc.items(), key=lambda x: (x[1], x[0]))
        stocks    = [s for s, _ in sorted_dc]
        ks        = [k for _, k in sorted_dc]
        palette   = {1:"#d73027", 2:"#f46d43", 3:"#fdae61",
                     5:"#fee090", 7:"#74add1", 10:"#4575b4", 14:"#313695"}

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.barh(stocks, ks,
                color=[palette.get(k, "grey") for k in ks],
                edgecolor="white")
        ax.set_xlabel("Decay half-life k (trading days)", fontsize=12)
        ax.set_title("Per-Stock Sentiment Decay Constants\n"
                     "(learned from training data only)", fontsize=13)
        ax.axvline(5, color="black", linestyle="--", alpha=0.6,
                   label="Global baseline k=5")
        ax.legend(fontsize=11)
        ax.set_xlim(0, 16)
        plt.tight_layout()

        out_path = os.path.join("evaluation", "results", "decay_constants_chart.png")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [OK] Chart saved → {out_path}")
    except ImportError:
        print("  [SKIP] matplotlib not installed")


# ── Step registry ─────────────────────────────────────────────────────────────

ALL_STEPS = [
    ("build_technical",   step_build_technical),
    ("build_fundamental", step_build_fundamental),
    ("build_news",        step_build_news),
    ("build_events",      step_build_events),
    ("finbert",           step_finbert),
    ("learn_decay",       step_learn_decay),
    ("merge_features",    step_merge_features),
    ("train_xgb",         step_train_xgb),
    ("train_lstm",        step_train_lstm),
    ("ensemble",          step_ensemble),
    ("evaluate",          step_evaluate),
    ("backtest",          step_backtest),
    ("explain",           step_explain),
    ("ablation",          step_ablation),
    ("decay_chart",       step_decay_chart),
]

_STEP_NAMES = [s[0] for s in ALL_STEPS]


def main() -> None:
    parser = argparse.ArgumentParser(description="Stock Prediction Pipeline")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--steps", nargs="+",
                       choices=_STEP_NAMES + ["all"],
                       help="Run a specific list of steps or 'all'",
                       )
    group.add_argument("--from-step", type=str, default=None, metavar="STEP",
                       help="Resume pipeline starting at STEP")
    parser.add_argument("--tune",  action="store_true", default=False,
                        help="Run Optuna hyperparameter search during train_xgb")
    parser.add_argument("--quick", action="store_true", default=False,
                        help="Use OPTUNA_QUICK_TRIALS instead of OPTUNA_TRIALS")
    args = parser.parse_args()

    # Determine steps to run
    if args.steps:
        if "all" in args.steps:
            steps_to_run = ALL_STEPS
        else:
            steps_to_run = [s for s in ALL_STEPS if s[0] in args.steps]
            missing = [s for s in args.steps if s not in _STEP_NAMES and s != "all"]
            if missing:
                print(f"[ERROR] Unknown steps requested: {missing}")
                print(f"        Available: {', '.join(_STEP_NAMES)}")
                sys.exit(1)
    elif args.from_step:
        if args.from_step not in _STEP_NAMES:
            print(f"[ERROR] Unknown step '{args.from_step}'")
            print(f"        Available: {', '.join(_STEP_NAMES)}")
            sys.exit(1)
        steps_to_run = ALL_STEPS[_STEP_NAMES.index(args.from_step):]
    else:
        steps_to_run = ALL_STEPS

    print("\n" + "=" * 60)
    print("  STOCK PREDICTION PIPELINE")
    print(f"  Steps ({len(steps_to_run)}): {[s[0] for s in steps_to_run]}")
    if args.tune:
        trials = "OPTUNA_QUICK_TRIALS" if args.quick else "OPTUNA_TRIALS"
        print(f"  Optuna tuning: ON ({trials})")
    print("=" * 60)

    for name, fn in steps_to_run:
        if name == "train_xgb":
            _step(name, lambda fn=fn: fn(tune=args.tune, quick=args.quick))
        else:
            _step(name, fn)

    print("\n" + "=" * 60 + "\n  PIPELINE COMPLETE\n" + "=" * 60)
    artifacts = {
        "XGBoost model":     "models/xgboost/saved/xgb_model.pkl",
        "LSTM model":        "models/lstm/saved/lstm_model.pt",
        "Ensemble results":  "evaluation/results/ensemble_results.csv",
        "Ablation table":    "evaluation/results/ablation_results.csv",
        "Decay chart":       "evaluation/results/decay_constants_chart.png",
        "SHAP summary":      "evaluation/results/shap_summary.png",
        "Watchlist":         "evaluation/results/watchlist_latest.csv",
    }
    if args.tune and os.path.exists("models/xgboost/saved/best_params.json"):
        artifacts["Optuna best params"] = "models/xgboost/saved/best_params.json"
    print()
    for label, path in artifacts.items():
        mark = "✓" if os.path.exists(path) else "✗"
        print(f"  {mark}  {label:<24}  {path}")
    print("\n  Launch UI: streamlit run app/app.py\n")


if __name__ == "__main__":
    main()
