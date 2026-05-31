#!/usr/bin/env python3
"""
tools/regen_readme_stats.py
===========================
Recomputes every headline statistic directly from the shipped artifacts in
evaluation/results/ and writes the results back to:
  - scratch/readme_subs.json   (canonical substitution source for README numbers)
  - scratch/bootstrap_ci.json  (bootstrap CI record for the headline ensemble run)
  - README.md                  (inline values patched to match)

Usage:
  python tools/regen_readme_stats.py           # recompute and write
  python tools/regen_readme_stats.py --check   # diff only; exit 1 on any drift

Test-set results table sources (all three rows are included):
  Row 1 (XGBoost base):             evaluation/results/xgboost_results.csv
  Row 2 (Ensemble, HEADLINE):       evaluation/results/ensemble_results.csv
  Row 3 (Walk-forward XGB):         evaluation/results/xgboost_walkforward_tuned.csv

The ensemble row is the default run_pipeline.py output and the headline for
bootstrap CI and Known Limitations prose. Bootstrap CI is computed on the
ensemble run only (seed=42, n=1000).

Walk-forward XGB is a separate experiment — it is NOT part of the default
run_pipeline.py step list; it must be run manually via
models/xgboost/train_walkforward.py.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# ── constants ──────────────────────────────────────────────────────────────────
BOOTSTRAP_SEED = 42
N_BOOTSTRAP    = 1000

ROOT = Path(__file__).resolve().parent.parent

XGB_CSV      = ROOT / "evaluation" / "results" / "xgboost_results.csv"
ENSEMBLE_CSV = ROOT / "evaluation" / "results" / "ensemble_results.csv"
WF_XGB_CSV   = ROOT / "evaluation" / "results" / "xgboost_walkforward_tuned.csv"
VAL_CSV      = ROOT / "evaluation" / "results" / "xgboost_results_val.csv"
ABLATION_CSV = ROOT / "evaluation" / "results" / "ablation_results.csv"
README_PATH  = ROOT / "README.md"
SUBS_JSON    = ROOT / "scratch" / "readme_subs.json"
CI_JSON      = ROOT / "scratch" / "bootstrap_ci.json"

# TRAIN_UP lives in the training split of the merged feature file, not in
# evaluation/results/.  We look for it here; if absent we flag and preserve.
MERGED_CSV = ROOT / "data" / "merged" / "merged_final.csv"
TRAIN_END  = "2024-06-30"


# ── computation ───────────────────────────────────────────────────────────────

def _bootstrap_auc(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    rng = np.random.RandomState(BOOTSTRAP_SEED)
    n = len(y_true)
    boot = [
        roc_auc_score(y_true[idx := rng.choice(n, n, replace=True)], y_prob[idx])
        for _ in range(N_BOOTSTRAP)
    ]
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def _csv_stats(path: Path) -> dict:
    """Compute acc, AUC, macro-F1, and always-UP baseline from a predictions CSV."""
    df = pd.read_csv(path)
    y_true = df["label"].values
    y_pred = df["Predicted"].values
    y_prob = df["prob_up"].values
    return {
        "n":   int(len(df)),
        "acc": float(accuracy_score(y_true, y_pred)),
        "auc": float(roc_auc_score(y_true, y_prob)),
        "f1":  float(f1_score(y_true, y_pred, average="macro", labels=[-1, 1])),
        "up":  float((y_true == 1).mean()),
    }


def compute_all() -> dict:
    """Return a dict of all computed statistics plus any flagged items."""
    flagged: dict[str, str] = {}

    for path in (XGB_CSV, ENSEMBLE_CSV, WF_XGB_CSV, VAL_CSV, ABLATION_CSV):
        if not path.exists():
            sys.exit(f"[ERROR] Required CSV not found: {path}")

    xgb_stats = _csv_stats(XGB_CSV)
    ens_stats  = _csv_stats(ENSEMBLE_CSV)
    wf_stats   = _csv_stats(WF_XGB_CSV)

    # Bootstrap CI computed on the headline (ensemble) run only
    ens_df = pd.read_csv(ENSEMBLE_CSV)
    auc_lo, auc_hi = _bootstrap_auc(ens_df["label"].values, ens_df["prob_up"].values)
    ens_stats["auc_lo"] = auc_lo
    ens_stats["auc_hi"] = auc_hi

    val_df = pd.read_csv(VAL_CSV)
    val_stats = {
        "n":   int(len(val_df)),
        "acc": float(accuracy_score(val_df["label"].values, val_df["Predicted"].values)),
        "auc": float(roc_auc_score(val_df["label"].values, val_df["prob_up"].values)),
    }

    ablation = pd.read_csv(ABLATION_CSV)

    # ── TRAIN_UP (training split; not in evaluation/results/) ─────────────────
    if MERGED_CSV.exists():
        merged = pd.read_csv(MERGED_CSV, usecols=["Date", "label"],
                             parse_dates=["Date"])
        train_up = float((merged.loc[merged["Date"] <= TRAIN_END, "label"] == 1).mean())
        train_up_src = str(MERGED_CSV.relative_to(ROOT))
    else:
        if SUBS_JSON.exists():
            existing = json.loads(SUBS_JSON.read_text())
            train_up = float(existing.get("TRAIN_UP", "nan"))
        else:
            train_up = float("nan")
        train_up_src = "(merged_final.csv absent — value preserved from scratch/readme_subs.json)"
        flagged["TRAIN_UP"] = (
            f"training-set UP rate: requires {MERGED_CSV.relative_to(ROOT)}, "
            "which was not found. Existing value preserved."
        )

    return {
        "xgb":      xgb_stats,
        "ensemble": ens_stats,
        "wf_xgb":   wf_stats,
        "val":      val_stats,
        "ablation": ablation,
        "train_up":     train_up,
        "train_up_src": train_up_src,
        "flagged":  flagged,
    }


# ── formatting ────────────────────────────────────────────────────────────────

def _fmt(v: float, d: int = 4) -> str:
    return f"{v:.{d}f}"

def _rows(n: int) -> str:
    return f"{n:,}"


def build_subs(stats: dict) -> dict:
    xgb = stats["xgb"]
    ens = stats["ensemble"]
    wf  = stats["wf_xgb"]
    val = stats["val"]
    abl = stats["ablation"]

    abl_rows = "\n".join(
        f"| {row['Configuration']} | {row['Accuracy']:.4f} | {row['AUC']:.4f} |"
        for _, row in abl.iterrows()
    )

    train_up_str = _fmt(stats["train_up"]) if not np.isnan(stats["train_up"]) else "N/A"

    return {
        # XGBoost base row
        "XGB_ROWS": _rows(xgb["n"]),
        "XGB_ACC":  _fmt(xgb["acc"]),
        "XGB_AUC":  _fmt(xgb["auc"]),
        "XGB_UP":   _fmt(xgb["up"]),
        "XGB_F1":   _fmt(xgb["f1"]),
        # Ensemble row (headline)
        "ENS_ROWS":   _rows(ens["n"]),
        "ENS_ACC":    _fmt(ens["acc"]),
        "ENS_AUC":    _fmt(ens["auc"]),
        "ENS_UP":     _fmt(ens["up"]),
        "ENS_AUC_LO": _fmt(ens["auc_lo"]),
        "ENS_AUC_HI": _fmt(ens["auc_hi"]),
        # Walk-forward XGB row
        "WF_ROWS": _rows(wf["n"]),
        "WF_ACC":  _fmt(wf["acc"]),
        "WF_AUC":  _fmt(wf["auc"]),
        "WF_UP":   _fmt(wf["up"]),
        # Val row
        "VAL_ROWS": _rows(val["n"]),
        "VAL_ACC":  _fmt(val["acc"]),
        "VAL_AUC":  _fmt(val["auc"]),
        # Train UP rate
        "TRAIN_UP": train_up_str,
        # Ablation
        "ABLATION_ROWS": abl_rows,
    }


def build_ci_json(stats: dict) -> dict:
    ens = stats["ensemble"]
    return {
        "headline_model":  "ensemble",
        "source_csv":      str(ENSEMBLE_CSV.relative_to(ROOT)),
        "test_auc_point":  ens["auc"],
        "test_auc_ci95":   [ens["auc_lo"], ens["auc_hi"]],
        "test_acc_point":  ens["acc"],
        "n_test":          ens["n"],
        "up_rate":         ens["up"],
        "n_bootstrap":     N_BOOTSTRAP,
        "bootstrap_seed":  BOOTSTRAP_SEED,
    }


# ── README patching ───────────────────────────────────────────────────────────

def _results_block(subs: dict) -> str:
    """Build the Results section body (inserted between ## Results and ### Ablation)."""
    xgb_n = int(subs["XGB_ROWS"].replace(",", ""))
    ens_n = int(subs["ENS_ROWS"].replace(",", ""))
    row_diff = xgb_n - ens_n
    return (
        f"| Model | Rows | Accuracy | AUC | Always-UP baseline |\n"
        f"|---|---|---|---|---|\n"
        f"| XGBoost base | {subs['XGB_ROWS']} | {subs['XGB_ACC']} | {subs['XGB_AUC']} | {subs['XGB_UP']} |\n"
        f"| **Ensemble (default pipeline output)** | **{subs['ENS_ROWS']}** | **{subs['ENS_ACC']}** | **{subs['ENS_AUC']}** | **{subs['ENS_UP']}** |\n"
        f"| Walk-forward XGB (separate experiment) | {subs['WF_ROWS']} | {subs['WF_ACC']} | {subs['WF_AUC']} | {subs['WF_UP']} |\n"
        f"\n"
        f"Notes:\n"
        f"- Ensemble coverage starts 2025-01-21 ({row_diff} fewer rows than XGBoost base) because the LSTM needs a 15-day sequence warmup.\n"
        f"- Ensemble accuracy ({subs['ENS_ACC']}) is below its own always-UP baseline ({subs['ENS_UP']}); AUC is the metric carrying any signal.\n"
        f"- Walk-forward XGB is a separate experiment — it is **not** part of the default `run_pipeline.py` step list and must be run manually via `models/xgboost/train_walkforward.py`.\n"
        f"\n"
        f"| Split | Rows | Accuracy | AUC |\n"
        f"|---|---|---|---|\n"
        f"| Val (2024 H2, honest, out-of-sample) | {subs['VAL_ROWS']} | {subs['VAL_ACC']} | {subs['VAL_AUC']} |\n"
        f"\n"
        f"- Ensemble test AUC bootstrap 95% CI ({N_BOOTSTRAP} resamples, seed={BOOTSTRAP_SEED}): **[{subs['ENS_AUC_LO']}, {subs['ENS_AUC_HI']}]**\n"
        f"- XGBoost base test macro-F1: {subs['XGB_F1']}"
    )


def patch_readme(text: str, subs: dict) -> str:
    """Replace all headline numeric values inline in README.md."""

    # Results section: replace everything between the ## Results header
    # and the ### Ablation header.  The lambda avoids backslash-escape issues
    # with special characters in new_block (backticks, asterisks, pipes).
    new_block = _results_block(subs)
    text = re.sub(
        r"(## Results \(measured from shipped artifacts, not claimed\)\n\n)"
        r".*?"
        r"(\n\n### Ablation)",
        lambda m: m.group(1) + new_block + m.group(2),
        text,
        flags=re.DOTALL,
    )

    # "Honest framing" blurb — update the AUC mention to reference ensemble.
    # Handles both the old form "Test AUC is ~Npp" and the current form.
    ens_pp = round((float(subs["ENS_AUC"]) - 0.5) * 100)
    text = re.sub(
        r"(Ensemble test AUC is ~|Test AUC is ~)[0-9]+pp above random\.",
        f"Ensemble test AUC is ~{ens_pp}pp above random.",
        text,
    )

    # Known-limitations item 1: signal weakness AUC mention
    text = re.sub(
        r"\*\*Signal is weak\.\*\* (Ensemble test AUC|Test AUC) ~[0-9.]+ is ~[0-9]+pp above random\.",
        f"**Signal is weak.** Ensemble test AUC ~{subs['ENS_AUC']} is ~{ens_pp}pp above random.",
        text,
    )

    # Known-limitations item 2: train/test prior drift UP rates
    if not np.isnan(float(subs.get("TRAIN_UP", "nan") or "nan")):
        text = re.sub(
            r"Train UP rate [0-9]+\.[0-9]+, (ensemble test UP rate|test UP rate) [0-9]+\.[0-9]+",
            f"Train UP rate {subs['TRAIN_UP']}, ensemble test UP rate {subs['ENS_UP']}",
            text,
        )

    # Ablation table (four consecutive rows, flexible whitespace/line endings)
    abl_pattern = (
        r"\| A - Baseline \| [\d.]+ \| [\d.]+ \|[ \t]*\r?\n"
        r"\| B - Learned Decay \| [\d.]+ \| [\d.]+ \|[ \t]*\r?\n"
        r"\| C - Adaptive Gate \| [\d.]+ \| [\d.]+ \|[ \t]*\r?\n"
        r"\| D - Both \(Proposed\) \| [\d.]+ \| [\d.]+ \|"
    )
    if re.search(abl_pattern, text):
        text = re.sub(abl_pattern, subs["ABLATION_ROWS"], text)
    else:
        print("  [WARN] ablation table pattern not found in README — "
              "manual update may be needed")

    return text


# ── drift checking ────────────────────────────────────────────────────────────

def _diff_subs(new_subs: dict, old_subs: dict) -> list[str]:
    diffs = []
    for key, new_val in new_subs.items():
        old_val = old_subs.get(key, "(missing)")
        if old_val != new_val:
            diffs.append(f"  {key}: subs.json has {old_val!r}, computed {new_val!r}")
    return diffs


def _diff_readme(new_subs: dict, readme_text: str) -> list[str]:
    """Check that README inline values match new_subs for all three test rows."""
    diffs = []

    def _chk(pattern: str, expected: str, label: str) -> None:
        m = re.search(pattern, readme_text)
        if m is None:
            diffs.append(f"  {label}: pattern not found in README")
            return
        found = m.group(1)
        if found != expected:
            diffs.append(f"  {label}: README has {found!r}, expected {expected!r}")

    # XGBoost base row (plain, no bold)
    _chk(
        r"\| XGBoost base \| [\d,]+ \| ([\d.]+) \| [\d.]+ \| [\d.]+ \|",
        new_subs["XGB_ACC"], "README XGB_ACC",
    )
    _chk(
        r"\| XGBoost base \| [\d,]+ \| [\d.]+ \| ([\d.]+) \| [\d.]+ \|",
        new_subs["XGB_AUC"], "README XGB_AUC",
    )
    _chk(
        r"\| XGBoost base \| [\d,]+ \| [\d.]+ \| [\d.]+ \| ([\d.]+) \|",
        new_subs["XGB_UP"], "README XGB_UP",
    )

    # Ensemble row (bold markers)
    _chk(
        r"\| \*\*Ensemble \(default pipeline output\)\*\* \| \*\*[\d,]+\*\* \| \*\*([\d.]+)\*\* \|",
        new_subs["ENS_ACC"], "README ENS_ACC",
    )
    _chk(
        r"\| \*\*Ensemble \(default pipeline output\)\*\* \| \*\*[\d,]+\*\* \| \*\*[\d.]+\*\* \| \*\*([\d.]+)\*\* \|",
        new_subs["ENS_AUC"], "README ENS_AUC",
    )
    _chk(
        r"\| \*\*Ensemble \(default pipeline output\)\*\* \| \*\*[\d,]+\*\* \| \*\*[\d.]+\*\* \| \*\*[\d.]+\*\* \| \*\*([\d.]+)\*\* \|",
        new_subs["ENS_UP"], "README ENS_UP",
    )

    # Walk-forward XGB row
    _chk(
        r"\| Walk-forward XGB \(separate experiment\) \| [\d,]+ \| ([\d.]+) \| [\d.]+ \| [\d.]+ \|",
        new_subs["WF_ACC"], "README WF_ACC",
    )
    _chk(
        r"\| Walk-forward XGB \(separate experiment\) \| [\d,]+ \| [\d.]+ \| ([\d.]+) \| [\d.]+ \|",
        new_subs["WF_AUC"], "README WF_AUC",
    )
    _chk(
        r"\| Walk-forward XGB \(separate experiment\) \| [\d,]+ \| [\d.]+ \| [\d.]+ \| ([\d.]+) \|",
        new_subs["WF_UP"], "README WF_UP",
    )

    # Val row
    _chk(
        r"\| Val \(2024 H2, honest, out-of-sample\) \| [\d,]+ \| ([\d.]+) \| [\d.]+ \|",
        new_subs["VAL_ACC"], "README VAL_ACC",
    )
    _chk(
        r"\| Val \(2024 H2, honest, out-of-sample\) \| [\d,]+ \| [\d.]+ \| ([\d.]+) \|",
        new_subs["VAL_AUC"], "README VAL_AUC",
    )

    # Ensemble bootstrap CI
    _chk(
        r"Ensemble test AUC bootstrap 95% CI [^[]+\[([\d.]+),",
        new_subs["ENS_AUC_LO"], "README ENS_AUC_LO",
    )
    _chk(
        r"Ensemble test AUC bootstrap 95% CI [^[]+\[[\d.]+, ([\d.]+)\]",
        new_subs["ENS_AUC_HI"], "README ENS_AUC_HI",
    )

    # XGBoost base macro-F1
    _chk(
        r"XGBoost base test macro-F1: ([\d.]+)",
        new_subs["XGB_F1"], "README XGB_F1",
    )

    # Ablation spot-check: config D accuracy
    abl_d_acc = new_subs["ABLATION_ROWS"].split("\n")[-1].split("|")[2].strip()
    _chk(
        r"\| D - Both \(Proposed\) \| ([\d.]+) \|",
        abl_d_acc, "README ablation D accuracy",
    )

    return diffs


# ── modes ─────────────────────────────────────────────────────────────────────

def _print_flagged(stats: dict) -> None:
    if stats["flagged"]:
        for key, msg in stats["flagged"].items():
            print(f"  [FLAG] {key}: {msg}")
    print(f"  [INFO] TRAIN_UP={stats['train_up']:.4f}  "
          f"source: {stats['train_up_src']}")


def _print_stale_warning(new_subs: dict) -> None:
    """Warn if bootstrap_ci.json has values from a different (e.g. uncommitted) run."""
    if not CI_JSON.exists():
        return
    old_ci = json.loads(CI_JSON.read_text())
    stale = {}
    if abs(old_ci.get("test_acc_point", 0) - float(new_subs["ENS_ACC"])) > 1e-6:
        stale["test_acc (ensemble)"] = (f"{old_ci['test_acc_point']:.4f}", new_subs["ENS_ACC"])
    if abs(old_ci.get("test_auc_point", 0) - float(new_subs["ENS_AUC"])) > 1e-6:
        stale["test_auc (ensemble)"] = (f"{old_ci['test_auc_point']:.4f}", new_subs["ENS_AUC"])

    if stale:
        print()
        print("  [STALE CI JSON] bootstrap_ci.json has values from a different run:")
        for stat, (old_v, new_v) in stale.items():
            print(f"    {stat}: stored {old_v!r}, computed from "
                  f"{ENSEMBLE_CSV.relative_to(ROOT)} -> {new_v!r}")
        print("  The script will overwrite bootstrap_ci.json with ensemble values.")
        print()


def write_mode() -> None:
    print("Sources:")
    print(f"  XGBoost base:        {XGB_CSV.relative_to(ROOT)}")
    print(f"  Ensemble (headline): {ENSEMBLE_CSV.relative_to(ROOT)}")
    print(f"  Walk-forward XGB:    {WF_XGB_CSV.relative_to(ROOT)}")
    print()

    stats    = compute_all()
    new_subs = build_subs(stats)
    new_ci   = build_ci_json(stats)

    _print_flagged(stats)
    _print_stale_warning(new_subs)

    old_subs = json.loads(SUBS_JSON.read_text()) if SUBS_JSON.exists() else {}

    changed = []
    for k, v in new_subs.items():
        ov = old_subs.get(k, "(missing)")
        if ov != v:
            changed.append((k, ov, v))

    if changed:
        print("  Values changing in scratch/readme_subs.json:")
        for k, ov, nv in changed:
            print(f"    {k}: {ov!r} -> {nv!r}")
    else:
        print("  scratch/readme_subs.json already up-to-date.")
    print()

    SUBS_JSON.write_text(json.dumps(new_subs, indent=2) + "\n")
    CI_JSON.write_text(json.dumps(new_ci, indent=2) + "\n")
    print("  Wrote scratch/readme_subs.json")
    print("  Wrote scratch/bootstrap_ci.json")

    original = README_PATH.read_text(encoding="utf-8")
    patched  = patch_readme(original, new_subs)
    if patched != original:
        README_PATH.write_text(patched, encoding="utf-8")
        print("  Patched README.md")
    else:
        print("  README.md already up-to-date.")


def check_mode() -> None:
    print("Sources:")
    print(f"  XGBoost base:        {XGB_CSV.relative_to(ROOT)}")
    print(f"  Ensemble (headline): {ENSEMBLE_CSV.relative_to(ROOT)}")
    print(f"  Walk-forward XGB:    {WF_XGB_CSV.relative_to(ROOT)}")
    print()

    stats    = compute_all()
    new_subs = build_subs(stats)

    _print_flagged(stats)

    readme_text = README_PATH.read_text(encoding="utf-8")
    readme_diffs = _diff_readme(new_subs, readme_text)

    # Also check subs.json if it has been written with the current schema
    subs_diffs = []
    if SUBS_JSON.exists():
        old_subs = json.loads(SUBS_JSON.read_text())
        if "ENS_ACC" in old_subs:
            subs_diffs = _diff_subs(new_subs, old_subs)

    all_diffs = subs_diffs + readme_diffs
    if all_diffs:
        print("DRIFT DETECTED — run without --check to fix:\n")
        for d in all_diffs:
            print(d)
        sys.exit(1)
    else:
        print("OK: all README and JSON values match the shipped CSVs.")


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Recompute README headline stats from all three shipped test CSVs.\n"
            "Default: write updated values to scratch/*.json and README.md.\n"
            "--check: diff only, exit 1 on any drift (for CI)."
        )
    )
    ap.add_argument(
        "--check", action="store_true",
        help="Diff computed values against current README/JSON; exit 1 on drift.",
    )
    args = ap.parse_args()

    if args.check:
        check_mode()
    else:
        write_mode()


if __name__ == "__main__":
    main()
