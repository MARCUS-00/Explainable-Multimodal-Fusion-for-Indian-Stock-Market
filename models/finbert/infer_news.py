"""
models/finbert/infer_news.py
=============================
Runs ProsusAI/finbert on news headlines and saves per-stock/per-day
sentiment scores to data/news/finbert_scores.csv.

Columns output:
  Date, Stock, finbert_pos, finbert_neg, finbert_neu

* Batched inference for speed.
* Falls back gracefully if GPU unavailable.
* Skips if finbert_scores.csv already exists and --force not passed.
"""

import argparse
import logging
import os

import numpy as np
import pandas as pd

from config.settings import (
    DATE_END, DATE_START, FINBERT_BATCH_SIZE, FINBERT_MAX_LEN,
    FINBERT_MODEL, NEWS_CSV,
)

FINBERT_CSV = os.path.join(os.path.dirname(NEWS_CSV), "finbert_scores.csv")

logging.basicConfig(level=logging.INFO, format="  [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def _require_safe_torch() -> None:
    import torch

    version_tuple = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
    if version_tuple < (2, 6):
        raise RuntimeError(
            f"torch=={torch.__version__} is blocked due to CVE-2025-32434. "
            "Upgrade to torch>=2.6 before running FinBERT inference."
        )


def _norm_ticker(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.(NS|BO)$", "", regex=True).str.strip()


def _headline_column(news_df: pd.DataFrame) -> str:
    for col in ("headline", "News_Text", "news_text", "text"):
        if col in news_df.columns:
            return col
    raise KeyError("No headline text column found. Expected one of: headline, News_Text, news_text, text")


def _load_finbert_pipe(device: int):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

    _require_safe_torch()

    _tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    _model = AutoModelForSequenceClassification.from_pretrained(
        FINBERT_MODEL, torch_dtype=torch.float32,
    )
    return pipeline(
        "text-classification", model=_model, tokenizer=_tokenizer,
        top_k=None, device=device, truncation=True, max_length=FINBERT_MAX_LEN,
    )


def _neutral_finbert_rows(batch):
    return [
        [{"label": "neutral", "score": 1.0},
         {"label": "positive", "score": 0.0},
         {"label": "negative", "score": 0.0}]
        for _ in batch
    ]


def run_finbert(news_df: pd.DataFrame) -> pd.DataFrame:
    """Run FinBERT.
    """
    try:
        import torch
    except ImportError:
        raise ImportError(
            "transformers and torch are required for FinBERT inference.\n"
            "Install: pip install transformers torch>=2.6 sentencepiece safetensors"
        )

    _require_safe_torch()

    device = 0 if torch.cuda.is_available() else -1
    print(f"  FinBERT device: {'GPU' if device == 0 else 'CPU'}")

    try:
        pipe = _load_finbert_pipe(device)
        _FINBERT_OK = True
        log.info("FinBERT loaded successfully")
    except Exception as e:
        log.warning(f"FinBERT failed to load ({e}) — will use neutral placeholders")
        pipe = None
        _FINBERT_OK = False

    text_col = _headline_column(news_df)
    headlines = news_df[text_col].fillna("").tolist()
    results   = []
    total     = len(headlines)
    for i in range(0, total, FINBERT_BATCH_SIZE):
        batch = headlines[i : i + FINBERT_BATCH_SIZE]
        if not _FINBERT_OK or pipe is None:
            # return neutral placeholder scores
            out = _neutral_finbert_rows(batch)
        else:
            try:
                out = pipe(batch, truncation=True, max_length=FINBERT_MAX_LEN, top_k=None)
            except Exception as e:
                log.warning(f"FinBERT inference failed on batch ({e}) — neutral fallback")
                out = _neutral_finbert_rows(batch)
        results.extend(out)
        if i % (FINBERT_BATCH_SIZE * 10) == 0:
            print(f"    Processed {min(i+FINBERT_BATCH_SIZE, total)}/{total}")

    rows = []
    for r in results:
        row_dict = {item["label"].lower(): item["score"] for item in r}
        rows.append({
            "finbert_pos": row_dict.get("positive", 0.0),
            "finbert_neg": row_dict.get("negative", 0.0),
            "finbert_neu": row_dict.get("neutral",  0.0),
        })
    return pd.DataFrame(rows, index=news_df.index)


def main(force: bool = False):
    print("\n" + "=" * 55)
    print("  FINBERT INFERENCE")
    print("=" * 55)

    _require_safe_torch()

    if os.path.exists(FINBERT_CSV) and not force:
        print(f"  [SKIP] {FINBERT_CSV} already exists. Pass --force to rerun.")
        return

    if not os.path.exists(NEWS_CSV):
        print(f"  [WARN] {NEWS_CSV} not found — creating empty finbert_scores.csv")
        pd.DataFrame(columns=["Date", "Stock", "finbert_pos", "finbert_neg", "finbert_neu"]
                     ).to_csv(FINBERT_CSV, index=False)
        return

    news = pd.read_csv(NEWS_CSV, parse_dates=["Date"])
    news["Stock"] = _norm_ticker(news["Stock"])
    text_col = _headline_column(news)
    news = news[(news["Date"] >= DATE_START) & (news["Date"] <= DATE_END)].copy()
    news[text_col] = news[text_col].fillna("").astype(str)
    news = news[news[text_col].str.len() > 0]

    if news.empty:
        print("  [WARN] No news rows after date filter — empty output")
        pd.DataFrame(columns=["Date", "Stock", "finbert_pos", "finbert_neg", "finbert_neu"]
                     ).to_csv(FINBERT_CSV, index=False)
        return

    print(f"  Running FinBERT on {len(news)} headlines ...")
    try:
        scores = run_finbert(news)
    except Exception as e:
        log.warning(f"FinBERT pipeline failed ({e})")
        if os.path.exists(FINBERT_CSV):
            log.warning("Using cached finbert_scores.csv as fallback")
            cached = pd.read_csv(FINBERT_CSV, parse_dates=["Date"])
            required = {"Date", "Stock", "finbert_pos", "finbert_neg", "finbert_neu"}
            if required.issubset(set(cached.columns)):
                cached.to_csv(FINBERT_CSV, index=False)
                print(f"  [OK] Using cached scores: {FINBERT_CSV}  shape={cached.shape}")
                return
        log.warning("No valid cached FinBERT scores found — using neutral placeholders")
        scores = pd.DataFrame(
            {
                "finbert_pos": np.zeros(len(news), dtype=float),
                "finbert_neg": np.zeros(len(news), dtype=float),
                "finbert_neu": np.ones(len(news), dtype=float),
            }
        )

    out = news[["Date", "Stock"]].copy().reset_index(drop=True)
    out = pd.concat([out, scores.reset_index(drop=True)], axis=1)

    # Average per (Date, Stock) — multiple headlines → single daily score
    out = (
        out.groupby(["Date", "Stock"], as_index=False)
           [["finbert_pos", "finbert_neg", "finbert_neu"]].mean()
    )
    out.to_csv(FINBERT_CSV, index=False)
    print(f"  [OK] Saved: {FINBERT_CSV}  shape={out.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if finbert_scores.csv already exists")
    args = parser.parse_args()
    main(force=args.force)
