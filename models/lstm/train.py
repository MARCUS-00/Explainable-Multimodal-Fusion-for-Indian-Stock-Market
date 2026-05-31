import logging
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from config.settings import (
    MERGED_CSV, RANDOM_SEED, TEST_START, TRAIN_END, VAL_END,
    VAL_START, XGBOOST_FEATURES, SEQUENCE_LENGTH,
    INT_TO_EXT,
)


def _seed_everything(seed: int = RANDOM_SEED) -> torch.Generator:
    """FIX B4: deterministic LSTM training.
    Must be called at the top of main() BEFORE any random op (incl. DataLoader)."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    g = torch.Generator()
    g.manual_seed(seed)
    return g

FEATURE_COLS  = XGBOOST_FEATURES
CLASS_NAMES = ["DOWN", "UP"]

from models.lstm.dataset import StockSequenceDataset
from models.lstm.model import StockLSTM
from models.artifact_io import safe_dump

logging.basicConfig(level=logging.INFO, format=" [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SEQ_LENGTH = SEQUENCE_LENGTH
BATCH_SIZE = 256
EPOCHS = 30
LEARNING_RATE = 1e-3
NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _dec(y_int_arr):
    return np.vectorize(INT_TO_EXT.get)(y_int_arr)


def load_data() -> pd.DataFrame:
    if not os.path.exists(MERGED_CSV):
        raise FileNotFoundError(f"Missing {MERGED_CSV}. Run features/build_features.py first.")
    df = pd.read_csv(MERGED_CSV, parse_dates=["Date"])
    return df.sort_values(["Stock", "Date"]).reset_index(drop=True)


def time_split(df):
    train = df[df["Date"] <= TRAIN_END].copy()
    val = df[(df["Date"] >= VAL_START) & (df["Date"] <= VAL_END)].copy()
    test = df[df["Date"] >= TEST_START].copy()
    log.info(f"Splits raw sizes — train={len(train)} val={len(val)} test={len(test)}")
    return train, val, test


def compute_class_weights(dataset):
    labels = dataset.labels
    counts = np.bincount(labels, minlength=2).astype(float)
    w = len(labels) / (2 * np.maximum(counts, 1.0))
    return torch.FloatTensor(w).to(DEVICE)


def _predict_loader(model, loader):
    all_probs, all_preds = [], []
    with torch.no_grad():
        for batch_x, _ in loader:
            batch_x = batch_x.to(DEVICE)
            logits = model(batch_x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_preds.append(np.argmax(probs, axis=1))
    if not all_probs:
        return np.empty((0, 2)), np.empty((0,), dtype=int)
    return np.vstack(all_probs), np.concatenate(all_preds)


def _save_val_predictions(val_loader, val_dataset, model, out_path):
    if val_loader is None or val_dataset is None or len(val_dataset) == 0:
        return
    val_probs, val_preds = _predict_loader(model, val_loader)
    val_df = pd.DataFrame({
        "Date": val_dataset.dates,
        "Stock": val_dataset.stocks,
        "label": _dec(val_dataset.labels),
        "lstm_pred": _dec(val_preds),
        "lstm_prob_down": val_probs[:, 0] if val_probs.size else np.array([]),
        "lstm_prob_up": val_probs[:, 1] if val_probs.size else np.array([]),
    })
    val_out_path = out_path.replace(".csv", "_val.csv")
    val_df.to_csv(val_out_path, index=False)
    log.info(f"LSTM VAL results saved -> {val_out_path}")


def evaluate_and_save(model, test_loader, test_dataset, best_model_path,
                      val_loader=None, val_dataset=None):
    log.info("Evaluating Best Model on Test Set...")
    from models.artifact_io import safe_load
    checkpoint = safe_load(best_model_path)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    test_probs, test_preds = _predict_loader(model, test_loader)

    from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

    y_test_int = test_dataset.labels
    y_test_ext = _dec(y_test_int)
    y_pred_ext = _dec(test_preds)

    acc = accuracy_score(y_test_ext, y_pred_ext)
    auc = float('nan')
    if len(np.unique(y_test_ext)) > 1 and test_probs.size > 0:
        auc = roc_auc_score((y_test_ext == 1).astype(int), test_probs[:, 1])

    print(f"\n{'=' * 20} LSTM Test Set {'=' * 20}")
    print(f"Accuracy={acc:.4f} Macro-OVR AUC={auc:.4f}")
    print(classification_report(y_test_ext, y_pred_ext, labels=[-1, 1], target_names=CLASS_NAMES, digits=3))

    results_df = pd.DataFrame({
        "Date": test_dataset.dates,
        "Stock": test_dataset.stocks,
        "label": y_test_ext,
        "lstm_pred": y_pred_ext,
        "lstm_prob_down": test_probs[:, 0] if test_probs.size else np.array([]),
        "lstm_prob_up": test_probs[:, 1] if test_probs.size else np.array([]),
    })

    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "evaluation", "results", "lstm_results.csv"
    )
    results_df.to_csv(out_path, index=False)
    log.info(f"LSTM Test results saved -> {out_path}")

    _save_val_predictions(val_loader, val_dataset, model, out_path)


def main():
    print("\n" + "=" * 60)
    print(f" LSTM TRAIN (Seq={SEQ_LENGTH}, Device={DEVICE})")
    print("=" * 60)

    # FIX B4: seed every RNG before anything else (incl. DataLoader)
    g = _seed_everything(RANDOM_SEED)

    # 1. Load Data
    df = load_data()
    
    # Pre-check feature availability
    available_feats = [c for c in FEATURE_COLS if c in df.columns]
    
    train_df, val_df, test_df = time_split(df)

    # Fit a StandardScaler on TRAIN rows ONLY (no test/val data touched).
    # FIX B7: ffill MUST be grouped by Stock; otherwise a stock's first row
    # inherits the previous stock's last value. The merged_final.csv currently
    # has zero-filled numerics upstream so this is a no-op today, but the
    # latent bug activates the moment any future change leaves NaNs.
    log.info("Fitting StandardScaler on training rows only ...")
    _train_sorted = train_df.sort_values(["Stock", "Date"]).reset_index(drop=True)
    train_clean = (
        _train_sorted.groupby("Stock", group_keys=False)[available_feats]
        .apply(lambda g: g.ffill())
        .fillna(0.0)
        .astype(np.float32)
    )
    scaler = StandardScaler().fit(train_clean.values)

    def _apply_scaler(split_df: pd.DataFrame) -> pd.DataFrame:
        out = split_df.sort_values(["Stock", "Date"]).reset_index(drop=True).copy()
        feat_df = (
            out.groupby("Stock", group_keys=False)[available_feats]
               .apply(lambda g: g.ffill())
               .fillna(0.0)
               .astype(np.float32)
        )
        out[available_feats] = scaler.transform(feat_df.values)
        return out

    train_df = _apply_scaler(train_df)
    val_df   = _apply_scaler(val_df)
    test_df  = _apply_scaler(test_df)

    # 2. Build Datasets
    log.info("Building Sequence Datasets (this may take a few seconds)...")
    train_dataset = StockSequenceDataset(train_df, available_feats, seq_length=SEQ_LENGTH)
    val_dataset = StockSequenceDataset(val_df, available_feats, seq_length=SEQ_LENGTH)
    test_dataset = StockSequenceDataset(test_df, available_feats, seq_length=SEQ_LENGTH)

    log.info(f"Sequences generated — Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              drop_last=True, num_workers=NUM_WORKERS, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # 3. Model Setup
    input_dim = len(available_feats)
    model = StockLSTM(input_dim=input_dim, hidden_dim=64, num_layers=2, num_classes=2, dropout=0.3).to(DEVICE)

    class_weights = compute_class_weights(train_dataset)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # 4. Training Loop with Early Stopping
    best_val_loss  = float("inf")
    patience = 5
    patience_counter = 0
    save_path = os.path.join(os.path.dirname(__file__), "saved")
    os.makedirs(save_path, exist_ok=True)
    best_model_path = os.path.join(save_path, "lstm_model.pt")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        # FIX B8: accumulate train accuracy DURING the training pass instead of
        # running a second forward pass over train_loader (which doubled epoch
        # time and reported a different-mode metric).
        train_correct = 0
        train_total   = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
            with torch.no_grad():
                train_correct += (torch.argmax(logits, 1) == batch_y).sum().item()
                train_total   += batch_x.size(0)

        train_loss /= len(train_dataset)
        train_acc_epoch = train_correct / max(train_total, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                val_loss += loss.item() * batch_x.size(0)

                preds = torch.argmax(logits, dim=1)
                correct += (preds == batch_y).sum().item()
                total += batch_x.size(0)

        val_loss /= len(val_dataset)
        val_acc = correct / total

        log.info(f"Epoch {epoch+1:02d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc_epoch:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            safe_dump({
                "model_state_dict": model.state_dict(),
                "feature_cols":     available_feats,
                "seq_len":          SEQ_LENGTH,
                "hidden_size":      model.hidden_dim,
                "num_layers":       model.num_layers,
                "dropout":          model.lstm.dropout,
                "scaler_mean":      scaler.mean_.tolist(),
                "scaler_scale":     scaler.scale_.tolist(),
            }, best_model_path)
            log.info(f"  >> Saved best model (Val Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log.info("Early stopping triggered.")
                break

    evaluate_and_save(model, test_loader, test_dataset, best_model_path,
                      val_loader=val_loader, val_dataset=val_dataset)


if __name__ == "__main__":
    main()
