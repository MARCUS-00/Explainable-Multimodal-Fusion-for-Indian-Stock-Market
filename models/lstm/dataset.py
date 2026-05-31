import numpy as np
import torch
from torch.utils.data import Dataset

class StockSequenceDataset(Dataset):
    def __init__(self, df, feature_cols, seq_length=15):
        """
        Creates a time-series sequence dataset per stock.
        """
        self.seq_length = seq_length
        self.features = []
        self.labels = []
        self.dates = []
        self.stocks = []
        
        # Sort values properly just to be safe
        df = df.sort_values(by=["Stock", "Date"]).reset_index(drop=True)
        
        # Binary: DOWN=-1→0, UP=1→1. Ambiguous rows are dropped upstream.
        EXT_TO_INT = {-1: 0, 1: 1}
        
        # We group by stock to avoid sequences bleeding across stocks
        for stock_code, group in df.groupby("Stock"):
            group = group.sort_values("Date").reset_index(drop=True)
            
            # Extract underlying Numpy arrays for speed
            x_data = group[feature_cols].copy()
            # Forward-fill any potential NaNs, then fill 0 (do NOT backfill)
            x_data.replace([np.inf, -np.inf], np.nan, inplace=True)
            x_data = x_data.ffill().fillna(0).values.astype(np.float32)
            
            y_ext_data = group["label"].values.astype(int)
            dates_data = group["Date"].values
            stocks_data = group["Stock"].values
            
            num_rows = len(group)
            if num_rows < seq_length:
                continue
                
            # Create rolling sequences
            for i in range(num_rows - seq_length + 1):
                end_idx = i + seq_length
                # 3D tensor context shape: (seq_length, features)
                seq_x = x_data[i:end_idx, :]
                
                # Label is linked to the prediction made on the 'last' day of the sequence
                # (which itself predicts LABEL_HORIZON days ahead)
                target_ext = y_ext_data[end_idx - 1]
                target_int = EXT_TO_INT.get(target_ext, 0)  # Default DOWN if unknown
                
                self.features.append(seq_x)
                self.labels.append(target_int)
                self.dates.append(dates_data[end_idx - 1])
                self.stocks.append(stocks_data[end_idx - 1])
                
        self.features = np.array(self.features, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.longlong)
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        x = torch.from_numpy(self.features[idx])
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y
