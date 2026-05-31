import torch
import torch.nn as nn

class StockLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, num_classes=2, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # The LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout
        )
        
        # Fully connected block for classification
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        # x is of shape (batch, sequence_length, input_dim)
        lstm_out, (_, _) = self.lstm(x)
        
        # Take the output of the last time step from the top layer
        last_step_out = lstm_out[:, -1, :] 
        
        # Pass through linear block to get class logits
        out = self.fc(last_step_out)
        return out
