import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size=4, hidden_size=50, num_layers=1, dropout_rate=0.2):
        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.dense = nn.Linear(in_features=hidden_size, out_features=1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        final_output = lstm_out[:, -1, :]
        dropout_out = self.dropout(final_output)
        dense_out = self.dense(dropout_out)
        output = torch.sigmoid(dense_out)
        
        return output