import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size):
        super(LSTMModel, self).__init__()

        self.lstm_1 = nn.LSTM(
            input_size=input_size, 
            hidden_size=100,
            num_layers=1,
            batch_first=True
        )

        self.lstm_2 = nn.LSTM(
            input_size=100, 
            hidden_size=50, 
            num_layers=1,
            batch_first=True
        )

        self.dropout = nn.Dropout(0.2)
        self.dense = nn.Linear(in_features=50, out_features=1)
        
    def forward(self, x):
        lstm_out1, _ = self.lstm_1(x)
        lstm_out2, _ = self.lstm_2(lstm_out1)
        final_output = lstm_out2[:, -1, :]
        dropout_out = self.dropout(final_output)
        dense_out = self.dense(dropout_out)
        # output = torch.sigmoid(dense_out)
        output = dense_out

        return output