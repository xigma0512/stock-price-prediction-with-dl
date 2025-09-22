import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size):
        super(LSTMModel, self).__init__()

        self.lstm_1 = nn.LSTM(
            input_size=input_size, 
            hidden_size=128,
            num_layers=1,
            batch_first=True
        )

        self.lstm_2 = nn.LSTM(
            input_size=128, 
            hidden_size=64, 
            num_layers=1,
            batch_first=True
        )

        self.lstm_3 = nn.LSTM(
            input_size=64, 
            hidden_size=32, 
            num_layers=1,
            batch_first=True
        )

        self.dropout_1 = nn.Dropout(0.2)
        self.dropout_2 = nn.Dropout(0.3)
        self.dropout_3 = nn.Dropout(0.3)
        self.dense = nn.Linear(in_features=32, out_features=1)
        
    def forward(self, x):
        output = x

        output, _ = self.lstm_1(output)
        output = torch.relu(output)
        output = self.dropout_1(output)

        output, _ = self.lstm_2(output)
        output = torch.relu(output)
        output = self.dropout_2(output)

        output, _ = self.lstm_3(output)
        output = torch.relu(output)
        output = self.dropout_3(output)
        
        output = output[:, -1, :]
        output = self.dense(output)

        return output