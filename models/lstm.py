import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size):
        super(LSTMModel, self).__init__()

        self.lstm_1 = nn.LSTM(
            input_size=input_size, 
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )

        self.lstm_2 = nn.LSTM(
            input_size=64, 
            hidden_size=32, 
            num_layers=1,
            batch_first=True
        )

        self.dropout = nn.Dropout(0.2)
        self.dense = nn.Linear(in_features=32, out_features=1)
        
    def forward(self, x):
        output, _ = self.lstm_1(x)
        output, _ = self.lstm_2(output)
        output = output[:, -1, :]
        output = self.dropout(output)
        output = self.dense(output)

        return output