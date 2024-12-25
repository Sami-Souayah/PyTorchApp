from torch import nn


class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        input_size=3
        hidden_size=50
        num_layers=2
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 7) 

    def forward(self, x):
        lstm_out, _ = self.lstm(x) 
        return self.fc(lstm_out[:, -1, :])

