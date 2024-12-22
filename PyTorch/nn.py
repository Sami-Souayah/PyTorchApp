import torch
from torch import nn
from torch import optim
import dataset


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # Output layer for price prediction

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # LSTM output and hidden state
        return self.fc(lstm_out[:, -1, :])
    