import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from Dsets.training_dataset import Training_Dataset
import numpy as np
import torch


class Dataset():
    def __init__(self, data):
        pop = Training_Dataset()
        self.data = yf.download(data, period='2y')
        self.seql = len(self.data['Close'])-3
        self.X_input = None
        self.scaler = pop.scaler

    
    def create_input(self):
        closing_prices=self.data['Close']
        vol_dat = self.data['Volume']
        volatil_dat = self.data['High']-self.data['Low']
        normalized_volatil = self.scaler.fit_transform(volatil_dat.values.reshape(-1,1))
        normalized_vol = self.scaler.fit_transform(vol_dat.values.reshape(-1,1))
        normalized_prices = self.scaler.fit_transform(closing_prices.values.reshape(-1, 1))
        combined = np.column_stack((normalized_prices,normalized_vol, normalized_volatil))
        X_input = combined[-self.seql:].reshape(1, self.seql, 3)
        self.X_input = torch.tensor(X_input, dtype=torch.float32)
