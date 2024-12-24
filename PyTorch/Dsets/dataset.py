import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import numpy as np


class Dataset():
    def __init__(self, data):
        self.seql = 50
        self.data = yf.download(data, period='6mo')
        self.X_input = None

    
    def create_input(self):
        closing_prices=self.data['Close']
        scaler = MinMaxScaler(feature_range=(0, 1))
        normalized_prices = scaler.fit_transform(closing_prices.values.reshape(-1, 1))
        normalized_prices = np.array(normalized_prices)
        self.X_input = normalized_prices[-self.seql:].reshape(1, self.seql, 1)
