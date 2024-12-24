import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from datetime import datetime as dt
import numpy as np
import pandas as pd


class Training_Dataset():
    def __init__(self):
        self.current = dt.now().date()
        self.data = yf.download("BTC-USD", period='6mo')
        self.seql = 180
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.X_train = None
        self.X_test =None
        self.Y_test= None
        self.Y_train = None

    def create_data(self):
        closing_prices = self.data['Close']
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        normalized_prices = self.scaler.fit_transform(closing_prices.values.reshape(-1, 1))
        normalized_prices = pd.DataFrame(normalized_prices, columns=["Close"])
        X_train, Y_train = [], []
        for i in range(len(normalized_prices) - self.seql):
            X_train.append(normalized_prices.iloc[i:i + self.seql].values)  
            Y_train.append(normalized_prices.iloc[i + self.seql].values[0])  
        X_train = np.array(X_train)  
        Y_train = np.array(Y_train) 

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)


