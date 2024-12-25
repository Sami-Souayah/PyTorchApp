import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from datetime import datetime as dt
import numpy as np
import pandas as pd


class Training_Dataset():
    def __init__(self):
        self.current = dt.now().date()
        self.data = yf.download("AAPL", period='2y')
        self.seql = len(self.data['Close'])-10
        self.ahead = 7
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.X_train = None
        self.X_test =None
        self.Y_test= None
        self.Y_train = None

    def create_data(self):
        closing_prices = self.data['Close']
        vol_dat = self.data['Volume']
        volatil_dat = self.data['High']-self.data['Low']
        normalized_volatil = self.scaler.fit_transform(volatil_dat.values.reshape(-1,1))
        normalized_volatil = pd.DataFrame(normalized_volatil,columns=['Daily Range'])
        normalized_vol = self.scaler.fit_transform(vol_dat.values.reshape(-1,1))
        normalized_vol = pd.DataFrame(normalized_vol,columns=['Volume'])
        normalized_prices = self.scaler.fit_transform(closing_prices.values.reshape(-1, 1))
        normalized_prices = pd.DataFrame(normalized_prices, columns=["Close"])
        X_train, Y_train = [], []
        for i in range(len(normalized_prices) - self.seql-self.ahead):
            closeprices = normalized_prices.iloc[i:i + self.seql].values
            volumedat = normalized_vol.iloc[i:i + self.seql].values
            volatildat = normalized_volatil.iloc[i:i+self.seql].values
            combined = np.column_stack((closeprices,volumedat,volatildat))
            X_train.append(combined)
            Y_train.append(normalized_prices.iloc[i + self.seql:i+self.seql+self.ahead].values.flatten())  

        X_train = np.array(X_train)  
        Y_train = np.array(Y_train)

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)