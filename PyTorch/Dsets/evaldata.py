import torch
import yfinance as yf
from Dsets.transformations import Transformations
import numpy as np
import pandas as pd




class EvalData():
    def __init__(self, ticker):
        inst = Transformations()
        inst.load_data()
        inst.build_features()
        inst.split_data()
        inst.fit_scalers()
        self.price_scaler = inst.price_scaler
        self.volatility_scaler = inst.volatility_scaler
        self.volume_scaler = inst.volume_scaler
        self.ticker = ticker
        self.df = None
        self.X_input = None
        self.feature_df = None
        self.seql = inst.seql

    def find_ticker(self):
        ticker = yf.Ticker(self.ticker)
        df = ticker.history(period="5y")
        self.df = df
    
    def build_eval_features(self):
        df = self.df.copy()
        df["Volatility"] = df["High"] - df["Low"]
        df = df[["Date", "Ticker", "Close", "Volume", "Volatility"]]
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values(by=["Ticker", "Date"])
        self.feature_df = df

    def apply_scalers(self):
        self.feature_df["Close_norm"] = self.price_scaler.transform(self.feature_df[["Close"]])
        self.feature_df["Volatility_norm"] = self.volatility_scaler.transform(self.feature_df[["Volatility"]])
        self.feature_df["Volume_norm"] = self.volume_scaler.transform(self.feature_df[["Volume"]])

    def create_eval_sequence(self):
        X = []
        for i, j in self.feature_df:
            j = j.sort_values("Date")

            features = j[["Close_norm", "Volatility_norm", "Volume_norm"]].values

            max_val = len(j) - self.seql + 1

            if max_val <= 0:
                raise ValueError(f"Not enough data to create a sequence of length {self.seql} for ticker {i}")
            
            x_window = features[max_val:]

            X.append(x_window)
        self.X_input = np.array(X, dtype=np.float32)

    def to_tensor(self):
        self.X_input = torch.tensor(self.X_input, dtype=torch.float32)