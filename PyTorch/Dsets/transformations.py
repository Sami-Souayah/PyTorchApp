import pandas as pd
import numpy as np


class Transformations():
    def __init__(self):
        self.df = None
        self.feature_df = None
        pass


    def load_data(self):
        path = "cache/training_data.csv"
        df = pd.read_csv(path)
        self.df = df
    
    def build_features(self):
        df = self.df.copy()
        df["Volatility"] = df["High"] - df["Low"]
        df = df[["Date", "Ticker", "Close", "Volume", "Volatility"]]
        df = df.sort_values(by=["Ticker", "Date"])
        self.feature_df = df
