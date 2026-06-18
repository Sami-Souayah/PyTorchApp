import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch


class Transformations():
    def __init__(self):
        self.df = None
        self.feature_df = None
        self.training_dat = None
        self.testing_dat = None
        self.price_scaler = MinMaxScaler()
        self.volatility_scaler = MinMaxScaler()
        self.volume_scaler = MinMaxScaler()
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None
        self.ahead = 7
        self.seql = 365



    def load_data(self):
        path = "cache/training_data.csv"
        df = pd.read_csv(path)
        self.df = df
    
    def build_features(self):
        df = self.df.copy()
        df["Volatility"] = df["High"] - df["Low"]
        df = df[["Date", "Ticker", "Close", "Volume", "Volatility"]]
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values(by=["Ticker", "Date"])
        self.feature_df = df

    def split_data(self):
        train = []
        test = []
        train_ratio = 0.8
        for i,j in self.feature_df.groupby("Ticker"):
            j = j.sort_values("Date")
            idx = int(len(j) * train_ratio)
            train.append(j.iloc[:idx])
            test.append(j.iloc[idx:])

        self.training_dat = pd.concat(train, ignore_index=True)
        self.testing_dat = pd.concat(test, ignore_index=True)

    def fit_scalers(self):
        self.price_scaler.fit(self.training_dat[["Close"]])
        self.volatility_scaler.fit(self.training_dat[["Volatility"]])
        self.volume_scaler.fit(self.training_dat[["Volume"]])

        self.training_dat["Close_norm"] = self.price_scaler.transform(self.training_dat[["Close"]])
        self.training_dat["Volatility_norm"] = self.volatility_scaler.transform(self.training_dat[["Volatility"]])
        self.training_dat["Volume_norm"] = self.volume_scaler.transform(self.training_dat[["Volume"]])

        self.testing_dat["Close_norm"] = self.price_scaler.transform(self.testing_dat[["Close"]])
        self.testing_dat["Volatility_norm"] = self.volatility_scaler.transform(self.testing_dat[["Volatility"]])
        self.testing_dat["Volume_norm"] = self.volume_scaler.transform(self.testing_dat[["Volume"]])

    def create_sequences(self):
        for df in [self.training_dat, self.testing_dat]:
            X,Y = [], []
            for i, j in df.groupby("Ticker"):
                j = j.sort_values("Date")

                features = j[["Close_norm", "Volatility_norm", "Volume_norm"]].values
                targets = j["Close_norm"].values

                max_val = len(j) - self.seql - self.ahead + 1

                for k in range(max_val):
                    x_window = features[k:k+self.seql]
                    y_window = targets[k+self.seql:k+self.seql+self.ahead]

                    if len(x_window) == self.seql and len(y_window) == self.ahead:
                        X.append(x_window)
                        Y.append(y_window)
            if df is self.training_dat:
                self.X_train = np.array(X, dtype=np.float32)
                self.Y_train = np.array(Y, dtype=np.float32)
            else:
                self.X_test = np.array(X, dtype=np.float32)
                self.Y_test = np.array(Y, dtype=np.float32)

    def to_tensor(self):
        self.X_train = torch.tensor(self.X_train, dtype=torch.float32)
        self.Y_train = torch.tensor(self.Y_train, dtype=torch.float32)
        self.X_test = torch.tensor(self.X_test, dtype=torch.float32)
        self.Y_test = torch.tensor(self.Y_test, dtype=torch.float32)