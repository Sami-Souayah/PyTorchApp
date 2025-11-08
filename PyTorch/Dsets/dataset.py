import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from Dsets.training_dataset import Training_Dataset
import numpy as np
import torch
from datetime import datetime as dt
from dateutil.relativedelta import relativedelta as delta
import os
import pandas as pd
import pandas_datareader.data as web




class Dataset():
    def __init__(self, data):
        pop = Training_Dataset()
        self.X_input = None
        self.CACHE_DIR = "cache"
        _ ,self.data = self.cache_stuff(data)
        self.price_scaler = MinMaxScaler()
        self.volume_scaler = MinMaxScaler()
        self.volatility_scaler = MinMaxScaler()
        self.seql = 30
        self.ahead = 7
    
    def cache_stuff(self, ticker):
        cache_file = f"{self.CACHE_DIR}/{ticker}.csv"
        if os.path.exists(cache_file):
            try:
                df= pd.read_csv(cache_file, index_col=0, parse_dates=True)
                if not df.empty:
                    print(f"Cache good: {ticker}")
                    return ticker, df
            except Exception as e:
                print(f"Cache read fail: {ticker} error: {e}")
        try:

            end = dt.now()
            start = end - delta(years=5)
            dat = web.DataReader(ticker, "stooq", start, end)
            df = dat.ffill().bfill()[::-1]
            df.to_csv(cache_file)
            return ticker, df
        except Exception as e:
            print(f"Fetch failed: {ticker} error: {e}")
            return ticker, pd.DataFrame()

    def create_input(self):
        closing_prices=self.data['Close']
        vol_dat = self.data['Volume']
        volatil_dat = self.data['High']-self.data['Low']
        normalized_volatil = self.volatility_scaler.fit_transform(volatil_dat.values.reshape(-1,1))
        normalized_vol = self.volume_scaler.fit_transform(vol_dat.values.reshape(-1,1))
        normalized_prices = self.price_scaler.fit_transform(closing_prices.values.reshape(-1, 1))
        combined = np.column_stack((
        normalized_prices[-self.seql:], 
        normalized_vol[-self.seql:], 
        normalized_volatil[-self.seql:]
        ))
        self.X_input = torch.tensor(combined.reshape(1, self.seql, 3), dtype=torch.float32)
