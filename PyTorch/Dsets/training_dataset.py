from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from datetime import datetime as dt
import numpy as np
import pandas_datareader.data as web
from dateutil.relativedelta import relativedelta as delta
from concurrent.futures import ThreadPoolExecutor, as_completed
from tickers import tickers
import pandas as pd
import os




class Training_Dataset():
    def __init__(self):
        self.current = dt.now().date()
        self.tickers = tickers
        self.seql = 0
        self.ahead = 7
        self.price_scaler = MinMaxScaler()
        self.volume_scaler = MinMaxScaler()
        self.volatility_scaler = MinMaxScaler()
        self.X_train = None
        self.X_test =None
        self.Y_test= None
        self.Y_train = None
        self.CACHE_DIR = "cache"
        os.makedirs(self.CACHE_DIR, exist_ok=True)

    def cached_tickers(self, ticker):
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

    
    def compile_data(self):
        result = []
        def concur(ticker):
                try:
                    ticker, df = self.cached_tickers(ticker)
                    if not df.empty:
                        return ticker, df
                except Exception as e:
                    print(f"Stock failed: {ticker} because {e}")
        
        maxthread = min(8, len(self.tickers))
        with ThreadPoolExecutor(max_workers=maxthread) as executor:
            futures = [executor.submit(concur, j) for j in self.tickers]
            for future in as_completed(futures):
                res = future.result()
                if res:
                    ticker, df = res
                    result.append(df)
    
        self.seql = 30
        all_close = np.concatenate([df["Close"].values for df in result]).reshape(-1, 1)
        all_volume = np.concatenate([df["Volume"].values for df in result]).reshape(-1, 1)
        all_volatility = np.concatenate([(df["High"] - df["Low"]).values for df in result]).reshape(-1, 1)
        self.price_scaler.fit(all_close)
        self.volatility_scaler.fit(all_volatility)
        self.volume_scaler.fit(all_volume)
        print(f"Fetched {len(result)} tickers successfully.")
        print(result)
        return result

    def create_data(self):
        X_train, Y_train = [], []
        data1 = self.compile_data()
        for data in data1:
            closing_prices = data['Close'].values.reshape(-1,1)
            vol_dat = data['Volume'].values.reshape(-1,1)
            volatil_dat = (data['High']-data['Low']).values.reshape(-1,1)

            normalized_volatil = self.volatility_scaler.transform(volatil_dat)
            normalized_vol = self.volume_scaler.transform(vol_dat)
            normalized_prices = self.price_scaler.transform(closing_prices)
            for i in range(len(normalized_prices) - self.seql-self.ahead):
                closeprices = normalized_prices[i:i + self.seql]
                volumedat = normalized_vol[i:i + self.seql]
                volatildat = normalized_volatil[i:i+self.seql]
                combined = np.column_stack((closeprices,volumedat,volatildat))
                X_train.append(combined)
                Y_train.append(normalized_prices[i + self.seql:i+self.seql+self.ahead].flatten())  
        X_train = np.array(X_train)  
        Y_train = np.array(Y_train)

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)



