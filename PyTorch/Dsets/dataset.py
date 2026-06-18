import os

import pandas as pd
import yfinance as yf
from yfinance import EquityQuery
import time



class Dataset():
    def __init__(self):
        self.tickers = self.get_tickers() 
        self.df = self.download_prices()
    

    def get_tickers(self):
        q = EquityQuery("and", [EquityQuery("eq", ["region", "us"]), EquityQuery("is-in", ["exchange", "NMS", "NYQ"]),
                                EquityQuery("gt", ["intradayprice", 5]), EquityQuery("gt", ["avgdailyvol3m", 200000])])
        
        symbols = set()
        offset = 0

        while True:
            poo = yf.screen(q, offset=offset, size=250)
            aaa = poo.get("quotes", [])
            if not aaa:
                break

            for i in aaa:
                s = i.get("symbol") or i.get("ticker")
                if s:
                    symbols.add(s)
                
            if len(aaa) < 250:
                break

            offset += 250

        return sorted(symbols)
    
    def download_prices(self):
        all = []
        for i in range(0, len(self.tickers), 200):
            batch = self.tickers[i:i+200]
            df = yf.download(
                tickers=batch,
                period="5y",
                interval = "1d",
                group_by="ticker",
                auto_adjust=False,
                threads=False,
                timeout=60,
            )
            all.append(df)
            print(f"{i+200} of {len(self.tickers)} completed")
            time.sleep(3)
        return pd.concat(all, axis=1)
    
    def reshape_data(self):
        rows = []
        for i in self.df.columns.levels[0]:
            pee = self.df[i].copy()
            if pee.empty:
                continue
            if pee[["Open", "High", "Low", "Close", "Volume"]].isnull().all().all():
                continue
            pee["Ticker"] = i
            pee = pee.reset_index()
            rows.append(pee)
        return pd.concat(rows, ignore_index=True)
    
    def save_csv(self):
        path = "cache/training_data.csv"
        df = self.reshape_data()

        print("Null counts by column:")
        print(df.isnull().sum())

        df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
        counts = df.groupby("Ticker").size()
        good = counts[counts >= 1250].index
        df = df[df["Ticker"].isin(good)]

        if df.empty:
            print("No usable data to save.")
            return df

        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values(["Ticker", "Date"])
        df.to_csv(path, index=False)
        print(f"Saved CSV to {path}")
        return df
            

if __name__ == "__main__":
    dataset = Dataset()
    dataset.save_csv()