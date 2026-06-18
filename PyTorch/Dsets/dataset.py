from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
from datetime import datetime as dt
from dateutil.relativedelta import relativedelta as delta
import os
import pandas as pd
import yfinance as yf
from yfinance import EquityQuery



class Dataset():
    def __init__(self):
        self.tickers = self.get_tickers() 
        self.df = self.download_prices()
    

    def get_tickers(self):
        q = EquityQuery("and", [EquityQuery("eq", ["region", "US"]), EquityQuery("is-in", ["Exchange", "NMS", "NYQ"]),
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
        df = yf.download(
            tickers=self.tickers,
            period="5y",
            interval = "1d",
            group_by="ticker",
            auto_adjust=False,
            threads=True
        )
        return df
    
    def reshape_data(self):
        rows = []
        for i in self.df.columns.levels[0]:
            pee = self.df[i].copy()
            pee["Ticker"] = i
            pee = pee.reset_index()
            rows.append(pee)
        return pd.concat(rows, ignore_index=True)
            

