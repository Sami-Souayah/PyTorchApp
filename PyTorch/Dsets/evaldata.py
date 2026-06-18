import torch
import yfinance as yf
from transformations import Transformations
import numpy as np




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
        self.seql = inst.seql

    def find_ticker(self):
        ticker = yf.Search(self.ticker)
        df = ticker.history(period="5y")
        self.df = df 