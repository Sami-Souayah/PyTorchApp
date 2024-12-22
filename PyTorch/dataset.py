import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime as dt

current = dt.now().date()
data = yf.download("AAPL", period='6mo')
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))
