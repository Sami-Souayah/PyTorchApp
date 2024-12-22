import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from datetime import datetime as dt
import numpy as np
import pandas as pd

current = dt.now().date()
data = yf.download("AAPL", period='6mo')
closing_prices = data['Close']
scaler = MinMaxScaler(feature_range=(0, 1))
normalized_prices = scaler.fit_transform(closing_prices.values.reshape(-1, 1))
normalized_prices = pd.DataFrame(normalized_prices, columns=["Close"])
seq_length = 50

X_train, Y_train = [], []
for i in range(len(normalized_prices) - seq_length):
    X_train.append(normalized_prices.iloc[i:i + seq_length].values)  
    Y_train.append(normalized_prices.iloc[i + seq_length].values[0])  

X_train = np.array(X_train)  
Y_train = np.array(Y_train) 

print(f"X_train shape: {X_train.shape}")
print(f"Y_train shape: {Y_train.shape}")
