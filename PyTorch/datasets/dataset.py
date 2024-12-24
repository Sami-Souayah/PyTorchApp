import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import numpy as np

seql = 50

data = yf.download("COIN", period='6mo')

closing_prices = data['Close']
scaler = MinMaxScaler(feature_range=(0, 1))
normalized_prices = scaler.fit_transform(closing_prices.values.reshape(-1, 1))

normalized_prices = np.array(normalized_prices)
X_input = normalized_prices[-seql:].reshape(1, seql, 1)

