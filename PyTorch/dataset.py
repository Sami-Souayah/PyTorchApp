import yfinance as yf

data = yf.download("AAPL", start="2015-01-01", end="2024-12-22")
print(data.tail())