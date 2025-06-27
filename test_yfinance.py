import yfinance as yf

data = yf.download("AMZN", start="2023-01-01", end="2023-12-31")
print(data)
