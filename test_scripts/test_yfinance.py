import yfinance as yf
data = yf.download("AAPL", start="2025-04-01", end="2025-04-30") #or use period
print(data.head())
data.to_csv("data/us/AAPL_data.csv")