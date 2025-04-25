import yfinance as yf
import pandas as pd
from datetime import datetime

TICKERS = ["AAPL", "MSFT", "GOOGL"] # Add more as needed
DATA_DIR = "data/us/"

def fetch_data(ticker, years=1): # Fetch data for the last # years
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - pd.DateOffset(years=years)).strftime('%Y-%m-%d')
    data = yf.download(ticker, start=start_date, end=end_date)
    data.to_csv(f"{DATA_DIR}{ticker.lower()}.csv")

for ticker in TICKERS:
    fetch_data(ticker)
    print(f"Saved{ticker} data/us")