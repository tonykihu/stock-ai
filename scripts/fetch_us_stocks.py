import os
import yfinance as yf
import pandas as pd
from datetime import datetime

TICKERS = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA", "INTC", "GC00", "SOXS"]
DATA_DIR = "data/us/"

def fetch_data(ticker, years=1):
    """Fetch historical stock data for a given ticker and save to CSV."""
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - pd.DateOffset(years=years)).strftime('%Y-%m-%d')

    data = yf.download(ticker, start=start_date, end=end_date)
    os.makedirs(DATA_DIR, exist_ok=True)
    data.to_csv(f"{DATA_DIR}{ticker.lower()}.csv")
    print(f"Saved {ticker} to {DATA_DIR}")

def fetch_all():
    """Fetch data for all configured tickers."""
    for ticker in TICKERS:
        fetch_data(ticker)

if __name__ == "__main__":
    fetch_all()
