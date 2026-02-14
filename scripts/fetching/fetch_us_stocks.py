import os
import sys
import yfinance as yf
import pandas as pd
from datetime import datetime

# Add project root to path for utils import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.tickers import get_tickers_by_country

TICKERS = get_tickers_by_country("US")
DATA_DIR = "data/us/"

def fetch_data(ticker, years=5):
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
