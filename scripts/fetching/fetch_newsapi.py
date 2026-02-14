"""
Fetch real financial news headlines from NewsAPI.org.
Free tier: 100 requests/day, 1 month historical.

Usage:
  export NEWSAPI_KEY="your_key_here"
  python scripts/fetching/fetch_newsapi.py
"""
import os
import requests
import pandas as pd
from datetime import datetime, timedelta

NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY", "")
BASE_URL = "https://newsapi.org/v2/everything"

# Map tickers to search queries
TICKER_QUERIES = {
    "AAPL": "Apple stock OR AAPL",
    "MSFT": "Microsoft stock OR MSFT",
    "GOOGL": "Google stock OR Alphabet OR GOOGL",
    "NVDA": "Nvidia stock OR NVDA",
    "TSLA": "Tesla stock OR TSLA",
    "INTC": "Intel stock OR INTC",
    "SCOM": "Safaricom OR NSE Kenya",
}


def fetch_headlines(ticker, days_back=30, page_size=100):
    """
    Fetch headlines for a ticker from NewsAPI.

    Args:
        ticker: Stock ticker symbol
        days_back: How many days back to search (max 30 on free tier)
        page_size: Max results per request (max 100 on free tier)

    Returns:
        DataFrame with columns: Date, Headline, Source, Ticker
    """
    if not NEWSAPI_KEY:
        print("NEWSAPI_KEY not set. Set it as an environment variable.")
        print("Sign up at https://newsapi.org/register for a free key.")
        return pd.DataFrame()

    query = TICKER_QUERIES.get(ticker, ticker)
    from_date = (datetime.now() - timedelta(days=min(days_back, 30))).strftime("%Y-%m-%d")

    params = {
        "q": query,
        "from": from_date,
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": page_size,
        "apiKey": NEWSAPI_KEY,
    }

    try:
        response = requests.get(BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        articles = response.json().get("articles", [])
    except requests.RequestException as e:
        print(f"  Error fetching {ticker}: {e}")
        return pd.DataFrame()

    rows = []
    for article in articles:
        if article.get("title"):
            rows.append({
                "Date": article["publishedAt"][:10],
                "Headline": article["title"],
                "Source": article.get("source", {}).get("name", "Unknown"),
                "Ticker": ticker,
            })

    return pd.DataFrame(rows)


def fetch_all_headlines():
    """Fetch headlines for all configured tickers."""
    os.makedirs("data/processed", exist_ok=True)
    all_frames = []

    for ticker in TICKER_QUERIES:
        print(f"Fetching headlines for {ticker}...")
        df = fetch_headlines(ticker)
        if not df.empty:
            all_frames.append(df)
            print(f"  Got {len(df)} headlines")
        else:
            print(f"  No headlines found")

    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        combined.to_csv("data/processed/raw_headlines.csv", index=False)
        print(f"\nSaved {len(combined)} total headlines to data/processed/raw_headlines.csv")
        return combined

    print("No headlines fetched. Check your NEWSAPI_KEY.")
    return pd.DataFrame()


if __name__ == "__main__":
    fetch_all_headlines()
