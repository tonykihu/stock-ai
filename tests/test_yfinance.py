"""Test yfinance data fetching."""
import yfinance as yf

data = yf.download("AAPL", period="1mo")
print(f"Downloaded {len(data)} rows for AAPL")
print(data.head())

# Save test data (using consistent lowercase naming)
data.to_csv("data/us/aapl.csv")
print("Saved to data/us/aapl.csv")
