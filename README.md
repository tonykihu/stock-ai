# Stock AI

An AI-powered stock price prediction system that combines historical market data with news sentiment analysis to generate buy/sell signals for US and Kenyan (NSE) stocks.

## Overview

Stock AI uses a hybrid approach to predict stock price movements:

1. **Historical Data Analysis** - Fetches and processes past stock data to compute technical indicators (RSI, SMA, MACD, OBV)
2. **News Sentiment Analysis** - Analyzes financial news headlines using a pre-trained NLP model (BERTweet) to gauge market sentiment
3. **Hybrid Prediction** - Combines technical indicators and sentiment scores in a Random Forest classifier to predict whether a stock's price will rise or fall

The system supports both **US stocks** (via Yahoo Finance) and **Kenyan NSE stocks** (via web scraping).

## Architecture

```
stock-ai/
├── app.py                  # Streamlit dashboard
├── data/
│   ├── kenya/              # NSE market data
│   └── us/                 # US stock data (AAPL, MSFT, GOOGL, NVDA, TSLA, etc.)
├── models/
│   ├── train_technical.py  # Technical-only model (RSI, SMA, OBV)
│   └── train_hybrid.py     # Hybrid model (technical + sentiment)
├── scripts/
│   ├── fetch_us_stocks.py  # Fetch US stock data via yfinance
│   ├── scrape_nse.py       # Scrape Nairobi Stock Exchange data
│   ├── fetch_news.py       # Fetch and analyze news sentiment
│   ├── preprocess_data.py  # Feature engineering and data processing
│   ├── backtest.py         # Backtesting strategy performance
│   ├── retrain_models.py   # Model retraining pipeline
│   ├── alerts.py           # Price alert triggers
│   └── monitor.py          # System monitoring
├── utils/
│   └── alerts.py           # Email and Discord alert utilities
├── test_scripts/           # Test suite
└── .github/workflows/      # CI/CD pipelines
```

## How It Works

### Data Pipeline
1. **Fetch** - US stock data is pulled from Yahoo Finance; Kenyan stock data is scraped from the NSE
2. **Preprocess** - Raw data is cleaned and enriched with technical indicators (RSI-14, SMA-50, MACD, OBV) using the `ta` library
3. **Sentiment** - Financial news headlines are scored using a transformer-based sentiment model

### Models
- **Technical Model** - Random Forest classifier trained on technical indicators only (RSI, SMA, OBV). Predicts whether price will rise the next day.
- **Hybrid Model** - Random Forest classifier that combines technical indicators with news sentiment scores for improved accuracy.

### Dashboard
A Streamlit web app that provides:
- Ticker selection (US and Kenyan stocks)
- Model selection (Technical or Hybrid)
- Real-time buy/sell signals
- Price charts
- CSV upload for custom NSE data
- Email and Discord alert configuration

## Getting Started

### Prerequisites
- Python 3.9+

### Installation

```bash
git clone https://github.com/tonykihu/stock-ai.git
cd stock-ai
pip install -r requirements.txt
```

### Usage

**1. Fetch stock data:**
```bash
python scripts/fetch_us_stocks.py
```

**2. Preprocess data and generate features:**
```bash
python scripts/preprocess_data.py
```

**3. Fetch news and run sentiment analysis:**
```bash
python scripts/fetch_news.py
```

**4. Train models:**
```bash
python models/train_technical.py
python models/train_hybrid.py
```

**5. Run the dashboard:**
```bash
streamlit run app.py
```

## Supported Tickers

### US Stocks
AAPL, MSFT, GOOGL, NVDA, TSLA, INTC, SOXS, GC00

### Kenyan Stocks (NSE)
Scraped from the Nairobi Securities Exchange live feed.

## Alerts

Stock AI supports notifications through:
- **Email** - Gmail SMTP (requires app password)
- **Discord** - Webhook integration

## CI/CD

GitHub Actions workflows handle:
- **Code quality checks** on push/PR to main
- **Automated data fetching** on schedule
- **Model retraining** on schedule

## Tech Stack

| Component | Technology |
|---|---|
| ML Models | scikit-learn (Random Forest) |
| Sentiment Analysis | Hugging Face Transformers (BERTweet) |
| Data Fetching | yfinance, BeautifulSoup |
| Feature Engineering | ta (Technical Analysis) |
| Dashboard | Streamlit |
| Alerts | SMTP, Discord Webhooks |
| CI/CD | GitHub Actions |

## License

This project is open source.
