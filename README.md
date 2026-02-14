# Stock AI

An AI-powered stock price prediction system that combines historical market data with news sentiment analysis to generate buy/sell signals for US and Kenyan (NSE) stocks.

## Overview

Stock AI uses a hybrid approach to predict stock price movements:

1. **Historical Data Analysis** - Fetches 5 years of stock data and computes 13 technical indicators (RSI, SMA, MACD, Bollinger Bands, ATR, OBV, and more)
2. **News Sentiment Analysis** - Fetches real headlines via NewsAPI and scores them with BERTweet sentiment analysis
3. **Hybrid Prediction** - Combines technical indicators and sentiment scores in a Random Forest classifier to predict next-day price direction
4. **Walk-Forward Evaluation** - All models are evaluated using time-series cross-validation to prevent data leakage

The system supports both **US stocks** (via Yahoo Finance) and **Kenyan NSE stocks** (historical CSV data).

## Architecture

```
stock-ai/
├── app.py                              # Streamlit dashboard
├── data/
│   ├── kenya/                          # NSE market data
│   ├── us/                             # US stock CSVs (5 years)
│   └── processed/                      # Preprocessed features and sentiment
├── models/                             # Trained model files (.pkl)
├── logs/                               # Metrics JSON and backtest results
├── scripts/
│   ├── fetching/
│   │   ├── fetch_us_stocks.py          # Fetch US data via yfinance
│   │   ├── fetch_newsapi.py            # Fetch headlines from NewsAPI
│   │   └── fetch_news.py              # Score headlines with BERTweet
│   ├── preprocessing/
│   │   └── preprocess_data.py          # Feature engineering (13 indicators)
│   ├── training/
│   │   ├── train_technical.py          # Technical model (walk-forward CV)
│   │   ├── train_hybrid.py             # Hybrid model (technical + sentiment)
│   │   ├── retrain_models.py           # Automated retraining pipeline
│   │   └── compare_models.py          # RF vs GradientBoosting vs XGBoost
│   └── backtesting/
│       └── backtest.py                 # Walk-forward backtest with costs
├── utils/
│   ├── features.py                     # Centralized feature definitions
│   ├── evaluation.py                   # Walk-forward CV and metrics
│   └── alerts.py                       # Email and Discord alerts
├── tests/                              # Test suite
└── .github/workflows/                  # CI/CD pipelines
```

## How It Works

### Data Pipeline
1. **Fetch** - US stock data (5 years) is pulled from Yahoo Finance; Kenyan stock data loaded from historical CSVs
2. **Preprocess** - Raw data is cleaned and enriched with 13 technical indicators computed in `utils/features.py`
3. **Sentiment** - Headlines fetched via NewsAPI, scored with BERTweet, aggregated to daily averages per ticker

### Features (13 Technical Indicators)
RSI-14, SMA-50, OBV, MACD line, MACD signal, Bollinger upper/lower bands, ATR-14, 1/5/10-day returns, 10-day volatility, 20-day volume ratio

### Models
- **Technical Model** - RandomForest trained on 13 technical indicators with walk-forward cross-validation
- **Hybrid Model** - RandomForest using 13 technical indicators + sentiment score (14 features total)
- **Model Comparison** - `compare_models.py` benchmarks RandomForest vs GradientBoosting vs XGBoost

### Backtesting
Walk-forward backtesting with:
- 10 basis points transaction costs per trade
- Sharpe ratio, max drawdown, win rate metrics
- Buy-and-hold benchmark comparison

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

### Configuration

Copy `.env.example` to `.env` and add your API keys:
```bash
cp .env.example .env
```

- `NEWSAPI_KEY` - Get a free key at [newsapi.org](https://newsapi.org) (required for real sentiment data)

### Usage

**1. Fetch stock data:**
```bash
python scripts/fetching/fetch_us_stocks.py
```

**2. Preprocess data and generate features:**
```bash
python scripts/preprocessing/preprocess_data.py
```

**3. Fetch and score news sentiment (requires NewsAPI key):**
```bash
python scripts/fetching/fetch_newsapi.py
python scripts/fetching/fetch_news.py
```

**4. Train models:**
```bash
python scripts/training/train_technical.py
python scripts/training/train_hybrid.py
```

**5. Compare models:**
```bash
python scripts/training/compare_models.py
```

**6. Run backtesting:**
```bash
python scripts/backtesting/backtest.py
```

**7. Run the dashboard:**
```bash
streamlit run app.py
```

## Supported Tickers

### US Stocks
AAPL, MSFT, GOOGL, NVDA, TSLA, INTC, SOXS

### Kenyan Stocks (NSE)
Historical data loaded from CSV files (e.g., SCOM from nse_2020.csv).

## Alerts

Stock AI supports notifications through:
- **Email** - Gmail SMTP (requires app password)
- **Discord** - Webhook integration

## CI/CD

GitHub Actions workflows:
- **fetch_us.yml** - Daily US stock data fetch
- **retrain.yml** - Weekly model retrain (Sundays) with preprocessing
- **deploy.yml** - Streamlit deployment
- **fetch_data.yml** - Kenya data fetch

## Tech Stack

| Component | Technology |
|---|---|
| ML Models | scikit-learn (RandomForest, GradientBoosting), XGBoost |
| Evaluation | Walk-forward cross-validation (TimeSeriesSplit) |
| Sentiment Analysis | Hugging Face Transformers (BERTweet) |
| News Data | NewsAPI |
| Data Fetching | yfinance, BeautifulSoup |
| Feature Engineering | Custom indicators in utils/features.py |
| Backtesting | Walk-forward with transaction costs |
| Dashboard | Streamlit |
| Alerts | SMTP, Discord Webhooks |
| CI/CD | GitHub Actions |

## License

This project is open source.
