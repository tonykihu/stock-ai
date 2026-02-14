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
├── app.py                              # Streamlit dashboard (dynamic sidebar)
├── data/
│   ├── kenya/                          # NSE market data (2020-2024)
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
│   ├── tickers.py                      # Centralized ticker registry (country/sector/name)
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
A Streamlit web app with a **dynamic sidebar** for filtering:
- **Country filter** - US or Kenya (auto-detected from processed data)
- **Sector filter** - Technology, Banking/Finance, Healthcare, Consumer, ETFs, etc.
- **Ticker selector** - Shows only tickers with processed data, displayed as "AAPL - Apple"
- Model selection (Technical or Hybrid)
- Real-time buy/sell signals
- Interactive price charts with period selector (1W, 1M, 3M, 6M, 1Y, All) — live data for US, historical for Kenya
- CSV upload for custom NSE data
- Email and Discord alert configuration

The dashboard **auto-discovers** available tickers from `features.csv` — no manual updates needed when new stocks are added.

## Supported Tickers

All tickers are defined in `utils/tickers.py`. To add a new ticker, edit that one file.

### US Stocks (30)

| Sector | Tickers |
|---|---|
| Technology | AAPL, MSFT, GOOGL, NVDA, TSLA, INTC, META, AMZN, CRM, ORCL, ADBE, AMD |
| Banking / Finance | JPM, BAC, GS, MS, WFC, V, MA |
| Healthcare | JNJ, UNH, PFE |
| Consumer / Industrials | KO, PG, WMT, DIS, HD |
| ETFs | SOXS, SPY, QQQ |

### Kenyan Stocks (10)

| Sector | Tickers |
|---|---|
| Telecom | SCOM (Safaricom) |
| Banking / Finance | EQTY, KCB, ABSA, COOP, SCBK, NCBA |
| Consumer | EABL, BAT |
| Industrials | BAMB |

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

### Adding New Tickers

Edit `utils/tickers.py` and add entries to the `TICKER_REGISTRY` dict:
```python
"Technology": [
    {"ticker": "AAPL", "name": "Apple"},
    {"ticker": "NEW_TICKER", "name": "New Company"},  # Add here
],
```

Then re-run the pipeline:
```bash
python scripts/fetching/fetch_us_stocks.py      # Fetch data
python scripts/preprocessing/preprocess_data.py  # Compute features
```

The dashboard will automatically pick up the new ticker.

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
| Ticker Management | Centralized registry in utils/tickers.py |
| Backtesting | Walk-forward with transaction costs |
| Dashboard | Streamlit (dynamic sidebar with country/sector filtering) |
| Alerts | SMTP, Discord Webhooks |
| CI/CD | GitHub Actions |

## License

This project is open source.
