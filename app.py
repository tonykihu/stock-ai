import os
import streamlit as st
import pandas as pd
import joblib
import yfinance as yf
from datetime import datetime
from utils.features import compute_features, TECHNICAL_FEATURES, HYBRID_FEATURES
from utils.alerts import send_email_alert, send_discord_alert
from utils.tickers import (
    TICKER_REGISTRY,
    get_ticker_name,
    get_ticker_metadata,
    get_countries,
    get_sectors,
    get_tickers_by_country,
    get_tickers_by_sector,
)

# --- Page config (must be first st command) ---
st.set_page_config(page_title="AI Stock Signal Dashboard", layout="wide")

FEATURES_PATH = "data/processed/features.csv"


# --- Load models safely ---
@st.cache_resource
def load_models():
    """Load trained models with error handling."""
    models = {}
    try:
        models["technical"] = joblib.load("models/technical_model.pkl")
    except FileNotFoundError:
        pass
    try:
        models["hybrid"] = joblib.load("models/hybrid_model.pkl")
    except FileNotFoundError:
        pass
    return models


@st.cache_data(ttl=3600)
def discover_available_tickers():
    """
    Auto-discover tickers from features.csv.
    Returns (set of ticker symbols, dict of ticker->market).
    Falls back to TICKER_REGISTRY if features.csv doesn't exist.
    """
    if not os.path.exists(FEATURES_PATH):
        all_tickers = set()
        ticker_markets = {}
        for country, sectors in TICKER_REGISTRY.items():
            for sector_entries in sectors.values():
                for entry in sector_entries:
                    all_tickers.add(entry["ticker"])
                    ticker_markets[entry["ticker"]] = country
        return all_tickers, ticker_markets

    df = pd.read_csv(FEATURES_PATH, usecols=["Ticker", "Market"])
    available = set(df["Ticker"].unique())
    # Map Market column values to country names
    market_to_country = {"US": "US", "Kenya": "Kenya"}
    ticker_markets = {}
    for _, row in df.drop_duplicates("Ticker").iterrows():
        ticker_markets[row["Ticker"]] = market_to_country.get(row["Market"], row["Market"])
    return available, ticker_markets


def build_sidebar_filters(available_tickers, ticker_markets):
    """
    Sidebar filters: Country -> Sector -> Ticker.
    Only shows tickers that are actually in features.csv.
    Returns (selected_ticker, selected_country) or (None, None).
    """
    st.sidebar.header("Filters")

    # Country filter — from available data
    available_countries = sorted(set(ticker_markets.values()))
    if not available_countries:
        available_countries = get_countries()
    selected_country = st.sidebar.selectbox("Country", available_countries)

    # Sector filter — from registry metadata
    sectors = get_sectors(selected_country)
    sector_options = ["All Sectors"] + sectors
    selected_sector = st.sidebar.selectbox("Sector", sector_options)

    # Build filtered ticker list
    if selected_sector == "All Sectors":
        registry_tickers = get_tickers_by_country(selected_country)
    else:
        registry_tickers = get_tickers_by_sector(selected_country, selected_sector)

    # Intersect with what's actually processed in features.csv
    filtered_tickers = [t for t in registry_tickers if t in available_tickers]

    # Also include tickers from features.csv that are NOT in the registry
    # (future-proofing for dynamically added tickers)
    if selected_sector == "All Sectors":
        extra = [
            t for t in available_tickers
            if ticker_markets.get(t) == selected_country
            and t not in filtered_tickers
        ]
        if extra:
            filtered_tickers.extend(sorted(extra))

    if not filtered_tickers:
        st.sidebar.warning(f"No processed data for {selected_country} / {selected_sector}")
        st.sidebar.info("Run preprocessing: `python scripts/preprocessing/preprocess_data.py`")
        return None, selected_country

    # Ticker selectbox with human-readable labels: "AAPL - Apple"
    ticker_labels = {}
    for t in filtered_tickers:
        name = get_ticker_name(t)
        ticker_labels[t] = f"{t} - {name}" if name != t else t

    selected_label = st.sidebar.selectbox("Ticker", list(ticker_labels.values()))

    # Reverse lookup: label -> ticker symbol
    selected_ticker = next(t for t, label in ticker_labels.items() if label == selected_label)

    # Info
    st.sidebar.caption(
        f"{len(filtered_tickers)} tickers in {selected_country}"
        + (f" / {selected_sector}" if selected_sector != "All Sectors" else "")
    )

    return selected_ticker, selected_country


def fetch_latest_data(ticker, country):
    """Fetch latest stock data and compute all features."""
    if country == "Kenya":
        # Try yfinance with .NR suffix for Nairobi
        yf_ticker = f"{ticker}.NR"
        data = yf.download(yf_ticker, period="6mo")
        if data.empty:
            # Fallback: mock data (replace with real NSE source later)
            mock = pd.DataFrame({
                "Date": [pd.Timestamp.now()],
                "Close": [15.50],
                "High": [16.00],
                "Low": [15.00],
                "Volume": [1000000.0],
                "sentiment_score": [0.5],
            })
            mock = compute_features(mock)
            return mock.iloc[-1:].reset_index(drop=True)

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data = data.reset_index()
        if "Date" not in data.columns and "index" in data.columns:
            data = data.rename(columns={"index": "Date"})
        data = compute_features(data)
        return data.iloc[-1:].reset_index(drop=True)
    else:
        # US ticker
        data = yf.download(ticker, period="6mo")
        if data.empty:
            st.error(f"No data returned for {ticker}")
            return None

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data = data.reset_index()
        if "Date" not in data.columns and "index" in data.columns:
            data = data.rename(columns={"index": "Date"})
        data = compute_features(data)
        return data.iloc[-1:].reset_index(drop=True)


# --- Discover tickers and build filters ---
models = load_models()
available_tickers, ticker_markets = discover_available_tickers()
selected_ticker, selected_country = build_sidebar_filters(available_tickers, ticker_markets)

# --- Main content ---
st.title("AI Stock Signal Dashboard")
st.markdown("""
**US & Kenyan Market Predictions**
*Data updates EOD | Models retrain weekly*
""")

if selected_ticker is None:
    st.info("Select a ticker from the sidebar to see predictions.")
    if not models:
        st.warning("No models found. Train first:\n"
                    "`python scripts/training/train_technical.py`")
    st.stop()

# Model selector
model_type = st.radio("Model", ["Technical Only", "Hybrid (Technical + Sentiment)"])

# Fetch real data
data = fetch_latest_data(selected_ticker, selected_country)

# Predict
if data is not None:
    if model_type == "Technical Only" and "technical" in models:
        available = [f for f in TECHNICAL_FEATURES if f in data.columns]
        features = data[available].dropna()
        if not features.empty and len(available) == len(TECHNICAL_FEATURES):
            prediction = models["technical"].predict(features)
            st.metric("Signal", "BUY" if prediction[0] == 1 else "HOLD/SELL")
        else:
            st.warning("Not enough data to compute indicators.")
    elif model_type == "Hybrid (Technical + Sentiment)" and "hybrid" in models:
        if "sentiment_score" not in data.columns:
            data["sentiment_score"] = 0.5
        available = [f for f in HYBRID_FEATURES if f in data.columns]
        features = data[available].dropna()
        if not features.empty and len(available) == len(HYBRID_FEATURES):
            prediction = models["hybrid"].predict(features)
            st.metric("Signal", "BUY" if prediction[0] == 1 else "HOLD/SELL")
        else:
            st.warning("Not enough data to compute indicators.")
    else:
        st.info("Model not available. Please train the model first.")

    # Show price chart
    if selected_country != "Kenya":
        chart_data = yf.download(selected_ticker, period="1mo")
        if not chart_data.empty:
            if isinstance(chart_data.columns, pd.MultiIndex):
                chart_data.columns = chart_data.columns.get_level_values(0)
            st.line_chart(chart_data["Close"])
    else:
        # Kenya: show historical from features.csv
        if os.path.exists(FEATURES_PATH):
            hist = pd.read_csv(FEATURES_PATH)
            hist = hist[hist["Ticker"] == selected_ticker].tail(60)
            if not hist.empty:
                hist["Date"] = pd.to_datetime(hist["Date"])
                hist = hist.set_index("Date")
                st.line_chart(hist["Close"])
            else:
                st.info(f"No historical chart data for {selected_ticker}")

# --- Upload NSE Data ---
st.subheader("Upload NSE Kenya Data")
uploaded_file = st.file_uploader("Upload NSE Kenya Data (CSV)")
if uploaded_file:
    nse_data = pd.read_csv(uploaded_file)
    st.write("Latest NSE Data:", nse_data.tail())

# --- Feedback ---
st.subheader("Feedback")
feedback = st.text_area("How can we improve?")
if st.button("Submit Feedback"):
    os.makedirs("feedback", exist_ok=True)
    with open("feedback/log.txt", "a") as f:
        f.write(f"{datetime.now()}: {feedback}\n")
    st.success("Thanks for your input!")

# --- Alert section ---
st.subheader("Set Up Alerts")
alert_type = st.radio("Alert Channel", ["Email", "Discord"], key="alert_channel")

if alert_type == "Email":
    if st.button("Enable Email Alerts", key="email_alert_btn"):
        send_email_alert("BUY", selected_ticker)
elif alert_type == "Discord":
    webhook_url = st.text_input("Enter Discord Webhook URL")
    if st.button("Test Discord Alert", key="discord_alert_btn"):
        send_discord_alert("BUY", selected_ticker, webhook_url)
