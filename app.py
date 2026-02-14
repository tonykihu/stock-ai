import os
import streamlit as st
import pandas as pd
import joblib
import yfinance as yf
from datetime import datetime
from utils.features import compute_features, TECHNICAL_FEATURES, HYBRID_FEATURES
from utils.alerts import send_email_alert, send_discord_alert

# --- Load models safely ---
@st.cache_resource
def load_models():
    """Load trained models with error handling."""
    models = {}
    try:
        models["technical"] = joblib.load("models/technical_model.pkl")
    except FileNotFoundError:
        st.warning("Technical model not found. Train it first with: python scripts/training/train_technical.py")
    try:
        models["hybrid"] = joblib.load("models/hybrid_model.pkl")
    except FileNotFoundError:
        st.warning("Hybrid model not found. Train it first with: python scripts/training/train_hybrid.py")
    return models

models = load_models()


def fetch_latest_data(ticker):
    """Fetch latest stock data and compute all features."""
    if ".NSE" in ticker:
        # Kenyan ticker - return mock data (replace with real NSE data source)
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
    else:
        # US ticker - fetch 6 months for indicator warmup (SMA-50 needs 50+ bars)
        data = yf.download(ticker, period="6mo")
        if data.empty:
            st.error(f"No data returned for {ticker}")
            return None

        # Flatten multi-level columns from yfinance (e.g. ('Close','AAPL') -> 'Close')
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data.reset_index()
        if "Date" not in data.columns and "index" in data.columns:
            data = data.rename(columns={"index": "Date"})

        data = compute_features(data)
        return data.iloc[-1:].reset_index(drop=True)


# --- UI ---
st.title("AI Stock Signal Dashboard")
st.markdown("""
**US & Kenyan Market Predictions**
*Data updates EOD | Models retrain weekly*
""")

# Inputs
ticker = st.selectbox("Select Ticker", ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA", "SCOM.NSE"])
model_type = st.radio("Model", ["Technical Only", "Hybrid (Technical + Sentiment)"])

# Fetch real data
data = fetch_latest_data(ticker)

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
            data["sentiment_score"] = 0.5  # Neutral default
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
    if ".NSE" not in ticker:
        chart_data = yf.download(ticker, period="1mo")
        if not chart_data.empty:
            if isinstance(chart_data.columns, pd.MultiIndex):
                chart_data.columns = chart_data.columns.get_level_values(0)
            st.line_chart(chart_data["Close"])

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
        send_email_alert("BUY", ticker)
elif alert_type == "Discord":
    webhook_url = st.text_input("Enter Discord Webhook URL")
    if st.button("Test Discord Alert", key="discord_alert_btn"):
        send_discord_alert("BUY", ticker, webhook_url)
