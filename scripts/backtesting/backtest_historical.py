# scripts/backtest_historical.py
import pandas as pd
import joblib

def backtest_strategy():
    """Backtest the historical model on Kenya NSE data."""
    # Load the trained model
    model = joblib.load("models/historical_model.pkl")

    data = pd.read_csv("data/processed/kenya_processed.csv")

    # Drop rows with NaN in the feature columns
    data = data.dropna(subset=["sma_50", "rsi_14"])

    data["Prediction"] = model.predict(data[["sma_50", "rsi_14"]])

    # Calculate returns
    data["Strategy_Return"] = data["Prediction"] * data["Price"].pct_change().shift(-1)
    data["Buy_Hold_Return"] = data["Price"].pct_change().shift(-1)

    # Drop NaN rows
    data = data.dropna()

    # Cumulative returns
    strategy_total = (1 + data["Strategy_Return"]).cumprod().iloc[-1]
    buy_hold_total = (1 + data["Buy_Hold_Return"]).cumprod().iloc[-1]

    print(f"Strategy Return: {strategy_total:.2f}x")
    print(f"Buy-and-Hold Return: {buy_hold_total:.2f}x")

if __name__ == "__main__":
    backtest_strategy()
