# scripts/backtest_historical.py
import pandas as pd

def backtest_strategy():
    data = pd.read_csv("data/processed/kenya_processed.csv")
    data["Prediction"] = model.predict(data[["sma_50", "rsi_14"]])
    
    # Calculate returns
    data["Strategy_Return"] = data["Prediction"] * data["Price"].pct_change().shift(-1)
    data["Buy_Hold_Return"] = data["Price"].pct_change().shift(-1)
    
    # Cumulative returns
    strategy_total = (1 + data["Strategy_Return"]).cumprod().iloc[-1]
    buy_hold_total = (1 + data["Buy_Hold_Return"]).cumprod().iloc[-1]
    
    print(f"Strategy Return: {strategy_total:.2f}x")
    print(f"Buy-and-Hold Return: {buy_hold_total:.2f}x")

backtest_strategy()