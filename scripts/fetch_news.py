from transformers import pipeline
import pandas as pd

# Load headlines (example: mock data)
headlines = [
    "Apple stock surges after earnings report",
    "NSE Kenya faces volatility due to election uncertainty"
]

# Analyze sentiment
sentiment_analyzer = pipeline("sentiment-analysis", model="finiteautomata/bertweet-base-sentiment-analysis")
results = [sentiment_analyzer(headline) for headline in headlines]

# Save to CSV
pd.DataFrame({
    "Headline": headlines,
    "Sentiment": [r[0]["label"] for r in results],
    "Score": [r[0]["score"] for r in results]
}).to_csv("data/processed/news_sentiment.csv", index=False)