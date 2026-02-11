from transformers import pipeline
import pandas as pd
from datetime import datetime

def fetch_and_analyze_news(headlines=None):
    """
    Analyze sentiment of financial news headlines.
    Pass a list of headlines, or uses example headlines if none provided.
    Uses the BERTweet model for financial sentiment analysis.
    """
    if headlines is None:
        # Example headlines (replace with real news API integration)
        headlines = [
            "Apple stock surges after earnings report",
            "NSE Kenya faces volatility due to election uncertainty"
        ]

    # Analyze sentiment
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="finiteautomata/bertweet-base-sentiment-analysis"
    )
    results = [sentiment_analyzer(headline) for headline in headlines]

    # Build DataFrame with Date column for merging with features
    df = pd.DataFrame({
        "Date": [datetime.now().strftime("%Y-%m-%d")] * len(headlines),
        "Headline": headlines,
        "Sentiment": [r[0]["label"] for r in results],
        "sentiment_score": [r[0]["score"] for r in results]
    })

    df.to_csv("data/processed/news_sentiment.csv", index=False)
    print(f"Sentiment analysis complete. {len(df)} headlines processed.")
    return df

if __name__ == "__main__":
    fetch_and_analyze_news()
