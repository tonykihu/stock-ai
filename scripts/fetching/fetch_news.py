"""
Score news headlines with BERTweet sentiment analysis.

Reads raw headlines (from fetch_newsapi.py or manual CSV) and produces
daily average sentiment scores per (Date, Ticker).

Usage:
  python scripts/fetching/fetch_news.py
"""
import os
import sys
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def load_headlines(path="data/processed/raw_headlines.csv"):
    """Load raw headlines CSV. Expected columns: Date, Headline, Ticker."""
    if not os.path.exists(path):
        print(f"No headlines file at {path}")
        print("Run fetch_newsapi.py first, or place a CSV with Date, Headline, Ticker columns.")
        return pd.DataFrame()
    df = pd.read_csv(path)
    required = {"Date", "Headline"}
    if not required.issubset(df.columns):
        print(f"Headlines CSV missing columns: {required - set(df.columns)}")
        return pd.DataFrame()
    return df


def score_headlines(df):
    """
    Run BERTweet sentiment on each headline.

    Returns DataFrame with added columns: Sentiment, raw_score, sentiment_score.
    sentiment_score is mapped to [0, 1]: NEG→low, NEU→0.5, POS→high.
    """
    if df.empty:
        return df

    from transformers import pipeline

    print("Loading BERTweet sentiment model...")
    analyzer = pipeline(
        "sentiment-analysis",
        model="finiteautomata/bertweet-base-sentiment-analysis",
    )

    # Score in batches to avoid memory issues
    batch_size = 32
    headlines = df["Headline"].tolist()
    all_results = []

    for i in range(0, len(headlines), batch_size):
        batch = headlines[i : i + batch_size]
        # Truncate long headlines (BERTweet max ~128 tokens)
        batch = [h[:280] if isinstance(h, str) else "" for h in batch]
        results = analyzer(batch)
        all_results.extend(results)
        if (i + batch_size) % 100 == 0:
            print(f"  Scored {min(i + batch_size, len(headlines))}/{len(headlines)} headlines")

    # Map labels to sentiment_score
    scores = []
    labels = []
    for r in all_results:
        label = r["label"]
        confidence = r["score"]
        labels.append(label)

        if label == "POS":
            scores.append(0.5 + 0.5 * confidence)  # 0.5 → 1.0
        elif label == "NEG":
            scores.append(0.5 - 0.5 * confidence)  # 0.5 → 0.0
        else:  # NEU
            scores.append(0.5)

    df = df.copy()
    df["Sentiment"] = labels
    df["raw_score"] = [r["score"] for r in all_results]
    df["sentiment_score"] = scores

    return df


def aggregate_daily_sentiment(df):
    """
    Aggregate to daily average sentiment per (Date, Ticker).

    Returns DataFrame with columns: Date, Ticker, sentiment_score, headline_count.
    """
    if df.empty or "sentiment_score" not in df.columns:
        return pd.DataFrame()

    group_cols = ["Date"]
    if "Ticker" in df.columns:
        group_cols.append("Ticker")

    agg = df.groupby(group_cols, as_index=False).agg(
        sentiment_score=("sentiment_score", "mean"),
        headline_count=("Headline", "count"),
    )

    return agg


def main():
    """Full pipeline: load headlines → score → aggregate → save."""
    headlines = load_headlines()
    if headlines.empty:
        return

    print(f"Loaded {len(headlines)} headlines")

    # Score with BERTweet
    scored = score_headlines(headlines)
    if scored.empty:
        return

    # Save detailed scores
    os.makedirs("data/processed", exist_ok=True)
    scored.to_csv("data/processed/scored_headlines.csv", index=False)
    print(f"Detailed scores saved to data/processed/scored_headlines.csv")

    # Aggregate to daily sentiment
    daily = aggregate_daily_sentiment(scored)
    daily.to_csv("data/processed/news_sentiment.csv", index=False)
    print(f"\nDaily sentiment saved to data/processed/news_sentiment.csv")
    print(f"  {len(daily)} date-ticker pairs")
    print(f"  Mean sentiment: {daily['sentiment_score'].mean():.3f}")


if __name__ == "__main__":
    main()
