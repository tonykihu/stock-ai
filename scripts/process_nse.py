import pandas as pd
import glob
import os

def process_nse():
    """Find and process the latest NSE Kenya data file."""
    pattern = "data/kenya/nse_*.csv"
    files = glob.glob(pattern)

    if not files:
        print(f"No files matching '{pattern}' found.")
        return

    latest_file = max(files, key=os.path.getctime)
    df = pd.read_csv(latest_file)
    df.to_csv("data/kenya/nse_processed.csv", index=False)
    print(f"Processed file: {latest_file} -> data/kenya/nse_processed.csv ({len(df)} rows)")

if __name__ == "__main__":
    process_nse()
