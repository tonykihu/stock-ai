import pandas as pd

def validate(file_path):
    df = pd.read_csv(file_path)
    assert not df.empty, "Data is empty!"
    assert df["Close"].isna().sum() == 0, "Missing values detected"
    print(f"✅ Validation passed for {file_path}")

# Validate the data files
validate("data/us/googl.csv")
validate("data/us/msft.csv")
validate("data/kenya/nse_processed.csv")