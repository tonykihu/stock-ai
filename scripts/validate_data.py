import os
import pandas as pd

def validate(file_path, price_column="Close"):
    """
    Validate a data file for completeness.
    Args:
        file_path: Path to the CSV file
        price_column: Name of the price column to check (default: "Close")
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return False

    df = pd.read_csv(file_path)
    assert not df.empty, f"Data is empty in {file_path}!"

    if price_column in df.columns:
        assert df[price_column].isna().sum() == 0, f"Missing values in '{price_column}' column of {file_path}"

    print(f"Validation passed for {file_path}")
    return True

def validate_all():
    """Validate all data files."""
    files_to_validate = [
        ("data/us/googl.csv", "Close"),
        ("data/us/msft.csv", "Close"),
        ("data/us/aapl.csv", "Close"),
        ("data/kenya/nse_latest.csv", "Price"),
    ]

    all_passed = True
    for file_path, price_col in files_to_validate:
        try:
            if not validate(file_path, price_col):
                all_passed = False
        except Exception as e:
            print(f"Validation failed: {e}")
            all_passed = False

    if all_passed:
        print("All validations passed!")
    else:
        print("Some validations failed. Check logs above.")

if __name__ == "__main__":
    validate_all()
