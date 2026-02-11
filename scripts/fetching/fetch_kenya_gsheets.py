# scripts/fetch_kenya_gsheets.py
import os
import pandas as pd

def get_gsheets_data(creds_path=None):
    """
    Fetches data from Google Sheets for AI training.
    Requires gspread and a Google service account credentials.json.
    """
    try:
        import gspread
    except ImportError:
        print("gspread not installed. Run: pip install gspread")
        return None

    if creds_path is None:
        creds_path = os.environ.get("GSHEETS_CREDENTIALS", "credentials.json")

    if not os.path.exists(creds_path):
        print(f"Credentials file not found at {creds_path}")
        return None

    gc = gspread.service_account(filename=creds_path)
    sheet = gc.open("NSE Kenya Manual Upload").sheet1
    data = pd.DataFrame(sheet.get_all_records())
    data.to_csv("data/kenya/nse_gsheets.csv", index=False)
    print(f"Fetched {len(data)} rows from Google Sheets.")
    return data

def load_historical_data(sheet_url, creds_path=None):
    """
    Loads your Google Sheets data into a pandas DataFrame.
    Args:
        sheet_url (str): URL of your Google Sheet
        creds_path (str): Path to credentials.json (optional)
    """
    try:
        import gspread
    except ImportError:
        print("gspread not installed. Run: pip install gspread")
        return None

    if creds_path is None:
        creds_path = os.environ.get("GSHEETS_CREDENTIALS", "credentials.json")

    if not os.path.exists(creds_path):
        print(f"Credentials file not found at {creds_path}")
        return None

    gc = gspread.service_account(filename=creds_path)
    sheet = gc.open_by_url(sheet_url).sheet1
    data = pd.DataFrame(sheet.get_all_records())

    # Convert date strings to datetime
    data["Date"] = pd.to_datetime(data["Date"], format="%Y-%m-%d")
    data.sort_values("Date", inplace=True)
    return data

if __name__ == "__main__":
    # Fetch from named sheet
    get_gsheets_data()

    # Or load from a specific URL (uncomment and replace with your URL):
    # df = load_historical_data("https://docs.google.com/spreadsheets/d/YOUR_SHEET_ID")
    # df.to_csv("data/kenya/nse_historical.csv", index=False)
