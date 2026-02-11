import os
import pandas as pd

def upload():
    """
    Upload processed data to Google Sheets.
    Requires gspread and a credentials.json file.
    Set GSHEETS_CREDENTIALS env var to the path of your credentials file.
    """
    try:
        import gspread
    except ImportError:
        print("gspread not installed. Skipping Google Sheets upload.")
        print("Install with: pip install gspread")
        return

    creds_path = os.environ.get("GSHEETS_CREDENTIALS", "credentials.json")
    if not os.path.exists(creds_path):
        print(f"Credentials file not found at {creds_path}. Skipping upload.")
        return

    try:
        gc = gspread.service_account(filename=creds_path)
        sheet = gc.open("NSE Kenya Manual Upload").sheet1

        # Load the latest data
        data = pd.read_csv("data/kenya/nse_latest.csv")

        # Clear and update sheet
        sheet.clear()
        sheet.update([data.columns.values.tolist()] + data.values.tolist())
        print("Data uploaded to Google Sheets successfully.")

    except Exception as e:
        print(f"Google Sheets upload failed: {e}")

if __name__ == "__main__":
    upload()
