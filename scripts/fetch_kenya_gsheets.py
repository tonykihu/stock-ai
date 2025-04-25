# scripts/fetch_kenya_gsheets.py
import gspread
import pandas as pd

def get_gsheets_data():
    """
    Fetches data from Google Sheets for AI training.
    """
    gc = gspread.service_account(filename='credentials.json')
    sheet = gc.open("NSE Kenya Manual Upload").sheet1
    data = pd.DataFrame(sheet.get_all_records())
    data.to_csv('data/kenya/nse_gsheets.csv', index=False)
    return data


# scripts/load_gsheets.py
import gspread
import pandas as pd

def load_historical_data(sheet_url):
    """
    Loads your Google Sheets data into a pandas DataFrame.
    Args:
        sheet_url (str): URL of your Google Sheet (e.g., https://docs.google.com/.../)
    """
    gc = gspread.service_account("credentials.json")  # Auth file from Phase 1
    sheet = gc.open_by_url(sheet_url).sheet1
    data = pd.DataFrame(sheet.get_all_records())
    
    # Convert date strings to datetime
    data["Date"] = pd.to_datetime(data["Date"], format="%Y-%m-%d")  
    data.sort_values("Date", inplace=True)
    return data

# Example: Load your uploaded data
df = load_historical_data("YOUR_SHEET_URL")
df.to_csv("data/kenya/nse_historical.csv", index=False)