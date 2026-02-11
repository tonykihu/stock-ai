"""Test Google Sheets authentication."""
import os

try:
    import gspread
except ImportError:
    print("gspread not installed. Run: pip install gspread")
    exit(1)

creds_path = os.environ.get("GSHEETS_CREDENTIALS", "credentials.json")
if not os.path.exists(creds_path):
    print(f"Credentials file not found at: {creds_path}")
    print("Download it from Google Cloud Console.")
    exit(1)

gc = gspread.service_account(filename=creds_path)
print("Authentication successful!" if gc else "Failed")
