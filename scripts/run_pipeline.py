from fetch_us_stocks import fetch_data
#from scrape_nse import scrape_nse
from upload_to_gsheets import upload
import validate_data

try:
    fetch_data()  # US
    #scrape_nse()  # Kenya
    upload()      # Google Sheets
    validate_data.validate_all()
except Exception as e:
    print(f"🚨 Pipeline failed: {e}")
    # Send alert via Twilio/Discord (optional)