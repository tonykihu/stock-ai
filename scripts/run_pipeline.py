import sys
import os

# Add scripts directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fetch_us_stocks import fetch_all
from scrape_nse import scrape_nse
from upload_to_gsheets import upload
from validate_data import validate_all

def run_pipeline():
    """Run the full data pipeline: fetch, scrape, upload, validate."""
    try:
        print("Step 1: Fetching US stock data...")
        fetch_all()

        print("Step 2: Scraping NSE Kenya data...")
        scrape_nse()

        print("Step 3: Uploading to Google Sheets...")
        upload()

        print("Step 4: Validating data...")
        validate_all()

        print("Pipeline completed successfully!")

    except Exception as e:
        print(f"Pipeline failed: {e}")

if __name__ == "__main__":
    run_pipeline()
