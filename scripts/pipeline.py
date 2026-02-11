import subprocess
import sys


def run_pipeline():
    """Run the full data pipeline: fetch, scrape, upload, validate."""
    steps = [
        ("Step 1: Fetching US stock data...", "scripts/fetching/fetch_us_stocks.py"),
        ("Step 2: Scraping NSE Kenya data...", "scripts/fetching/scrape_nse.py"),
        ("Step 3: Uploading to Google Sheets...", "scripts/upload_to_gsheets.py"),
        ("Step 4: Validating data...", "scripts/validate_data.py"),
    ]

    for label, script in steps:
        print(label)
        result = subprocess.run([sys.executable, script], capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(f"Pipeline failed at: {label}")
            print(result.stderr)
            return

    print("Pipeline completed successfully!")


if __name__ == "__main__":
    run_pipeline()
