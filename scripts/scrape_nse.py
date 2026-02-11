import requests
from bs4 import BeautifulSoup
import pandas as pd

def scrape_nse():
    """
    Scrapes the NSE website for live trading data and saves it to a CSV file.
    Requires: pip install requests beautifulsoup4 pandas
    Adjust the URL and HTML selectors based on the actual structure of the NSE website.
    """
    url = "https://live.mystocks.co.ke"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract table (adjust selectors based on NSE's HTML)
    table = soup.find("table", {"class": "table"})
    if table is None:
        print("No table found on the page. Check the HTML structure.")
        print(soup.prettify()[:500])
        return

    rows = table.find_all("tr")

    data = []
    for row in rows[1:]:  # Skip header
        cols = row.find_all("td")
        if len(cols) >= 3:
            try:
                data.append({
                    "Ticker": cols[0].text.strip(),
                    "Price": float(cols[1].text.strip()),
                    "Volume": int(cols[2].text.replace(",", ""))
                })
            except (ValueError, IndexError) as e:
                print(f"Skipping row due to parsing error: {e}")
                continue

    if data:
        pd.DataFrame(data).to_csv("data/kenya/nse_latest.csv", index=False)
        print(f"Scraped {len(data)} rows and saved to data/kenya/nse_latest.csv")
    else:
        print("No data scraped. Check the page structure.")

if __name__ == "__main__":
    scrape_nse()
