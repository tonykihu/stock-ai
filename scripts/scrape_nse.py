import requests
from bs4 import BeautifulSoup
import pandas as pd

def scrape_nse():
    url = "<https://live.mystocks.co.ke>"
    headers = {"User-Agent": "Mozilla/5.0"}  # Avoid bot blocks
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract table (adjust selectors based on NSE's HTML)
    table = soup.find("table", {"class": "table"})  # Update class name
    rows = table.find_all("tr")

    data = []
    for row in rows[1:]:  # Skip header
        cols = row.find_all("td")
        if len(cols) >= 3:
            data.append({
                "Ticker": cols[0].text.strip(),
                "Price": float(cols[1].text.strip()),
                "Volume": int(cols[2].text.replace(",", ""))
            })

    pd.DataFrame(data).to_csv("data/kenya/nse_latest.csv", index=False)

scrape_nse()

print(soup.prettify())


# This script scrapes the NSE website for live trading data and saves it to a CSV file.
# Ensure you have the required libraries installed:
# pip install requests beautifulsoup4 pandas
# Adjust the URL and HTML selectors based on the actual structure of the NSE website.   