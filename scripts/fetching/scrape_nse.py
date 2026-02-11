import argparse
import requests
from bs4 import BeautifulSoup
import pandas as pd

SOURCES = {
    "afx": {
        "name": "AFX Kwayisi",
        "url": "https://afx.kwayisi.org/nse/",
        "method": "static",
    },
    "mystocks": {
        "name": "myStocks",
        "url": "https://live.mystocks.co.ke",
        "method": "selenium",
    },
    "fib": {
        "name": "Faida Investment Bank",
        "url": "https://fib.co.ke/live-markets/",
        "method": "selenium",
    },
}

HEADERS = {"User-Agent": "Mozilla/5.0"}
OUTPUT_FILE = "data/kenya/nse_latest.csv"


def scrape_afx():
    """
    Scrape NSE data from afx.kwayisi.org/nse/
    Static HTML — fast, no browser needed.
    Requires html5lib parser for the unclosed HTML tags on this site.
    """
    url = SOURCES["afx"]["url"]
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()

    # html5lib handles AFX's unclosed <td>/<tr> tags correctly
    soup = BeautifulSoup(response.text, "html5lib")

    # Find the table with stock listings (has thead with "Ticker")
    table = None
    for t in soup.find_all("table"):
        thead = t.find("thead")
        if thead and "Ticker" in thead.get_text():
            table = t
            break

    if table is None:
        print("No stock data table found on AFX.")
        return []

    tbody = table.find("tbody")
    rows = tbody.find_all("tr") if tbody else table.find_all("tr")[1:]

    data = []
    for row in rows:
        cols = row.find_all("td")
        if len(cols) >= 5:
            try:
                ticker = cols[0].get_text(strip=True)
                if not ticker:
                    continue

                volume_text = cols[2].get_text(strip=True).replace(",", "")
                price_text = cols[3].get_text(strip=True).replace(",", "")
                change_text = cols[4].get_text(strip=True)

                data.append({
                    "Ticker": ticker,
                    "Name": cols[1].get_text(strip=True),
                    "Volume": int(volume_text) if volume_text else 0,
                    "Price": float(price_text) if price_text else 0.0,
                    "Change": change_text if change_text else "0",
                })
            except (ValueError, IndexError) as e:
                print(f"Skipping row: {e}")
                continue
    return data


def _get_selenium_driver():
    """Create a headless Chrome Selenium driver."""
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
    except ImportError:
        print("Selenium not installed. Run: pip install selenium")
        print("Also requires ChromeDriver in PATH.")
        return None

    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    return webdriver.Chrome(options=options)


def scrape_mystocks():
    """
    Scrape NSE data from live.mystocks.co.ke
    Data loads via JavaScript — requires Selenium.
    """
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC

    driver = _get_selenium_driver()
    if driver is None:
        return []

    try:
        driver.get(SOURCES["mystocks"]["url"])

        # Wait for stock data table to load
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "table.table td"))
        )

        soup = BeautifulSoup(driver.page_source, "html.parser")
        table = soup.find("table", {"class": "table"})
        if table is None:
            print("No stock table found on myStocks after loading.")
            return []

        data = []
        for row in table.find_all("tr")[1:]:
            cols = row.find_all("td")
            if len(cols) >= 3:
                try:
                    data.append({
                        "Ticker": cols[0].get_text(strip=True),
                        "Price": float(cols[1].get_text(strip=True).replace(",", "")),
                        "Volume": int(cols[2].get_text(strip=True).replace(",", "")),
                    })
                except (ValueError, IndexError) as e:
                    print(f"Skipping row: {e}")
                    continue
        return data

    except Exception as e:
        print(f"myStocks scraping failed: {e}")
        return []
    finally:
        driver.quit()


def scrape_fib():
    """
    Scrape NSE data from fib.co.ke/live-markets/
    Data loads via JavaScript — requires Selenium.
    """
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC

    driver = _get_selenium_driver()
    if driver is None:
        return []

    try:
        driver.get(SOURCES["fib"]["url"])

        # Wait for the table to load
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "table td"))
        )

        soup = BeautifulSoup(driver.page_source, "html.parser")
        table = soup.find("table")
        if table is None:
            print("No table found on FIB after loading.")
            return []

        data = []
        for row in table.find_all("tr")[1:]:
            cols = row.find_all("td")
            if len(cols) >= 3:
                try:
                    data.append({
                        "Ticker": cols[0].get_text(strip=True),
                        "Price": float(cols[1].get_text(strip=True).replace(",", "")),
                        "Volume": int(cols[2].get_text(strip=True).replace(",", "")),
                    })
                except (ValueError, IndexError) as e:
                    print(f"Skipping row: {e}")
                    continue
        return data

    except Exception as e:
        print(f"FIB scraping failed: {e}")
        return []
    finally:
        driver.quit()


def scrape_nse(source="afx"):
    """
    Scrape NSE Kenya data from the chosen source.

    Args:
        source: One of 'afx', 'mystocks', or 'fib'
    """
    if source not in SOURCES:
        print(f"Unknown source: '{source}'")
        print(f"Available sources: {', '.join(SOURCES.keys())}")
        return

    info = SOURCES[source]
    print(f"Scraping from {info['name']} ({info['url']})...")
    if info["method"] == "selenium":
        print("(This source requires Selenium + ChromeDriver)")

    scrapers = {
        "afx": scrape_afx,
        "mystocks": scrape_mystocks,
        "fib": scrape_fib,
    }

    data = scrapers[source]()

    if data:
        df = pd.DataFrame(data)
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"Scraped {len(df)} rows and saved to {OUTPUT_FILE}")
    else:
        print("No data scraped. Try a different source.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape NSE Kenya stock data")
    parser.add_argument(
        "--source",
        choices=list(SOURCES.keys()),
        default="afx",
        help="Data source to scrape from (default: afx)",
    )
    args = parser.parse_args()

    print("Available sources:")
    for key, info in SOURCES.items():
        method_tag = "static" if info["method"] == "static" else "requires Selenium"
        marker = " <-- selected" if key == args.source else ""
        print(f"  {key:10s} -> {info['name']} ({method_tag}){marker}")
    print()

    scrape_nse(args.source)
