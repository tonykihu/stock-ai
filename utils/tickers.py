"""
Centralized ticker configuration for US and Kenya markets.

All scripts import ticker lists and metadata from here.
To add a new ticker, add it to TICKER_REGISTRY below.

In the future, this module can be extended to pull from
Google Sheets, Baserow, or a database instead of a static dict.
"""

# Master ticker registry: country -> sector -> list of {ticker, name}
TICKER_REGISTRY = {
    "US": {
        "Technology": [
            {"ticker": "AAPL", "name": "Apple"},
            {"ticker": "MSFT", "name": "Microsoft"},
            {"ticker": "GOOGL", "name": "Alphabet (Google)"},
            {"ticker": "NVDA", "name": "NVIDIA"},
            {"ticker": "TSLA", "name": "Tesla"},
            {"ticker": "INTC", "name": "Intel"},
            {"ticker": "META", "name": "Meta Platforms"},
            {"ticker": "AMZN", "name": "Amazon"},
            {"ticker": "CRM", "name": "Salesforce"},
            {"ticker": "ORCL", "name": "Oracle"},
            {"ticker": "ADBE", "name": "Adobe"},
            {"ticker": "AMD", "name": "AMD"},
        ],
        "Banking / Finance": [
            {"ticker": "JPM", "name": "JPMorgan Chase"},
            {"ticker": "BAC", "name": "Bank of America"},
            {"ticker": "GS", "name": "Goldman Sachs"},
            {"ticker": "MS", "name": "Morgan Stanley"},
            {"ticker": "WFC", "name": "Wells Fargo"},
            {"ticker": "V", "name": "Visa"},
            {"ticker": "MA", "name": "Mastercard"},
        ],
        "Healthcare": [
            {"ticker": "JNJ", "name": "Johnson & Johnson"},
            {"ticker": "UNH", "name": "UnitedHealth"},
            {"ticker": "PFE", "name": "Pfizer"},
        ],
        "Consumer / Industrials": [
            {"ticker": "KO", "name": "Coca-Cola"},
            {"ticker": "PG", "name": "Procter & Gamble"},
            {"ticker": "WMT", "name": "Walmart"},
            {"ticker": "DIS", "name": "Disney"},
            {"ticker": "HD", "name": "Home Depot"},
        ],
        "ETFs": [
            {"ticker": "SOXS", "name": "Semiconductor Bear ETF"},
            {"ticker": "SPY", "name": "S&P 500 ETF"},
            {"ticker": "QQQ", "name": "Nasdaq 100 ETF"},
        ],
    },
    "Kenya": {
        "Telecom": [
            {"ticker": "SCOM", "name": "Safaricom"},
        ],
        "Banking / Finance": [
            {"ticker": "EQTY", "name": "Equity Group"},
            {"ticker": "KCB", "name": "KCB Group"},
            {"ticker": "ABSA", "name": "ABSA Bank Kenya"},
            {"ticker": "COOP", "name": "Co-operative Bank"},
            {"ticker": "SCBK", "name": "Standard Chartered Bank Kenya"},
            {"ticker": "NCBA", "name": "NCBA Group"},
        ],
        "Consumer": [
            {"ticker": "EABL", "name": "East African Breweries"},
            {"ticker": "BAT", "name": "BAT Kenya"},
        ],
        "Industrials": [
            {"ticker": "BAMB", "name": "Bamburi Cement"},
        ],
    },
}


# --- Convenience helper functions ---

def get_countries():
    """Return list of country names. e.g. ['US', 'Kenya']"""
    return list(TICKER_REGISTRY.keys())


def get_sectors(country):
    """Return list of sector names for a country."""
    return list(TICKER_REGISTRY.get(country, {}).keys())


def get_tickers_by_country(country):
    """Return flat list of ticker symbols for a country."""
    tickers = []
    for sector_entries in TICKER_REGISTRY.get(country, {}).values():
        tickers.extend(entry["ticker"] for entry in sector_entries)
    return tickers


def get_tickers_by_sector(country, sector):
    """Return list of ticker symbols for a country+sector."""
    entries = TICKER_REGISTRY.get(country, {}).get(sector, [])
    return [entry["ticker"] for entry in entries]


def get_all_tickers():
    """Return flat list of all ticker symbols across all countries."""
    tickers = []
    for country in TICKER_REGISTRY:
        tickers.extend(get_tickers_by_country(country))
    return tickers


def get_ticker_name(ticker_symbol):
    """Look up human-readable name. Returns the symbol itself if not found."""
    for country_data in TICKER_REGISTRY.values():
        for sector_entries in country_data.values():
            for entry in sector_entries:
                if entry["ticker"] == ticker_symbol:
                    return entry["name"]
    return ticker_symbol


def get_ticker_metadata(ticker_symbol):
    """Return {ticker, name, country, sector} or None if not in registry."""
    for country, sectors in TICKER_REGISTRY.items():
        for sector, entries in sectors.items():
            for entry in entries:
                if entry["ticker"] == ticker_symbol:
                    return {
                        "ticker": entry["ticker"],
                        "name": entry["name"],
                        "country": country,
                        "sector": sector,
                    }
    return None
