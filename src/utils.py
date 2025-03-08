"""Generic Helper functions"""

import random
from datetime import timedelta
from typing import TYPE_CHECKING

import pandas as pd
import pytimeparse
from scrapling.defaults import Fetcher


def gen_stock_list(
    url: str = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
) -> dict[str, str]:
    """Generate a stock list by randomly select a S&P500 stock from each GICS Sector.

    Args:
        url (str):
            URL to download complete list of S&P500 stocks
            (Default: "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies").

    Returns:
        stock_dict (dict[str, str]):
            Dictionary containing 11 stocks selected from each of 11 GICS Sector.
    """

    # Get DataFrame containing info on S&P500 stocks
    df_info, _ = pd.read_html(url)

    stock_dict = {}
    for sector in df_info["GICS Sector"].unique():
        # Get list of tickers in the same GICS Sector
        sector_list = df_info.loc[df_info["GICS Sector"] == sector, "Symbol"].to_list()

        # Randomly select a single stock from 'sector_list'
        stock_dict[sector] = random.choice(sector_list)

    return stock_dict


def get_gics_sector(
    tickers: list[str],
    url: str = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
) -> dict[str, str]:
    """Get GICS Sector for given list of stock tickers.

    Args:
        tickers (list[str]):
            List of stock tickers.
        url (str):
            URL to download complete list of S&P500 stocks
            (Default: "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies").

    Returns:
        (dict[str, str]): Dictionary mapping stock ticker to its GICS Sector.
    """

    # Get DataFrame containing info on S&P500 stocks
    df_info, _ = pd.read_html(url)

    return {
        ticker: df_info.loc[df_info["Symbol"] == ticker, "GICS Sector"].item()
        for ticker in tickers
    }


def extract_all_text(url: str) -> str:
    """Extract all text content from all HTML tags except script and style
    from given url."""

    # Create an Adaptor instance via GET request to url
    page = Fetcher.get(url)

    # Get all text content from all HTML tags except 'script' and 'style'
    all_text = page.get_all_text(ignore_tags=["script", "style"])

    return str(all_text)


def extract_news_briefs(url: str) -> list[dict[str, str]]:
    """Extract news brief including news heading and publishing info for given url."""

    # Create an Adaptor instance via GET request to url
    page = Fetcher.get(url)

    # Get list of CSS elements containing news brief
    results = page.find_all("div", class_="content yf-82qtw3")

    news_info = []
    for result in results:
        # TextHandlers with 2 elements i.e. news headings and news brief
        news = result.css(".clamp::text")

        # TextHandlers with 2 elements i.e. publisher and published period
        publisher_info = result.css(".publishing::text")
        period_str = str(publisher_info[1].clean())

        # Remove extra whitespace and save publisher, period (how long ago) and news
        # content as dictionary
        news_info.append(
            {
                "publisher": str(publisher_info[0].clean()),
                "period": convert_to_timedelta(period_str),
                "title": news[0].clean(),
                "content": news[1].clean(),
            }
        )

    return news_info


def convert_to_timedelta(period_str: str) -> timedelta | None:
    """Convert time period string into timedelta object."""

    if period_str == "yesterday":
        period_str = "1 day"
    else:
        # Remove 'ago' in 'period_str'
        period_str = period_str.replace("ago", "").strip()

    # Get time period in seconds
    period_in_seconds = pytimeparse.parse(period_str)

    if period_in_seconds is None:
        print(f"Invalid time period string : {period_str}")
        return

    return timedelta(seconds=period_in_seconds)
