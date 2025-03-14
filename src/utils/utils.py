"""Generic helper functions"""

import random
from datetime import datetime
from pathlib import Path

import pandas as pd


def get_current_dt(fmt: str = "%Y%m%d_%H%M") -> str:
    """Return current datetime as string with 'specific' format."""

    return datetime.now().strftime(fmt)


def save_html(html_content: str, file_name: str, data_dir: str = "./data") -> None:
    """Save HTML content as html file under 'data_dir' folder."""

    file_path = f"{data_dir}/{file_name}"

    with open(file_path, "w") as file:
        file.write(html_content)


def create_folder(data_dir: str | Path) -> None:
    """Create folder if not exist."""

    data_dir = Path(data_dir)

    if not data_dir.is_dir():
        data_dir.mkdir(parents=True, exist_ok=True)


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
