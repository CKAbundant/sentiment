"""Generic helper functions"""

import random
from collections import Counter
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd


def get_current_dt(fmt: str = "%Y%m%d_%H%M") -> str:
    """Return current datetime as string with 'specific' format."""

    return datetime.now().strftime(fmt)


def save_html(html_content: str, file_name: str, data_dir: str = "./data/html") -> None:
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


def count_total_words(news_list: list[str]) -> int:
    """Count total number of words in news list"""

    total_count = 0

    for news in news_list:
        # Split each text in 'news_list' into words
        word_list = news.split()

        # Perform word count for each text in 'news_list'
        counter = Counter(word_list)
        word_count = sum(counter.values())

        total_count += word_count

    return total_count


def save_csv(df: pd.DataFrame, file_path: str, save_index: bool = False) -> None:
    """Convert numeric columns to Decimal type before saving DataFrame
    as csv file."""

    # Get numeric columns
    num_cols = df.select_dtypes(include=np.number).columns.to_list()

    # Convert numbers to Decimal type
    for col in num_cols:
        df[col] = df[col].map(lambda num: Decimal(str(num)))

    # Save DataFrame as 'trade_results.csv'
    df.to_csv(file_path, index=save_index)


def load_csv(
    file_path: str,
    header: list[int] | None = "infer",
    index_col: list[int] | None = None,
) -> pd.DataFrame:
    """Load DataFrame and convert numeric columns to Decimal type.

    Args:
        file_path (str):
            Relative patht to csv file.
        header (list[int] | str = "infer"):
            If provided, list of row numbers containing column labels
            (Default: "infer").
        index_col (list[int] | None = None):
            If provided, list of columns to use as row labels.

    Returns:
        df (pd.DataFrame): Loaded DataFrame (including multi-level).
    """

    # Load DataFrame from 'trade_results.csv'
    df = pd.read_csv(file_path, header=header, index_col=index_col)

    # Ensure all numbers are set to Decimal type and all dates are set
    # to datetime.date type
    df = set_decimal_type(df)
    df = set_date_type(df)

    return df


def set_decimal_type(data: pd.DataFrame) -> pd.DataFrame:
    """Ensure all numeric types in DataFrame are Decimal type."""

    df = data.copy()
    num_cols = df.select_dtypes(include=np.number).columns.to_list()

    # Convert numbers to Decimal type
    for col in num_cols:
        df[col] = df[col].map(lambda num: Decimal(str(num)))

    return df


def set_date_type(data: pd.DataFrame) -> pd.DataFrame:
    """Ensure all datetime objects in DataFrame are set to datetime.date type."""

    df = data.copy()
    date_cols = [col for col in df.columns if "date" in col.lower()]

    # Convert date to datetime.date type
    for col in date_cols:
        df[col] = pd.to_datetime(df[col]).dt.date

    return df
