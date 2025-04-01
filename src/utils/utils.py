"""Generic helper functions"""

import random
from collections import Counter, defaultdict
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd


def get_current_dt(fmt: str = "%Y%m%d_%H%M", tz: str = "America/New_York") -> str:
    """Return current datetimeas string with 'specific' format."""

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


def gen_snp500_list(url: str, ignore_list: list[str]) -> list[str]:
    """Generate updated list of S&P500 stocks from given url.

    Args:
        url (str):
            URL to download complete list of S&P500 stocks
            (Default: "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies").
        ignore_list (list[str]):
            List of stock tickers, which do not have OHLCV data from yfinance.

    Returns:
        (list[str]): list of S&P 500 stocks.
    """

    # Get DataFrame containing info on S&P500 stocks
    df_info, _ = pd.read_html(url)

    # Remove stocks in 'self.ignore_list' from list of S&P500 stocks
    return [stock for stock in df_info["Symbol"].to_list() if stock not in ignore_list]


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

    if isinstance(header, list):
        # Remove 'Unnamed' from multi-level columns
        return remove_unnamed_cols(df)

    return df


def remove_unnamed_cols(data: pd.DataFrame) -> pd.DataFrame:
    """Set column label containing 'Unnamed:' to empty string for multi-level
    columns DataFrame."""

    df = data.copy()
    formatted_cols = []

    if any([isinstance(col, str) for col in df.columns]):
        # No amendments made since columns are not multi-level
        return df

    for col_tuple in df.columns:
        col_levels = []
        for col in col_tuple:
            if "unnamed:" in col.lower():
                col_levels.append("")
            else:
                col_levels.append(col)
        formatted_cols.append(col_levels)

    df.columns = pd.MultiIndex.from_tuples(formatted_cols)

    return df


def set_decimal_type(data: pd.DataFrame, to_round: bool = False) -> pd.DataFrame:
    """Ensure all numeric types in DataFrame are Decimal type.

    Args:
        DataFrame (pd.DataFrame):
            Both normal and multi-level columns DataFrame.
        to_round (bool):
            Whether to round float to 6 decimal places before converting to
            Decimal (Default: False).

    Returns:
        df (pd.DataFrame): DataFrame containing numbers of Decimal type only.
    """

    df = data.copy()

    # Convert numbers to Decimal type
    for col in df.columns:
        # Check if any item in Panda Series is float type
        if any([isinstance(item, float) for item in df[col].to_list()]):
            df[col] = df[col].map(
                lambda num: (
                    Decimal(str(round(num, 6))) if to_round else Decimal(str(num))
                )
            )

    return df


def set_date_type(data: pd.DataFrame) -> pd.DataFrame:
    """Ensure all datetime objects in DataFrame are set to datetime.date type."""

    df = data.copy()

    # Check if 'date' is found in column for nomral and multi-level DataFrame
    date_cols = [
        col
        for col in df.columns
        if (isinstance(col, str) and "date" in col.lower())
        or (isinstance(col, tuple) and "date" in col[0].lower())
    ]

    # Convert date to datetime.date type
    for col in date_cols:
        df[col] = pd.to_datetime(df[col]).dt.date

    return df


def display_divergent_rating(
    df: pd.DataFrame,
    cols: list[str] = [
        "pub_date",
        "ticker",
        "publisher",
        "period",
        "title",
        "content",
        "prosusai",
        "yiyanghkust",
        "ziweichen",
        "aventiq_ai",
    ],
    models: list[str] = ["prosusai", "yiyanghkust", "ziweichen", "aventiq_ai"],
) -> pd.DataFrame:
    """Display diver

    Args:
        data (pd.DataFrame):
            DataFrame containing sentiment rating of news.
        cols (list[str]):
            List of columns to be displayed (Defaults: ["pub_date", "ticker",
            "publisher", "period", "title", "content", "prosusai", "yiyanghkust",
            "ziweichen", "aventiq_ai"]).
        models (list[str]):
            List of models for comparision (Default: ["prosusai", "yiyanghkust",
            "ziweichen", "aventiq_ai"].

    Returns:
        df_divergent (pd.DataFrame):
            Formatted DataFrame (words wrapped inside cell) to be displayed in Jupyter notebook.
    """

    # Create 'temp' DataFrame to contain only sentiment ratings for selected models
    temp = df.loc[:, models]

    # Append 'combined' column which contains list of sentiment ratings
    temp["combined"] = temp.values.tolist()

    # Filter DataFrame i.e. news with both postive (rating 5) and negative (rating 1)
    cond = temp["combined"].map(lambda x: (5 in x) & (1 in x))
    df_divergent = df.loc[cond, cols]
    print(f"divergent news : {len(df_divergent)}")

    # Ensure words are wrapped for 'title' and 'content' columns
    return df_divergent.style.set_properties(
        subset=["title", "content"], **{"width": "500px", "white-space": "normal"}
    )
