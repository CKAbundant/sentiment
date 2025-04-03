"""helper functions for plotting news-related graphs."""

import re
import string
from collections import Counter

import pandas as pd

from src.utils import utils


def gen_date_dist_df(df_news: pd.DataFrame) -> pd.DataFrame:
    """Generate DataFrame containing min and max datetime for each ticker."""

    df = df_news.copy()

    # Set 'pub_date' as datetime object
    df["pub_date"] = pd.to_datetime(df["pub_date"])

    # Group by 'ticker' before getting min and max 'pub_date'
    df_dt = df.groupby("ticker")["pub_date"].agg(["min", "max"]).reset_index()

    return df_dt


def gen_time_dist_df(df_news: pd.DataFrame) -> pd.DataFrame:
    """Generate DataFrame containing min and max time for each ticker."""

    df = df_news.copy()

    # Set 'pub_date' as datetime object
    df["pub_date"] = pd.to_datetime(df["pub_date"])

    # Insert 'pub_time' by extracting time component from 'pub_date'
    df.insert(1, "pub_time", df["pub_date"].dt.time)

    # Group by 'ticker' before getting min and max 'pub_date'
    df_time = df.groupby("ticker")["pub_time"].agg(["min", "max"]).reset_index()

    # Append 'min_minutes' and 'max_minutes' i.e. Amount of time passed in minutes
    # from midnight
    df_time["min_minutes"] = df_time["min"].map(
        lambda time: time.hour * 60 + time.minute
    )
    df_time["max_minutes"] = df_time["max"].map(
        lambda time: time.hour * 60 + time.minute
    )

    return df_time


def append_word_count(df_news: pd.DataFrame) -> pd.DataFrame:
    """Append word count for news title, news content and combined news title
    and content."""

    df = df_news.copy()

    df["title_wc"] = df["title"].map(lambda text: len(text.split()))
    df["content_wc"] = df["content"].map(lambda text: len(text.split()))
    df["combined_wc"] = df["title_wc"] + df["content_wc"]

    return df


def gen_word_counter(text: str) -> Counter:
    """Generate counter for extracted words excluding punctuations at beginning and end of text string."""

    text_list = [re.sub(r"(?<=\w)\W+$", "", word) for word in text.split()]
    text_list = [re.sub(r"^\W+(?=\w)", "", word) for word in text_list]
    text_list = [
        word.lower().strip() for word in text_list if re.search(r"[a-z]", word.lower())
    ]

    return Counter(text_list)


def gen_punct_counter(text: str) -> Counter:
    """Generate counter for punctuations used in text string."""

    punct_list = [
        punct.strip()
        for punct in re.findall(r"\W+", text)
        if punct != " " and punct.strip() in string.punctuation
    ]

    return Counter(punct_list)


def gen_special_counter(text: str) -> Counter:
    """Generate counter for special characters in text string."""

    special_list = [
        special.strip()
        for special in re.findall(r"\W", text)
        if special != " " and special.strip() not in string.punctuation
    ]

    return Counter(special_list)


def get_top_n(df: pd.DataFrame, col: str, top_n: int) -> pd.Series:
    """Return top N items for selected column in DataFrame as Panda Series.

    Args:
        data (pd.DataFrame): DataFrame of concern.
        col (str): Column in DataFrame to perform 'value_count.
        top_n (int): Top N unique item in selected column by frequency.

    Returns:
        (pd.Series): Sorted value count by 'top_n'.
    """

    # Get frequency of unique items and filter top N by frequency
    value_count = df[col].value_counts()

    return value_count.head(top_n)
