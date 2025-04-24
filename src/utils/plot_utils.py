"""helper functions for plotting news-related graphs."""

import re
import string
from collections import Counter, defaultdict
from decimal import Decimal
from itertools import product
from pathlib import Path
from typing import Any, Literal, get_args, get_origin

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from omegaconf import DictConfig, ListConfig

from config.variables import StratComponent
from src.utils import utils


def set_palette(
    df_combined: pd.DataFrame,
    strat_comp: StratComponent,
    palette: str = "dark",
) -> dict[str, str]:
    """Set color palette for seaborn histogram plot.

    Args:
        df_combined (pd.DataFrame):
            DataFrame containing overall summary for all strategy combinations.
        strat_comp (StratComponent):
            Component making up trading strategy.
        palette (str):
            Seaborn color palette (Default: "dark").

    Returns:
        (dict[str, str]):
            Dictionary mapping categories to unique color.
    """

    # Get all categories for selected strat component
    categories = df_combined[strat_comp].unique()

    # Get colors from palette based on number of categories
    palette = sns.color_palette(palette, len(categories))

    return dict(zip(categories, palette))


def hide_unused_ax(ax_list: list[Axes], item_list: list[str]) -> None:
    """Hide unused axes when 'ax_list' contains more items than 'item_list'."""

    if (diff := len(ax_list) - len(item_list)) > 0:
        # Get unused axes i.e. starting from last item in list
        for idx in range(diff):
            ax_list[-idx - 1].set_visible(False)


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


def format_breakdown_df(
    breakdown_path: str,
    strat_comp_list: list[str],
    min_trading_period: int = 2,
    top_n: int = 2,
    strat_comp_cols: list[StratComponent] = [
        "entry_type",
        "entry_struct",
        "exit_struct",
        "stop_method",
        "hf_model",
        "coint_corr_fn",
        "period",
    ],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Format DataFrame loaded from 'breakdown_summary.csv' to extract top N
    ticker pairs with highest daily return.

    Args:
        breakdown_path (str):
            Relative path to 'breakdown_summary.csv' for specific model
            at specific date.
        strat_comp_list (list[str]):
            List of components making up the strategy i.e. entry type, entry
            structure, exit structure, stop method, hf model, cointegration/correlation
            function and period.
        min_trading_period (int):
            Minimum number of trading period to be considered for top N computation (Default: 2).
        top_n (int):
            Top N ticker pair with highest daily return (Default: 15).
        strat_comp_cols (list[StratComponent]]):
            List of strat component names (Default: ["entry_type", "entry_struct",
            "exit_struct", "stop_method", "hf_model", "coint_corr_fn", "period"].

    Returns:
        df (pd.DataFrame):
            DataFrame containing top N ticker pairs with highest daily return for
            each strategy.
        df_pivot (pd.DataFrame):
            Pivoted DataFrame where columns are top N tickers.
    """

    # Load DataFrame from 'breakdown_summary.csv'
    df_breakdown = utils.load_csv(breakdown_path, tz="America/New_York")

    # Filter only trading periods more than 'min_trading_period' days;
    # sort 'overall_daily_ret' by descending order
    req_cols = [
        "news_ticker",
        "ticker",
        "overall_daily_ret",
        "days_held_max",
        "trading_period",
        "win_rate",
        "win_count",
        "neg_ret_mean",
        "neg_ret_max",
    ]
    df = df_breakdown.loc[
        df_breakdown["trading_period"] >= min_trading_period, req_cols
    ]
    df = df.sort_values(by="overall_daily_ret", ascending=False).reset_index(drop=True)

    # Get top N ticker pairs
    df = df.head(top_n)

    # Insert 'ticker_pair', strategy component columns
    df.insert(0, "ticker_pair", df["news_ticker"] + "_" + df["ticker"])
    df = insert_strat_cols(df, strat_comp_list, strat_comp_cols)

    return df


def format_breakdown_pivot_df(
    df_pre_pivot: pd.DataFrame, top_n: int = 15
) -> pd.DataFrame:
    """Convert filtered breakdown DataFrame ('df_pre_pivot') into pivot format.

    Args:
        df_pre_pivot (pd.DataFrame):
            Filtered DataFrame containing breakdown summary for ticker pairs.
        top_n (int):
            Top N ticker pair with highest daily return (Default: 15).

    Returns:
        (pd.DataFrame):
            Pivoted DataFrame where index is strategy and columns are top N ticker
            pairs.
    """

    df = df_pre_pivot.copy()

    # Pivot DataFrame by 'strategy' index and "ticker_pair" values
    df_pivot = df.pivot(index="strategy", columns="ticker_pair", values="ticker_pair")
    df_pivot.columns = [f"top_{idx+1}" for idx in range(top_n)]

    return df_pivot


def get_top_pairs(
    top_n_path: str,
    top_n: int = 15,
    disp_cols: list[str] = [
        "ticker_pair",
        "overall_daily_ret",
        "trading_period",
        "win_rate",
        "win_count",
        "neg_ret_mean",
        "neg_ret_max",
    ],
) -> pd.DataFrame:
    """Get top 'top_n' ticker pairs for all strategy components.

    Args:
        top_n_path (str):
            Relative path to csv file containing top N tickers with highest daily returns.
        top_n (int):
            Top N ticker pair with highest daily return (Default: 15).
        disp_cols (list[str]):
            Subset of columns for display in console (Default: ["ticker_pair",
            "overall_daily_ret", "trading_period", "win_rate", "win_count",
            "neg_ret_mean", "neg_ret_max",]).

    Returns:
        (pd.DataFrame):
            DataFrame containing top N ticker pairs with highest daily returns.
    """

    top_n_path = Path(top_n_path)

    if not top_n_path.is_file():
        raise FileNotFoundError(
            f"{top_n_path.name} is not present in '{top_n_path.as_posix()}'."
        )

    df = utils.load_csv(top_n_path)
    df = df.loc[:, disp_cols]
    df = df.sort_values(by="overall_daily_ret", ascending=False)
    df = df.drop_duplicates(subset="ticker_pair")

    return df.head(top_n).reset_index(drop=True)


def get_top_pairs_by_comp(
    strat_comp: StratComponent,
    top_n_path: str,
    top_n: int = 15,
    disp_cols: list[str] = [
        "ticker_pair",
        "overall_daily_ret",
        "trading_period",
        "win_rate",
        "win_count",
        "neg_ret_mean",
        "neg_ret_max",
    ],
) -> dict[str, pd.DataFrame]:
    """Get top ticker pairs for each strategy component e.g. 'hf_model'.

    Args:
        strat_comp (StratComponent):
            Components making up trading strategy e.g. 'entry_struct'.
        top_n_path (str):
            Relative path to csv file containing top N tickers with highest daily returns.
        top_n (int):
            Top N ticker pair with highest daily return (Default: 15).
        disp_cols (list[str]):
            Subset of columns for display in console (Default: ["ticker_pair",
            "overall_daily_ret", "trading_period", "win_rate", "win_count",
            "neg_ret_mean", "neg_ret_max",]).

    Returns:
        (dict[str, pd.DataFrame]):
            Dictionary mapping strategy component to top N ticker pairs.
    """

    # load 'top_15_tickers.csv' if available
    top_n_path = Path(top_n_path)

    if not top_n_path.is_file():
        raise FileNotFoundError(
            f"{top_n_path.name} is not present in '{top_n_path.as_posix()}'."
        )

    df = utils.load_csv(top_n_path)

    # Filter unique category for selected strat component
    pairs_dict = {}
    for comp in df[strat_comp].unique():
        df_filter = df.loc[df[strat_comp] == comp, disp_cols]
        df_filter = df_filter.sort_values(by="overall_daily_ret", ascending=False)
        df_filter = df_filter.drop_duplicates(subset="ticker_pair")
        pairs_dict[comp] = df_filter.head(top_n).reset_index(drop=True)

    return pairs_dict


def insert_strat_cols(
    df_filter: pd.DataFrame,
    strat_comp_list: list[str],
    strat_comp_cols: list[StratComponent] = [
        "entry_type",
        "entry_struct",
        "exit_struct",
        "stop_method",
        "hf_model",
        "coint_corr_fn",
        "period",
    ],
) -> pd.DataFrame:
    """Insert the individual components making up the trading strategy as column and
    insert name of strategy by appending all strat components by "_".

    Args:
        df_filter (pd.DataFrame):
            Filtered breakdown DataFrame.
        strat_comp_list (list[str]):
            List of components making up the strategy.
        strat_comp_cols (list[StratComponent]):
            List of strat component names (Default: ["entry_type", "entry_struct",
            "exit_struct", "stop_method", "hf_model", "coint_corr_fn", "period"].


    Returns:
        df (pd.DataFrame): Filtered DataFrame appended with strat component columns.
    """

    # Create DataFrame from 'strat_comp_list'
    df_comp = pd.DataFrame([strat_comp_list] * len(df_filter), columns=strat_comp_cols)

    # Concatenate 'df_comp' and 'df_filter' column-wise
    df = pd.concat([df_comp, df_filter], axis=1).reset_index(drop=True)

    # Insert strategy column
    df.insert(
        0, "strategy", "_".join([str(strat_comp) for strat_comp in strat_comp_list])
    )

    return df


def gen_stats_comp_dict(
    combined_path: str,
    strat_comp_cols: list[str] = [
        "entry_type",
        "entry_struct",
        "exit_struct",
        "stop_method",
        "hf_model",
        "coint_corr_fn",
        "period",
    ],
    analysis_list: list[str] = [
        "annualized_return",
        "win_rate",
        "max_days_held",
        "highest_neg_percent_return",
    ],
) -> defaultdict[str, dict[str, pd.DataFrame]]:
    """Generate min, max, mean, median of items in 'analysis_list' for all strat
    components.

    Args:
        combined_path (str):
            Relative path to 'combined_overall.csv' for specific date.
        strat_comp_cols (list[str]):
            List of component for strategies
            (Default: ["entry_type", "entry_struct", "exit_struct",
            "stop_method", "hf_model", "coint_corr_fn", "period"]).
        analysis_list: (list[str]):
            List of columns for analysis (Default: ["annualized_return",
            "win_rate", "max_days_held", "highest_neg_percent_return"])

    Returns:
        df_dict (defaultdict[str, dict[str, pd.DataFrame]]):
            Dictionary mapping strat component to statistics of selected
            attributes.
    """

    # Load csv file if exist
    combined_path = Path(combined_path)
    date_dir = combined_path.parent

    if not combined_path.is_file():
        raise FileNotFoundError(
            f"'combined_overall.csv' is not found at '{combined_path}'."
        )

    # Filter columns to 'analysis_list'
    df = utils.load_csv(combined_path, tz="America/New_York")
    df_dict = defaultdict(dict)

    for strat_comp in strat_comp_cols:
        for attribute in analysis_list:
            # Group by strat component and determine min, max, mean, median of
            # selected attributes
            df_comp = df.groupby(by=[strat_comp]).agg(
                {attribute: ["min", "max", "mean", "median"]}
            )

            # Flatten multi-level columns by keeping the 2nd level
            df_comp.columns = [col_tuple[1] for col_tuple in df_comp.columns]

            # Convert all values to Decimal type before updating 'df_dict'
            # because mean and median generates float instead of Decimal object
            df_comp = df_comp.map(
                lambda val: Decimal(str(val)).quantize(Decimal("1.000000"))
            )
            df_dict[strat_comp][attribute] = df_comp

    # Save dictionary as pickle object
    utils.save_pickle(df_dict, date_dir.joinpath("strat_comp_dict.pkl"))

    return df_dict


def format_overall_df(overall_path: str) -> pd.DataFrame:
    """Format DataFrame loaded from 'overall_summary.csv' such that indexes are
    columns with 'strategy' column present."""

    # Load DataFrame from 'overall_summary.csv'
    df_overall = utils.load_csv(overall_path, index_col=0)

    # Ensure valid strings are converted to Decimal type
    df_overall = convert_to_valid_type(df_overall)

    # Transpose DataFrame, reset index, and remove 'index' column
    df_overall = df_overall.T.reset_index()
    df_overall = df_overall.drop(columns=["index"])

    return df_overall


def convert_to_valid_type(df_overall: pd.DataFrame) -> pd.DataFrame:
    """Ensure numeric data is set to Decimal type and rest to string."""

    df = df_overall.copy()

    # Convert numeric types from string to Decimal
    df["Overall Statistics"] = df["Overall Statistics"].map(
        lambda text: (
            Decimal(text)
            if isinstance(text, str) and re.search(r"^-$\d+\.\d+$", text)
            else text
        )
    )

    return df


def gen_pair_counts_df(
    df_pivot_top_n: pd.DataFrame, top_n: int, display_num: int = 10
) -> pd.DataFrame:
    """Generate DataFrame containing frequency count of ticker pairs found in
    top N ticker.

    Args:
        df_pivot (pd.DataFrame):
            Pivoted DataFrame where columns are top N tickers.
        top_n (int):
            Top N ticker pairs with highest daily returns.
        display_n (int):
            Number of ticker pairs to display in each bar plot (Default: 10).

    Returns:
        df_pair_counts (pd.DataFrame):
            DataFrame containing frequency count of ticker pairs found in
            top N ticker.
    """

    df = df_pivot_top_n.copy()

    # Extract DataFrame containing top N tickers columns from 'df_pivot_top_n'
    df_top_n = df.loc[:, df.columns[:top_n]]

    # Extract ticker pairs into 1D numpy array
    ticker_pairs = np.ravel(df_top_n.to_numpy())

    # Apply Counter to 'ticker_pairs' numpy array and convert to DataFrame
    pairs_counter = Counter(ticker_pairs)
    df_pair_counts = pd.DataFrame.from_dict(
        pairs_counter, orient="index", columns=["count"]
    )

    # Sort count by descending order
    df_pair_counts = df_pair_counts.sort_values(by="count", ascending=False)

    return df_pair_counts.head(display_num)


def get_combi_list(full: ListConfig) -> list[list[str]]:
    """Get nested list of all possible combinations of strategy components.

    Args:
        full (ListConfig):
            OmegaConf ListConfig containing list of combinations for long,
            short and long-short strategies.

    Returns:
        (list[list[str]]):
            Nested list of all possible combinations of strategy components.
    """

    # Get list of combinations for long, short and long-short strategies
    combi_list = [list(product(*strat)) for strat in full]

    return [combi for sub_list in combi_list for combi in sub_list]


def validate_mapping(
    mapping: DictConfig | dict[str, str], literal: Any = StratComponent
) -> DictConfig:
    """Validate whether keys in drilldown_mapping is the same as 'StratComponent' literal."""

    if get_origin(literal) is not Literal:
        raise ValueError(f"'literal' variable is not Literal type.")

    assert sorted(list(get_args(literal))) == sorted(list(mapping.keys()))

    return mapping


def get_top_n(df: pd.DataFrame, col: str, top_n: int = 10) -> pd.Series:
    """Get top frequently occuring categories for selected col."""

    value_count = df[col].value_counts()

    return value_count.head(top_n)
