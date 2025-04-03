"""Class to analyze strategies of different combinations of:

- FinBert sentiment classifier (ProsusAI, yiyangkhust, ziweichen, AventiqAI).
- Cointegration/correlation (Engle-Granger, Pearson, Spearman, Kendall).
- Periods (1, 3, 5 years).
"""

from collections import Counter
from decimal import Decimal
from itertools import product
from pathlib import Path
from typing import get_args

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config.variables import COINT_CORR_FN, HF_MODEL
from src.utils import plot_utils, utils


class PlotStrategies:
    """Plot graphs to analyze performance for all combinations of FinBERT, cointegration/correlation,
    and periods.

    - Overall performance evaluation
    - Performance evaluation for different FinBERT
    - Performance evaluation for different cointegration/correlation
    - Performance evaluation for different time period

    Usage:
        >>> plot_strategies = PlotStrategies()
        >>> plot_strategies.run()

    Args:
        date (str):
            If provided, date when news are scraped.
        periods (list[iint]):
            Number of past years records for cointegration/correlation computation
            (Default: [1, 2, 3]).
        min_trading_period (int):
            Minimum trading period to be considered for top N computation (Default: 2).
        strategy_subcols (list[str])
            List of column names representing components of strategy
            (Default: ["hf_model", "coint_corr_fn", "period"].
        analysis_cols (list[str]):
            List of columns required for analysis
            (Default: ["annualized_return", "win_rate", "mean_days_held", "trading_period"]).
        top_n (int):
            Top N ticker pair with highest daily return (Default: 15).
        results_dir (str):
            Relative path of folder containing news for all dates
            (Default: "./data/results).
        coint_corr_dir (str):
            Relative path of folder containing cointegration and correlation info
            for all dates (Default: "./data/coint_corr").
        graph_dir (str):
            Relative path of folder containing graphs for all dates
            (Default: "./data/graph").

    Attributes:
        date (str):
            If provided, date when news are scraped.
        periods (list[iint]):
            Number of past years records for cointegration/correlation computation
            (Default: [1, 2, 3]).
        min_trading_period (int):
            Minimum trading period to be considered for top N computation (Default: 2).
        strategy_subcols (list[str])
            List of column names representing components of strategy
            (Default: ["hf_model", "coint_corr_fn", "period"].
        analysis_cols (list[str]):
            List of columns required for analysis
            (Default: ["annualized_return", "win_rate", "mean_days_held", "trading_period"]).
        top_n (int):
            Top N ticker pair with highest daily return (Default: 15).
        results_date_dir (str):
            Relative path to folder containing all strategies for specific date
            (Default: "./data/results").
        coint_corr_date_dir (str):
            Relative path to folder containing cointegration and correlation
            info for specific date.
        graph_date_dir (str):
            Relative path to folder to save generated graphs.
    """

    def __init__(
        self,
        date: str | None = None,
        periods: list[int] = [1, 3, 5],
        min_trading_period: int = 2,
        strategy_subcols: list[str] = ["hf_model", "coint_corr_fn", "period"],
        analysis_cols: list[str] = [
            "annualized_return",
            "win_rate",
            "mean_days_held",
            "trading_period",
        ],
        top_n: int = 15,
        results_dir: str = "./data/results/",
        coint_corr_dir: str = "./data/coint_corr",
        graph_dir: str = "./data/graph",
    ) -> None:
        self.date = date or utils.get_current_dt(fmt="%Y-%m-%d")
        self.periods = periods
        self.min_trading_period = min_trading_period
        self.strategy_subcols = strategy_subcols
        self.analysis_cols = analysis_cols
        self.top_n = top_n
        self.results_date_dir = f"{results_dir}/{self.date}"
        self.coint_corr_date_dir = f"{coint_corr_dir}/{self.date}"
        self.graph_date_dir = f"{graph_dir}/{self.date}"

    def run(self) -> None:
        # Combine 'overall_summary.csv' info for all combinations
        df_combined = self.combine_overall()

        # Generate DataFrame containing top 15 ticker pairs with highest
        # positive annualized return
        df_top_n, df_pivot_top_n = self.gen_top_n_pairs()

        # Plot histograms of annualized returns for all combinations, all FinBERT,
        # all cointegration/correlation and all time period
        # self.plot_all(df_combined)
        # self.plot_drill_down(df_combined, "hf_model")
        # self.plot_drill_down(df_combined, "coint_corr_fn")
        # self.plot_drill_down(df_combined, "period")

        # # Plot bar chart of common occurring ticker pairs among top_N pairs
        # self.plot_top_n(df_top_n)

        # # Plot bar chart of common occuring ticker pairs in increasing top N
        # # i.e. start from top 1 till top N
        # self.plot_successive_top_n(df_pivot_top_n)

        return df_combined, df_top_n, df_pivot_top_n

    def plot_all(self, df_combined: pd.DataFrame) -> None:
        """Plot histogram of annualized returns, mean days held, trading period,
        win_rate for all strategies i.e. overall performance."""

        df = df_combined.copy()

        _, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 15))

        for ax, col in zip(axes.flat, self.analysis_cols):
            sns.histplot(data=df, x=col, ax=ax, kde=True, bins=10)

            # Format text for graphical representation
            col_msg = " ".join([col.title() for col in col.split("_")])

            ax.set_title(f"Frequency Distribution of {col_msg}")
            ax.set_xlabel(col_msg)
            ax.set_ylabel("Frequency")

        # Create folder if not exist
        utils.create_folder(self.graph_date_dir)

        plt.tight_layout()
        plt.savefig(f"{self.graph_date_dir}/combined_overall.png")

    def plot_drill_down(self, df_combined: pd.DataFrame, drill_down: str) -> None:
        """Plot histogram of annualized returns, mean days held, trading period,
        win_rate for all strategies for various FinBert models.

        Args:
            df_combined (pd.DataFrame):
                DataFrame containing combined 'overall_summary.csv' info for
                specific date.
            drill_down (str):
                Drill down studies by either "hf_model", "coint_corr_fn", and "period".
        """

        if drill_down not in df_combined.columns:
            raise ValueError(
                f"'{drill_down}' is not a valid column name in combined DataFrame."
            )

        df = df_combined.copy()

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 15))

        for ax, col in zip(axes.flat, self.analysis_cols):
            sns.histplot(
                data=df, x=col, hue=drill_down, ax=ax, kde=True, legend=True, alpha=0.5
            )

            # Format text for graphical representation
            col_msg = " ".join([col.title() for col in col.split("_")])
            drill_down_mapping = {
                "hf_model": "HuggingFace FinBERT Sentiment Model",
                "coint_corr_fn": "Cointegration / Correlation Function",
                "period": "Time Period",
            }

            ax.set_title(
                f"Frequency Distribution of {col_msg} by {drill_down_mapping[drill_down]}"
            )
            ax.set_xlabel(col_msg)
            ax.set_ylabel("Frequency")

        # Create folder if not exist
        utils.create_folder(self.graph_date_dir)

        plt.tight_layout()
        plt.savefig(f"{self.graph_date_dir}/combined_{drill_down}.png")

    def plot_top_n(
        self,
        df_top_n: pd.DataFrame,
        col: str = "ticker_pair",
        top_n: int | None = None,
    ) -> None:
        """Generate bar chart of common occurring
        ticker pairs among top_N pairs"""

        df = df_top_n.copy()
        top_n = top_n or self.top_n

        # Get top N 'ticker_pair' with highest count
        top_n_pairs = plot_utils.get_top_n(df, col, top_n)

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))

        # Plot bar chart of top N ticker_pair
        sns.barplot(x=top_n_pairs.index, y=top_n_pairs, ax=ax)

        ax.set_title(f"Top {top_n} Ticker Pair by Frequency")
        ax.set_xlabel("Ticker Pair")
        ax.set_ylabel("Counts")

        # Create folder if not exist
        utils.create_folder(self.graph_date_dir)

        fig.autofmt_xdate()
        plt.tight_layout()
        plt.savefig(f"{self.graph_date_dir}/top_n_pairs.png")

    def plot_successive_top_n(
        self,
        df_pivot_top_n: pd.DataFrame,
        top_n: int | None = None,
        display_num: int = 10,
    ) -> None:
        """Start with top 1 ticker pair to plot bar chart; Proceed until top N reached."""

        # Top N is limited by number of columns in 'df_pivot_top_n'
        top_n = top_n or self.top_n
        top_n = min(10, len(df_pivot_top_n.columns))

        fig, axes = plt.subplots(nrows=round(top_n / 2), ncols=2, figsize=(20, 20))
        for ax, idx in zip(axes.flat, range(top_n)):
            df_pair_counts = self.gen_pair_counts_df(df_pivot_top_n, idx, display_num)

            sns.barplot(x=df_pair_counts.index, y=df_pair_counts["count"], ax=ax)

            ax.set_title(
                f"Top {display_num} Common Ticker-Pairs Extracted from Top {top_n} "
                "Ticker-Pairs with highest Daily Return."
            )
            ax.set_xlabel("Ticker-Pair")
            ax.set_ylabel("Counts")

        # Create folder if not exist
        utils.create_folder(self.graph_date_dir)

        fig.autofmt_xdate()
        plt.tight_layout()
        plt.savefig(f"{self.graph_date_dir}/successive_top_n_pairs.png")

    def gen_pair_counts_df(
        self, df_pivot_top_n: pd.DataFrame, top_n: int, display_num: int = 10
    ) -> pd.DataFrame:
        """Generate DataFrame containing frequency count of ticker pairs found in
        top N ticker.

        Args:
            df_pivot (pd.DataFrame):
                Pivoted DataFrame where columns are top N tickers.
            top_n (int):
                Top N ticker pairs with highest daily return.
            display_n (int):
                Number of ticker pairs to display in each bar plot.

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

    def combine_overall(self) -> pd.DataFrame:
        """Combine DataFrame generated from 'overall_summary.csv' for all
        combinations."""

        # Get FinBERT models, cointegration/correlation functions and time periods
        hf_models = get_args(HF_MODEL)
        coint_corr_fns = get_args(COINT_CORR_FN)

        df_list = []

        for hf_model, coint_corr_fn, period in product(
            hf_models, coint_corr_fns, self.periods
        ):
            strategy_dir = (
                f"{self.results_date_dir}/{hf_model}_{coint_corr_fn}_{period}"
            )
            overall_path = f"{strategy_dir}/overall_summary.csv"

            # Append formatted overall summary DataFrame to list
            df_list.append(self.format_overall_df(overall_path))

        # Concatenate row-wise and save DataFrame
        df_combined = pd.concat(df_list, axis=0).reset_index(drop=True)
        utils.save_csv(df_combined, f"{self.results_date_dir}/combined_overall.csv")

        return df_combined

    def gen_top_n_pairs(
        self, top_n: int | None = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Generate DataFrame where columns represent top N ticker pairs with highest
        annualized return.

        Args:
            top_n (int | None):
                If provided, top N number of ticker pairs with highest daily return.

        Returns:
            df (pd.DataFrame):
                DataFrame containing top N ticker pairs with highest daily return for
                each strategy.
            df_pivot (pd.DataFrame):
                Pivoted DataFrame where columns are top N tickers.
        """

        top_n = top_n or self.top_n

        # Get FinBERT models, cointegration/correlation functions and time periods
        hf_models = get_args(HF_MODEL)
        coint_corr_fns = get_args(COINT_CORR_FN)

        df_list, df_pivot_list = [], []

        for hf_model, coint_corr_fn, period in product(
            hf_models, coint_corr_fns, self.periods
        ):
            strategy_dir = (
                f"{self.results_date_dir}/{hf_model}_{coint_corr_fn}_{period}"
            )
            breakdown_path = f"{strategy_dir}/breakdown_summary.csv"

            # Extract top N ticker pairs with highest daily return
            df, df_pivot = self.format_breakdown_df(breakdown_path, top_n)

            if df.empty or df_pivot.empty:
                continue
            df_list.append(df)
            df_pivot_list.append(df_pivot)

        # Concantate row-wise' and reset index
        df_top_n = pd.concat(df_list, axis=0).reset_index(drop=True)
        utils.save_csv(df_top_n, f"{self.results_date_dir}/top_{top_n}_tickers.csv")

        # Generate pivot table where columns are top N tickers
        df_pivot_top_n = pd.concat(df_pivot_list, axis=0).reset_index()
        utils.save_csv(
            df_pivot_top_n, f"{self.results_date_dir}/pivot_top_{top_n}_tickers.csv"
        )

        return df_top_n, df_pivot_top_n

    def format_breakdown_df(
        self, breakdown_path: str, top_n: int | None = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Format DataFrame loaded from 'breakdown_summary.csv' to extract top N
        ticker pairs with highest daily return.

        Args:
            breakdown_path (str):
                Relative path to 'breakdown_summary.csv' for specific model
                at specific date.
            top_n (int | None):
                If provided, top N number of ticker pairs with highest daily return.

        Returns:
            df (pd.DataFrame):
                DataFrame containing top N ticker pairs with highest daily return for
                each strategy.
            df_pivot (pd.DataFrame):
                Pivoted DataFrame where columns are top N tickers.
        """

        top_n = top_n or self.top_n

        # Load DataFrame from 'breakdown_summary.csv'
        df_breakdown = utils.load_csv(breakdown_path, header=[0, 1], index_col=[0, 1])

        # Filter only trading periods more than 'self.min_trading_period' days;
        # sort 'overall_daily_ret' by descending order
        df = df_breakdown.loc[
            df_breakdown["trading_period"] >= self.min_trading_period,
            ["overall_daily_ret", "trading_period"],
        ]
        df = df.sort_values(by="overall_daily_ret", ascending=False).reset_index()

        # Insert 'strategy' (<hf_model>_<coint_corr_fn>_<period>) column
        strategy_name = Path(breakdown_path).parent.name
        df.insert(0, "strategy", strategy_name)

        # Insert 'ticker_pair' (<ticker>_<coint_corr_ticker>)
        df.insert(1, "ticker_pair", df["ticker"] + "_" + df["coint_ticker"])

        # Get top N ticker pairs
        df = df.head(top_n).reset_index(drop=True)

        # Pivot DataFrame by 'strategy' index and "ticker_pair" values
        df_pivot = df.pivot(
            index="strategy", columns="ticker_pair", values="ticker_pair"
        )
        df_pivot.columns = [f"top_{idx+1}" for idx in range(top_n)]

        return df, df_pivot

    def format_overall_df(self, overall_path: str) -> pd.DataFrame:
        """Format DataFrame loaded from 'overall_summary.csv' such that indexes are
        columns with 'strategy' column present."""

        # Load DataFrame from 'overall_summary.csv'
        df_overall = utils.load_csv(overall_path, index_col=0)

        # Ensure valid strings are converted to Decimal type
        df_overall = self.convert_to_valid_type(df_overall)

        # Transpose DataFrame, reset index, and remove 'index' column
        df_overall = df_overall.T.reset_index()
        df_overall = df_overall.drop(columns=["index"])

        # Insert 'strategy' column if not present
        if "strategy" not in df_overall.columns:
            df_overall.insert(0, "strategy", Path(overall_path).parent.name)

        # Ensure 'total_num_wins' and 'total_num_trades' are of integer type.
        # Append 'win_rate' column
        df_overall["win_rate"] = (
            df_overall["total_num_wins"] / df_overall["total_num_trades"]
        )
        df_overall["win_rate"] = df_overall["win_rate"].map(
            lambda val: val.quantize(Decimal("1.000000"))
        )

        # Split 'strategy' columns into its component i.e. 'hf_model',
        # 'coint_corr_fn' and 'period'
        df_overall = self.reorder_cols(df_overall)

        return df_overall

    def reorder_cols(self, df_overall: pd.DataFrame) -> pd.DataFrame:
        """Split 'strategy' columns into its component i.e. 'hf_model',
        'coint_corr_fn' and 'period'."""

        df = df_overall.copy()

        # Split 'strategy' column into 'hf_model', 'coint_corr_fn' and period
        df[self.strategy_subcols] = df["strategy"].str.rsplit("_", n=2, expand=True)
        df["period"] = df["period"].map(Decimal)

        no_strategy_cols = [
            col
            for col in df_overall.columns
            if col not in self.strategy_subcols + ["strategy"]
        ]
        new_cols = ["strategy"] + self.strategy_subcols + no_strategy_cols

        return df.loc[:, new_cols]

    def convert_to_valid_type(self, df_overall: pd.DataFrame) -> pd.DataFrame:
        """Ensure numeric data is set to Decimal type and rest to string.

        - Note that all data is set to string type (Needs further investigation).
        """

        df = df_overall.copy()

        non_num_indices = [
            "strategy",
            "stock_ticker_without_trades",
            "stock_ticker_with_trades",
            "first_trade",
            "last_trade",
        ]

        # Convert valid strings to Decimal type
        for idx in df.index:
            if idx not in non_num_indices:
                df.at[idx, "Overall Statistics"] = Decimal(
                    df.at[idx, "Overall Statistics"]
                )

        return df
