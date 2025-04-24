"""Class to analyze strategies of different combinations of:

- FinBert sentiment classifier (ProsusAI, yiyangkhust, ziweichen, AventiqAI).
- Cointegration/correlation (Engle-Granger, Pearson, Spearman, Kendall).
- Periods (1, 3, 5 years).
"""

import re
from collections import Counter, defaultdict
from decimal import Decimal
from itertools import product
from pathlib import Path
from typing import get_args

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from omegaconf import DictConfig, ListConfig
from tqdm import tqdm

from config.variables import StratComponent
from src.utils import plot_utils, utils

# Set default fontsize for labels and ticks
plt.rcParams["font.size"] = 14
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14

import warnings

# Suppress missing glyph user warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
)


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
        path (DictConfig):
            OmegaConf DictConfig containing required file and directory paths.
        full (ListConfig):
            OmegaConf ListConfig object containing parameters for running all
            strategies.
        drilldown (DictConfig):
            OmegaConf DictConfig mapping strategy component to title representation.
        date (str):
            If provided, date when news are scraped.
        periods (list[iint]):
            Number of past years records for cointegration/correlation computation
            (Default: [1, 2, 3]).
        min_trading_period (int):
            Minimum number of trading period to be considered for top N computation (Default: 2).
        analysis_cols (list[str]):
            List of columns required for analysis
            (Default: ["annualized_return", "win_rate", "max_days_held",
            "neg_ret_max"]).
        top_n (int):
            Top N ticker pair with highest daily return (Default: 15).
        top_n_common (int):
            Top N most common ticker pair (Default: 10).
        palette (str):
            Seaborn color palette used for overlapping histogram plot (Default: "bright").
        disp_cols (list[str]):
            Subset of columns for display in console (Default: ["ticker_pair",
            "overall_daily_ret", "trading_period", "win_rate", "win_count",
            "neg_ret_mean", "neg_ret_max",]).

    Attributes:
        full (ListConfig):
            OmegaConf ListConfig object containing parameters for running all
            strategies.
        drilldown (DictConfig):
            OmegaConf DictConfig mapping strategy component to title representation.
        date (str):
            If provided, date when news are scraped.
        periods (list[iint]):
            Number of past years records for cointegration/correlation computation
            (Default: [1, 2, 3]).
        min_trading_period (int):
            Minimum number of trading period to be considered for top N computation (Default: 2).
        analysis_cols (list[str]):
            List of columns required for analysis
            (Default: ["annualized_return", "win_rate", "max_days_held",
            "neg_ret_max"]).
        top_n (int):
            Top N ticker pair with highest daily return (Default: 15).
        top_n_common (int):
            Top N most common ticker pair (Default: 10).
        palette (str):
            Seaborn color palette used for overlapping histogram plot (Default: "bright").
        disp_cols (list[str]):
            Subset of columns for display in console (Default: ["ticker_pair",
            "overall_daily_ret", "trading_period", "win_rate", "win_count",
            "neg_ret_mean", "neg_ret_max",]).
        date_dir (str):
            Relative path to folder for news generated on specific date.
        graph_date_dir (str):
            Relative path to folder to save generated graphs.
        combined_path (str):
            Relative path to csv file combining trade results data for all strategies.
        top_n_path (str):
            Relative path to csv file containing top N tickers with highest daily returns.
        pivot_top_n_path (str):
            Relative path to csv file containing top N tickers with highest daily
            returns for each strategy combination.
    """

    def __init__(
        self,
        path: DictConfig,
        full: ListConfig,
        drilldown_mapping: DictConfig,
        date: str | None = None,
        periods: list[int] = [1, 3, 5],
        min_trading_period: int = 2,
        analysis_cols: list[str] = [
            "annualized_return",
            "win_rate",
            "max_days_held",
            "highest_neg_percent_return",
        ],
        top_n: int = 15,
        top_n_common: int = 10,
        palette: str = "bright",
        disp_cols: list[str] = [
            "ticker_pair",
            "overall_daily_ret",
            "trading_period",
            "win_rate",
            "win_count",
            "days_held_max",
            "neg_ret_mean",
            "neg_ret_max",
        ],
    ) -> None:
        self.full = full
        self.drilldown_mapping = plot_utils.validate_mapping(
            drilldown_mapping, StratComponent
        )
        self.date = date or utils.get_current_dt(fmt="%Y-%m-%d")
        self.periods = periods
        self.min_trading_period = min_trading_period
        self.analysis_cols = analysis_cols
        self.top_n = top_n
        self.top_n_common = top_n_common
        self.palette = palette
        self.disp_cols = disp_cols

        # Get file and directory paths
        self.gen_paths(path)

    def gen_paths(self, path: DictConfig) -> None:
        """Generate required file and directory paths"""

        self.date_dir = f"{path.data_dir}/{self.date}"
        self.graph_date_dir = f"{path.graph_dir}/{self.date}"
        self.combined_path = f"{self.date_dir}/combined_overall.csv"
        self.top_n_path = f"{self.date_dir}/top_{self.top_n}_tickers.csv"
        self.pivot_top_n_path = f"{self.date_dir}/pivot_top_{self.top_n}_tickers.csv"

    def run(self) -> None:
        # # Combine 'overall_summary.csv' info for all combinations
        # self.combine_overall()

        # # Generate DataFrame containing top 15 ticker pairs with highest
        # # positive annualized return
        # self.gen_top_n_pairs()

        # Generate dictionary mapping strat component to mean annual returns
        stats_df_dict = plot_utils.gen_stats_comp_dict(
            self.combined_path,
            get_args(StratComponent),
            self.analysis_cols,
        )

        # # Plot histograms of annualized returns, win rate, number of trades and
        # # highet negative returns for all strategy combinations
        # self.plot_overall()

        # Plot top N tickers with highest daily return overall
        self.plot_top_ticker_pairs()

        # Plot histograms overlay, annualized returns statistics,
        # and top ticker pairs for each strategy component
        for strat_comp, comp_str in self.drilldown_mapping.items():
            self.plot_comp_hist(strat_comp, comp_str)
            self.plot_comp_ticker_pairs(strat_comp)

            for attribute in self.analysis_cols:
                self.plot_comp_stats(
                    stats_df_dict[strat_comp][attribute],
                    strat_comp,
                    attribute,
                    comp_str,
                )

        # Plot bar chart of common occurring ticker pairs among top_N pairs
        self.plot_common()

        # # Plot bar chart of common occuring ticker pairs in increasing top N
        # # i.e. start from top 1 till top N
        # self.plot_successive_common()

    def plot_overall(self) -> None:
        """Plot histogram of annualized returns, mean days held, trading period,
        win_rate for all strategies i.e. overall performance."""

        # Load csv file if exist
        combined_path = Path(self.combined_path)
        if not combined_path.is_file():
            raise FileNotFoundError(
                f"'combined_overall.csv' is not found at '{self.combined_path}'."
            )

        df = utils.load_csv(self.combined_path, tz="America/New_York")

        _, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 12))

        for ax, col in zip(axes.flat, self.analysis_cols):
            sns.histplot(data=df, x=col, ax=ax, kde=True, line_kws={"linewidth": 3})

            # Format text for graphical representation
            col_msg = " ".join([col.title() for col in col.split("_")])

            ax.set_title(f"Frequency Distribution of {col_msg}")
            ax.set_xlabel(col_msg)
            ax.set_ylabel("Frequency")

        # Hide unused axes
        plot_utils.hide_unused_ax(axes.flat, self.analysis_cols)

        # Create folder if not exist
        utils.create_folder(self.graph_date_dir)

        plt.tight_layout()
        plt.savefig(f"{self.graph_date_dir}/combined_overall.png")
        plt.close()

    def plot_comp_hist(self, strat_comp: StratComponent, comp_str: str) -> None:
        """Plot histogram of annualized returns, win_rate, num_trades and
        highest_neg_return for all strategy combinations.

        Args:
            strat_comp (StratComponent):
                Component that makes up the trading strategy e.g. "entry_struct".
            comp_str (str):
                Formatted strat component string to be used in title.
        """

        # Load csv file if exist
        combined_path = Path(self.combined_path)
        if not combined_path.is_file():
            raise FileNotFoundError(
                f"'combined_overall.csv' is not found at '{self.combined_path}'."
            )

        df = utils.load_csv(self.combined_path, tz="America/New_York")

        if strat_comp not in df.columns:
            raise ValueError(
                f"'{strat_comp}' is not a valid column name in combined DataFrame."
            )

        _, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 12))

        for ax, col in zip(axes.flat, self.analysis_cols):
            sns.histplot(
                data=df,
                x=col,
                hue=strat_comp,
                ax=ax,
                kde=True,
                line_kws={"linewidth": 3},
                legend=True,
                # alpha=0.7,
                palette=plot_utils.set_palette(df, strat_comp, self.palette),
            )

            # Format text for graphical representation
            col_msg = " ".join([col.title() for col in col.split("_")])

            ax.set_title(f"Distribution of {col_msg} by {comp_str}", fontsize=16)
            ax.set_xlabel(col_msg)
            ax.set_ylabel("Frequency")

        # Hide unused axes
        plot_utils.hide_unused_ax(axes.flat, self.analysis_cols)

        # Create folder if not exist
        utils.create_folder(self.graph_date_dir)

        plt.tight_layout()
        plt.savefig(f"{self.graph_date_dir}/combined_{strat_comp}.png")
        plt.close()

    def plot_top_ticker_pairs(self) -> None:
        """Plot top N ticker pairs with overall highest daily return."""

        df = plot_utils.get_top_pairs(self.top_n_path, self.top_n, self.disp_cols)
        print(f"Top N ticker pairs with highest daily return : \n\n{df}\n")

        _, ax = plt.subplots(figsize=(15, 8))
        sns.barplot(x=df["ticker_pair"], y=df["overall_daily_ret"], ax=ax)

        ax.set_title(
            f"Top {self.top_n} Ticker Pairs with Highest Daily Return", fontsize=24
        )
        ax.set_xlabel("Ticker Pair")
        ax.set_ylabel("Daily Returns")
        ax.tick_params(axis="x", rotation=30)

        # Annotate daily returns
        for bar in ax.patches:
            ax.annotate(
                bar.get_height(),
                (bar.get_x() + bar.get_width() / 2, bar.get_height() * 0.75),
                ha="center",
                va="bottom" if bar.get_height() >= 0 else "top",
                rotation=30,
            )

        # Create folder if not exist
        utils.create_folder(self.graph_date_dir)

        plt.tight_layout()
        plt.savefig(f"{self.graph_date_dir}/top_{self.top_n}_all.png")
        plt.close()

    def plot_comp_ticker_pairs(self, strat_comp: StratComponent) -> None:
        """Plot top ticker pairs with highest daily return for strategy component."""

        nrows = 3 if strat_comp == "exit_struct" else 2

        df_dict = plot_utils.get_top_pairs_by_comp(
            strat_comp, self.top_n_path, self.top_n, self.disp_cols
        )
        _, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(20, 12))

        for ax, (comp, df) in zip(axes.flat, df_dict.items()):
            sns.barplot(x=df["ticker_pair"], y=df["overall_daily_ret"], ax=ax)

            comp_str = f"period_{comp}y" if strat_comp == "period" else comp

            ax.set_title(f"Top {self.top_n} Ticker Pairs for '{comp_str}'")
            ax.set_xlabel("Ticker Pair")
            ax.set_ylabel("Daily Returns")
            ax.tick_params(axis="x", rotation=30)

            # Annotate daily returns
            for bar in ax.patches:
                ax.annotate(
                    bar.get_height(),
                    (bar.get_x() + bar.get_width() / 2, bar.get_height() * 0.75),
                    ha="center",
                    va="bottom" if bar.get_height() >= 0 else "top",
                    rotation=30,
                )

        # Hide unused axes
        plot_utils.hide_unused_ax(axes.flat, list(df_dict.keys()))

        # Create folder if not exist
        utils.create_folder(self.graph_date_dir)

        plt.tight_layout()
        plt.savefig(f"{self.graph_date_dir}/top_{self.top_n}_{strat_comp}.png")
        plt.close()

    def plot_comp_stats(
        self,
        df_ret: pd.DataFrame,
        strat_comp: StratComponent,
        attribute: str,
        comp_str: str,
    ) -> None:
        """Plot min, max, mean and median annualized returns for each
        strat components.

        Args:
            df_ret (pd.DataFrame):
                DataFrame containing min, max, mean, and median annual returns
                for specific strat component.
            strat_comp (StratComponent):
                Component making up trading strategy.
            attribute (str):
                Attributes for statistics generation e.g. 'win_rate'.
            comp_str (str):
                Formatted strat component string to be used in title.

        Returns:
            None.
        """

        df = df_ret.copy()

        # Create folder if not exist
        utils.create_folder(self.graph_date_dir)

        # Generate min, max, mean and median annualized return for
        # each strat component
        _, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 12))

        for ax, stat in zip(axes.flat, df.columns):
            # Sort by stat in descending order
            df_sorted = df.sort_values(by=[stat], ascending=False).reset_index()
            max_ret = df["max"].max()
            min_ret = df["min"].min()

            # Create color palette for positive and negative returns
            colors = [
                "red" if stat < 0 else "green" for stat in df_sorted[stat].to_list()
            ]

            # Setting hue is required if using palette
            sns.barplot(
                x=df_sorted[strat_comp],
                y=df_sorted[stat],
                hue=df_sorted[strat_comp],
                palette=colors,
                legend=False,
                ax=ax,
            )

            # Formatting for title
            stat_str = stat.title()
            attribute_str = " ".join([item.title() for item in attribute.split("_")])

            ax.set_title(f"{stat_str} {attribute_str} by {comp_str}")
            ax.set_xlabel(f"{comp_str}")
            ax.set_ylabel(f"{stat_str} Annualized Returns")
            ax.set_ylim(
                min_ret - Decimal("0.05"), max_ret + Decimal("0.05")
            )  # Ensure consistent y-axis for all subplots
            # ax.tick_params(axis="x", labelrotation=30)

            # Annotate annualized returns
            for bar in ax.patches:
                ax.annotate(
                    bar.get_height(),
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center",
                    va="bottom" if bar.get_height() >= 0 else "top",
                )

        plt.tight_layout()
        plt.savefig(f"{self.graph_date_dir}/{strat_comp}_{attribute}.png")
        plt.close()

    def plot_common(
        self,
        col: str = "ticker_pair",
    ) -> None:
        """Generate bar chart of common occurring
        ticker pairs among top_N pairs.

        Args:
            col (str): Column containing ticker pairs (Default: "ticker_pair").

        Returns:
            None.
        """

        top_n_path = Path(self.top_n_path)
        if not top_n_path.is_file():
            raise FileNotFoundError(
                f"'{top_n_path.name}' does not exist at '{self.top_n_path}'."
            )

        df = utils.load_csv(top_n_path, tz="America/New_York")

        # Get top N 'ticker_pair' with highest count
        top_n_pairs = plot_utils.get_top_n(df, col, self.top_n)

        # Plot bar chart of top N ticker_pair
        _, ax = plt.subplots(figsize=(20, 8))
        sns.barplot(x=top_n_pairs.index, y=top_n_pairs, ax=ax)

        ax.set_title(f"Top {self.top_n} Ticker Pair by Frequency")
        ax.set_xlabel("Ticker Pair")
        ax.set_ylabel("Counts")

        # Annotate daily returns
        for bar in ax.patches:
            ax.annotate(
                bar.get_height(),
                (bar.get_x() + bar.get_width() / 2, bar.get_height() * 0.75),
                ha="center",
                va="bottom" if bar.get_height() >= 0 else "top",
                rotation=30,
            )

        # Create folder if not exist
        utils.create_folder(self.graph_date_dir)

        plt.tight_layout()
        plt.savefig(f"{self.graph_date_dir}/common_pairs.png")
        plt.close()

    def plot_successive_common(
        self,
    ) -> None:
        """Start with top 1 ticker pair to plot bar chart; Proceed until top N reached."""

        pivot_top_n_path = Path(self.pivot_top_n_path)
        if not pivot_top_n_path.is_file():
            raise FileNotFoundError(
                f"'{pivot_top_n_path.name}' does not exist at '{self.pivot_top_n_path}'."
            )

        df = utils.load_csv(pivot_top_n_path, index_col=0)

        # Top N is limited by number of columns in 'df_pivot_top_n'
        top_n = min(self.top_n, len(df.columns))

        # Ensure all subplots are arranged in 2 columns
        _, axes = plt.subplots(nrows=round(top_n / 2), ncols=2, figsize=(20, 20))

        # Plot 'top_n_common' most common top 1 to 'top_n' ticker pair in each subplot
        for ax, idx in zip(axes.flat, range(top_n)):
            df_pair_counts = plot_utils.gen_pair_counts_df(
                df, self.top_n, self.top_n_common
            )

            sns.barplot(x=df_pair_counts.index, y=df_pair_counts["count"], ax=ax)

            ax.set_title(
                f"Top {self.top_n_common} Common Ticker-Pairs Extracted from Top {idx+1} "
                "Ticker-Pairs",
            )
            ax.set_xlabel("Ticker-Pair")
            ax.set_ylabel("Counts")
            ax.tick_params(axis="x", rotation=30)

        # Hide unused axes
        plot_utils.hide_unused_ax(axes.flat, list(range(top_n)))

        # Create folder if not exist
        utils.create_folder(self.graph_date_dir)

        plt.tight_layout()
        plt.savefig(f"{self.graph_date_dir}/successive_top_n_pairs.png")
        plt.close()

    def combine_overall(self) -> pd.DataFrame:
        """Combine DataFrame generated from 'overall_summary.csv' for all
        combinations."""

        combined_path = Path(self.combined_path)

        # Load csv file if exist
        if combined_path.is_file():
            print(f"\n{combined_path.name} exists at '{combined_path.as_posix()}'\n")
            return utils.load_csv(combined_path, tz="America/New_York")

        # Get list of combinations for long, short and long-short strategies
        combi_list = plot_utils.get_combi_list(self.full)
        df_list = []

        for (
            ent_type,
            ent_struct,
            ex_struct,
            stop_method,
            hf_model,
            coint_corr_fn,
            period,
        ) in tqdm(combi_list):
            # Load 'overall_summary.csv' for each combi
            model_dir = (
                f"{self.date_dir}/"
                f"{ent_type}_{ent_struct}_{ex_struct}_{stop_method}/"
                f"{hf_model}_{coint_corr_fn}_{period}"
            )
            overall_path = f"{model_dir}/overall_summary.csv"

            # Append formatted overall summary DataFrame to list
            df_list.append(plot_utils.format_overall_df(overall_path))

        # Concatenate row-wise and save DataFrame
        df_combined = pd.concat(df_list, axis=0).reset_index(drop=True)
        utils.save_csv(df_combined, combined_path)

        return df_combined

    def gen_top_n_pairs(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Generate DataFrame where columns represent top N ticker pairs with highest
        annualized return.

        Args:
            None.

        Returns:
            df (pd.DataFrame):
                DataFrame containing top N ticker pairs with highest daily return for
                each strategy.
            df_pivot (pd.DataFrame):
                Pivoted DataFrame where columns are top N tickers.
        """

        # Get list of combinations for long, short and long-short strategies
        combi_list = plot_utils.get_combi_list(self.full)
        df_list, df_pivot_list = [], []

        for strat_comp_list in tqdm(combi_list):
            (
                ent_type,
                ent_struct,
                ex_struct,
                stop_method,
                hf_model,
                coint_corr_fn,
                period,
            ) = strat_comp_list
            # Load breakdown_summary.csv for each combination
            model_dir = (
                f"{self.date_dir}/"
                f"{ent_type}_{ent_struct}_{ex_struct}_{stop_method}/"
                f"{hf_model}_{coint_corr_fn}_{period}"
            )
            breakdown_path = f"{model_dir}/breakdown_summary.csv"
            strat_comp_cols = get_args(StratComponent)

            # Extract top N ticker pairs with highest daily return
            df = plot_utils.format_breakdown_df(
                breakdown_path,
                strat_comp_list,
                self.min_trading_period,
                self.top_n,
                strat_comp_cols,
            )

            # Convert filtered breakdown DataFrame into pivot form
            df_pivot = plot_utils.format_breakdown_pivot_df(df, self.top_n)

            # Avoid appending empy DataFrame to 'df_list' and 'df_pivot_list'
            if df.empty or df_pivot.empty:
                print(f"No breakdown for '{model_dir}/breakdown_summary.csv'.")
                continue

            df_list.append(df)
            df_pivot_list.append(df_pivot)

        # Concantate row-wise' and reset index
        df_top_n = pd.concat(df_list, axis=0)
        df_top_n = df_top_n.reset_index(drop=True)

        # Generate pivot table where columns are top N tickers
        df_pivot_top_n = pd.concat(df_pivot_list, axis=0)

        # Save DataFrame as csv files
        utils.save_csv(df_top_n, self.top_n_path, save_index=False)
        utils.save_csv(df_pivot_top_n, self.pivot_top_n_path, save_index=True)

        return df_top_n, df_pivot_top_n
