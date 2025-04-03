"""Class to analyze strategies of different combinations of:

- FinBert sentiment classifier (ProsusAI, yiyangkhust, ziweichen, AventiqAI).
- Cointegration/correlation (Engle-Granger, Pearson, Spearman, Kendall).
- Periods (1, 3, 5 years).
"""

from itertools import product
from pathlib import Path
from typing import get_args

import pandas as pd

from config.variables import COINT_CORR_FN, HF_MODEL
from src.utils import utils


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
        results_dir: str = "./data/results/",
        coint_corr_dir: str = "./data/coint_corr",
        graph_dir: str = "./data/graph",
    ) -> None:
        self.date = date or utils.get_current_dt(fmt="%Y-%m-%d")
        self.periods = periods
        self.results_date_dir = f"{results_dir}/{self.date}"
        self.coint_corr_date_dir = f"{coint_corr_dir}/{self.date}"
        self.graph_date_dir = f"{graph_dir}/{self.date}"

    def run(self) -> None:
        # Combine 'overall_summary.csv' info for all combinations
        df_combined = self.combine_overall()

        # Generate DataFrame containing top 15 ticker pairs with highest
        # positive annualized return
        df_ticker_pairs = self.gen_top_n_pairs()

        # Plot histograms of annualized returns for all combinations, all FinBERT,
        # all cointegration/correlation and all time period
        self.plot_all(df_combined)
        self.plot_finbert(df_combined)
        self.plot_coint_corr(df_combined)
        self.plot_period(df_combined)

        # Plot bar chart of common occurring ticker pairs among top_N pairs
        self.plot_top_n(df_ticker_pairs)

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

    # def gen_top_n_pairs

    def format_overall_df(self, overall_path: str) -> pd.DataFrame:
        """Format DataFrame loaded from 'overall_summary.csv' such that indexes are
        columns with 'strategy' column present."""

        # Load DataFrame from 'overall_summary.csv'
        df_overall = utils.load_csv(overall_path, index_col=0)

        # Transpose DataFrame, reset index, and remove 'index' column
        df_overall = df_overall.T.reset_index()
        df_overall = df_overall.drop(columns=["index"])

        # Insert 'strategy' column if not present
        if "strategy" not in df_overall.columns:
            df_overall.insert(0, "strategy", Path(overall_path).parent.name)

        # Split 'strategy' columns into its component i.e. 'hf_model',
        # 'coint_corr_fn' and 'period'
        df_overall = self.reorder_cols(df_overall)

        return df_overall

    def reorder_cols(self, df_overall: pd.DataFrame) -> pd.DataFrame:
        """Split 'strategy' columns into its component i.e. 'hf_model',
        'coint_corr_fn' and 'period'."""

        df = df_overall.copy()

        # Split 'strategy' column into 'hf_model', 'coint_corr_fn' and period
        strategy_subcols = ["hf_model", "coint_corr_fn", "period"]
        df[strategy_subcols] = df["strategy"].str.rsplit("_", n=2, expand=True)

        no_strategy_cols = [
            col
            for col in df_overall.columns
            if col not in strategy_subcols + ["strategy"]
        ]
        new_cols = ["strategy"] + strategy_subcols + no_strategy_cols
