"""Class to plot graphs relating to cointegration and correlation on
S&P500 ticker pairs."""

from pathlib import Path
from typing import get_args

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from config.variables import CointCorrFn
from src.utils import utils


class PlotCointCorr:
    """Plot and save graphs relating to cointegration.

    Usage:
        >>> plot_coint_corr = PlotCointCorr()
        >>> plot_coint_corr.run()

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
        news_path (str):
            Relative path to DataFrame containing news scraped on specific date.
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
        self.news_path = f"{results_dir}/{self.date}/news.csv"
        self.coint_corr_date_dir = f"{coint_corr_dir}/{self.date}"
        self.graph_date_dir = f"{graph_dir}/{self.date}"

    def run(self) -> None:
        """Generate histogram plot for Engle-Granger (coint), Pearson, Spearman,
        Kendall Tau test on 1, 3 and 5 years period."""

        # Iterate through cointegration correlation csv files for
        # 1, 3, and 5 years period
        for period in tqdm(self.periods):
            df_coint_corr = pd.read_csv(
                f"{self.coint_corr_date_dir}/coint_corr_{period}y.csv"
            )

            # Generate histogram plot for all 4 cointegration and correlation info
            # in a single plot
            self.plot_hist(df_coint_corr, period)

    def plot_hist(self, df_coint_corr: pd.DataFrame, period: int) -> None:
        """Generate histogram plot for all 4 cointegration and correlation info
        for a specific period in a single plot."""

        df = df_coint_corr.copy()

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 15))

        # Iterate through each ax sequentially after flattening 'axes'
        for ax, analysis in zip(axes.flat, get_args(CointCorrFn)):
            sns.histplot(df, x=analysis, ax=ax, kde=True)

            # Customize graph format
            analysis_msg = (
                "Engle-Granger Cointegration"
                if analysis == "coint"
                else f"{analysis.title()} Correlation"
            )
            xlabel_msg = (
                "p-value" if analysis == "coint" else f"{analysis.title()} correlation"
            )
            year_msg = "year" if period == 1 else "years"

            ax.set_title(
                f"Histogram Plot of {analysis_msg} on {period} {year_msg} Period."
            )
            ax.set_xlabel(xlabel_msg)
            ax.set_ylabel("Frequency")

        # Create folder if not exist
        utils.create_folder(self.graph_date_dir)

        plt.tight_layout()
        plt.savefig(f"{self.graph_date_dir}/coint_corr_{period}y.png")
