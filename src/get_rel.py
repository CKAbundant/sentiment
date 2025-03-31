"""Class to perform cointegration for S&P 500 stocks"""

import csv
import time
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import yfinance as yf
from matplotlib.figure import Figure
from pyrate_limiter import Duration, Limiter, RequestRate
from requests import Session
from requests_cache import CacheMixin, SQLiteCache
from requests_ratelimiter import LimiterMixin, MemoryQueueBucket
from scipy import stats
from statsmodels.tsa.stattools import coint
from tqdm import tqdm
from yfinance.exceptions import YFPricesMissingError

from src.utils import utils


class GetRel:
    """Perform Cointegration check for each pair of S&P500 stocks:

    - 502 stocks in S&P500.
    - Save the p-value and test statistics for each combinations.

    Usage:
        >>> co_integrate = CoIntegrate()
        >>> co_integrate.run()

    Args:
        url (str):
            URL to download updated list of S&P500 stocks
            (Default: "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies").
        start_date (str):
            Start date to download daily stock OHLCV data (Default: "2020-01-01").
        end_date (str):
            End date to download daily stock OHLCV data. If None,
            current date will be used (Default: None).
        batch_size (int):
            Number of tickers to download concurrently (Default: 20).
        ignore_list (list[str]):
            List of stocks to ignore due to data inavailbility in yfinance
            (Default: ["BRK.B", "BF.B", "CTAS"]).
        stock_dir (str):
            Relative path to folder containing stocks OHLCV data
            (Default: "./data/stock").
        coint_dir (str):
            Relative path to folder containing cointegration information
            (Default: "./data/coint").

    Attributes:
        url (str):
            URL to download updated list of S&P500 stocks
            (Default: "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies").
        start_date (str):
            Start date to download daily stock OHLCV data (Default: "2020-01-01").
        end_date (str | None):
            If provided, end date to download daily stock OHLCV data. If None,
            current date will be used (Default: None).
        batch_size (int):
            Number of tickers to download concurrently (Default: 10).
        ignore_list (list[str]):
            List of stocks to ignore due to data inavailbility in yfinance
            (Default: ["BRK.B", "BF.B", "CTAS"]).
        stock_dir (str):
            Relative path to folder containing stocks OHLCV data
            (Default: "./data/stock").
        coint_dir (str):
            Relative path to folder containing cointegration information
            (Default: "./data/coint").
        coint_date_dir (str):
            Relative path to subfolder under 'coint_dir'.
        stock_list (list[str]):
            List containing S&P500 stocks.
        session (CachedLimiterSession):
            Instance of CachedLimiterSession which inherits from CacheMixin,
            LimiterMixin and Session.
        unsuccessful (list[str]):
            List of tickers that are unable to be downloaded successfully via yfinance.
    """

    def __init__(
        self,
        stock_dir: str = "./data/stock",
        coint_dir: str = "./data/coint",
    ) -> None:
        self.stock_dir = stock_dir
        self.coint_dir = coint_dir
        self.coint_date_dir = f"{coint_dir}/{self.end_date}"
        self.stock_list = self.gen_stock_list()
        self.session = None
        self.unsuccessful = []

    def run(self) -> None:
        """Calculate correlation and cointegration between all pair combinations
        of S&P500 stocks."""

        # Perform correlation and cointegration analysis for 5, 3 and 1 year
        for num_years in [5, 3, 1]:
            df_coint = self.cal_cointegration(num_years)
            df_coint = self.cal_correlation(df_coint, num_years)
            # self.plot_network_graph(df_coint, f"coint_{num_years}yr.png")

        return df_coint

    def cal_cointegration(self, num_years: int) -> pd.DataFrame:
        """Compute pvalue for cointegration for all pair combinations of S&P500 stocks
        using 'num_years' records.

        Args:
            num_years (int): Number of years required to compute cointegration.

        Returns:
            (pd.DataFrame):
                DataFrame containing all pair combinations of S&P500 stocks and its pvalue.
        """

        # Get list of tickers with enough data for cointegration computation
        updated_list = self.get_enough_period(num_years)

        # If file exist, skip computation of cointegration since it is already available
        subfolder = f"{self.coint_dir}/{self.end_date}"
        file_path = f"{subfolder}/coint_{num_years}y.csv"
        if Path(file_path).is_file():
            return pd.read_csv(file_path)

        # Ensure subfolder exist
        utils.create_folder(subfolder)

        fieldnames = [
            "ticker1",
            "ticker2",
            "pvalue",
            "coint_t",
            "percent_1",
            "percent_5",
            "percent_10",
            "cointegrate",
        ]

        if not Path(file_path).is_file():
            with open(file_path, "w", newline="") as file:
                # Create csv file with header if not exist
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()

        with open(file_path, "a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)

            for ticker1, ticker2 in combinations(updated_list, 2):
                print(f"{ticker1}-{ticker2}")

                try:
                    # Get equal length 'Close' price for both tickers
                    df_close = self.gen_timeseries(ticker1, ticker2, num_years)

                    # Get pvalue via 'coint'
                    coint_t, pvalue, crit_value = coint(
                        df_close[ticker1], df_close[ticker2]
                    )

                    # Append row to csv file
                    writer.writerow(
                        {
                            "ticker1": ticker1,
                            "ticker2": ticker2,
                            "pvalue": pvalue,
                            "coint_t": coint_t,
                            "percent_1": crit_value[0],
                            "percent_5": crit_value[1],
                            "percent_10": crit_value[2],
                            "cointegrate": 1 if pvalue <= 0.05 else 0,
                        }
                    )

                except Exception as e:
                    print(f"Error encountered for {ticker1}-{ticker2} : {e}")

        # Load DataFrame from csv file
        return pd.read_csv(file_path)

    def get_enough_period(self, num_years: int) -> list[str]:
        """Identify list of tickers that have enough data for 'num_years'
        number of years.

        Args:
            num_years (int): Number of years required to compute cointegration.

        Returns:
            (list[str]): List of tickers with enough data points.
        """

        return [
            ticker for ticker in self.stock_list if self.is_enough(ticker, num_years)
        ]

    def gen_timeseries(
        self, ticker1: str, ticker2: str, num_years: int
    ) -> pd.DataFrame | None:
        """Generate equal length timeseries of Pandas Series Type based on given
        tickers and number of years.

        Args:
            ticker1 (str): First stock ticker.
            ticker2 (str): Second stock ticker.
            num_years (int): Number of years required to compute cointegration.

        Returns:
            df (pd.DataFrame): DataFrame containing closing price for both tickers.
        """

        # Load DataFrame for both tickers
        df1 = pd.read_parquet(f"{self.stock_dir}/{ticker1}.parquet")
        df2 = pd.read_parquet(f"{self.stock_dir}/{ticker2}.parquet")

        # Filter DataFrame to contain past 'num_years' years of records
        df1 = self.filter_df(df1, num_years)
        df2 = self.filter_df(df2, num_years)

        # Ensure timeseries for both tickers are of equal length by concatenating
        # "Close" Pandas Series and removing null values
        df = pd.concat([df1["Close"], df2["Close"]], axis=1)
        df.columns = [ticker1, ticker2]
        df = df.dropna()

        return df

    def filter_df(self, df: pd.DataFrame, num_years: int) -> pd.DataFrame:
        """Filter DataFrame to contain past 'num_years' years of records."""

        # Latest date in DataFrame
        req_earliest = self.cal_req_earliest(df, num_years)

        return df.loc[df.index >= req_earliest, :].reset_index(drop=True)

    def cal_req_earliest(self, data: pd.DataFrame, num_years: int) -> pd.Timestamp:
        """Calculate required earliest date for DataFrame to contain past
        'num_years' years of records."""

        latest = data.index.max()
        return latest - pd.DateOffset(years=num_years)

    def is_enough(self, ticker: str, num_years: int) -> bool:
        """Check whether there is enough data for cointegration computation.

        Args:
            ticker (str): Stock ticker
            num_years (int): Number of years required to compute cointegration.

        Returns:
            (bool): True if there is enough data.
        """

        # Load DataFrame from parquet file
        df = pd.read_parquet(f"{self.stock_dir}/{ticker}.parquet")

        # Compute required earliest date
        earliest = df.index.min()
        latest = df.index.max()
        req_earliest = latest - pd.DateOffset(years=num_years)

        return earliest <= req_earliest

    def plot_network_graph(
        self, data: pd.DataFrame, png_name: str, threshold: float = 0.05
    ) -> Figure:
        """Plot network graph of ticker pairs with pvalue below 'threshold'."""

        # Filter DataFrame to include ticker pairs below threshold
        significant_df = data.loc[data["pvalue"] < threshold, :]

        # Append 'color' column. Remove 'pvalue' and 'cointegrate' column
        significant_df["color"] = significant_df["pvalue"].map(self._get_edge_color)
        significant_df = significant_df.drop(columns=["pvalue", "cointegrate"])

        # Create undirected graph
        G = nx.Graph()

        # Add nodes for each stock
        for ticker in self.stock_list:
            G.add_node(ticker)

        # Add edges for significant cointegration relationships
        for ticker1, ticker2, color in significant_df.itertuples(
            index=False, name=None
        ):
            G.add_edge(ticker1, ticker2, color=color)

        # Prepare edge colors for visualization
        edge_colors = [G[u][v]["color"] for u, v in G.edges()]

        # Create folder if not exist
        graph_dir = f"{self.data_dir}/graph"
        utils.create_folder(graph_dir)

        # Plot and save graph
        fig, ax = plt.subplots(figsize=(15, 15))
        nx.draw(
            G,
            with_labels=True,
            node_color="red",
            edge_colors=edge_colors,
            node_size=500,
            font_size=10,
            ax=ax,
        )
        ax.set_title("Cointegration Relationships for S&P500 Stocks")
        plt.savefig(f"{graph_dir}/{png_name}")

        return fig

    def _get_edge_color(self, pvalue: float) -> str | None:
        """Define color mapping based on pvalue ranges."""

        if 0.025 <= pvalue < 0.05:
            return "lightblue"
        if 0.01 <= pvalue < 0.025:
            return "blue"
        if 0 <= pvalue < 0.01:
            return "darkblue"

    def gen_correlation(self, df_coint: pd.DataFrame, num_years: int) -> None:
        """Generate DataFrame containing Pearson, Spearman and Kendall correltion
        for comparision with cointegration.

        Args:
            num_years (int): Number of years required to compute cointegration.

        Returns:
            (pd.DataFrame):
                DataFrame containing cointegration and correlation values
                for each ticker pair.
        """

        df = df_coint.loc[:, ["ticker1", "ticker2"]].copy()
        records = []

        for ticker1, ticker2 in df.itertuples(index=False, name=None):
            # Get equal length 'Close' price for both tickers
            df_close = self.gen_timeseries(ticker1, ticker2, num_years)

            # Compute Pearson, Spearman and Kendall correlation
            pearson_results = stats.pearsonr(df_close[ticker1], df_close[ticker2])
            spearman_results = stats.spearmanr(df_close[ticker1], df_close[ticker2])
            kendall_results = stats.kendalltau(df_close[ticker1], df_close[ticker2])

            record = {
                "pearson_corr": pearson_results.statistic,
                "pearson_pvalue": pearson_results.pvalue,
                "spearman_corr": spearman_results.statistic,
                "spearman_pvalue": spearman_results.pvalue,
                "kendall_corr": kendall_results.statistic,
                "kendall_pvalue": kendall_results.pvalue,
            }
            records.append(record)

        # Convert to DataFrame
        df_corr = pd.DataFrame(records)
        df_corr.to_csv(f"{self.coint_date_dir}/correlation.csv")
