"""Class to perform cointegration for S&P 500 stocks"""

import csv
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
from statsmodels.tsa.stattools import coint
from yfinance.exceptions import YFPricesMissingError

from src.utils import utils


class CoIntegrate:
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
        start (str):
            Start date to download daily stock OHLCV data (Default: "2020-01-01").
        end (str):
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

    Attributes:
        url (str):
            URL to download updated list of S&P500 stocks
            (Default: "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies").
        start (str):
            Start date to download daily stock OHLCV data (Default: "2020-01-01").
        end (str | None):
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
        stock_list (list[str]):
            List containing S&P500 stocks.
        session (CachedLimiterSession):
            Instance of CachedLimiterSession which inherits from CacheMixin,
            LimiterMixin and Session.
        cur_dt (str):
            Datetime when cointegration analysis is performed.
        unsuccessful (list[str]):
            List of tickers that are unable to be downloaded successfully via yfinance.
    """

    def __init__(
        self,
        url: str = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        start: str = "2020-01-01",
        end: str | None = None,
        batch_size: int = 10,
        ignore_list: list[str] = ["BRK.B", "BF.B", "CTAS"],
        stock_dir: str = "./data/stock",
    ) -> None:
        self.url = url
        self.start = start
        self.end = end
        self.batch_size = batch_size
        self.ignore_list = ignore_list
        self.stock_dir = stock_dir
        self.stock_list = self.gen_stock_list()
        self.session = None
        self.cur_dt = utils.get_current_dt(fmt="%Y%m%d")
        self.unsuccessful = []

    def run(self) -> None:
        """Download 5 years OHLCV data from yfinance to perform co-integration
        between all pair combinations of S&P500 stocks."""

        # # Download OHLCV data from Yahoo Finance
        # self.download_tickers()

        # tries = 0
        # max_tries = 3

        # while self.unsuccessful and tries < max_tries:
        #     # Attempt to download unsuccessful tickers after 20 seconds
        #     time.sleep(20)
        #     self.download_tickers(self.unsuccessful)
        #     tries += 1

        df_coint = self.cal_cointegration(5)

        # # Perform cointegration analysis for 5, 3 and 1 year
        # for num_years in [5, 3, 1]:
        #     df_coint = self.cal_cointegration(num_years)
        #     self.plot_network_graph(df_coint, f"coint_{num_years}yr.png")

    def _init_session(self) -> None:
        """Combine 'requests_cache' with rate-limiting to avoid Yahoo's rate-limiter.
        Refer to https://yfinance-python.org/advanced/caching.html#smarter-scraping"""

        class CachedLimiterSession(CacheMixin, LimiterMixin, Session):
            pass

        self.session = CachedLimiterSession(
            limiter=Limiter(
                RequestRate(1, Duration.SECOND * 10)
            ),  # max 1 requests per 10 seconds
            bucket_class=MemoryQueueBucket,
            backend=SQLiteCache("yfinance.cache"),
        )

    def gen_stock_list(self) -> list[str]:
        """Generate updated list of S&P500 stocks from given url."""

        # Get DataFrame containing info on S&P500 stocks
        df_info, _ = pd.read_html(self.url)

        # Remove stocks in 'self.ignore_list' from list of S&P500 stocks
        return [
            stock
            for stock in df_info["Symbol"].to_list()
            if stock not in self.ignore_list
        ]

    def download_tickers(self, stock_list: list[str] | None = None) -> None:
        """Download about 5 years OHLCV daily data for each S&P500 stocks in batches."""

        stock_list = stock_list or self.stock_list

        # Reset self.unsuccessful
        self.unsuccessful = []

        # Initiate rate limiting caching session if not initialized
        if not self.session:
            self._init_session()

        for idx, ticker in enumerate(stock_list):
            try:
                # Download OHLCV data in batches with rate limiting and caching
                df = yf.download(
                    ticker,
                    start=self.start,
                    end=self.end,
                    rounding=True,
                    session=self.session,
                    multi_level_index=False,
                )

                # Format DataFrame; drop duplicates and null values; and sort by index
                df = df.loc[:, ["Open", "High", "Low", "Close", "Volume"]]
                df.insert(0, "Ticker", [ticker] * len(df))
                df = df.dropna().drop_duplicates()
                df = df.sort_index(ascending=True)

                print(f"{idx+1:>5}) {ticker:<6} : {len(df)}\n")

                # Create subfolder under '/data/stock'
                subfolder_path = f"{self.stock_dir}/{self.cur_dt}"
                utils.create_folder(subfolder_path)

                # Save as parquet file
                df.to_parquet(f"{subfolder_path}/{ticker}.parquet", index=True)

            except YFPricesMissingError as e:
                print(e)
                self.unsuccessful.append(ticker)

    def cal_cointegration(
        self, num_years: int, date: str | None = None
    ) -> pd.DataFrame:
        """Compute pvalue for cointegration for all pair combinations of S&P500 stocks
        using 'num_years' records.

        Args:
            num_years (int): Number of years required to compute cointegration.
            date (str | None): If provided, date when required parquet files are generated.

        Returns:
            (pd.DataFrame):
                DataFrame containing all pair combinations of S&P500 stocks and its pvalue.
        """

        # Get list of tickers with enough data for cointegration computation
        updated_list = self.get_enough_period(num_years, date)

        file_path = f"{self.stock_dir}/coint_5y_{self.cur_dt}.csv"
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

        with open(file_path, "w", newline="") as file:
            if not Path(file_path).is_file():
                # Create csv file with header if not exist
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()

        with open(file_path, "a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)

            for ticker1, ticker2 in combinations(updated_list, 2):
                print(f"{ticker1}-{ticker2}")

                try:
                    # Get equal length 'Close' price for both tickers
                    df_close = self._gen_timeseries(ticker1, ticker2, num_years, date)

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

    def get_enough_period(self, num_years: int, date: str | None = None) -> list[str]:
        """Identify list of tickers that have enough data for 'num_years'
        number of years.

        Args:
            num_years (int): Number of years required to compute cointegration.
            date (str | None): If provided, date when required parquet files are generated.

        Returns:
            (list[str]): List of tickers with enough data points.
        """

        return [
            ticker
            for ticker in self.stock_list
            if self.is_enough(ticker, num_years, date)
        ]

    def _gen_timeseries(
        self, ticker1: str, ticker2: str, num_years: int, date: str | None = None
    ) -> pd.DataFrame | None:
        """Generate equal length timeseries of Pandas Series Type based on given
        tickers and number of years.

        Args:
            ticker1 (str): First stock ticker.
            ticker2 (str): Second stock ticker.
            num_years (int): Number of years required to compute cointegration.
            date (str | None): If provided, date when required parquet files are generated.

        Returns:
            df (pd.DataFrame): DataFrame containing closing price for both tickers.
        """

        date = date or self.cur_dt

        # Load DataFrame for both tickers
        df1 = pd.read_parquet(f"{self.stock_dir}/{date}/{ticker1}.parquet")
        df2 = pd.read_parquet(f"{self.stock_dir}/{date}/{ticker2}.parquet")

        # Concat Panda Series for closing price for both tickers to form a DataFrame
        df = pd.concat([df1["Close"], df2["Close"]], axis=1)
        df.columns = [ticker1, ticker2]

        # Drop null values
        df = df.dropna()

        return df

    def is_enough(self, ticker: str, num_years: int, date: str | None = None) -> bool:
        """Check whether there is enough data for cointegration computation.

        Args:
            ticker (str): Stock ticker
            num_years (int): Number of years required to compute cointegration.
            date (str | None): If provided, date when required parquet files are generated.

        Returns:
            (bool): True if there is enough data.
        """

        date = date or self.cur_dt

        # Load DataFrame from parquet file
        df = pd.read_parquet(f"{self.stock_dir}/{date}/{ticker}.parquet")

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
