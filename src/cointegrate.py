"""Class to perform cointegration for S&P 500 stocks"""

from itertools import combinations

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
            (Default: ["BRK.B", "BF.B]).
        data_dir (str):
            Relative path to data folder (Default: "./data").

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
            (Default: ["BRK.B", "BF.B]).
        stock_list (list[str]):
            List containing S&P500 stocks.
        session (CachedLimiterSession):
            Instance of CachedLimiterSession which inherits from CacheMixin,
            LimiterMixin and Session.
        cur_dt (str):
            Datetime when cointegration analysis is performed.
    """

    def __init__(
        self,
        url: str = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        start: str = "2020-01-01",
        end: str | None = None,
        batch_size: int = 10,
        ignore_list: list[str] = ["BRK.B", "BF.B"],
        data_dir: str = "./data",
    ) -> None:
        self.url = url
        self.start = start
        self.end = end
        self.batch_size = batch_size
        self.ignore_list = ignore_list
        self.data_dir = data_dir
        self.stock_list = self.gen_stock_list()
        self.session = None
        self.cur_dt = utils.get_current_dt()

    def run(self) -> None:
        """Download 5 years OHLCV data from yfinance to perform co-integration
        between all pair combinations of S&P500 stocks."""

        # Download OHLCV data from Yahoo Finance
        df = self.batch_download()

        # Perform cointegration analysis for 5, 3 and 1 year
        for num_years in [5, 3, 1]:
            df_coint = self.cal_cointegration(df, num_years)
            self.plot_network_graph(df_coint, f"coint_{num_years}yr.png")

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

    def batch_download(self) -> pd.DataFrame:
        """Download about 5 years OHLCV daily data for each S&P500 stocks in batches."""

        # Initiate rate limiting caching session if not initialized
        if not self.session:
            self._init_session()

        df_list = []

        for count, idx in enumerate(range(0, len(self.stock_list), self.batch_size)):
            ticker_batch = self.stock_list[idx : idx + self.batch_size]
            print(f"batch {count+1} : {ticker_batch}")

            # Download OHLCV data in batches with rate limiting and caching
            df = yf.download(
                ticker_batch,
                start=self.start,
                end=self.end,
                group_by="ticker",
                rounding=True,
                session=self.session,
            )

            # Flatten multi-level columns and sort DataFrame
            df = self._format_df(df)
            df_list.append(df)

        # Combine DataFrame in list to a single DataFrame; and save DataFrame
        df_combine = pd.concat(df_list, axis=0)
        df_combine.to_parquet(
            f"{self.data_dir}/ohlcv_{self.cur_dt}.parquet", index=True
        )

        return df_combine

    def _format_df(self, data: pd.DataFrame) -> pd.DataFrame:
        """Flatten multi-level columns and sort DataFrame by Ticker
        followed by Date."""

        df = data.copy()

        # Flattened multi-level columns by stacking by 'Ticker' (level 0)
        df = (
            df.stack(level=0, future_stack=True)
            .rename_axis(["Date", "Ticker"])
            .reset_index(level=1)
        )

        # Sort DataFrame by 'Ticker' followed by 'Date'
        df = df.reset_index()
        df = df.sort_values(by=["Ticker", "Date"], ascending=True)
        df = df.set_index("Date")

        # Remove name of columns i.e. 'Price'
        df.columns.name = None

        return df

    def cal_cointegration(self, data: pd.DataFrame, num_years: int) -> pd.DataFrame:
        """Compute pvalue for cointegration for all pair combinations of S&P500 stocks
        using 'num_years' records.

        Args:
            data (pd.DataFrame):
                DataFrame containing OHLCV prices of S&P500 stocks for past 5 years.

        Returns:
            (pd.DataFrame):
                DataFrame containing all pair combinations of S&P500 stocks and its pvalue.
        """

        info = []

        for ticker1, ticker2 in combinations(self.stock_list, 2):
            # Get equal length 'Close' price for both tickers
            close1, close2 = self._gen_timeseries(data, ticker1, ticker2, num_years)

            # Get pvalue via 'coint'
            _, pvalue, _ = coint(close1, close2)

            info.append(
                {
                    "ticker1": ticker1,
                    "ticker2": ticker2,
                    "pvalue": pvalue,
                    "cointegrate": 1 if pvalue <= 0.05 else 0,
                }
            )

        # Convert 'info' list to DataFrame; and save as csv file
        df_info = pd.DataFrame(info)
        df_info.to_csv(f"{self.data_dir}/coint_{self.cur_dt}.csv", index=False)

        return df_info

    def _gen_timeseries(
        self, data: pd.DataFrame, ticker1: str, ticker2: str, num_years: int
    ) -> tuple[pd.Series, pd.Series] | None:
        """Generate equal length timeseries of Pandas Series Type based on given
        tickers and number of years.

        Args:
            data (pd.DataFrame): DataFrame containing OHLCV of all S&P500 stocks.
            ticker1 (str): First stock ticker.
            ticker2 (str): Second stock ticker.
            num_years (int): Number of years required to compute cointegration.

        Returns:
            timeseries1 (pd.Series): If available, first time series.
            timeseries2 (pd.Series): If available, second time series.
        """

        df = data.copy()

        # Get latest date in DataFrame. All tickers will have the same latest date.
        latest_date = df.index.max()

        # Get required earliest date i.e. latest_date - 'num_years'
        req_earliest_date = latest_date - pd.DateOffset(years=num_years)

        # Get DataFrame for ticker1 and ticker2; and remove null values
        df1 = df.loc[(df["Ticker"] == ticker1) & (~df.isna()), ["Ticker", "Close"]]
        df2 = df.loc[(df["Ticker"] == ticker2) & (~df.isna()), ["Ticker", "Close"]]

        # Get list of earliest available date for ticker1 and ticker2
        earliest_list = [df1.index.min(), df2.index.min()]

        if any([earliest_date < req_earliest_date for earliest_date in earliest_list]):
            print(
                f"Not enough data for {num_years} years to compute cointegration for {ticker1}-{ticker2} pair!"
            )
            return

        # Filter DataFrames to start from req_earliest_date
        df1 = df1.loc[df1.index >= req_earliest_date, :]
        df2 = df2.loc[df2.index >= req_earliest_date, :]

        # Get common index for df1 and df2
        common_index = self._get_equal_timeseries(df1, df2)

        return df1.loc[common_index, "Close"], df2.loc[common_index, "Close"]

    def _get_equal_timeseries(
        self, df1: pd.DataFrame, df2: pd.DataFrame
    ) -> list[pd.Timestamp]:
        """Ensure both DataFrame have same datetime index value."""

        # Get the intersection of both index
        common_set = set(df1.index) & set(df2.index)

        # Convert to list and sort in ascending order
        return sorted(list(common_set))

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
