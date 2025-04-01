"""Download OHLCV data for S&P500 stocks via yfinance."""

import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf
from pyrate_limiter import Duration, Limiter, RequestRate
from requests import Session
from requests_cache import CacheMixin, SQLiteCache
from requests_ratelimiter import LimiterMixin, MemoryQueueBucket
from tqdm import tqdm
from yfinance.exceptions import YFPricesMissingError

from src.utils import utils


class DownloadOHLCV:
    """Download OHLCV data for S&P500 stocks via yfinance.

    - Update OHLCV data if existing OHLCV available.

    Usage:
        >>> download_ohlcv = DownloadOHLCV()
        >>> download_ohlcv.run()

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

    Attributes:
        snp500_list (list[str]):
            List containing S&P500 stocks.
        start_date (str):
            Start date to download daily stock OHLCV data (Default: "2020-01-01").
        end_date (str | None):
            If provided, end date to download daily stock OHLCV data. If None,
            current date will be used (Default: None).
        batch_size (int):
            Number of tickers to download concurrently (Default: 10).
        stock_dir (str):
            Relative path to folder containing stocks OHLCV data
            (Default: "./data/stock").
        session (CachedLimiterSession):
            Instance of CachedLimiterSession which inherits from CacheMixin,
            LimiterMixin and Session.
        unsuccessful (list[str]):
            List of tickers that are unable to be downloaded successfully via yfinance.
    """

    def __init__(
        self,
        snp500_list: list[str],
        start_date: str = "2020-01-01",
        end_date: str | None = None,
        batch_size: int = 10,
        stock_dir: str = "./data/stock",
    ) -> None:
        self.snp500_list = snp500_list
        self.start_date = start_date
        self.end_date = end_date or utils.get_current_dt(fmt="%Y-%m-%d")
        self.batch_size = batch_size
        self.stock_dir = stock_dir
        self.session = None
        self.unsuccessful = []

    def run(self) -> None:
        """Download tickers if new data is available."""

        # Determine if existing OHLCV data is the latest
        if self.is_latest():
            print(
                f"\nLatest OHLCV has already been downloaded. No futher action taken!"
            )
            return

        self.download_tickers()

        tries = 0
        max_tries = 3

        while self.unsuccessful and tries < max_tries:
            # Attempt to download unsuccessful tickers after 20 seconds
            time.sleep(20)
            self.download_tickers(self.unsuccessful)
            tries += 1

    def is_latest(self) -> bool:
        """Check if existing downloaded OHLCV data (using 'AAPL', 'WMT', and 'JPM')
        is the latest."""

        latest_dates = []
        for ticker in ["AAPL", "WMT", "JPM"]:
            df = pd.read_parquet(f"{self.stock_dir}/{ticker}.parquet")

            # Get latest date
            latest_date = datetime.strftime(df.index.max(), format="%Y-%m-%d")
            latest_dates.append(latest_date)

        # Get first item from sorted 'latest_dates'
        last_date = sorted("latest_dates", reverse=False)[0]

        return last_date >= self.end_date

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

    def download_tickers(self, stock_list: list[str] | None = None) -> None:
        """Download about 5 years OHLCV daily data for each S&P500 stocks in batches."""

        stock_list = stock_list or self.snp500_list

        # Create folder if not exist
        utils.create_folder(self.stock_dir)

        # Reset self.unsuccessful
        self.unsuccessful = []

        # Initiate rate limiting caching session if not initialized
        if not self.session:
            self._init_session()

        for idx, ticker in tqdm(
            enumerate(stock_list),
            desc="Downloading from yfinance",
            total=len(stock_list),
        ):
            try:
                file_path = f"{self.stock_dir}/{ticker}.parquet"

                if Path(file_path).is_file():
                    # Load and update existing OHLCV with latest data
                    df = self.update_ohlcv(ticker, file_path)
                else:
                    # Download full OHLCV data
                    df = self.download_ticker(ticker, start_date=self.start_date)

                print(f"{idx+1:>5}) {ticker:<6} : {len(df)}\n")

                # Save as parquet file
                df.to_parquet(file_path, index=True)

            except YFPricesMissingError as e:
                print(e)
                self.unsuccessful.append(ticker)

    def update_ohlcv(self, ticker: str, file_path: str) -> pd.DataFrame:
        """Update existing CSV file with latest OHLCV data for ticker."""

        # Load existing OHLCV data and get latest date
        df = pd.read_parquet(file_path)
        latest_date = df.index.max()

        # End date for input to yfinance has to be greater than latest date in DataFrame
        if pd.to_datetime(self.end_date) <= latest_date:
            return df

        # Download latest OHLCV data in batches with rate limiting and caching
        df_latest = self.download_ticker(ticker, start_date=latest_date)
        assert df.columns.to_list() == df_latest.columns.to_list()

        # Concatenate 'df_latest' to 'df' row-wise; remove duplicates and sort by index
        df = pd.concat([df, df_latest], axis=0)
        df = df.drop_duplicates()
        df = df.loc[~df.index.duplicated(keep="last"), :]
        df = df.sort_index(ascending=True)

        return df

    def download_ticker(self, ticker: str, start_date: str) -> pd.DataFrame:
        """Download latest OHLCV data if past OHLCV data exist else download
        all data."""

        # Download OHLCV data in batches with rate limiting and caching
        df = yf.download(
            ticker,
            start=start_date,
            end=self.end_date,
            rounding=True,
            session=self.session,
            multi_level_index=False,
        )

        # Format DataFrame; drop duplicates and null values; and sort by index
        df = df.loc[:, ["Open", "High", "Low", "Close", "Volume"]]
        df.insert(0, "Ticker", [ticker] * len(df))
        df = df.dropna().drop_duplicates()
        df = df.sort_index(ascending=True)

        return df
