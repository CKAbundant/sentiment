"""Generate price action based on sentiment rating for the day.

- Compute average sentiment rating for the day.
- Buy 1 co-integrated stock if average rating is more than 4.
- Sell 1 co-integrated stock if average rating is less than 2.
- Generate DataFrame for stock:
    1. num_news -> Number of news articles for the day.
    2. av_rating -> Average sentiment rating for all news articles.
    3. av_bull_bear -> Average sentiment rating ignoring news with ranking 3.
    4. <ticker>_close -> Closing price of stock
    4. <ticker>_close -> Closing price of co-integrated stock.
    5. action -> "buy", "sell", "nothing"

Considerations:

1. Actual time of news publishing is not known and estimated from time period lapsed
e.g. '23 minutes ago'. Therefore, we take the average sentiment rating for the stock
for the day.
2. We will trade the stock that has the lowest p-value for co-integration test.
3. We assume decision to buy/sell stock occurs before market closing. Hence we observe
only the closing price of co-integrated stock.
4. We take past 10 days data from 13 Mar 2025 to observe the price changes.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

from src.cointegrate import CoIntegrate
from src.utils import utils


class GenPriceAction:
    """Generate sentiment rating, closing price and price action for each day.

    Usage:
        >>> gen_price_action = GenPriceAction("AAPL")
        >>> df = gen_price_action()

    Args:
        date (str):
            If provided, date when cointegration is performed.
        senti_path (str):
            Relative path to CSV file containing news sentiment rating
            (Default: "./data/sentiment.csv").
        stock_dir (str):
            Relative path to 'stock' folder (Default: "./data/stock").
        results_dir (str):
            Relative path to 'results' folder (Default: "./data/results").
        model_name (str):
            Name of FinBERT model in HuggingFace (Default: "ziweichen").
        top_n (int):
            Top N number of stocks with lowest pvalue.

    Attributes:
        date (str):
            If provided, date when cointegration is performed.
        coint_path (str):
            If provided, relative path to CSV cointegration information (Default: None).
        senti_path (str):
            Relative path to CSV file containing news sentiment rating
            (Default: "./data/sentiment.csv").
        stock_dir (str):
            Relative path to stock folder (Default: "./data/stock").
        results_dir (str):
            Relative path to 'results' folder (Default: "./data/results").
        model_name (str):
            Name of FinBERT model in HuggingFace (Default: "ziweichen").
        top_n (int):
            Top N number of stocks with lowest pvalue.
    """

    def __init__(
        self,
        date: str | None = None,
        senti_path: str = "./data/sentiment.csv",
        stock_dir: str = "./data/stock",
        results_dir: str = "./data/results",
        model_name: str = "ziweichen",
        top_n: int = 10,
    ) -> None:
        self.date = date or utils.get_current_dt(fmt="Y%m%d")
        self.coint_path = f"./data/coint/{self.date}/coint_5y.csv"
        self.senti_path = senti_path
        self.stock_dir = stock_dir
        self.results_dir = results_dir
        self.model_name = model_name
        self.top_n = top_n

    def __call__(self) -> pd.DataFrame:
        """Generate DataFrame including average sentiment rating and closing price
        of stock and co-integrated stock."""

        # Load 'sentiment.csv'
        df_senti = pd.read_csv(self.senti_path)

        for ticker in df_senti["ticker"].unique():
            # Filter specific ticker
            df_ticker = df_senti.loc[df_senti["ticker"] == ticker, :].reset_index(
                drop=True
            )

            # Group by publication date and compute mean sentiment rating
            df_av = self.cal_mean_sentiment(df_ticker, ticker)

            # Append 'is_holiday', 'ticker' and 'weekday'
            df_av = self.append_is_holiday(df_av)
            df_av = self.append_dayname(df_av)
            df_av.insert(0, "ticker", [ticker] * len(df_av))

            # Append closing price of ticker
            df_av = self.append_close(df_av, ticker)

            # Append closing price of top N co-integrated stocks with lowest pvalue
            self.gen_topn_close(df_av, ticker)

    def load_coint(self) -> pd.DataFrame:
        """Load cointegration CSV file as DataFrame if available else generate
        cointegration csv file"""

        if Path(self.coint_path).is_file():
            return pd.read_csv(self.coint_path)

        # Generate cointegration file
        cointegrate = CoIntegrate()
        return cointegrate.run()

    def cal_mean_sentiment(self, data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Compute the average sentiment and average sentiment (excluding rating 3)
        for each trading day"""

        df = data.copy()

        # Generate 'date' column from 'pub_date' by extracting out the date
        # i.e. exclude time component
        df["pub_date"] = pd.to_datetime(df["pub_date"])
        df["date"] = df["pub_date"].dt.date

        # Exclude news with rating 3
        df_exclude = df.loc[df[self.model_name] != 3, :]

        # Get Pandas Series of mean ratings (with and without rating 3)
        series_incl = df.groupby(by=["date"])[self.model_name].mean()
        series_excl = df_exclude.groupby(by=["date"])[self.model_name].median()

        # Generate DataFrame by concatenating 'series_incl' and 'series_excl'
        df_av = pd.concat([series_incl, series_excl], axis=1)
        df_av.columns = ["av_rating", "median_rating_excl"]

        return df_av

    def append_dayname(self, data: pd.DataFrame) -> pd.DataFrame:
        """Append 'day_name' column i.e. from 'Monday' to 'Sunday' based on
        'date' index."""

        df = data.copy()

        # Set 'date' index as column; and set as datetime type
        df = df.reset_index()
        df["date"] = pd.to_datetime(df["date"])

        # Insert 'day' column and set 'date' as index
        day_name = df["date"].dt.day_name()
        df.insert(0, "day_name", day_name)
        df = df.set_index("date")

        return df

    def append_is_holiday(self, data: pd.DataFrame) -> pd.DataFrame:
        """Append 'is_holiday' column i.e. from 'Monday' to 'Sunday' based on
        'date' index."""

        df = data.copy()

        # Get holidays for NYSE since 2020
        nyse = mcal.get_calendar("NYSE")
        holidays = nyse.holidays().holidays
        holidays_since_2024 = [
            holiday for holiday in holidays if holiday >= np.datetime64("2024-01-01")
        ]

        # Extract date index as list of np.datetime64 objects
        date_list = [np.datetime64(dt.date()) for dt in df.index.to_list()]

        # Append whether date is holiday
        df.insert(0, "is_holiday", [dt in holidays_since_2024 for dt in date_list])

        return df

    def append_close(self, data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Append closing price of stock ticker."""

        df = data.copy()

        # Load OHLCV prices for ticker
        ohlcv_path = f"{self.stock_dir}/{ticker}.parquet"
        df_ohlcv = pd.read_parquet(ohlcv_path)
        df_ohlcv.index = pd.DatetimeIndex(df_ohlcv.index)

        # Append closing price of 'ticker'
        df[f"{ticker}_close"] = df_ohlcv.loc[df_ohlcv.index.isin(df.index), "Close"]

        return df

    def gen_topn_close(self, data: pd.DataFrame, ticker: str) -> None:
        """Generate Dataframe for each 'top_n' cointegrated stocks with lowest pvalue.

        Args:
            data (pd.DataFrame):
                DataFrame containing average sentiment rating and closing price
                for ticker.

        Returns:
            None.
        """

        df = data.copy()

        # Get list of cointegrated stocks with lowest pvalue
        coint_list = self.get_topn_tickers(ticker)

        for coint_ticker in coint_list:
            # Generate and save DataFrame for each cointegrated stock
            df_coint_ticker = self.append_coint_close(df, coint_ticker=coint_ticker)

            subfolder = f"{self.results_dir}/{self.date}"
            file_path = f"{subfolder}/{ticker}_{coint_ticker}.csv"
            df_coint_ticker.to_parquet(file_path, index=True)

    def get_topn_tickers(self, ticker: str) -> list[str]:
        """Get list of top N stock with lowest pvalue for cointegration test with 'ticker'."""

        # load cointegration CSV file if present
        df_coint = self.load_coint()

        # Filter top N cointegrated stocks with lowest pvalue
        cond = ((df_coint["cointegrate"] == 1) & (df_coint["ticker1"] == ticker)) | (
            (df_coint["cointegrate"] == 1) & (df_coint["ticker2"] == ticker)
        )
        df_topn = (
            df_coint.loc[cond, :]
            .sort_values(by=["pvalue"], ascending=True)
            .head(self.top_n)
        )

        # Get set of unique tickers from 'ticker1' and 'ticker2' columns
        coint_set = set(df_topn["ticker1"].to_list()) | set(
            df_topn["ticker2"].to_list()
        )

        # Convert set to sorted list excluding 'ticker'
        return [symb for symb in sorted(list(coint_set)) if symb != ticker]

    def append_coint_close(self, data: pd.DataFrame, coint_ticker: str) -> pd.DataFrame:
        """Append closing price of cointegrated stocks."""

        df = data.copy()

        # Load OHLCV prices for ticker
        ohlcv_path = f"{self.stock_dir}/{coint_ticker}.parquet"
        df_ohlcv = pd.read_parquet(ohlcv_path)

        # Append closing price of 'ticker'
        df[f"{coint_ticker}_close"] = df_ohlcv.loc[
            df_ohlcv.index.isin(df.index), "Close"
        ]

        return df
