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
from typing import Literal

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from tqdm import tqdm

from config.variables import COINT_CORR_FN, HF_MODEL
from src.cal_coint_corr import CalCointCorr
from src.utils import utils


class GenPriceAction:
    """Generate sentiment rating, closing price and price action for
    specific date, model, period, and correlation/cointegration combination.

    Usage:
        >>> gen_pa = GenPriceAction()
        >>> df = gen_pa.run()

    Args:
        date (str):
            If provided, date when news are scraped.
        hf_model (HF_MODEL):
            Name of FinBERT model in Huggi[ngFace (Default: "ziweichen").
        coint_corr_fn (COINT_CORR_FN):
            Name of function to perform either cointegration or correlation.
        period (int):
            Time period used to compute cointegration (Default: 5).
        top_n (int):
            Top N number of stocks with lowest pvalue.
        stock_dir (str):
            Relative path to 'stock' folder (Default: "./data/stock").
        results_dir (str):
            Relative path of folder containing price action for ticker pairs (i.e.
            stock ticker and its cointegrated ticker) (Default: "./data/results").

    Attributes:
        date (str):
            If provided, date when news are scraped.
        hf_model (str):
            Name of FinBERT model in HuggingFace (Default: "ziweichen").
        coint_corr_fn (COINT_CORR_FN):
            Name of function to perform either cointegration or correlation
            (Default: "coint").
        top_n (int):
            Top N number of stocks with lowest pvalue.
        stock_dir (str):
            Relative path to stock folder (Default: "./data/stock").
        coint_corr_path (str):
            If provided, relative path to CSV cointegration information (Default: None).
        senti_path (str):
            Relative path to CSV file containing news sentiment rating
            (Default: "./data/sentiment.csv").
        price_action_dir (str):
            Relative path of folder containing price action of ticker pairs for specific
            model and cointegration period.
    """

    def __init__(
        self,
        date: str | None = None,
        hf_model: HF_MODEL = "ziweichen",
        coint_corr_fn: COINT_CORR_FN = "coint",
        period: int = 5,
        top_n: int = 10,
        stock_dir: str = "./data/stock",
        results_dir: str = "./data/results",
        coint_corr_dir: str = "./data/coint_corr",
    ) -> None:
        self.date = date or utils.get_current_dt(fmt="%Y-%m-%d")
        self.hf_model = hf_model
        self.coint_corr_fn = coint_corr_fn
        self.top_n = top_n
        self.stock_dir = stock_dir

        coint_corr_date_dir = f"{coint_corr_dir}/{self.date}"
        results_date_dir = f"{results_dir}/{self.date}"

        self.coint_corr_path = f"{coint_corr_date_dir}/coint_corr_{period}y.csv"
        self.senti_path = f"{results_date_dir}/sentiment.csv"
        self.price_action_dir = (
            f"{results_date_dir}/{hf_model}_{coint_corr_fn}_{period}/price_actions"
        )

    def run(self) -> None:
        """Generate and save DataFrame including average sentiment rating and
        closing price of stock and co-integrated stock."""

        # Load 'sentiment.csv' and 'coint_5y.csv'
        df_senti = pd.read_csv(self.senti_path)
        df_coint_corr = self.load_coint_corr()

        for ticker in tqdm(df_senti["ticker"].unique()):
            # Filter specific ticker
            df_ticker = df_senti.loc[df_senti["ticker"] == ticker, :].reset_index(
                drop=True
            )

            # Group by publication date and compute mean sentiment rating
            df_av = self.cal_mean_sentiment(df_ticker)

            # Append 'is_holiday', 'ticker' and 'weekday'
            df_av = self.append_is_holiday(df_av)
            df_av = self.append_dayname(df_av)
            df_av.insert(0, "ticker", [ticker] * len(df_av))

            # Append closing price of ticker
            df_av = self.append_close(df_av, ticker)

            # Append closing price of top N co-integrated stocks with lowest pvalue
            self.gen_topn_close(df_av, df_coint_corr, ticker)

    def load_coint_corr(self) -> pd.DataFrame:
        """Load csv file containing cointegration and correlation info."""

        csv_path = Path(self.coint_corr_path)

        if not csv_path.is_file():
            print(
                f"'{csv_path.name}' is not available at '{csv_path}'. "
                f"Proceed to generate '{csv_path.name}'..."
            )
            cal_coint_corr = CalCointCorr(date=self.date)
            cal_coint_corr.run()

        return pd.read_csv(self.coint_corr_path)

    def cal_mean_sentiment(self, df_ticker: pd.DataFrame) -> pd.DataFrame:
        """Compute the average sentiment and average sentiment (excluding rating 3)
        for each trading day"""

        df = df_ticker.copy()

        # Generate 'date' column from 'pub_date' by extracting out the date
        # i.e. exclude time component
        df["pub_date"] = pd.to_datetime(df["pub_date"])
        df["date"] = df["pub_date"].dt.date

        # Exclude news with rating 3
        df_exclude = df.loc[df[self.hf_model] != 3, :]

        # Get Pandas Series of mean ratings (with and without rating 3)
        series_incl = df.groupby(by=["date"])[self.hf_model].mean()
        series_excl = df_exclude.groupby(by=["date"])[self.hf_model].median()

        # Generate DataFrame by concatenating 'series_incl' and 'series_excl'
        df_av = pd.concat([series_incl, series_excl], axis=1)
        df_av.columns = ["av_rating", "median_rating_excl"]

        # Replace null value in 'median_rating_excl' with 3.0 since all the news
        # articles have rating of 3 on the same day are excluded. Hence median will
        # return null.
        df_av = df_av.fillna(3)

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
        date_list = [np.datetime64(dt) for dt in df.index.to_list()]

        # Append whether date is holiday
        df.insert(0, "is_holiday", [dt in holidays_since_2024 for dt in date_list])

        return df

    def append_close(self, df_av: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Append closing price of stock ticker whose news are sentiment-rated."""

        df = df_av.copy()

        # Load OHLCV prices for ticker
        ohlcv_path = f"{self.stock_dir}/{ticker}.parquet"
        df_ohlcv = pd.read_parquet(ohlcv_path)
        df_ohlcv.index = pd.to_datetime(df_ohlcv.index)
        df.index = pd.to_datetime(df.index)

        # Append closing price of 'ticker'
        df[f"{ticker}_close"] = df_ohlcv.loc[df_ohlcv.index.isin(df.index), "Close"]

        return df

    def gen_topn_close(
        self,
        df_av: pd.DataFrame,
        df_coint_corr: pd.DataFrame,
        ticker: str,
    ) -> None:
        """Generate and save Dataframe for each 'top_n' cointegrated stocks with
        lowest pvalue.

        Args:
            df_av (pd.DataFrame):
                DataFrame containing average sentiment rating and closing price
                for ticker.
            df_coint_corr (pd.DataFrame):
                DataFrame containing cointegration and correlation info for
                stock ticker pairs.
            ticker (str):
                Stock ticker whose news are sentiment-rated.

        Returns:
            None.
        """

        df = df_av.copy()

        # Get list of cointegrated stocks with lowest pvalue
        coint_corr_list = self.get_topn_tickers(ticker, df_coint_corr)

        if coint_corr_list is None:
            return

        for coint_corr_ticker in coint_corr_list:
            # Generate and save DataFrame for each cointegrated stock
            df_coint_corr_ticker = self.append_coint_corr_ohlc(
                df, coint_corr_ticker=coint_corr_ticker
            )

            # Append price-action i.e. buy if rating is >=4; sell if rating is <=2
            df_coint_corr_ticker["action"] = df_coint_corr_ticker[
                "median_rating_excl"
            ].map(self.gen_price_action)

            # Create folder if not exist
            utils.create_folder(self.price_action_dir)

            # Ensure precision is maintained using Decimal object
            file_name = f"{ticker}_{coint_corr_ticker}.csv"
            file_path = f"{self.price_action_dir}/{file_name}"
            utils.save_csv(df_coint_corr_ticker, file_path, save_index=True)
            print(f"Saved '{file_name}' at '{file_path}'")

    def gen_price_action(self, rating: int) -> str:
        """Return 'buy' if rating > 4, 'sell' if rating <= 2  else 'wait'."""

        if rating >= 4:
            return "buy"
        if rating <= 2:
            return "sell"
        return "wait"

    def get_topn_tickers(
        self, ticker: str, df_coint_corr: pd.DataFrame
    ) -> list[str] | None:
        """Get list of top N stock with lowest pvalue for cointegration test with 'ticker'.

        Args:
            ticker (str):
                Stock ticker whose news are sentiment-rated.
            df_coint_corr (pd.DataFrame):
                DataFrame containing cointegration and correlation info for
                stock ticker pairs.

        Returns:
            (list[str]) | None:
                List of top N stocks with lowest cointegration pvalue or highest
                correlation if available.
        """

        # Filter records containing 'ticker' in 'ticker1' or 'ticker2' columns
        cond = (df_coint_corr["ticker1"] == ticker) | (
            df_coint_corr["ticker2"] == ticker
        )
        df = df_coint_corr.loc[cond, :].copy()

        if self.coint_corr_fn == "coint":
            # Ensure cointegration pvalue is less than 0.05
            df = df.loc[df[self.coint_corr_fn] < 0.05, :]
            sort_order = True

        else:
            # Ensure correlation is at more than 0.5
            df = df.loc[df[self.coint_corr_fn] > 0.5, :]
            sort_order = False

        if df.empty:
            print(
                f"Records doesn't meet minimum requirement for '{self.coint_corr_fn}' method."
            )
            return None

        # Get top N stocks based on 'self.coint_corr_fn'
        df_topn = df.sort_values(by=self.coint_corr_fn, ascending=sort_order).head(
            self.top_n
        )

        # Get set of unique tickers from 'ticker1' and 'ticker2' columns
        coint_set = set(df_topn["ticker1"].to_list()) | set(
            df_topn["ticker2"].to_list()
        )

        # Convert set to sorted list excluding 'ticker'
        return [symb for symb in sorted(list(coint_set)) if symb != ticker]

    def append_coint_corr_ohlc(
        self, df_av: pd.DataFrame, coint_corr_ticker: str
    ) -> pd.DataFrame:
        """Append OHLC data of cointegrated stocks."""

        df = df_av.copy()

        # Load OHLCV prices for ticker
        ohlcv_path = f"{self.stock_dir}/{coint_corr_ticker}.parquet"
        df_ohlcv = pd.read_parquet(ohlcv_path)

        # Ensure both df.index and df_ohlcv are datetime objects
        df.index = pd.to_datetime(df.index)
        df_ohlcv.index = pd.to_datetime(df_ohlcv.index)

        # Append OHLC data to DataFrame
        df[["coint_corr_ticker", "open", "high", "low", "close"]] = df_ohlcv.loc[
            df_ohlcv.index.isin(df.index), ["Ticker", "Open", "High", "Low", "Close"]
        ]

        return df
