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
from tqdm import tqdm

from src.get_rel import GetRel
from src.utils import utils


class GenPriceAction:
    """Generate sentiment rating, closing price and price action for each day.

    Usage:
        >>> gen_price_action = GenPriceAction("AAPL")
        >>> df = gen_price_action.run()

    Args:
        date (str):
            If provided, date when news are scraped.
        stock_dir (str):
            Relative path to 'stock' folder (Default: "./data/stock").
        results_dir (str):
            Relative path of folder containing price action for ticker pairs (i.e.
            stock ticker and its cointegrated ticker) (Default: "./data/results").
        model_name (str):
            Name of FinBERT model in Huggi[ngFace (Default: "ziweichen").
        period (int):
            Time period used to compute cointegration (Default: 5).
        top_n (int):
            Top N number of stocks with lowest pvalue.

    Attributes:
        date (str):
            If provided, date when news are scraped.
        coint_path (str):
            If provided, relative path to CSV cointegration information (Default: None).
        senti_path (str):
            Relative path to CSV file containing news sentiment rating
            (Default: "./data/sentiment.csv").
        stock_dir (str):
            Relative path to stock folder (Default: "./data/stock").
        model_name (str):
            Name of FinBERT model in HuggingFace (Default: "ziweichen").
        price_action_dir (str):
            Relative path of folder containing price action of ticker pairs for specific
            model and cointegration period.
        top_n (int):
            Top N number of stocks with lowest pvalue.
    """

    def __init__(
        self,
        date: str | None = None,
        stock_dir: str = "./data/stock",
        results_dir: str = "./data/results",
        model_name: str = "ziweichen",
        period: int = 5,
        top_n: int = 10,
    ) -> None:
        self.date = date or utils.get_current_dt(fmt="%Y-%m-%d")
        self.coint_path = f"./data/coint/{self.date}/coint_{period}y.csv"
        self.senti_path = f"./data/results/{self.date}/sentiment.csv"
        self.stock_dir = stock_dir
        self.model_name = model_name
        self.price_action_dir = (
            f"{results_dir}/{self.date}/{model_name}_{period}/price_actions"
        )
        self.top_n = top_n

    def run(self) -> None:
        """Generate and save DataFrame including average sentiment rating and
        closing price of stock and co-integrated stock."""

        # Load 'sentiment.csv' and 'coint_5y.csv'
        df_senti = pd.read_csv(self.senti_path)
        df_coint = self.load_coint()

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
            self.gen_topn_close(df_av, df_coint, ticker)

    def load_coint(self) -> pd.DataFrame:
        """Load cointegration CSV file as DataFrame if available else generate
        cointegration csv file"""

        if Path(self.coint_path).is_file():
            return pd.read_csv(self.coint_path)

        # Generate cointegration file
        cointegrate = CoIntegrate()
        return cointegrate.run()

    def cal_mean_sentiment(self, data: pd.DataFrame) -> pd.DataFrame:
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

    def append_close(self, data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Append closing price of stock ticker whose news are sentiment-rated."""

        df = data.copy()

        # Load OHLCV prices for ticker
        ohlcv_path = f"{self.stock_dir}/{ticker}.parquet"
        df_ohlcv = pd.read_parquet(ohlcv_path)
        df_ohlcv.index = pd.to_datetime(df_ohlcv.index)
        df.index = pd.to_datetime(df.index)

        # Append closing price of 'ticker'
        df[f"{ticker}_close"] = df_ohlcv.loc[df_ohlcv.index.isin(df.index), "Close"]

        return df

    def gen_topn_close(
        self, df_av: pd.DataFrame, df_coint: pd.DataFrame, ticker: str
    ) -> None:
        """Generate and save Dataframe for each 'top_n' cointegrated stocks with
        lowest pvalue.

        Args:
            data (pd.DataFrame):
                DataFrame containing average sentiment rating and closing price
                for ticker.
            df_coint (pd.DataFrame):
                DataFrame containing cointegration info for stock ticker pairs.
            ticker (str):
                Stock ticker whose news are sentiment-rated.

        Returns:
            None.
        """

        df = df_av.copy()

        # Get list of cointegrated stocks with lowest pvalue
        coint_list = self.get_topn_tickers(ticker, df_coint)

        for coint_ticker in coint_list:
            # Generate and save DataFrame for each cointegrated stock
            df_coint_ticker = self.append_coint_close(df, coint_ticker=coint_ticker)

            # Append price-action i.e. buy if rating is >=4; sell if rating is <=2
            df_coint_ticker["action"] = df_coint_ticker["median_rating_excl"].map(
                self.gen_price_action
            )

            # Create folder if not exist
            utils.create_folder(self.price_action_dir)

            file_name = f"{ticker}_{coint_ticker}.csv"
            file_path = f"{self.price_action_dir}/{file_name}"
            utils.save_csv(df_coint_ticker, file_path, save_index=True)
            print(f"Saved '{file_name}' at '{file_path}'")

    def gen_price_action(self, rating: int) -> str:
        """Return 'buy' if rating > 4, 'sell' if rating <= 2  else 'wait'."""

        if rating >= 4:
            return "buy"
        if rating <= 2:
            return "sell"
        return "wait"

    def get_topn_tickers(self, ticker: str, df_coint: pd.DataFrame) -> list[str]:
        """Get list of top N stock with lowest pvalue for cointegration test with 'ticker'.

        Args:
            ticker (str):
                Stock ticker whose news are sentiment-rated.
            df_coint (pd.DataFrame):
                DataFrame containing cointegration info for stock ticker pairs.

        Returns:
            (list[str]): LIst of top N stocks with lowest cointegration pvalue.
        """

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

        # Ensure both df.index and df_ohlcv are datetime objects
        df.index = pd.to_datetime(df.index)
        df_ohlcv.index = pd.to_datetime(df_ohlcv.index)

        # Append closing price of 'ticker'
        df[f"{coint_ticker}_close"] = df_ohlcv.loc[
            df_ohlcv.index.isin(df.index), "Close"
        ]

        return df
