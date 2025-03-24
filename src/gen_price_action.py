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

import pandas as pd

from src.cointegrate import CoIntegrate
from src.utils import utils


class GenPriceAction:
    """Generate sentiment rating, closing price and price action for each day.

    Usage:
        >>> gen_price_action = GenPriceAction("AAPL")
        >>> df = gen_price_action()

    Args:
        ticker (str):
            Stock ticker e.g. "AAPL".
        date (str):
            If provided, date when cointegration is performed.
        senti_path (str):
            Relative path to CSV file containing news sentiment rating
            (Default: "./data/sentiment.csv").
        model_name (str):
            Name of FinBERT model in HuggingFace (Default: "ziweichen").

    Attributes:
        ticker (str):
            Stock ticker e.g. "AAPL".
        date (str):
            If provided, date when cointegration is performed.
        coint_path (str):
            If provided, relative path to CSV cointegration information (Default: None).
        senti_path (str):
            Relative path to CSV file containing news sentiment rating
            (Default: "./data/sentiment.csv").
        model_name (str):
            Name of FinBERT model in HuggingFace (Default: "ziweichen").
    """

    def __init__(
        self,
        ticker: str,
        date: str | None = None,
        senti_path: str = "./data/sentiment.csv",
        model_name: str = "ziweichen",
    ) -> None:
        self.ticker = ticker
        self.date = date or utils.get_current_dt(fmt="Y%m%d")
        self.coint_path = f"./data/stock/{self.date}/coint_5y.csv"
        self.senti_path = senti_path
        self.model_name = model_name

    def __call__(self) -> pd.DataFrame:
        """Generate DataFrame including average sentiment rating and closing price
        of stock and co-integrated stock."""

        # Load 'sentiment.csv'
        df_senti = pd.read_csv(self.senti_path)

        # load cointegration CSV file if present
        df_coint = self.load_coint()

        for ticker in df_senti["ticker"].unique():
            # Filter specific ticker
            df_ticker = df_senti.loc[df_senti["ticker"] == ticker, :].reset_index(
                drop=True
            )

            # Group by publication date and compute mean sentiment rating
            df_av = self.cal_mean_sentiment(df_ticker)

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
        ticker = df.at[0, "ticker"]

        # Generate 'date' column from 'pub_date' by extracting out the date
        # i.e. exclude time component
        df["pub_date"] = pd.to_datetime(df["pub_date"])
        df["date"] = df["pub_date"].dt.date()

        # Exclude news with rating 3
        df_exclude = df.loc[df[self.model_name] != 3, :]

        # Get Pandas Series of mean ratings (with and without rating 3)
        series_incl = df.groupby(by=["date"])[self.model_name].mean()
        series_excl = df_exclude.groupby(by=["date"])[self.model_name].mean()

        # Generate DataFrame by concatenating 'series_incl' and 'series_excl'
        df_av = pd.concat([series_incl, series_excl], axis=1)
        df_av.columns = ["av_rating", "av_rating_excl"]
        df_av.insert(0, "ticker", [ticker] * len(df_av))

        return df_av

    def append_close(self, data: pd.DataFrame) -> pd.DataFrame:
        """Append closing price of stock and co-integrated stock with lowest p-value."""

        df = data.copy()
        ticker = df["ticker"].unique()[0]

        # Load cointegration CSV file
        df_coint = pd.read_csv()
