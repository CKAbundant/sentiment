"""Concrete implementation of 'EntrySignal' abstract class."""

import pandas as pd

from config.variables import EntryType
from src.strategy import base


class SentiEntry(base.EntrySignal):
    """Use daily median sentiment rating for stock ticker news to execute
    entry signal for cointegrated/correlated stock.

    Args:
        entry_type (EntryType):
            Either "long_only", "short_only", "long_or_short".
        coint_corr_ticker (str):
            Ticker for cointegrated/correlated ticker to news ticker.
        rating_col (str):
            Name of column containing sentiment rating to generate price action.

    Attributes:
        entry_type (EntryType):
            Either "long_only", "short_only", "long_or_short".
        coint_corr_ticker (str):
            Ticker for cointegrated/correlated ticker to news ticker.
        rating_col (str):
            Name of column containing sentiment rating to generate price action.
        req_cols (list[str]):
            List of columns that is required by the strategy.

    """

    def __init__(
        self,
        entry_type: EntryType,
        coint_corr_ticker: str,
        rating_col: str = "median_rating_excl",
    ) -> None:
        super().__init__(entry_type)
        self.coint_corr_ticker = coint_corr_ticker
        self.rating_col = rating_col
        self.req_cols = [f"{coint_corr_ticker}_close", rating_col]

    def gen_entry_signal(self, df_senti: pd.DataFrame) -> pd.DataFrame:
        """Append entry signal (i.e. 'buy', 'sell', 'wait') to DataFrame based
        on 'entry_type'.

        Args:
            df_senti (pd.DataFrame):
                DataFrame containing median daily rating (i.e. 'median_rating_excl'
                column) and closing price of cointegrated/correlated stock.

        Returns:
            df (pd.DataFrame):
                DataFrame with 'entry_signal' column appended.
        """

        df = df_senti.copy()

        if any(req_col not in df.columns for req_col in self.req_cols):
            raise ValueError(f"Missing required columns : {self.req_cols}")

        # Ensure 'median_rating_excl' is integer type
        df["median_rating_excl"] = df["median_rating_excl"].astype(int)

        if self.entry_type == "long_only":
            df["entry_signal"] = df["median_rating_excl"].map(self._gen_long_signal)

        elif self.entry_type == "short_only":
            df["entry_signal"] = df["median_rating_excl"].map(self._gen_short_signal)

        else:
            df["entry_signal"] = df["median_rating_excl"].map(
                self._gen_long_short_signal
            )

        self._validate_entry_signal(df)

        return df

    def _gen_long_signal(self, rating: int) -> str:
        """Generate buy signal if rating >= 4"""

        return "buy" if rating >= 4 else "wait"

    def _gen_short_signal(self, rating: int) -> str:
        """Generate sell signal if rating <= 2"""

        return "sell" if rating <= 2 else "wait"

    def _gen_long_short_signal(self, rating: int) -> str:
        """Generate buy signal if rating >= 4 or sell signal if rating <= 2."""

        if rating >= 4:
            return "buy"

        if rating <= 2:
            return "sell"

        return "wait"
