"""Concrete implementation of 'EntrySignal' abstract class."""

import pandas as pd

from config.variables import EntryType
from src.strategy import base


class SentiEntry(base.EntrySignal):
    """Use daily median sentiment rating for stock ticker news to execute
    entry signal for cointegrated/correlated stock.

    Args:
        entry_type (EntryType):
            Either "long", "short", "longshort".
        rating_col (str):
            Name of column containing sentiment rating to generate price action.

    Attributes:
        rating_col (str):
            Name of column containing sentiment rating to generate price action
    """

    def __init__(
        self,
        entry_type: EntryType,
        rating_col: str = "median_rating_excl",
    ) -> None:
        super().__init__(entry_type)
        self.rating_col = rating_col

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

        if self.rating_col not in df.columns:
            raise ValueError(f"'{self.rating_col}' column is not available!")

        if self.entry_type == "long":
            df["entry_signal"] = df[self.rating_col].map(self._gen_long_signal)

        elif self.entry_type == "short":
            df["entry_signal"] = df[self.rating_col].map(self._gen_short_signal)

        else:
            df["entry_signal"] = df[self.rating_col].map(self._gen_long_short_signal)

        self._validate_entry_signal(df)

        return df

    def _gen_long_signal(self, rating: int) -> str:
        """Generate buy signal if rating >= 4"""

        return "buy" if rating and rating >= 4 else "wait"

    def _gen_short_signal(self, rating: int) -> str:
        """Generate sell signal if rating <= 2"""

        return "sell" if rating and rating <= 2 else "wait"

    def _gen_long_short_signal(self, rating: int) -> str:
        """Generate buy signal if rating >= 4 or sell signal if rating <= 2."""

        if rating and rating >= 4:
            return "buy"

        if rating and rating <= 2:
            return "sell"

        return "wait"
