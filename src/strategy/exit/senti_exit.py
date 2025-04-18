"""Concrete implementation of 'ProfitExitSignal' abstract class."""

import numpy as np
import pandas as pd

from config.variables import EntryType
from src.strategy import base


class SentiExit(base.ExitSignal):
    """Use daily median sentiment rating for stock ticker news to execute
    exit signal for cointegrated/correlated stock.

    Args:
        entry_type (EntryType):
            Either "long", "short", "longshort".
        rating_col (str):
            Name of column containing sentiment rating to generate price action.

    Attributes:
        rating_col (str):
            Name of column containing sentiment rating to generate price action.
    """

    def __init__(
        self,
        entry_type: EntryType,
        rating_col: str = "median_rating_excl",
    ) -> None:
        super().__init__(entry_type)
        self.rating_col = rating_col

    def gen_exit_signal(self, df_senti: pd.DataFrame) -> pd.DataFrame:
        """Append exit signal (i.e. 'buy', 'sell', 'wait') to DataFrame based
        on 'entry_type'.

        Args:
            df_senti (pd.DataFrame):
                DataFrame containing median daily rating (i.e. 'median_rating_excl'
                column) and closing price of cointegrated/correlated stock.

        Returns:
            df (pd.DataFrame):
                DataFrame with 'exit_signal' column appended.
        """

        df = df_senti.copy()

        if any(col not in df.columns for col in [self.rating_col, "entry_signal"]):
            raise ValueError(
                f"'{self.rating_col}' or 'entry_signal' columns are not available!"
            )

        if self.entry_type == "long":
            df["exit_signal"] = df[self.rating_col].map(self._gen_exit_long_signal)

        elif self.entry_type == "short":
            df["exit_signal"] = df[self.rating_col].map(self._gen_exit_short_signal)

        else:
            df["exit_signal"] = df[self.rating_col].map(
                self._gen_exit_long_short_signal
            )

        self._validate_exit_signal(df)

        return df

    def _gen_exit_long_signal(self, rating: int) -> str:
        """Generate sell signal to close long position if rating <= 4"""

        return "sell" if rating and rating <= 2 else "wait"

    def _gen_exit_short_signal(self, rating: int) -> str:
        """Generate buy signal to close short position if rating <= 2"""

        return "buy" if rating and rating >= 4 else "wait"

    def _gen_exit_long_short_signal(self, rating: int) -> str:
        """Generate buy signal if rating >= 4 or sell signal if rating <= 2."""

        if rating and rating >= 4:
            return "buy"

        if rating and rating <= 2:
            return "sell"

        return "wait"
