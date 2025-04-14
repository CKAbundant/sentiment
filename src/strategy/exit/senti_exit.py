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
            Either "long_only", "short_only", "long_or_short".
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

        # Ensure 'median_rating_excl' is integer type
        df[self.rating_col] = df[self.rating_col].astype(int)

        if self.entry_type == "long_only":
            # Ensure there is at least a 'buy' in df['entry_signal']
            if not (df["entry_signal"] == "buy").any():
                raise ValueError(
                    f"At least 1 open long position required to initiate sell."
                )

            df["exit_signal"] = df[self.rating_col].map(self._gen_exit_long_signal)

        elif self.entry_type == "short_only":
            # Ensure there is at least a 'buy' in df['entry_signal']
            if not (df["entry_signal"] == "sell").any():
                raise ValueError(
                    f"At least 1 open short position required to initiate buy."
                )

            df["exit_signal"] = df[self.rating_col].map(self._gen_exit_short_signal)

        else:
            # Ensure there is at least a 'buy' in df['entry_signal']
            if not (df["entry_signal"].isin(["buy", "sell"])).any():
                raise ValueError(f"At least 1 open long or short position required.")

            df["exit_signal"] = df[self.rating_col].map(
                self._gen_exit_long_short_signal
            )

        self._validate_exit_signal(df)

        return df

    def _gen_exit_long_signal(self, rating: int) -> str:
        """Generate sell signal to close long position if rating <= 4"""

        return "sell" if rating <= 2 else "wait"

    def _gen_exit_short_signal(self, rating: int) -> str:
        """Generate buy signal to close short position if rating <= 2"""

        return "buy" if rating >= 4 else "wait"

    def _gen_exit_long_short_signal(self, rating: int) -> str:
        """Generate buy signal if rating >= 4 or sell signal if rating <= 2."""

        if rating >= 4:
            return "buy"

        if rating <= 2:
            return "sell"

        return "wait"
