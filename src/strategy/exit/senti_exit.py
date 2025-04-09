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
        self.req_cols = [f"{coint_corr_ticker}_close", rating_col, "entry_signal"]

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

        if any(req_col not in df.columns for req_col in self.req_cols):
            raise ValueError(f"Missing required columns : {self.req_cols}")

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

        self._validate_exit_signal()

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


class SentiExitDrawdown(base.ExitSignal):
    """Use daily median sentiment rating for stock ticker news and percentage
    drawdownto execute exit signal for cointegrated/correlated stock.

    Args:
        entry_type (EntryType):
            Either "long_only", "short_only", "long_or_short".
        coint_corr_ticker (str):
            Ticker for cointegrated/correlated ticker to news ticker.
        rating_col (str):
            Name of column containing sentiment rating to generate price action.
        percent_drawdown (float):
            Percentage drawdown before triggering stop loss (Default: 0.2).

    Attributes:
        entry_type (EntryType):
            Either "long_only", "short_only", "long_or_short".
        coint_corr_ticker (str):
            Ticker for cointegrated/correlated ticker to news ticker.
        rating_col (str):
            Name of column containing sentiment rating to generate price action.
        req_cols (list[str]):
            List of columns that is required by the strategy.
        percent_drawdown (float):
            Percentage drawdown before triggering stop loss (Default: 0.2).
    """

    def __init__(
        self,
        entry_type: EntryType,
        coint_corr_ticker: str,
        rating_col: str = "median_rating_excl",
        percent_drawdown: float = 0.2,
        drawdown_type: str = "mean",
    ) -> None:
        super().__init__(entry_type)
        self.coint_corr_ticker = coint_corr_ticker
        self.rating_col = rating_col
        self.req_cols = ["close", rating_col, "entry_signal"]
        self.percent_drawdown = percent_drawdown
        self.drawdown_type = drawdown_type

    def gen_exit_signal(self, df_senti: pd.DataFrame) -> pd.DataFrame:
        """Append exit signal (i.e. 'buy', 'sell', 'wait') to DataFrame based
        on 'entry_type'.

        - rating <= 2 or drawdown hit based on close -> 'sell' to close long position.
        - rating >= 4 or drawdown hit based on close -> 'buy' to close short position.

        Args:
            df_senti (pd.DataFrame):
                DataFrame containing median daily rating (i.e. 'median_rating_excl'
                column) and closing price of cointegrated/correlated stock.

        Returns:
            df (pd.DataFrame):
                DataFrame with 'exit_signal' column appended.
        """

        # Filter out null values for OHLC due to weekends and holiday
        df = df_senti.loc[~df_senti["close"].isna(), self.req_cols].copy()

        long_stop_prices = []
        short_stop_prices = []
        exit_action = []

        for close, rating, ent_sig in df.itertuples(index=False, name=None):
            # For long position
            if ent_sig == "buy":
                # Compute stop price
                long_stop_prices.append(close * (1 - self.percent_drawdown))

            # For short position
            elif ent_sig == "sell":
                # Compute stop price
                short_stop_prices.append(close * (1 + self.percent_drawdown))

            max_long_stop_price = np.max(long_stop_prices) if long_stop_prices else 0
            max_short_stop_price = (
                np.max(short_stop_prices) if av_short_stop_prices else 0
            )

            # stop loss hit or rating <= 2
            if (long_stop_prices and close <= max_long_stop_price) or rating <= 2:
                exit_action.append("sell")

                # Reset stop prices
                long_stop_prices = []
                av_long_stop_price = 0

            # stop loss hit or rating >= 4
            elif (short_stop_prices and close >= max_short_stop_price) or rating >= 4:
                exit_action.append("buy")

                # Reset stop prices
                short_stop_prices = []
                av_short_stop_prices = 0

            else:
                exit_action.append("wait")

        df["exit_signal"] = exit_action

        return df
