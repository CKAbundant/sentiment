"""Abstract classes for different aspect of trading strategy:

- 'EntrySignal' -> Generate signal to initiate new position.
- 'ExitSignal' -> Generate signal to exit position with either profit or loss.
- 'GenTrades' -> Generate completed trades based on entry and exit signal.
"""

from abc import ABC, abstractmethod
from typing import get_args

import pandas as pd

from config.variables import EntryType


class TradeSignal(ABC):
    """Abstract base class to generate entry and exit trade signal.

    Args:
        entry_type (EntryType):
            Whether to allow long ("long"), short ("short") or
            both long and short position ("longshort").

    Attributes:
        entry_type (EntryType):
            Whether to allow long ("long"), short ("short") or
            both long and short position ("longshort").
    """

    def __init__(self, entry_type: EntryType) -> None:
        self.entry_type = self._validate_entry_type(entry_type)

    def _validate_entry_type(self, entry_type: EntryType) -> EntryType:
        if entry_type not in get_args(EntryType):
            raise ValueError(f"'{entry_type}' is not a valid 'EntryType'.")

        return entry_type


class EntrySignal(TradeSignal, ABC):
    """Abstract class to generate entry signal and number of lots to execute to
    initiate new position"""

    @abstractmethod
    def gen_entry_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        """Append 'entry_signal' (i.e. 'buy', 'sell', or 'wait')
        column to DataFrame containing prices and any info required to generate
        entry signal.

        - 'long' -> only 'buy' or 'wait' signal allowed.
        - 'short' -> only 'sell' or 'wait' signal allowed.
        - 'longshort' -> 'buy', 'sell', or 'wait' signal allowed.
        """

        pass

    def _validate_entry_signal(self, df: pd.DataFrame) -> None:
        """Ensure that entry action is aligned with 'entry_type'."""
        if "entry_signal" not in df.columns:
            raise ValueError(f"'entry_signal' column not found!")

        if self.entry_type == "long" and (df["entry_signal"] == "sell").any():
            raise ValueError("Long only strategy cannot generate sell entry signals")

        if self.entry_type == "short" and (df["entry_signal"] == "buy").any():
            raise ValueError("Short only strategy cannot generate buy entry signals")


class ExitSignal(TradeSignal, ABC):
    """Abstract class to generate take profit signal to execute
    i.e. exit existing position."""

    @abstractmethod
    def gen_exit_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        """Append 'profit_signal' (i.e. 'buy', 'sell', or 'wait')
        column to DataFrame containing prices and any info required to generate
        entry signal.

        - 'long' -> only 'sell' or 'wait' exit signal allowed.
        - 'short' -> only 'buy' or 'wait' exit signal allowed.
        - 'longshort' -> 'buy', 'sell', or 'wait' exit signal allowed.
        """

        pass

    def _validate_exit_signal(self, df: pd.DataFrame) -> None:
        """Ensure that entry action is aligned with 'entry_type'."""
        if "exit_signal" not in df.columns:
            raise ValueError(f"'exit_signal' column not found!")

        if self.entry_type == "long" and (df["exit_signal"] == "buy").any():
            raise ValueError("Long only strategy cannot generate buy exit signals.")

        if self.entry_type == "short" and (df["exit_signal"] == "sell").any():
            raise ValueError("Short only strategy cannot generate sell exit signals.")
