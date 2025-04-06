"""Create class for different trading strategy while adhering to SOLID principle:

1. Abstract class for entry and exit i.e. 'EntrySignal', 'ProfitSignal', 'StopSignal'.
2. 'TradingStrategy' class to use composition rather than inherit from abstract class.

Note that:
- Append 'entry_signal', 'profit_signal' and 'stop_signal' columns to DataFrame
containing prices and information required to generate buy/sell signal such as TA,
sentiment rating, etc.
"""

from abc import ABC, abstractmethod
from typing import get_args

import pandas as pd

from config.variables import EntryType
from src.utils import utils


class TradeSignal(ABC):
    """Abstract base class to generate entry and exit trade signal."""

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
        """Append 'entry_signal' (i.e. 'buy', 'sell', or 'wait') and 'entry_lots'
        columns to DataFrame containing prices and any info required to generate
        entry signal."""

        pass

    def _validate_entry_signal(self, df: pd.DataFrame) -> None:
        """Ensure that entry action is aligned with 'entry_type'."""
        if "entry_signal" not in df.columns:
            raise ValueError(f"'entry_signal' column not found!")

        if "entry_lots" not in df.columns:
            raise ValueError(f"'entry_lots' column not found!")

        if self.entry_type == "long_only" and (df["entry_signal"] == "sell").any():
            raise ValueError("Long only strategy cannot generate sell entry signals")

        if self.entry_type == "short_only" and (df["entry_signal"] == "buy").any():
            raise ValueError("Short only strategy cannot generate buy entry signals")


class ProfitExitSignal(ABC):
    """Abstract class to generate take profit signal and number of lots to execute
    i.e. exit existing position with profit."""

    def __init_(self, entry_type: EntryType) -> None:
        self.entry_type = entry_type

    @abstractmethod
    def gen_profit_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        """Append 'profit_signal' (i.e. 'buy', 'sell', or 'wait') and 'profit_lots'
        columns to DataFrame containing prices and any info required to generate
        entry signal."""

        pass

    def _validate_exit_signal(self, df: pd.DataFrame) -> None:
        """Ensure that entry action is aligned with 'entry_type'."""
        if "profit_signal" not in df.columns:
            raise ValueError(f"'profit_signal' column not found!")

        if "profit_lots" not in df.columns:
            raise ValueError(f"'profit_lots' column not found!")

        if self.entry_type == "long_only" and (df["profit_signal"] == "buy").any():
            raise ValueError("Long only strategy cannot generate buy exit signals.")

        if self.entry_type == "short_only" and (df["profit_signal"] == "buy").any():
            raise ValueError("Short only strategy cannot generate sell exit signals.")


class StopExitSignal(ABC):
    """Abstract class to generate stop loss signal and number of lots to execute
    i.e. exit existing position with loss."""

    def __init_(self, entry_type: EntryType) -> None:
        self.entry_type = entry_type

    @abstractmethod
    def gen_stop_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        """Append 'stop_signal' (i.e. 'buy', 'sell', or 'wait') and 'stop_lots'
        columns to DataFrame containing prices and any info required to generate
        entry signal."""

        pass

    def _validate_exit_signal(self, df: pd.DataFrame) -> None:
        """Ensure that entry action is aligned with 'entry_type'."""
        if "stop_signal" not in df.columns:
            raise ValueError(f"'profit_signal' column is not found!")

        if "stop_lots" not in df.columns:
            raise ValueError(f"'stop_lots' column not found!")

        if self.entry_type == "long_only" and (df["stop_signal"] == "buy").any():
            raise ValueError("Long only strategy cannot generate buy exit signals.")

        if self.entry_type == "short_only" and (df["stop_signal"] == "buy").any():
            raise ValueError("Short only strategy cannot generate sell exit signals.")


class TradingStrategy:
    """Combine entry, profit and stop loss strategy as a complete trading strategy.

    Usage:
        >>> strategy = TradingStrategy(
                entry=SentimentRater
                profit_exit=SentimentRater
                stop_exit = SentimentorMaxDrawDown
                entry_type = "long_only"
            )
        >>> strategy.run()

    Args:
        entry_type (EntryType):
            Types of open positions allowed either 'long_only', 'short_only' or
            'long_or_short'.
        entry (EntrySignal):
            Instance of concrete implementation of 'EntrySignal' abstract class.
        profit_exit (ProfitExitSignal):
            Instance of concrete implementation of 'ProfitExitSignal' abstract class.
        stop_exit (StopExitSignal):
            Instance of concrete implementation of 'StopExitSignal' abstract class.

    Attributes:
        entry_type (EntryType):
            Types of open positions allowed either 'long_only', 'short_only' or
            'long_or_short'.
        entry (EntrySignal):
            Instance of concrete implementation of 'EntrySignal' abstract class.
        profit_exit (ProfitExitSignal):
            Instance of concrete implementation of 'ProfitExitSignal' abstract class.
        stop_exit (StopExitSignal):
            Instance of concrete implementation of 'StopExitSignal' abstract class.

    """

    def __init__(
        self,
        entry_type: EntryType,
        entry: type[EntrySignal],
        profit_exit: type[ProfitExitSignal],
        stop_exit: type[StopExitSignal],
        num_lots: int,
    ) -> None:
        self.entry_type = entry_type
        self.entry = entry(entry_type)
        self.profit_exit = profit_exit(entry_type)
        self.stop_exit = stop_exit(entry_type)
        self.num_lots = num_lots

    def append_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Append entry, profit, stop loss and required number of lots
        to DataFrame."""

        df = self.entry.gen_entry_signal(df)
        df = self.profit_exit.gen_profit_signal(df)
        df = self.stop_exit.gen_profit_signal(df)

        return df
