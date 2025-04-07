"""Create class for different trading strategy while adhering to SOLID principle:

1. Abstract class for entry and exit i.e. 'EntrySignal', 'ProfitSignal', 'StopSignal'.
2. 'TradingStrategy' class to use composition rather than inherit from abstract class.

Note that:
- Append 'entry_signal', 'profit_signal' and 'stop_signal' columns to DataFrame
containing prices and information required to generate buy/sell signal such as TA,
sentiment rating, etc.
"""

from abc import ABC, abstractmethod
from datetime import date
from decimal import Decimal
from typing import Any, Optional, get_args

import pandas as pd
from pydantic import BaseModel, Field, computed_field, field_validator

from config.variables import EntryStruct, EntryType, ProfitStruct
from src.utils import utils


class StockTrade(BaseModel):
    ticker: str = Field(description="Stock ticker")
    coint_ticker: str = Field(description="Cointegrated stock ticker.")
    action: str = Field(description="Either 'buy' or 'sell'", default="buy")
    entry_date: date = Field(description="Date when opening long position")
    entry_price: Decimal = Field(description="Price when opening long position")
    exit_date: Optional[date] = Field(
        description="Date when exiting long position", default=None
    )
    exit_price: Optional[Decimal] = Field(
        description="Price when exiting long position", default=None
    )

    @computed_field(description="Number of days held for trade")
    def days_held(self) -> Optional[int]:
        if self.exit_date is not None and self.entry_date is not None:
            days_held = self.exit_date - self.entry_date
            return days_held.days
        return

    @computed_field(description="Profit/loss when trade completed")
    def profit_loss(self) -> Optional[Decimal]:
        if self.exit_price is not None and self.entry_price is not None:
            profit_loss = self.exit_price - self.entry_price
            return profit_loss
        return

    @computed_field(description="Percentage return of trade")
    def percent_ret(self) -> Optional[Decimal]:
        if self.exit_price is not None and self.entry_price is not None:
            percent_ret = (self.exit_price - self.entry_price) / self.entry_price
            return percent_ret.quantize(Decimal("1.000000"))
        return

    @computed_field(description="daily percentage return of trade")
    def daily_ret(self) -> Optional[Decimal]:
        if self.percent_ret is not None and self.days_held is not None:
            daily_ret = (1 + self.percent_ret) ** (1 / Decimal(str(self.days_held))) - 1
            return daily_ret.quantize(Decimal("1.000000"))
        return

    @computed_field(description="Whether trade is profitable")
    def win(self) -> Optional[int]:
        if (pl := self.percent_ret) is not None:
            return int(pl > 0)
        return

    model_config = {"validate_assignment": True}

    @field_validator("exit_date")
    def validate_exit_date(
        cls, exit_date: Optional[date], info: dict[str, Any]
    ) -> Optional[date]:
        if exit_date and (entry_date := info.data.get("entry_date")):
            if exit_date < entry_date:
                raise ValueError("Exit date must be after entry date!")
        return exit_date


class TradeSignal(ABC):
    """Abstract base class to generate entry and exit trade signal.

    Args:
        entry_type (EntryType):
            Whether to allow long ("long_only"), short ("short_only") or
            both long and short position ("long_or_short").

    Attributes:
        entry_type (EntryType):
            Whether to allow long ("long_only"), short ("short_only") or
            both long and short position ("long_or_short").
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

        - 'long_only' -> only 'buy' or 'wait' signal allowed.
        - 'short_only' -> only 'sell' or 'wait' signal allowed.
        - 'long_or_short' -> 'buy', 'sell', or 'wait' signal allowed.
        """

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


class ExitSignal(TradeSignal, ABC):
    """Abstract class to generate take profit signal to execute
    i.e. exit existing position."""

    @abstractmethod
    def gen_exit_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        """Append 'profit_signal' (i.e. 'buy', 'sell', or 'wait')
        column to DataFrame containing prices and any info required to generate
        entry signal.

        - 'long_only' -> only 'sell' or 'wait' exit signal allowed.
        - 'short_only' -> only 'buy' or 'wait' exit signal allowed.
        - 'long_or_short' -> 'buy', 'sell', or 'wait' exit signal allowed.
        """

        pass

    def _validate_exit_signal(self, df: pd.DataFrame) -> None:
        """Ensure that entry action is aligned with 'entry_type'."""
        if "exit_signal" not in df.columns:
            raise ValueError(f"'exit_signal' column not found!")

        if self.entry_type == "long_only" and (df["exit_signal"] == "buy").any():
            raise ValueError("Long only strategy cannot generate buy exit signals.")

        if self.entry_type == "short_only" and (df["exit_signal"] == "buy").any():
            raise ValueError("Short only strategy cannot generate sell exit signals.")


class GetTrades(ABC):
    """Abstract class to generate completed trades for given strategy.

    Args:
        entry_struct (EntryStruct):
            Whether to allow multiple open position ("mulitple") or single
            open position at a time ("single").
        profit_struct (ProfitStruct):
            Whether to apply first-in-first-out ("fifo"), last-in-first-out ("lifo"),
            take profit for half open positions repeatedly ("half_life") or
            take profit for all open positions ("take_all").
        num_lots (int):
            Number of lots to initiate new position each time (Default: 1).

    Attributes:
        entry_struct (EntryStruct):
            Whether to allow multiple open position ("mulitple") or single
            open position at a time ("single").
        profit_struct (ProfitStruct):
            Whether to apply first-in-first-out ("fifo"), last-in-first-out ("lifo"),
            take profit for half open positions repeatedly ("half_life") or
            take profit for all open positions ("take_all").
        num_lots (int):
            Number of lots to initiate new position each time (Default: 1).
    """

    def __init__(
        self,
        entry_struct: EntryStruct = "multiple",
        profit_struct: ProfitStruct = "take_all",
        num_lots: int = 1,
    ) -> None:
        self.entry_struct = entry_struct
        self.profit_struct = profit_struct
        self.num_lots = num_lots

    @abstractmethod
    def gen_trades(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate DataFrame containing completed trades for given strategy"""

        pass


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
            Class instance of concrete implementation of 'EntrySignal' abstract class.
        trades (GetTrades):
            Class instance of concrete implementation of 'GetTrades' abstract class.
        exit (ExitSignal):
            If provided, Class instance of concrete implementation of 'ExitSignal'
            abstract class. If None, standard profit and stop loss will be applied via
            'gen_trades'.
        num_lots (int):
            Number of lots to open new position (Default: 1).

    Attributes:
        entry_type (EntryType):
            Types of open positions allowed either 'long_only', 'short_only' or
            'long_or_short'.
        entry (EntrySignal):
            Class instance of concrete implementation of 'EntrySignal' abstract class.
        trades (GetTrades):
            Class instance of concrete implementation of 'GetTrades' abstract class.
        exit (ExitSignal | None):
            If provided, Class instance of concrete implementation of
            'ExitSignal' abstract class.
    """

    def __init__(
        self,
        entry_type: EntryType,
        entry: type[EntrySignal],
        trades: type[GetTrades],
        exit: type[ExitSignal] | None = None,
        num_lots: int = 1,
    ) -> None:
        self.entry_type = entry_type
        self.entry = entry(entry_type)
        self.gen_trades = trades(num_lots)
        self.exit = None if exit is None else exit(entry_type)
        self.open_trades = []

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate completed trades based on trading strategy i.e.
        combination of entry, profit exit and stop exit."""

        # Append 'entry_s ignal' column
        df = self.entry.gen_entry_signal(df)

        if self.exit is not None:
            # Append 'exit_signal' and 'exit_type' if 'self.exit' exist
            df = self.exit.gen_exit_signal(df)

        # Generate trades
        df_trades = self.gen_trades.gen_trades(df)

        return df_trades
