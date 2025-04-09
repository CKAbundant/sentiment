"""Create class for different trading strategy while adhering to SOLID principle:

1. Abstract class for entry and exit i.e. 'EntrySignal', 'ProfitSignal', 'StopSignal'.
2. 'TradingStrategy' class to use composition rather than inherit from abstract class.

Note that:
- Append 'entry_signal', 'profit_signal' and 'stop_signal' columns to DataFrame
containing prices and information required to generate buy/sell signal such as TA,
sentiment rating, etc.
"""

import math
from abc import ABC, abstractmethod
from collections import deque
from datetime import date
from decimal import Decimal
from typing import Any, Optional, get_args

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, computed_field, field_validator

from config.variables import EntryStruct, EntryType, ExitStruct, FixedPL, PriceAction
from src.utils import utils


class StockTrade(BaseModel):
    ticker: str = Field(description="Stock ticker to be traded")
    entry_date: date = Field(description="Date when opening long position")
    entry_action: str = Field(description="Either 'buy' or 'sell'", default="buy")
    entry_lots: Decimal = Field(
        description="Number of lots to open new open position", default=Decimal("1")
    )
    entry_price: Decimal = Field(description="Price when opening long position")
    exit_date: Optional[date] = Field(
        description="Date when exiting long position", default=None
    )
    exit_action: Optional[str] = Field(
        description="Opposite of 'entry_action'", default="sell"
    )
    exit_lots: Optional[Decimal] = Field(
        description="Number of lots to close open position"
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
        # Get entry_date from StockTrade object
        entry_date = info.data.get("entry_date")

        if exit_date is not None and entry_date is not None:
            if exit_date < entry_date:
                raise ValueError("Exit date must be after entry date!")
        return exit_date

    @field_validator("exit_lots")
    def validate_exit_lots(
        cls, exit_lots: Optional[Decimal], info: dict[str, Any]
    ) -> Optional[Decimal]:
        # Get entry_lots from StockTrade object
        entry_lots = info.data.get("entry_lots")

        if exit_lots is not None and entry_lots is not None:
            if exit_lots > entry_lots:
                raise ValueError("Exit lots must be equal or less than entry lots.")

            if exit_lots < 0 or entry_lots < 0:
                raise ValueError(f"Entry lots and exit lots must be positive.")

        return exit_lots


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
            Whether to allow multiple open position ("multiple") or single
            open position at a time ("single") (Default: "multiple").
        exit_struct (ExitStruct):
            Whether to apply first-in-first-out ("fifo"), last-in-first-out ("lifo"),
            take profit for half open positions repeatedly ("half_life") or
            take profit for all open positions ("take_all") (Default: "take_all").
        num_lots (int):
            Number of lots to initiate new position each time (Default: 1).

    Attributes:
        entry_struct (EntryStruct):
            Whether to allow multiple open position ("mulitple") or single
            open position at a time ("single").
        exit_struct (ExitStruct):
            Whether to apply first-in-first-out ("fifo"), last-in-first-out ("lifo"),
            take profit for half open positions repeatedly ("half_life") or
            take profit for all open positions ("take_all").
        num_lots (int):
            Number of lots to initiate new position each time (Default: 1).
        net_pos (int):
            Net position for stock ticker (Default: 0).
        open_trades (deque[StockTrade]):
            List of open trades containing StockTrade pydantic object.
        completed_trades (list[StockTrade]):
            List of completed trades i.e. completed StockTrade pydantic object.

    """

    def __init__(
        self,
        entry_struct: EntryStruct = "multiple",
        exit_struct: ExitStruct = "take_all",
        num_lots: int = 1,
    ) -> None:
        self.entry_struct = utils.validate_literal(
            entry_struct, EntryStruct, "EntryStruct"
        )
        self.exit_struct = utils.validate_literal(exit_struct, ExitStruct, "ExitStruct")
        self.num_lots = num_lots
        self.net_pos: int = 0
        self.open_trades = deque()
        self.completed_trades = []

    @abstractmethod
    def gen_trades(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate DataFrame containing completed trades for given strategy"""

        pass

    def open_new_pos(
        self, dt: date, ticker: str, entry_price: float, ent_sig: PriceAction
    ) -> None:
        """Open new position by creating new StockTrade object.

        - If entry_struct == "multiple", multiple open positions are allowed.
        - If entry_struct == "single", new open position can only be initiated after existing position is closed.

        Args:
            dt (date):
                Trade date object.
            ticker (str):
                Stock ticker to be traded.
            entry_price (float):
                Entry price for stock ticker.
            ent_sig (PriceAction):
                Entry signal i.e. "buy", "sell" or "wait" to create new position.

        Returns:
            None.
        """

        if (
            self.entry_struct == "single" and self.net_pos == 0
        ) or self.entry_struct == "multiple":
            # Create StockTrade object to record new long/short position
            # based on 'ent_sig'
            stock_trade = StockTrade(
                ticker=ticker,
                entry_date=dt,
                entry_action=ent_sig,
                entry_lots=Decimal(str(self.num_lots)),
                entry_price=Decimal(str(entry_price)),
            )
            self.open_trades.append(stock_trade)
            self.net_pos += self.num_lots if ent_sig == "buy" else -self.num_lots

    def close_pos_with_profit(
        self,
        dt: date,
        exit_price: float,
        ex_sig: PriceAction,
    ) -> None:
        """Close existing position by updating StockTrade object in 'self.open_trades'.

        - 'fifo' -> First-in-First-out.
        - 'lifo' -> Last-in-Last-out.
        - 'half_fifo' -> Reduce open position by half via First-in-First-out.
        - 'half_lifo' -> Reduce open position by half via Last-in-Last-out.
        - 'take_all' -> Exit all open positions.

        Args:
            dt (date):
                Trade date object.
            exit_price (float):
                Exit price for stock ticker.
            ex_sig (PriceAction):
                Exit signal i.e. "buy", "sell" or "wait" to close existing position.

        Returns:
            None.
        """

        match self.exit_struct:
            case "fifo":
                self.completed_trades.extend(
                    self.update_via_fifo_or_lifo(dt, ex_sig, exit_price, "fifo")
                )
            case "lifo":
                self.completed_trades.extend(
                    self.update_via_fifo_or_lifo(dt, ex_sig, exit_price, "lifo")
                )
            case "half_fifo":
                self.completed_trades.extend(
                    self.update_via_half_fifo(dt, ex_sig, exit_price)
                )
            case "half_lifo":
                self.completed_trades.extend(
                    self.update_via_half_fifo(dt, ex_sig, exit_price)
                )
            case _:
                self.completed_trades.extend(
                    self.update_via_take_all(dt, ex_sig, exit_price)
                )

    def update_via_take_all(
        self, dt: date, ex_sig: PriceAction, exit_price: float
    ) -> list[dict[str, Any]]:
        """Update existing StockTrade objects and remove from self.open_trades.

        Args:
            dt (date):
                datetime.date object of trade date.
            ex_sig (PriceAction):
                Action to close open position either "buy" or "sell".
            exit_price (float):
                Exit price of stock ticker.

        Returns:
            (list[dict[str, Any]]):
                List of dictionary containing required fields to generate DataFrame.
        """

        updated_trades = []

        for trade in self.open_trades:
            # Update StockTrade objects with exit info
            trade.exit_date = dt
            trade.exit_action = ex_sig
            trade.exit_lots = trade.entry_lots  # Ensure entry lots matches exit lots
            trade.exit_price = Decimal(str(exit_price))

            # Convert StockTrade to dictionary only if all fields are populated
            # i.e. trade completed.
            if self._validate_completed_trades(trade):
                updated_trades.append(trade.model_dump())

        # Reset self.open_trade and self.net_pos
        if len(updated_trades) == len(self.open_trades):
            self.open_trades.clear()
            self.net_pos = 0

        return updated_trades

    def update_via_fifo_or_lifo(
        self, dt: date, ex_sig: PriceAction, exit_price: float, fifo_or_lifo: str
    ) -> list[dict[str, Any]]:
        """Update earliest entry to 'self.open_trades'.

        Args:
            dt (date):
                datetime.date object of trade date.
            ex_sig (PriceAction):
                Action to close open position either "buy" or "sell".
            exit_price (float):
                Exit price for stock ticker.
            fifo_or_lifo (str):
                Either "fifo" or "lifo".

        Returns:
            updated_trades (list[dict[str, Any]]):
                Empty list or List containing dictionary, which contain required fields
                to generate DataFrame.
        """

        updated_trades = []

        if fifo_or_lifo == "fifo":
            # Remove earliest StockTrade object from non-empty queue
            trade = self.open_trades.popleft()

        else:
            # Remove latest StockTrade object from non-empty queue
            trade = self.open_trades.pop()

        # Update StockTrade object with exit info
        trade.exit_date = dt
        trade.exit_action = ex_sig
        trade.exit_lots = trade.entry_lots
        trade.exit_price = Decimal(str(exit_price))

        # Convert StockTrade to dictionary only if all fields are populated
        # i.e. trade completed.
        if not self._validate_completed_trades(trade):
            updated_trades.append(trade.model_dump())

        # Update self.net_pos
        self.net_pos += trade.exit_lots if ex_sig == "buy" else -trade.exit_lots

        return updated_trades

    def update_via_half_fifo(
        self, dt: date, ex_sig: PriceAction, exit_price: float
    ) -> list[dict[str, Any]]:
        """Update existing StockTrade objects by reducing earliest trade by half.

        Args:
            dt (date):
                datetime.date object of trade date.
            ex_sig (PriceAction):
                Action to close open position either "buy" or "sell".
            exit_price (float):
                Exit price of stock ticker.

        Returns:
            (list[dict[str, Any]]):
                List of dictionary containing required fields to generate DataFrame.
        """

        updated_trades = []
        updated_open_trades = deque()

        # Half of net position
        half_pos = math.ceil(abs(self.net_pos) / 2)

        for trade in self.open_trades:
            # Determine quantity to close based on 'half_pos'
            lots_to_exit, net_exit_lots = self._cal_exit_lots(
                half_pos, trade.entry_lots, trade.exit_lots
            )

            # Update StockTrade objects with exit info
            trade.exit_date = dt
            trade.exit_action = ex_sig
            trade.exit_lots = net_exit_lots
            trade.exit_price = Decimal(str(exit_price))

            if self._validate_completed_trades(trade):
                # Convert StockTrade to dictionary only if all fields are
                # populated i.e. trade completed.
                updated_trades.append(trade.model_dump())

            else:
                # Append uncompleted trade (i.e. partially close) to
                # 'updated_open_trades'
                updated_open_trades.append(trade)

            # Update remaining positions required to be closed and net position
            half_pos -= lots_to_exit
            self.net_pos += lots_to_exit if ex_sig == "buy" else -lots_to_exit

            if half_pos <= 0:
                # Half of positions already closed. No further action required.
                break

        # Update 'self.open_trades' with 'updated_open_trades'
        self.open_trades = updated_open_trades

        return updated_trades

    def update_via_half_lifo(
        self, dt: date, ex_sig: PriceAction, exit_price: float
    ) -> list[dict[str, Any]]:
        """Update existing StockTrade objects by reducing latest trades by half.

        Args:
            dt (date):
                datetime.date object of trade date.
            ex_sig (PriceAction):
                Action to close open position either "buy" or "sell".
            exit_price (float):
                Exit price of stock ticker.

        Returns:
            (list[dict[str, Any]]):
                List of dictionary containing required fields to generate DataFrame.
        """

        updated_trades = []
        updated_open_trades = deque()

        # Reverse copy of 'self.open_trades'
        open_trades_list = list(self.open_trades)
        reversed_open_trades = open_trades_list[::-1]

        # Half of net position
        half_pos = math.ceil(abs(self.net_pos) / 2)

        for trade in reversed_open_trades:
            # Determine quantity to close based on 'half_pos'
            lots_to_exit, net_exit_lots = self._cal_exit_lots(
                half_pos, trade.entry_lots, trade.exit_lots
            )

            # Update StockTrade objects with exit info
            trade.exit_date = dt
            trade.exit_action = ex_sig
            trade.exit_lots = net_exit_lots
            trade.exit_price = Decimal(str(exit_price))

            if self._validate_completed_trades(trade):
                # Convert StockTrade to dictionary only if all fields are
                # populated i.e. trade completed.
                updated_trades.append(trade.model_dump())

            else:
                # Append uncompleted trade (i.e. partially close) to
                # 'updated_open_trades'
                updated_open_trades.appendleft(trade)

            # Update remaining positions required to be closed and net position
            half_pos -= lots_to_exit
            self.net_pos += lots_to_exit if ex_sig == "buy" else -lots_to_exit

            if half_pos <= 0:
                # Half of positions already closed. No further action required.
                break

        # Update 'self.open_trades'
        self.open_trades = updated_open_trades

        return updated_trades

    def _cal_exit_lots(
        self, closed_qty: int, entry_lots: Decimal, existing_exit_lots: Decimal
    ) -> Decimal:
        """Compute number of lots to exit from open position allowing for partial fill.

        Args:
            closed_qty (int): Number of lots to close.
            entry_lots (Decimal): Number of lots enter for trade.
            existing_exit_lots (Decimal): Number of lots that have been closed already for trade.

        Returns:
            lots_to_close (Decimal):
                Number of lots to close for specific trade.
            net_exit_lots (Decimal):
                Net exit_lots after closing required number of lots
        """

        # Compute number of lots to close for specific trade
        if closed_qty < (entry_lots - existing_exit_lots):
            lots_to_close = closed_qty
        else:
            lots_to_close = entry_lots - existing_exit_lots

        # Compute net exit_lots after closing required number of lots
        net_exit_lots = existing_exit_lots + lots_to_close

        return Decimal(str(lots_to_close)), Decimal(str(net_exit_lots))

    def _is_loss(self, exit_price: float, ex_sig: PriceAction) -> bool:
        """Check if latest trade is running at a loss.

        Args:
            exit_price (float): Exit price of stock ticker.
            ex_sig (PriceAction): Price action to close existing position.

        Returns:
            (bool): Whether the latest trade is running at a loss.
        """

        if len(self.open_trades) == 0:
            raise ValueError("No open trades are available!")

        # Latest trade is the last item in deque
        latest_trade = self.open_trades[-1]

        if ex_sig == "buy":
            return exit_price >= latest_trade.entry_price

        return exit_price <= latest_trade.entry_price

    def _validate_completed_trades(self, stock_trade: StockTrade) -> bool:
        """Check if all the fields in StockTrade object are not null and
        entry_lots matches exit_lots."""

        is_no_null_field = all(
            field is not None for field in stock_trade.model_dump().values()
        )

        is_lots_matched = stock_trade.entry_lots == stock_trade.exit_lots

        return is_no_null_field and is_lots_matched


class TradingStrategy:
    """Combine entry, profit and stop loss strategy as a complete trading strategy.

    Usage:
        >>> strategy = TradingStrategy(
                entry_type="long_only",
                entry=SentiEntry,
                exit=SentiExit,
                trades=SentiTrades,
            )
        >>> strategy.run()

    Args:
        entry_type (EntryType):
            Types of open positions allowed either 'long_only', 'short_only' or
            'long_or_short'.
        entry (EntrySignal):
            Class instance of concrete implementation of 'EntrySignal' abstract class.
        exit (ExitSignal):
            If provided, Class instance of concrete implementation of 'ExitSignal'
            abstract class. If None, standard profit and stop loss will be applied via
            'gen_trades'.
        trades (GetTrades):
            Class instance of concrete implementation of 'GetTrades' abstract class.
        fixed_pl (FixedPL | None):
            If provided, "mean_drawdown", or "max_drawdown" to stop loss based on price
            movement.

    Attributes:
        entry_type (EntryType):
            Types of open positions allowed either 'long_only', 'short_only' or
            'long_or_short'.
        entry (EntrySignal):
            Class instance of concrete implementation of 'EntrySignal' abstract class.
        exit (ExitSignal):
            Class instance of concrete implementation of 'ExitSignal' abstract class.
        trades (GetTrades):
            Instance of concrete implementation of 'GetTrades' abstract class.
        fixed_pl (FixedPL | None):
            If provided, "mean_drawdown", or "max_drawdown" to stop loss based on price
            movement.
    """

    def __init__(
        self,
        entry_type: EntryType,
        entry: type[EntrySignal],
        exit: type[ExitSignal],
        trades: GetTrades,
        fixed_pl: FixedPL | None = None,
        percent_drawdown: float = 0.2,
    ) -> None:
        self.entry_type = entry_type
        self.entry = entry(entry_type)
        self.exit = exit(entry_type)
        self.trades = trades
        self.fixed_pl = fixed_pl
        self.percent_drawdown = percent_drawdown

    def __call__(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Generate completed trades based on trading strategy i.e.
        combination of entry, profit exit and stop exit.

        Args:
            df (pd.DataFrame): DataFrame without price action.

        Returns:
            df_trades (pd.DataFrame):
                DataFrame containing completed trades info.
            df_pa (pd.DataFrame):
                DataFrame with price action (i.e. 'entry_signal', 'exit_signal').
        """

        # Append 'entry_signal' column
        df_pa = self.entry.gen_entry_signal(df)

        if self.exit is not None:
            # Append 'exit_signal' if 'self.exit' exist
            df_pa = self.exit.gen_exit_signal(df_pa)

        match self.fixed_pl:
            case "all_drawdown":
                df_pa = self.update_drawdown_mean(df_pa)
            case "max_drawdown":
                df_pa = self.update_drawdown_max(df_pa)
            case _:
                pass

        # Generate trades
        df_trades, df_pa = self.trades.gen_trades(df_pa)

        return df_trades, df_pa

    def update_drawdown(self, df_pa: pd.DataFrame) -> pd.DataFrame:
        """Update 'exit_signal' based on drawdown computed from open posiitons.

        - Get total investment value based on all open positions.
        - Calculate minimum accepted investment value based on percentage drawdown.
        - Calculate stop price to achieve minimum accepted investment value.

        Args:
            df_pa (pd.DataFrame): DataFrame containing both entry and exit signals.

        Returns:
            df_pa (pd.DataFrame): DataFrame updated with drawdown info.
        """

        # Filter out null values for OHLC due to weekends and holiday
        df = df_pa.loc[~df_pa["close"].isna(), ["close", "ent_sig"]].copy()

        long_stop_prices = []
        short_stop_prices = []
        long_entry_price_list = []
        short_entry_price_list = []
        exit_action = []

        for close, ent_sig in df.itertuples(index=False, name=None):
            # For long position
            if ent_sig == "buy":
                # Compute stop price
                long_stop_prices.append(close * (1 - self.percent_drawdown))
                long_entry_price_list.append(close)

            # For short position
            elif ent_sig == "sell":
                # Compute stop price
                short_stop_prices.append(close * (1 + self.percent_drawdown))
                short_entry_price_list.append(close)

            max_long_stop_price = np.max(long_stop_prices) if long_stop_prices else 0
            max_short_stop_price = (
                np.max(short_stop_prices) if av_short_stop_prices else 0
            )

            # stop loss hit or rating <= 2
            if long_stop_prices and close <= max_long_stop_price:
                exit_action.append("sell")

                # Reset stop prices
                long_stop_prices = []
                av_long_stop_price = 0

            # stop loss hit or rating >= 4
            elif short_stop_prices and close >= max_short_stop_price:
                exit_action.append("buy")

                # Reset stop prices
                short_stop_prices = []
                av_short_stop_prices = 0

            else:
                exit_action.append("wait")

        df["exit_signal"] = exit_action

        return df

    def update_drawdown_max(self, df_pa: pd.DataFrame) -> pd.DataFrame:
        """Update 'exit_signal' based on highest drawdown for open positions.

        - Long position -> highest stop price among all open positions.
        - Short position -> lowest stop price among all open positions.

        Args:
            df_pa (pd.DataFrame): DataFrame containing both entry and exit signals.

        Returns:
            df_pa (pd.DataFrame): DataFrame updated with drawdown info.
        """

        pass
