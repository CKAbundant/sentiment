"""Abstract classes for generating completed trades."""

from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime
from decimal import Decimal
from pprint import pformat
from typing import Any

import pandas as pd

from config.variables import EntryMethod, ExitMethod, PriceAction
from src.utils.strategy_utils import get_class_instance, get_net_pos, get_std_field
from src.utils.utils import display_open_trades, set_decimal_type

from .stock_trade import StockTrade


class GenTrades(ABC):
    """Abstract class to generate completed trades for given strategy.

    Args:
        entry_struct (EntryMethod):
            Whether to allow multiple open position ("mulitple") or single
            open position at a time ("single").
        exit_struct (ExitMethod):
            Whether to apply first-in-first-out ("fifo"), last-in-first-out ("lifo"),
            take profit for half open positions repeatedly ("half_life") or
            take profit for all open positions ("take_all").
        num_lots (int):
            Number of lots to initiate new position each time (Default: 1).
        monitor_close (bool):
            Whether to monitor close price ("close") or both high and low price
            (Default: True).
        percent_loss (float):
            Percentage loss allowed for investment (Default: 0.05).
        stop_method (ExitMethod):
            Exit method to generate stop price (Default: "no_stop").
        trigger_trail (float):
            If provided, percentage profit required to trigger trailing profit.
            No trailing profit if None.
        step (float):
            If provided, percent profit increment to trail profit. If None,
            increment set to current high - trigger_trail_level.
        entry_struct_path (str):
            Relative path to 'entry_struct.py'
            (Default: "./src/strategy/base/entry_struct.py").
        exit_struct_path (str):
            Relative path to 'exit_struct.py'
            (Default: "./src/strategy/base/exit_struct.py").
        stop_method_path (str):
            Relative path to 'cal_exit_price.py'
            (Default: "./src/strategy/base/cal_exit_price.py").

    Attributes:
        entry_struct (EntryMethod):
            Whether to allow multiple open position ("mulitple") or single
            open position at a time ("single").
        exit_struct (ExitMethod):
            Whether to apply first-in-first-out ("fifo"), last-in-first-out ("lifo"),
            take profit for half open positions repeatedly ("half_life") or
            take profit for all open positions ("take_all").
        num_lots (int):
            Number of lots to initiate new position each time (Default: 1).
        monitor_close (bool):
            Whether to monitor close price ("close") or both high and low price
            (Default: True).
        percent_loss (float):
            Percentage loss allowed for investment (Default: 0.05).
        stop_method (ExitMethod):
            Exit method to generate stop price (Default: "no_stop").
        trigger_trail (Decimal):
            If provided, percentage profit required to trigger trailing profit.
            No trailing profit if None.
        step (Decimal):
            If provided, percent profit increment to trail profit. If None,
            increment set to current high - trigger_trail_level.
        entry_struct_path (str):
            Relative path to 'entry_struct.py'
            (Default: "./src/strategy/base/entry_struct.py").
        exit_struct_path (str):
            Relative path to 'exit_struct.py'
            (Default: "./src/strategy/base/exit_struct.py").
        stop_method_path (str):
            Relative path to 'cal_exit_price.py'
            (Default: "./src/strategy/base/cal_exit_price.py").
        req_cols (list[str]):
            List of required columns to generate trades.
        open_trades (list[StockTrade]):
                List of 'StockTrade' pydantic objects representing open positions.
        stop_info_list (list[dict[str, datetime | str | Decimal]]):
            List to record datetime, stop price and whether stop price is triggered.
        trail_info_list (list[dict[str, datetime | str | Decimal]]):
            List to record datetime, trail profit price and whether trailing profit
            is triggered.
        trigger_trail_level (float):
            Actual price required to trigger trailing profit. Calculated from
            'trigger_trail'.
        trailing_profit (float):
            Trailing profit price level. If triggered, all open positions
            will be closed.
    """

    def __init__(
        self,
        entry_struct: EntryMethod,
        exit_struct: ExitMethod,
        num_lots: int,
        monitor_close: bool = True,
        percent_loss: float = 0.05,
        stop_method: ExitMethod = "no_stop",
        trigger_trail: float | None = None,
        step: float | None = None,
        entry_struct_path: str = "./src/strategy/base/entry_struct.py",
        exit_struct_path: str = "./src/strategy/base/exit_struct.py",
        stop_method_path: str = "./src/strategy/base/cal_exit_price.py",
    ) -> None:
        self.entry_struct = entry_struct
        self.exit_struct = exit_struct
        self.num_lots = num_lots
        self.monitor_close = monitor_close
        self.percent_loss = percent_loss
        self.stop_method = stop_method
        self.trigger_trail = (
            Decimal(str(trigger_trail)) if trigger_trail is not None else None
        )  # Convert to Decimal
        self.step = (
            Decimal(str(step)) if step is not None else None
        )  # Convert to Decimal
        self.entry_struct_path = entry_struct_path
        self.exit_struct_path = exit_struct_path
        self.stop_method_path = stop_method_path
        self.req_cols = [
            "date",
            "high",
            "low",
            "close",
            "entry_signal",
            "exit_signal",
        ]
        self.open_trades: deque[StockTrade] = deque()
        self.stop_info_list = []
        self.trail_info_list = []
        self.trigger_trail_level = None
        self.trailing_profit = None

    @abstractmethod
    def gen_trades(self, df_signals: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Generate DataFrame containing completed trades for given strategy.

        Args:
            df_signals (pd.DataFrame):
                DataFrame containing entry and exit signals for specific ticker.

        Returns:
            df_trades (pd.DataFrame):
                DataFrame containing completed trades.
            df_signals (pd.DataFrame):
                DataFrame containing updated exit signals price-related stops.
        """

        pass

    def iterate_df(
        self, ticker: str, df_signals: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Iterate through DataFrame containing buy and sell signals to
        populate completed trades.

        - Close off all open positions at end of trading period.
        - Check to cut loss for all open position.
        - Check to take profit.
        - Check to enter new position.
        - Update DataFrame with stop price and whether if triggered.

        Args:
            ticker (str):
                Stock ticker for back testing.
            df_signals (pd.DataFrame):
                DataFrame containing entry and exit signals.

        Returns:
            df_trades (pd.DataFrame):
                DataFrame containing completed trades.
            df_signals (pd.DataFrame):
                DataFrame containing updated exit signals based on price-related stops.
        """

        # Filter required columns i.e. date, high, low, close, entry and exit signal
        df = df_signals.copy()
        df = df.loc[:, self.req_cols]

        # Convert numeric type to Decimal
        df = set_decimal_type(df)
        completed_list = []

        for record in df.itertuples(index=True, name=None):
            # Create mapping for attribute to its values and check if end of DataFrame
            info = self.gen_mapping(record)
            is_end = info["idx"] >= len(df) - 1

            idx = info["idx"]
            dt = info["date"]
            close = info["close"]
            ent_sig = info["entry_signal"]
            ex_sig = info["exit_signal"]

            print(f"idx : {idx}")
            print(f"dt : {dt}")
            print(f"close : {close}")
            print(f"ent_sig : {ent_sig}")
            print(f"ex_sig : {ex_sig}")
            print(f"net_pos : {get_net_pos(self.open_trades)}")

            # Close off all open positions at end of trading period
            # Skip creating new open positions after all open positions closed
            if is_end:
                completed_list = self.exit_all_end(completed_list, info)
                continue

            # Check whether to cut loss, take profit and open new position sequentially
            completed_list = self.check_stop_loss(completed_list, info)
            completed_list = self.check_profit(completed_list, info)
            completed_list = self.check_trailing_profit(completed_list, info)
            self.check_new_pos(ticker, info)

            print(f"net_pos after update : {get_net_pos(self.open_trades)}")
            print(f"len(self.open_trades) : {len(self.open_trades)}")
            print(f"self.trigger_trail_level after update: {self.trigger_trail_level}")
            display_open_trades(self.open_trades)

        # Append stop loss price and trailing price if available
        df_signals = self.append_info(df_signals, self.stop_info_list)
        df_signals = self.append_info(df_signals, self.trail_info_list)

        # Convert 'completed_list' to DataFrame; append 'ticker'
        df_trades = pd.DataFrame(completed_list)

        return df_trades, df_signals

    def exit_all_end(
        self,
        completed_list: list[dict[str, Any]],
        record: dict[str, float | datetime],
    ) -> list[dict[str, Any]]:
        """Exit all open positions at end of testing/trading period.

        - Close off all position if end of testing period.
        - No new postiion at end of testing period.

        Args:
            completed_list (list[dict[str, Any]]):
                List of dictionary containing required fields to generate DataFrame.
            record (dict[str, float | datetime]):
                Dictionary mapping required attributes to its values.

        Returns:
            completed_list (list[dict[str, Any]]):
                List of dictionary containing required fields to generate DataFrame.
        """

        if len(self.open_trades) == 0:
            # Return 'completed_list' unamended
            return completed_list

        # Close off all open positions at end of trading period
        completed_list.extend(self.exit_all(record["date"], record["close"]))

        return completed_list

    def check_stop_loss(
        self,
        completed_list: list[dict[str, Any]],
        record: dict[str, float | datetime],
    ) -> list[dict[str, Any]]:
        """Check if stop loss condition met

        Args:
            completed_list (list[dict[str, Any]]):
                List of dictionary containing required fields to generate DataFrame.
            record (dict[str, float | datetime]):
                Dictionary mapping required attributes to its values.

        Returns:
            completed_list (list[dict[str, Any]]):
                List of dictionary containing required fields to generate DataFrame.
        """

        # Return 'completed_list' unamended if no net position or no stop loss set
        if len(self.open_trades) == 0 or self.stop_method == "no_stop":
            return completed_list

        dt = record["date"]
        close = record["close"]
        low = record["low"]
        high = record["high"]

        # Get standard 'entry_action' from 'self.open_trades'; aand stop price
        entry_action = get_std_field(self.open_trades, "entry_action")
        stop_price = self.cal_stop_price()

        cond_list = [
            self.monitor_close and entry_action == "buy" and close < stop_price,
            self.monitor_close and entry_action == "sell" and close > stop_price,
            not self.monitor_close and entry_action == "buy" and low < stop_price,
            not self.monitor_close and entry_action == "sell" and high > stop_price,
        ]

        # Exit price is closing price if monitor based on closing price
        # else stop price
        exit_price = close if self.monitor_close else stop_price

        # Exit all open positions if any condition in 'cond_list' is true
        if any(cond_list):
            exit_action = "sell" if entry_action == "buy" else "buy"
            print(f"\nStop triggered -> {exit_action} @ stop price {stop_price}\n")

            completed_list.extend(self.exit_all(dt, exit_price))
            self.stop_info_list.append(
                {"date": dt, "stop_price": stop_price, "stop_triggered": Decimal("1")}
            )

        else:
            self.stop_info_list.append(
                {"date": dt, "stop_price": stop_price, "stop_triggered": Decimal("0")}
            )

        return completed_list

    def check_profit(
        self,
        completed_list: list[dict[str, Any]],
        record: dict[str, float | datetime],
    ) -> list[dict[str, Any]]:
        """Check whether take profit condition is met and update completed_list.

        Args:
            completed_list (list[dict[str, Any]]):
                List of dictionary containing required fields to generate DataFrame.
            record (dict[str, float | datetime]):
                Dictionary mapping required attributes to its values.

        Returns:
            completed_list (list[dict[str, Any]]):
                List of dictionary containing required fields to generate DataFrame.
        """

        ent_sig = record["entry_signal"]
        ex_sig = record["exit_signal"]
        dt = record["date"]
        close = record["close"]

        # Return 'completed_list' unamended if no open position or not exit signals
        if len(self.open_trades) == 0 or (ex_sig != "sell" and ex_sig != "buy"):
            return completed_list

        # Get standard 'entry_action' from 'self.open_trades'
        entry_action = get_std_field(self.open_trades, "entry_action")

        # Exit all open position in order to flip position
        # If entry_action == 'buy', then ex_sig must be 'sell'
        # ex_sig != entry_action
        if ex_sig == ent_sig and ex_sig != entry_action:
            completed_list.extend(self.exit_all(dt, close))
        else:
            completed_list.extend(self.take_profit(dt, ex_sig, close))

        return completed_list

    def check_trailing_profit(
        self,
        completed_list: list[dict[str, Any]],
        record: dict[str, float | datetime],
    ) -> list[dict[str, Any]]:
        """Check if stop loss condition met

        Args:
            completed_list (list[dict[str, Any]]):
                List of dictionary containing required fields to generate DataFrame.
            record (dict[str, float | datetime]):
                Dictionary mapping required attributes to its values.

        Returns:
            completed_list (list[dict[str, Any]]):
                List of dictionary containing required fields to generate DataFrame.
        """

        print(f"self.trigger_trail : {self.trigger_trail}")
        print(f"self.step : {self.step}")
        print(f"self.trigger_trail_level : {self.trigger_trail_level}")
        print(f"high : {record["high"]}")
        print(
            f"first entry price : {self.open_trades[0].entry_price if len(self.open_trades)>0 else None}"
        )
        print(f"self.trailing_profit : {self.trailing_profit}\n")

        # Update trailing profit
        self.cal_trailing_profit(record)
        print(f"self.trailing_profit after update : {self.trailing_profit}")

        # Return completed list if trailing profit is still None after update
        if self.trailing_profit is None:
            return completed_list

        trail = self.trailing_profit
        dt = record["date"]
        high = record["high"]
        low = record["low"]
        close = record["close"]

        # Get standard 'entry_action' from 'self.open_trades'
        entry_action = get_std_field(self.open_trades, "entry_action")

        cond_list = [
            self.monitor_close and entry_action == "buy" and close < trail,
            self.monitor_close and entry_action == "sell" and close > trail,
            not self.monitor_close and entry_action == "buy" and low < trail,
            not self.monitor_close and entry_action == "sell" and high > trail,
        ]

        # Exit price is closing price if monitor based on closing price
        # else stop price
        exit_price = close if self.monitor_close else trail

        # Exit all open positions if any condition in 'cond_list' is true
        if any(cond_list):
            exit_action = "sell" if entry_action == "buy" else "buy"
            print(f"Trail triggered -> {exit_action} @ trail price {trail}\n")

            # Trailing profit price triggered
            completed_list.extend(self.exit_all(dt, exit_price))
            self.trail_info_list.append(
                {
                    "date": dt,
                    "trail_price": trail,
                    "trail_triggered": Decimal("1"),
                }
            )
        else:
            self.trail_info_list.append(
                {
                    "date": dt,
                    "trail_price": trail,
                    "trail_triggered": Decimal("0"),
                }
            )

        return completed_list

    def check_new_pos(
        self,
        ticker: str,
        record: dict[str, float | datetime],
    ) -> None:
        """Create new open position based on 'self.entry_struct' method.

        Args:
            ticker (str):
                Stock ticker to be traded.
            record (dict[str, float | datetime]):
                Dictionary mapping required attributes to its values.

        Returns:
            None.
        """

        dt = record["date"]
        ent_sig = record["entry_signal"]

        # Set entry price as closing price
        entry_price = record["close"]

        if ent_sig != "buy" and ent_sig != "sell":
            # No entry signal
            return

        # Get initialized instance of concrete class implementation
        entry_instance = get_class_instance(
            self.entry_struct, self.entry_struct_path, num_lots=self.num_lots
        )

        # Update 'self.open_trades' with new open position
        self.open_trades = entry_instance.open_new_pos(
            self.open_trades, ticker, dt, ent_sig, entry_price
        )

        if len(self.open_trades) == 0:
            raise ValueError("No open positions created!")

        if self.trigger_trail_level is None:
            self.trigger_trail_level = self.cal_trigger_trail_level()

    def cal_trailing_profit(self, record: dict[str, float | datetime]) -> None:
        """Compute trailing profit level.

        Args:
            record (dict[str, float | datetime]):
                Dictionary mapping required attributes to its values.

        Returns:
            None.
        """

        # Skip computation if trailing profit is not activated or no open positions
        if self.trigger_trail is None or len(self.open_trades) == 0:
            self.trigger_trail_level = None
            return

        # Compute trigger_trail_level if trailing profit enabled and open
        # positions present
        if self.trigger_trail_level is None:
            self.trigger_trail_level = self.cal_trigger_trail_level()

        high = record["high"]
        low = record["low"]

        # Get standard 'entry_action' from 'self.open_trades'
        entry_action = get_std_field(self.open_trades, "entry_action")

        # Compute amount of profit increment to shift trailing profit
        first_price = self.open_trades[0].entry_price
        step_level = first_price * self.step if self.step is not None else None

        # Current high must be higher than trigger_trail_level for long positions
        if entry_action == "buy" and (excess := high - self.trigger_trail_level) >= 0:
            computed_trailing = (
                first_price + excess
                if step_level is None
                else first_price + (excess // step_level) * step_level
            )

            if self.trailing_profit is None or computed_trailing > self.trailing_profit:
                # Update trailing profit level if higher than previous level
                self.trailing_profit = computed_trailing

        # Current low must be lower than trigger_trail_level for short positions
        elif entry_action == "sell" and (excess := self.trigger_trail_level - low) >= 0:
            computed_trailing = (
                first_price - excess
                if step_level is None
                else first_price - (excess // step_level * step_level)
            )

            if self.trailing_profit is None or computed_trailing < self.trailing_profit:
                # Update trailing profit level if lower than previous level
                self.trailing_profit = computed_trailing

    def cal_trigger_trail_level(self) -> float | None:
        """Compute price level to activate trailing profit."""

        if len(self.open_trades) == 0 or self.trigger_trail is None:
            self.trigger_trail_level = None
            self.trailing_profit = None

            return

        # Get standard 'entry_action' from 'self.open_trades' and first entry price
        entry_action = get_std_field(self.open_trades, "entry_action")
        first_price = self.open_trades[0].entry_price

        return (
            first_price * (1 + self.trigger_trail)
            if entry_action == "buy"
            else first_price * (1 - self.trigger_trail)
        )

    def take_profit(
        self,
        dt: datetime,
        ex_sig: PriceAction,
        exit_price: float,
    ) -> list[dict[str, Any]]:
        """Close existing open positions based on 'self.exit_struct' method.

        Args:
            dt (datetime):
                Trade datetime object.
            ex_sig (PriceAction):
                Exit signal generated by 'ExitSignal' class either 'buy', 'sell'
                or 'wait'.
            exit_price (float):
                Exit price of stock ticker.

        Returns:
            completed_list (list[dict[str, Any]]):
                List of dictionary containing required fields to generate DataFrame.
        """

        # Get standard 'entry_action' from 'self.open_trades'
        entry_action = get_std_field(self.open_trades, "entry_action")

        if (ex_sig == "buy" and entry_action == "buy") or (
            ex_sig == "sell" and entry_action == "sell"
        ):
            # No completed trades if exit signal is same as entry action
            return []

        # Get initialized instance of concrete class implementation
        exit_instance = get_class_instance(self.exit_struct, self.exit_struct_path)

        # Update open trades and generate completed trades
        self.open_trades, completed_list = exit_instance.close_pos(
            self.open_trades, dt, exit_price
        )

        return completed_list

    def exit_all(
        self,
        dt: datetime,
        exit_price: float,
    ) -> list[dict[str, Any]]:
        """Close all open positions via 'TakeAllExit.close_pos' method.

        Args:
            dt (datetime):
                Trade datetime object.
            exit_price (float):
                Exit price of stock ticker.

        Returns:
            completed_list (list[dict[str, Any]]):
                List of dictionary containing required fields to generate DataFrame.
        """

        # Get initialized instance of concrete class implementation
        take_all_exit = get_class_instance("TakeAllExit", self.exit_struct_path)

        # Update open trades and generate completed trades
        self.open_trades, completed_list = take_all_exit.close_pos(
            self.open_trades, dt, exit_price
        )

        if len(self.open_trades) != 0:
            raise ValueError("Open positions are not closed completely.")

        # Reset trigger trail level and trailing profit
        self.trigger_trail_level = None
        self.trailing_profit = None

        return completed_list

    def cal_stop_price(self) -> Decimal:
        """Compute price to trigger stop loss."""

        stop_loss = get_class_instance(
            self.stop_method, self.stop_method_path, percent_loss=self.percent_loss
        )

        return stop_loss.cal_exit_price(self.open_trades)

    def append_info(
        self,
        df_signals: pd.DataFrame,
        info_list: list[dict[str, str | datetime | Decimal]] | None = None,
    ) -> pd.DataFrame:
        """Convert 'info_list' (i.e. stop loss or trailing profit info) to DataFrame
        and append to 'df_signals'."""

        if len(info_list) == 0:
            return df_signals

        # Convert 'info_list' to DataFrame
        df_info = pd.DataFrame(info_list)

        # Ensure both 'date' column in 'df' and 'df_info' are time zone naive
        df_signals = self.set_naive_tz(df_signals)
        df_info = self.set_naive_tz(df_info)

        # Perform join via index
        df = df_signals.join(df_info)

        # Convert 'date' index to column
        df = df.reset_index()

        return df

    def gen_mapping(self, record: tuple[Any]) -> dict[str, Any]:
        """Generate mapping for record generated by 'itertuples' method."""

        # Record include row index and required fields
        fields = ["idx", *self.req_cols]

        return {attr: val for attr, val in zip(fields, record)}

    def set_naive_tz(self, data: pd.DataFrame) -> pd.DataFrame:
        """Set the 'data' column to be time zone naive and as index to faciliate join."""

        # Set 'date' column as index
        if "date" not in data.columns:
            raise ValueError("'date' column is not found.")

        df = data.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["date"] = df["date"].map(
            lambda dt: dt.replace(hour=0, minute=0, tzinfo=None)
        )
        df = df.set_index("date")

        return df
