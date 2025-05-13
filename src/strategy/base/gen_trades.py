"""Abstract classes for generating completed trades."""

from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime
from decimal import Decimal
from typing import Any

import pandas as pd

from config.variables import (
    EXIT_PRICE_MAPPING,
    STRUCT_MAPPING,
    EntryMethod,
    ExitMethod,
    PriceAction,
)
from src.utils.utils import get_class_instance, get_std_field

from .stock_trade import StockTrade


class GenTrades(ABC):
    """Abstract class to generate completed trades for given strategy.

    Args:
        entry_struct (EntryMethod):
            Whether to allow multiple open position ("mulitple") or single
            open position at a time ("single") (Default: "multiple").
        exit_struct (ExitMethod):
            Whether to apply first-in-first-out ("fifo"), last-in-first-out ("lifo"),
            take profit for half open positions repeatedly ("half_life") or
            take profit for all open positions ("take_all") (Default: "take_all").
        num_lots (int):
            Number of lots to initiate new position each time (Default: 1).
        monitor_close (bool):
            Whether to monitor close price ("close") or both high and low price
            (Default: True).
        entry_struct_path (str):
            Relative path to 'entry_struct.py'
            (Default: "./src/strategy/base/entry_struct.py").
        exit_struct_path (str):
            Relative path to 'exit_struct.py'
            (Default: "./src/strategy/base/exit_struct.py").

    Attributes:
        entry_struct (EntryMethod):
            Whether to allow multiple open position ("mulitple") or single
            open position at a time ("single") (Default: "multiple").
        exit_struct (ExitMethod):
            Whether to apply first-in-first-out ("fifo"), last-in-first-out ("lifo"),
            take profit for half open positions repeatedly ("half_life") or
            take profit for all open positions ("take_all") (Default: "take_all").
        num_lots (int):
            Number of lots to initiate new position each time (Default: 1).
        monitor_close (bool):
            Whether to monitor close price ("close") or both high and low price
            (Default: True).
        entry_struct_path (str):
            Relative path to 'entry_struct.py'
            (Default: "./src/strategy/base/entry_struct.py").
        exit_struct_path (str):
            Relative path to 'exit_struct.py'
            (Default: "./src/strategy/base/exit_struct.py").
        req_cols (list[str]):
            List of required columns to generate trades.
    """

    def __init__(
        self,
        entry_struct: EntryMethod,
        exit_struct: ExitMethod,
        num_lots: int,
        monitor_close: bool = True,
        entry_struct_path: str = "./src/strategy/base/entry_struct.py",
        exit_struct_path: str = "./src/strategy/base/exit_struct.py",
    ) -> None:
        self.entry_struct = entry_struct
        self.exit_struct = exit_struct
        self.num_lots = num_lots
        self.monitor_close = monitor_close
        self.open_trades: deque[StockTrade] = deque()
        self.entry_struct_path = entry_struct_path
        self.exit_struct_path = exit_struct_path
        self.req_cols = [
            "date",
            "high",
            "low",
            "close",
            "entry_signal",
            "exit_signal",
        ]

    @abstractmethod
    def gen_trades(self, df_signals: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Generate DataFrame containing completed trades for given strategy.

        Args:
            df_signals (pd.DataFrame): DataFrame containing entry and exit signals.

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

        completed_list = []

        for idx, dt, high, low, close, ent_sig, ex_sig in df.itertuples(
            index=True, name=None
        ):
            # Get net position and whether end of DataFrame
            net_pos = self.get_net_pos()
            is_end = idx >= len(df) - 1

            # Close off all open positions at end of trading period
            if is_end and net_pos != 0:
                completed_list.extend(self.exit_all(dt, close))

                # Skip creating new open positions after all open positions closed
                continue

            # Check to cut loss for all open position
            if net_pos != 0 and self.stop_method != "no_stop":
                completed_list.extend(self.stop_loss(dt, high, low, close))

            # Check to take profit
            if (ex_sig == "sell" or ex_sig == "buy") and net_pos != 0:
                # Get standard 'entry_action' from 'self.open_trades'
                entry_action = get_std_field(self.open_trades, "entry_action")

                # Exit all open position in order to flip position
                # If entry_action == 'buy', then ex_sig must be 'sell'
                # ex_sig != entry_action
                if ex_sig == ent_sig and ex_sig != entry_action:
                    completed_list.extend(self.exit_all(dt, close))
                else:
                    completed_list.extend(self.take_profit(dt, ex_sig, close))

            # Check to enter new position
            if ent_sig == "buy" or ent_sig == "sell":
                self.open_pos(ticker, dt, ent_sig, close)

        # Convert 'stop_info_list' to DataFrame and append to 'df_senti'
        # 'self.stop_list and 'self.trigger_list' are not empty
        if self.stop_info_list:
            df_signals = self.append_stop_info(df_signals)

        # Convert 'completed_list' to DataFrame
        df_trades = pd.DataFrame(completed_list)

        return df_trades, df_signals

    def open_pos(
        self,
        ticker: str,
        dt: datetime | str,
        ent_sig: PriceAction,
        entry_price: float,
    ) -> None:
        """Create new open position based on 'self.entry_struct' method.

        Args:
            ticker (str):
                Stock ticker to be traded.
            dt (datetime | str):
                Trade datetime object or string in "YYYY-MM-DD" format.
            ent_sig (PriceAction):
                Entry signal i.e. "buy", "sell" or "wait" to create new position.
            entry_price (float):
                Entry price for stock ticker.

        Returns:
            None.
        """

        # Name of concrete class implementation of 'EntryStruct'
        class_name = STRUCT_MAPPING.get(self.entry_struct)

        # Get initialized instance of concrete class implementation
        entry_instance = get_class_instance(
            class_name, self.entry_struct_path, num_lots=self.num_lots
        )

        # Update 'self.open_trades' with new open position
        self.open_trades = entry_instance.open_new_pos(
            self.open_trades, ticker, dt, ent_sig, entry_price
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
            completed_trades (list[dict[str, Any]]):
                List of dictionary containing required fields to generate DataFrame.
        """

        # Get standard 'entry_action' from 'self.open_trades'
        entry_action = get_std_field(self.open_trades, "entry_action")

        if (ex_sig == "buy" and entry_action == "buy") or (
            ex_sig == "sell" and entry_action == "sell"
        ):
            # No completed trades if exit signal is same as entry action
            return []

        # Name of concrete class implementation of 'ExitStruct'
        class_name = STRUCT_MAPPING.get(self.exit_struct)

        # Get initialized instance of concrete class implementation
        exit_instance = get_class_instance(class_name, self.exit_struct_path)

        # Update open trades and generate completed trades
        self.open_trades, completed_trades = exit_instance.close_pos(
            self.open_trades, dt, exit_price
        )

        return completed_trades

    def stop_loss(
        self,
        dt: datetime,
        high: float,
        low: float,
        close: float,
    ) -> list[dict[str, Any]]:
        """Close all open positions if computed stop price is triggered.

        - Stop price is computed via concrete implementation of 'CalExitPrice'.

        Args:
            dt (datetime):
                Trade datetime object.
            high (float):
                Current day high of cointegrated/correlated stock ticker.
            low (float):
                Current day low of cointegrated/correlated stock ticker.
            close (float):
                Current day open of cointegrated/correlated stock ticker.

        Returns:
            completed_trades (list[dict[str, Any]]):
                List of dictionary containing required fields to generate DataFrame.
        """

        completed_trades = []

        # Compute stop loss price based on 'self.stop_method'
        stop_price = self.cal_stop_price()
        entry_action = get_std_field(self.open_trades, "entry_action")
        # display_stop_price(
        #     self.monitor_close, stop_price, entry_action, high, low, close
        # )

        cond_list = [
            self.monitor_close and entry_action == "buy" and close < stop_price,
            self.monitor_close and entry_action == "sell" and close > stop_price,
            not self.monitor_close and entry_action == "buy" and low < stop_price,
            not self.monitor_close and entry_action == "sell" and high > stop_price,
        ]

        # Exit all open positions if any condition in 'cond_list' is true
        if any(cond_list):
            # exit_action = "sell" if entry_action == "buy" else "buy"
            # print(f"\nStop triggered -> {exit_action} @ stop price {stop_price}\n")

            completed_trades.extend(self.exit_all(dt, stop_price))
            self.stop_info_list.append(
                {"date": dt, "stop_price": stop_price, "triggered": Decimal("1")}
            )

        else:
            self.stop_info_list.append(
                {"date": dt, "stop_price": stop_price, "triggered": Decimal("0")}
            )

        return completed_trades

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
            completed_trades (list[dict[str, Any]]):
                List of dictionary containing required fields to generate DataFrame.
        """

        # Get initialized instance of concrete class implementation
        take_all_exit = get_class_instance("TakeAllExit", self.exit_struct_path)

        # Update open trades and generate completed trades
        self.open_trades, completed_trades = take_all_exit.close_pos(
            self.open_trades, dt, exit_price
        )

        return completed_trades

    def cal_stop_price(self) -> Decimal:
        """Compute stop price via concrete implementation of 'CalExitPrice'.

        Args:
            None.

        Returns:
            (Decimal): Stop price to monitor.
        """

        # Name of concrete class implemenation of 'CalExitPrice'
        class_name = EXIT_PRICE_MAPPING.get(self.stop_method)

        # Get initialized instance of concrete class implementation
        class_inst = get_class_instance(
            class_name, self.stop_method_path, percent_loss=self.percent_loss
        )

        return class_inst.cal_exit_price(self.open_trades)

    def append_stop_info(self, df_senti: pd.DataFrame) -> pd.DataFrame:
        """Append stop info (i.e. stop price and whether triggered) to
        'df_senti' DataFrame."""

        df = df_senti.copy()

        # Convert 'self.stop_info_list' to DataFrame
        df_stop = pd.DataFrame(self.stop_info_list)

        # Set time to 0000 hrs for 'dt' column in order to perform join
        df_stop["date"] = df_stop["date"].map(
            lambda dt: dt.replace(hour=0, minute=0, tzinfo=None)
        )
        df_stop = df_stop.set_index("date")

        # Set 'date' column as index
        if "date" not in df.columns:
            raise ValueError("'date' column is not found.")

        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

        # Perform join via index
        df = df.join(df_stop)

        return df

    def get_net_pos(self) -> int:
        """Get net positions from 'self.open_trades'."""

        return sum(
            (
                trade.entry_lots - trade.exit_lots
                if trade.entry_action == "buy"
                else -(trade.entry_lots - trade.exit_lots)
            )
            for trade in self.open_trades
        )
