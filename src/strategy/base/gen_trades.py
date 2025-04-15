"""Abstract classes for generating completed trades."""

from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime
from typing import Any

import pandas as pd

from config.variables import STRUCT_MAPPING, EntryMethod, ExitMethod, PriceAction
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
        req_cols (list[str]):
            List of required columns to generate trades.
        monitor_close (bool):
            Whether to monitor close price ("close") or both high and low price
            (Default: True).
        strategy_dir (str):
            Relative path to strategy folder containing subfolders for implementing
            trading strategy (Default: "./src/strategy").

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
        req_cols (list[str]):
            List of required columns to generate trades.
        monitor_close (bool):
            Whether to monitor close price ("close") or both high and low price
            (Default: True).
        open_trades (deque[StockTrade]):
                Deque list of StockTrade pydantic object to record open trades.
    """

    def __init__(
        self,
        entry_struct: EntryMethod,
        exit_struct: ExitMethod,
        num_lots: int,
        req_cols: list[str],
        monitor_close: bool = True,
        strategy_dir: str = "./src/strategy",
    ) -> None:
        self.entry_struct = entry_struct
        self.exit_struct = exit_struct
        self.num_lots = num_lots
        self.req_cols = req_cols
        self.monitor_close = monitor_close
        self.strategy_dir = strategy_dir
        self.open_trades: deque[StockTrade] = deque()

    @abstractmethod
    def gen_trades(
        self, df_signals: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
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

        # File path to concrete class implementation
        entry_struct_path = f"{self.strategy_dir}/base/entry_struct.py"

        # Get initialized instance of concrete class implementation
        entry_instance = get_class_instance(
            class_name, entry_struct_path, num_lots=self.num_lots
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

        # File path to concrete class implementation
        exit_struct_path = f"{self.strategy_dir}/base/exit_struct.py"

        # Get initialized instance of concrete class implementation
        exit_instance = get_class_instance(class_name, exit_struct_path)

        # Update open trades and generate completed trades
        self.open_trades, completed_trades = exit_instance.close_pos(
            self.open_trades, dt, exit_price
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

        # File path to concrete class implementation
        exit_struct_path = f"{self.strategy_dir}/base/exit_struct.py"

        # Get initialized instance of concrete class implementation
        take_all_exit = get_class_instance("TakeAllExit", exit_struct_path)

        # Update open trades and generate completed trades
        self.open_trades, completed_trades = take_all_exit.close_pos(
            self.open_trades, dt, exit_price
        )

        return completed_trades

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
