"""Abstract class to generate completed trades."""

import math
from abc import ABC, abstractmethod
from collections import deque
from datetime import date
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd

from config.variables import STRUCT_MAPPING, EntryMethod, ExitMethod, PriceAction
from src.utils import utils

from .stock_trade import StockTrade


class GenTrades(ABC):
    """Abstract class to generate completed trades for given strategy.

    Args:
        num_lots (int):
            Number of lots to initiate new position each time (Default: 1).

    Attributes:
        num_lots (int):
            Number of lots to initiate new position each time (Default: 1).
        open_trades (deque[StockTrade]):
            List of open trades containing StockTrade pydantic object.
        completed_trades (list[StockTrade]):
            List of completed trades i.e. completed StockTrade pydantic object.
    """

    def __init__(
        self,
        num_lots: int = 1,
    ) -> None:
        self.num_lots = num_lots
        self.open_trades = deque()
        self.completed_trades = []

    @abstractmethod
    def gen_trades(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate DataFrame containing completed trades for given strategy"""

        pass

    def get_net_pos(self) -> int:
        """Get net positions from 'self.open_trades'."""

        # Get list of entry and exit lots from 'self.open_trades'
        # Set exit lots to 0 if None
        entry_lots_list = [trade.entry_lots for trade in self.open_trades]
        exit_lots_list = [
            0 if trade.exit_lots is None else trade.exit_lots
            for trade in self.open_trades
        ]

        return sum(ent - ex for ent, ex in zip(entry_lots_list, exit_lots_list))

    # def open_new_pos(
    #     self, ticker: str, dt: date, entry_price: float, ent_sig: PriceAction
    # ) -> None:
    #     """Open new positions based on 'self.entry_struc'

    #     Args:
    #         ticker (str):
    #             Stock ticker to be traded.
    #         dt (date):
    #             Trade date object.
    #         entry_price (float):
    #             Entry price for stock ticker.
    #         ent_sig (PriceAction):
    #             Entry signal i.e. "buy", "sell" or "wait" to create new position.

    #     Returns:
    #         None.
    #     """

    #     # Get path to script containing concrete implementation of 'EntryStruct'
    #     # and 'ExitStruct'
    #     ent_path = f"{self.strategy_dir}/base/entry_struct.py"
    #     ex_path = f"{self.strategy_dir}/base/exit_struct.py"

    #     # if (
    #     #     self.entry_struct == "single" and self.net_pos == 0
    #     # ) or self.entry_struct == "multiple":
    #     #     # Create StockTrade object to record new long/short position
    #     #     # based on 'ent_sig'
    #     #     stock_trade = StockTrade(
    #     #         ticker=ticker,
    #     #         entry_date=dt,
    #     #         entry_action=ent_sig,
    #     #         entry_lots=Decimal(str(self.num_lots)),
    #     #         entry_price=Decimal(str(entry_price)),
    #     #     )
    #     #     self.open_trades.append(stock_trade)
    #     #     self.net_pos += self.num_lots if ent_sig == "buy" else -self.num_lots

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
        """Validate whether StockTrade object is properly updated with no null values."""

        # Check for null fields
        is_no_null_field = all(
            field is not None for field in stock_trade.model_dump().values()
        )

        # Check if number of entry lots must equal number of exit lots
        is_lots_matched = stock_trade.entry_lots == stock_trade.exit_lots

        return is_no_null_field and is_lots_matched

    def cal_stop_price(self, percent_loss: float = 0.2) -> Decimal | None:
        """Compute required stop price to keep investment losses within stipulated
        percentage loss.

        Args:
            percent_loss (float):
                Percentage loss in investment value (Default: 0.2).

        Returns:
            (Decimal | None): If available, required stop price to monitor.
        """

        if len(self.open_trades) == 0:
            # No open positions hence no stop price
            return

        entry_action = self.open_trades[0].entry_action
        open_lots = []
        entry_prices = []

        # Extract entry_lots, entry_price, exit_lots from 'self.open_trades'
        for open_trade in self.open_trades:
            open_lots.append(open_trade.entry_lots - open_trade.exit_lots)
            entry_prices.append(open_trade.entry_price)

        # Compute total_investment
        total_investment = sum(
            entry_price * lots for entry_price, lots in zip(entry_prices, open_lots)
        )

        # Compute value after stipulated percent loss for long and short position
        value_after_loss = (
            total_investment * (1 - percent_loss)  # sell at lower price
            if entry_action == "buy"
            else total_investment * (1 + percent_loss)  # buy at higher price
        )

        # Compute stop price to meet stipulated percent loss
        stop_price = value_after_loss / sum(open_lots)

        return Decimal(str(round(stop_price, 2)))
