"""Abstract class and concrete implementation of various
exit stuctures."""

import math
from abc import ABC, abstractmethod
from collections import deque
from datetime import date
from decimal import Decimal
from typing import Any

from config.variables import PriceAction

from .stock_trade import StockTrade


class ExitStruct(ABC):
    """Abstract class to populate 'StockTrade' pydantic object to close
    existing open positions fully or partially.

    - Exit open position with either profit or loss.
    - Incorporates fixed percentage gain and percentage loss.

    Args:
        None.

    Attributes:
        None.
    """

    @abstractmethod
    def close_pos(
        self,
        open_trades: deque[StockTrade],
        dt: date,
        exit_price: float,
        ex_sig: PriceAction,
    ) -> tuple[deque[StockTrade], list[dict[str, Any]]]:
        """Update existing StockTrade objects (still open); and remove completed
        StockTrade objects in 'open_trades'.

        Args:
            open_trades (deque[StockTrade]):
                Deque list of StockTrade pydantic object to record open trades.
            dt (date):
                Trade date object.
            ex_sig (PriceAction):
                Action to close open position either "buy" or "sell".
            exit_price (float):
                Exit price of stock ticker.

        Returns:
            open_trades (deque[StockTrade]):
                Updated deque list of 'StockTrade' objects.
            completed_trades (list[dict[str, Any]]):
                List of dictionary containing required fields to generate DataFrame.
        """

        pass

    def _validate_completed_trades(self, stock_trade: StockTrade) -> bool:
        """Validate whether StockTrade object is properly updated with no null values."""

        # Check for null fields
        is_no_null_field = all(
            field is not None for field in stock_trade.model_dump().values()
        )

        # Check if number of entry lots must equal number of exit lots
        is_lots_matched = stock_trade.entry_lots == stock_trade.exit_lots

        return is_no_null_field and is_lots_matched


class FIFOExit(ExitStruct):
    """Take profit for earliest open trade. For example:

    - Long stock at 50 (5 lots) -> 60 (3 lots) -> 70 (2 lots).
    - Signal to take profit -> Sell 5 lots bought at 50 (i.e. FIFO).
    """

    def close_pos(
        self,
        open_trades: deque[StockTrade],
        dt: date,
        ex_sig: PriceAction,
        exit_price: float,
    ) -> tuple[deque[StockTrade], list[dict[str, Any]]]:
        """Update existing StockTrade objects (still open); and remove completed
        StockTrade objects in 'open_trades'.

        Args:
            open_trades (deque[StockTrade]):
                Deque list of StockTrade pydantic object to record open trades.
            dt (date):
                Trade date object.
            ex_sig (PriceAction):
                Action to close open position either "buy" or "sell".
            exit_price (float):
                Exit price of stock ticker.

        Returns:
            open_trades (deque[StockTrade]):
                Updated deque list of 'StockTrade' objects.
            completed_trades (list[dict[str, Any]]):
                List of dictionary containing required fields to generate DataFrame.
        """

        if len(open_trades) == 0:
            # No open trades to close
            return open_trades, []

        completed_trades = []

        # Remove earliest StockTrade object from non-empty queue
        trade = open_trades.popleft()

        # Update earliest StockTrade object
        # Ensure exit lots = entry lots i.e. fully closed.
        trade.exit_date = dt
        trade.exit_action = ex_sig
        trade.exit_lots = trade.entry_lots
        trade.exit_price = Decimal(str(exit_price))

        # Convert StockTrade to dictionary only if all fields are populated
        # i.e. trade completed.
        if self._validate_completed_trades(trade):
            completed_trades.append(trade.model_dump())

        return open_trades, completed_trades


class LIFOExit(ExitStruct):
    """Take profit for latest open trade. For example:

    - Long stock at 50 (5 lots) -> 60 (3 lots) -> 70 (2 lots).
    - Signal to take profit -> Sell 2 lots bought at 70 (i.e. FIFO).
    """

    def close_pos(
        self,
        open_trades: deque[StockTrade],
        dt: date,
        ex_sig: PriceAction,
        exit_price: float,
    ) -> tuple[deque[StockTrade], list[dict[str, Any]]]:
        """Update existing StockTrade objects (still open); and remove completed
        StockTrade objects in 'open_trades'.

        Args:
            open_trades (deque[StockTrade]):
                Deque list of StockTrade pydantic object to record open trades.
            dt (date):
                Trade date object.
            ex_sig (PriceAction):
                Action to close open position either "buy" or "sell".
            exit_price (float):
                Exit price of stock ticker.

        Returns:
            open_trades (deque[StockTrade]):
                Updated deque list of 'StockTrade' objects.
            completed_trades (list[dict[str, Any]]):
                List of dictionary containing required fields to generate DataFrame.
        """

        if len(open_trades) == 0:
            # No open trades to close
            return open_trades, []

        completed_trades = []

        # Remove earliest StockTrade object from non-empty queue
        trade = open_trades.pop()

        # Update earliest StockTrade object
        # Ensure exit lots = entry lots i.e. fully closed.
        trade.exit_date = dt
        trade.exit_action = ex_sig
        trade.exit_lots = trade.entry_lots
        trade.exit_price = Decimal(str(exit_price))

        # Convert StockTrade to dictionary only if all fields are populated
        # i.e. trade completed.
        if self._validate_completed_trades(trade):
            completed_trades.append(trade.model_dump())

        return open_trades, completed_trades


class HalfFIFOExit(ExitStruct):
    """keep taking profit by exiting half of existing positions. For example:

    - Long stock at 50 (5 lots) -> 60 (3 lots) -> 70 (2 lots).
    - Signal to take profit -> Sell 5 lots (50% of total 10 lots) bought at 50
    i.e. left 60 (3 lots), 70 (2 lots)
    - Signal to take profit again -> sell 3 lots (50% of total 5 lots) bought at 60
    i.e. left 70 (2 lots).
    """

    def close_pos(
        self,
        open_trades: deque[StockTrade],
        dt: date,
        ex_sig: PriceAction,
        exit_price: float,
    ) -> tuple[deque[StockTrade], list[dict[str, Any]]]:
        """Update existing StockTrade objects (still open); and remove completed
        StockTrade objects in 'open_trades'.

        Args:
            open_trades (deque[StockTrade]):
                Deque list of StockTrade pydantic object to record open trades.
            dt (date):
                Trade date object.
            ex_sig (PriceAction):
                Action to close open position either "buy" or "sell".
            exit_price (float):
                Exit price of stock ticker.

        Returns:
            updated_open_trades (deque[StockTrade]):
                Updated deque list of 'StockTrade' objects.
            completed_trades (list[dict[str, Any]]):
                List of dictionary containing required fields to generate DataFrame.
        """

        completed_trades = []
        updated_open_trades = deque()

        # Half of net position
        half_pos = math.ceil(abs(self.net_pos) / 2)

        for trade in open_trades:
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
                completed_trades.append(trade.model_dump())

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

        return updated_open_trades, completed_trades

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
