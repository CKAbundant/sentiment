"""Abstract class and concrete implementation of various
exit stuctures."""

import math
from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime
from decimal import Decimal
from typing import Any

from pydantic import ValidationError

from config.variables import ExitMethod, PriceAction

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
        dt: datetime,
        exit_price: float,
    ) -> tuple[deque[StockTrade], list[dict[str, Any]]]:
        """Update existing StockTrade objects (still open); and remove completed
        StockTrade objects in 'open_trades'.

        Args:
            open_trades (deque[StockTrade]):
                Deque list of StockTrade pydantic object to record open trades.
            dt (datetime):
                Trade datetime object.
            exit_price (float):
                Exit price of stock ticker.

        Returns:
            open_trades (deque[StockTrade]):
                Updated deque list of 'StockTrade' objects.
            completed_trades (list[dict[str, Any]]):
                List of dictionary containing required fields to generate DataFrame.
        """

        pass

    def _update_pos(
        self,
        trade: StockTrade,
        dt: datetime,
        exit_price: float,
        exit_lots: int | None = None,
    ) -> StockTrade:
        """Update existing StockTrade objects (still open).

        Args:
            trade (StockTrade):
                Existing StockTrade object for open trade.
            dt (datetime):
                Trade datetime object.
            exit_price (float):
                Exit price of stock ticker.
            exit_lots (int | None):
                If provided, number of lot to exit open position.

        Returns:
            (StockTrade): StockTrade object updated with exit info
        """

        # set exit_lots to be net entry and exit lots to ensure
        exit_lots = exit_lots or trade.entry_lots

        # Get exit action to update position
        exit_action = "sell" if trade.entry_action == "buy" else "buy"
        updated_trade = trade.model_copy()

        try:
            updated_trade.exit_datetime = dt
            updated_trade.exit_action = exit_action
            updated_trade.exit_lots = exit_lots
            updated_trade.exit_price = Decimal(str(exit_price))

            return updated_trade

        except ValidationError as e:
            print(f"Validation Error : {e}")
            return trade

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

    - Long stock at 50 (5 lots) -> 60 (3 lots) -> 70 2 lots).
    - Signal to take profit -> Sell 5 lots bought at 50 (i.e. FIFO).
    """

    def close_pos(
        self,
        open_trades: deque[StockTrade],
        dt: datetime,
        exit_price: float,
    ) -> tuple[deque[StockTrade], list[dict[str, Any]]]:
        """Update existing StockTrade objects (still open); and remove completed
        StockTrade objects in 'open_trades'.

        Args:
            open_trades (deque[StockTrade]):
                Deque list of StockTrade pydantic object to record open trades.
            dt (datetime):
                Trade datetime object.
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
        earliest_trade = open_trades[0]

        # Update earliest StockTrade object
        earliest_trade = self._update_pos(earliest_trade, dt, exit_price)

        # Convert StockTrade to dictionary only if all fields are populated
        # i.e. trade completed.
        if self._validate_completed_trades(earliest_trade):
            # Remove earliest StockTrade object since it is completed
            open_trades.popleft()

            completed_trades.append(earliest_trade.model_dump())

        return open_trades, completed_trades


class LIFOExit(ExitStruct):
    """Take profit for latest open trade. For example:

    - Long stock at 50 (5 lots) -> 60 (3 lots) -> 70 (2 lots).
    - Signal to take profit -> Sell 2 lots bought at 70 (i.e. FIFO).
    """

    def close_pos(
        self,
        open_trades: deque[StockTrade],
        dt: datetime,
        exit_price: float,
    ) -> tuple[deque[StockTrade], list[dict[str, Any]]]:
        """Update existing StockTrade objects (still open); and remove completed
        StockTrade objects in 'open_trades'.

        Args:
            open_trades (deque[StockTrade]):
                Deque list of StockTrade pydantic object to record open trades.
            dt (datetime):
                Trade datetime object.
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
        latest_trades = open_trades[-1]

        # Update earliest StockTrade object
        latest_trades = self._update_pos(latest_trades, dt, exit_price)

        # Convert StockTrade to dictionary only if all fields are populated
        # i.e. trade completed.
        if self._validate_completed_trades(latest_trades):
            # Remove latest StockTrade object from 'open_trades'
            open_trades.pop()

            completed_trades.append(latest_trades.model_dump())

        return open_trades, completed_trades


class HalfFIFOExit(ExitStruct):
    """keep taking profit by exiting half of earliest positions. For example:

    - Long stock at 50 (5 lots) -> 60 (3 lots) -> 70 (2 lots).
    - Signal to take profit -> Sell 5 lots (50% of total 10 lots) bought at 50
    i.e. left 60 (3 lots), 70 (2 lots)
    - Signal to take profit again -> sell 3 lots (50% of total 5 lots) bought at 60
    i.e. left 70 (2 lots).
    """

    def close_pos(
        self,
        open_trades: deque[StockTrade],
        dt: datetime,
        exit_price: float,
    ) -> tuple[deque[StockTrade], list[dict[str, Any]]]:
        """Update existing StockTrade objects (still open); and remove completed
        StockTrade objects in 'open_trades'.

        Args:
            open_trades (deque[StockTrade]):
                Deque list of StockTrade pydantic object to record open trades.
            dt (datetime):
                Trade datetime object.
            exit_price (float):
                Exit price of stock ticker.

        Returns:
            new_open_trades (deque[StockTrade]):
                Updated deque list of 'StockTrade' objects.
            completed_trades (list[dict[str, Any]]):
                List of dictionary containing required fields to generate DataFrame.
        """

        completed_trades = []
        new_open_trades = deque()

        if len(open_trades) == 0:
            # No open trades to close
            return open_trades, []

        # Get net position and half of net position from 'open_trades'
        net_pos = self.get_net_pos(open_trades)
        half_pos = math.ceil(abs(net_pos) / 2)

        for trade in open_trades:
            initial_exit_lots = trade.exit_lots

            # Update trade only if haven't reach half of net position
            if half_pos > 0:
                # Determine quantity to close based on 'half_pos'
                lots_to_exit = min(half_pos, trade.entry_lots - initial_exit_lots)

                # Update StockTrade objects with exit info
                trade = self._update_pos(
                    trade, dt, exit_price, initial_exit_lots + lots_to_exit
                )

                # Break loop if trade is not updated properly i.e.
                # trade.exit_lots = initial_exit_lots
                if trade.exit_lots == initial_exit_lots:
                    return open_trades, []

                # Only update 'new_open_trades' if trades are still partially closed
                if not self._validate_completed_trades(trade):
                    new_open_trades.append(trade)

                # Create completed trades using 'lots_to_exit'
                completed_trades.append(self._gen_completed_trade(trade, lots_to_exit))

                # Update remaining positions required to be closed and net position
                half_pos -= lots_to_exit

            # trade not updated
            else:
                new_open_trades.append(trade)

        return new_open_trades, completed_trades

    def _gen_completed_trade(
        self, trade: StockTrade, lots_to_exit: Decimal
    ) -> dict[str, Any]:
        """Generate StockTrade object with completed trade from 'StockTrade'
        and convert to dictionary."""

        # Create a shallow copy of the updated trade
        completed_trade = trade.model_copy()

        # Update the 'entry_lots' and 'exit_lots' to be same as 'lots_to_exit'
        completed_trade.entry_lots = lots_to_exit
        completed_trade.exit_lots = lots_to_exit

        if not self._validate_completed_trades(completed_trade):
            raise ValueError("Completed trades not properly closed.")

        return completed_trade.model_dump()

    def get_net_pos(self, open_trades: deque[StockTrade]) -> int:
        """Get net positions from 'open_trades'."""

        return sum(
            (
                trade.entry_lots - trade.exit_lots
                if trade.entry_action == "buy"
                else -(trade.entry_lots - trade.exit_lots)
            )
            for trade in open_trades
        )


class HalfLIFOExit(ExitStruct):
    """keep taking profit by exiting half of latest positions . For example:

    - Long stock at 50 (5 lots) -> 60 (3 lots) -> 70 (2 lots).
    - Signal to take profit -> Sell 2 lots bought at 70 and 3 lots bought at 60
    i.e. left 50 (5 lots)
    - Signal to take profit again -> sell 3 lots bought at 50
    i.e. left 50 (2 lots).
    """

    def close_pos(
        self,
        open_trades: deque[StockTrade],
        dt: datetime,
        exit_price: float,
    ) -> tuple[deque[StockTrade], list[dict[str, Any]]]:
        """Update existing StockTrade objects (still open); and remove completed
        StockTrade objects in 'open_trades'.

        Args:
            open_trades (deque[StockTrade]):
                Deque list of StockTrade pydantic object to record open trades.
            dt (datetime):
                Trade datetime object.
            exit_price (float):
                Exit price of stock ticker.

        Returns:
            new_open_trades (deque[StockTrade]):
                Updated deque list of 'StockTrade' objects.
            completed_trades (list[dict[str, Any]]):
                List of dictionary containing required fields to generate DataFrame.
        """

        completed_trades = []
        new_open_trades = deque()

        if len(open_trades) == 0:
            # No open trades to close
            return open_trades, []

        # Reverse copy of 'self.open_trades'
        open_trades_list = list(open_trades)
        reversed_open_trades = open_trades_list[::-1]

        # Get net position and half of net position from 'open_trades'
        net_pos = self.get_net_pos(open_trades)
        half_pos = math.ceil(abs(net_pos) / 2)

        for trade in reversed_open_trades:
            initial_exit_lots = trade.exit_lots

            # Update trade only if haven't reach half of net position
            if half_pos > 0:
                # Determine quantity to close based on 'half_pos'
                lots_to_exit = min(half_pos, trade.entry_lots - initial_exit_lots)

                # Update StockTrade objects with exit info
                trade = self._update_pos(
                    trade, dt, exit_price, initial_exit_lots + lots_to_exit
                )

                # Break loop if trade is not updated properly i.e.
                # trade.exit_lots = initial_exit_lots
                if trade.exit_lots == initial_exit_lots:
                    return open_trades, []

                # Only update 'new_open_trades' if trades are still partially closed
                if not self._validate_completed_trades(trade):
                    new_open_trades.append(trade)

                # Create completed trades using 'lots_to_exit'
                completed_trades.append(self._gen_completed_trade(trade, lots_to_exit))

                # Update remaining positions required to be closed and net position
                half_pos -= lots_to_exit

            # trade not updated
            else:
                new_open_trades.appendleft(trade)

        return new_open_trades, completed_trades

    def _gen_completed_trade(
        self, trade: StockTrade, lots_to_exit: Decimal
    ) -> dict[str, Any]:
        """Generate StockTrade object with completed trade from 'StockTrade'
        and convert to dictionary."""

        # Create a shallow copy of the updated trade
        completed_trade = trade.model_copy()

        # Update the 'entry_lots' and 'exit_lots' to be same as 'lots_to_exit'
        completed_trade.entry_lots = lots_to_exit
        completed_trade.exit_lots = lots_to_exit

        if not self._validate_completed_trades(completed_trade):
            raise ValueError("Completed trades not properly closed.")

        return completed_trade.model_dump()

    def get_net_pos(self, open_trades: deque[StockTrade]) -> int:
        """Get net positions from 'open_trades'."""

        return sum(
            (
                trade.entry_lots - trade.exit_lots
                if trade.entry_action == "buy"
                else -(trade.entry_lots - trade.exit_lots)
            )
            for trade in open_trades
        )


class TakeAllExit(ExitStruct):
    """Exit all open positions at a loss."""

    def close_pos(
        self,
        open_trades: deque[StockTrade],
        dt: datetime,
        exit_price: float,
    ) -> tuple[deque[StockTrade], list[dict[str, Any]]]:
        """Update existing StockTrade objects (still open); and remove completed
        StockTrade objects in 'open_trades'.

        Args:
            open_trades (deque[StockTrade]):
                Deque list of StockTrade pydantic object to record open trades.
            dt (datetime):
                Trade datetime object.
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

        for trade in open_trades:
            initial_exit_lots = trade.exit_lots

            # Update to close open position
            trade = self._update_pos(trade, dt, exit_price)

            # Break loop if trade is not updated properly i.e.
            # trade.exit_lots = initial_exit_lots
            if trade.exit_lots == initial_exit_lots:
                # Return original 'open_trades' and empty completed trades list
                # if trade is None i.e. ValidationError
                return open_trades, []

            if initial_exit_lots > 0:
                # Number of lots to close partial open position
                lots_to_exit = trade.entry_lots - initial_exit_lots

                # Create completed trades using 'lots_to_exit'
                completed_trades.append(self._gen_completed_trade(trade, lots_to_exit))

            else:
                # Generate completed trades for fully open position
                if self._validate_completed_trades(trade):
                    completed_trades.append(trade.model_dump())

        if len(completed_trades) != len(open_trades):
            raise ValueError(f"Open positions failed to close completely.")

        # Reset open_trades
        open_trades.clear()

        return open_trades, completed_trades

    def _gen_completed_trade(
        self, trade: StockTrade, lots_to_exit: Decimal
    ) -> dict[str, Any]:
        """Generate StockTrade object with completed trade from 'StockTrade'
        and convert to dictionary."""

        # Create a shallow copy of the updated trade
        completed_trade = trade.model_copy()

        # Update the 'entry_lots' and 'exit_lots' to be same as 'lots_to_exit'
        completed_trade.entry_lots = lots_to_exit
        completed_trade.exit_lots = lots_to_exit

        if not self._validate_completed_trades(completed_trade):
            raise ValueError("Completed trades not properly closed.")

        return completed_trade.model_dump()
