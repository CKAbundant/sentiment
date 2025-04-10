"""Abstract class and concrete implementation of various
entry stuctures."""

import math
from abc import ABC, abstractmethod
from collections import deque
from datetime import date
from decimal import Decimal

from config.variables import PriceAction

from .stock_trade import StockTrade


class EntryStruct(ABC):
    """Abstract class to populate 'StockTrade' pydantic object to record
    open trades.

    Args:
        num_lots (int):
            Default number of lots to enter each time.

    Attributes:
        num_lots (int):
            Default number of lots to enter each time.
    """

    def __init__(self, num_lots: int) -> None:
        self.num_lots = num_lots

    @abstractmethod
    def open_new_pos(
        self,
        open_trades: deque[StockTrade],
        ticker: str,
        dt: date,
        entry_price: float,
        ent_sig: PriceAction,
    ) -> deque[StockTrade]:
        """Generate new 'StockTrade' object populating 'ticker', 'entry_date',
        'entry_lots' and 'entry_price'.

        Args:
            open_trades (deque[StockTrade]):
                Deque list of StockTrade pydantic object to record open trades.
            ticker (str):
                Stock ticker to be traded.
            dt (date):
                Trade date object.
            entry_price (float):
                Entry price for stock ticker.
            ent_sig (PriceAction):
                Entry signal i.e. "buy", "sell" or "wait" to create new position.

        Returns:
            open_trades (deque[StockTrade]):
                Updated deque list of 'StockTrade' objects.
        """
        pass

    def _validate_open_trades(sel, open_trades: deque[StockTrade]) -> None:
        """Validate StockTrade objects in 'self.open_trade'.

        -'entry_action' fields are  same for all StockTrade objects.
        - 'ticker' fields are same for all StockTrade objects.
        - 'entry_date' should be later than the latest StockTrade object
        in 'open_trades'.
        """

        if len(open_trades) == 0:
            raise ValueError(
                "'open_trades' is still empty after creating new position."
            )

        # Get 'entry_action' and 'ticker' from 1st item in 'open_trades'
        first_action = open_trades[0].entry_action
        first_ticker = open_trades[0].ticker

        # Get entry dates in 'open_trades'
        entry_dates = [open_trade.entry_date for open_trade in open_trades]

        if any(open_trade.entry_action != first_action for open_trade in open_trades):
            raise ValueError(
                "'entry_action' field is not the same for all open trades."
            )

        if any(open_trade.ticker != first_ticker for open_trade in open_trades):
            raise ValueError("'ticker' field is not the same for all open trades.")

        if any(
            entry_dates[idx] > entry_dates[idx + 1]
            for idx in range(len(entry_dates) - 1)
        ):
            raise ValueError(
                "'entry_date' field is not sequential i.e. entry_date is lower than the entry_date in the previous item."
            )


class MultiEntry(EntryStruct):
    """Allows multiple positions of same ticker i.e. new 'StockTrade' pydantic
    objects with same entry lots.

    - Add new long/short positions even if existing long/short positions.
    - Number of lots entered are fixed to 'self.num_lots'.

    Usage:
        >>> ticker = "AAPL"
        >>> dt = date(2025, 4, 11)
        >>> entry_price = 200.0
        >>> ent_sig = "buy"
        >>> multi_entry = MultiEntry(num_lots=1)
        >>> open_trades = multi_entry.open_new_pos(ticker, dt, entry_price, ent_sig)

    Args:
        num_lots (int):
            Default number of lots to enter each time (Default: 1).

    Attributes:
        num_lots (int):
            Default number of lots to enter each time (Default: 1).
    """

    def __init__(self, num_lots: int = 1) -> None:
        super().__init__(num_lots)

    def open_new_pos(
        self,
        open_trades: deque[StockTrade],
        ticker: str,
        dt: date,
        entry_price: float,
        ent_sig: PriceAction,
    ):
        """Generate new 'StockTrade' object populating 'ticker', 'entry_date',
        'entry_lots' and 'entry_price'.

        Args:
            open_trades (deque[StockTrade]):
                Deque list of StockTrade pydantic object to record open trades.
            ticker (str):
                Stock ticker to be traded.
            dt (date):
                Trade date object.
            entry_price (float):
                Entry price for stock ticker.
            ent_sig (PriceAction):
                Entry signal i.e. "buy", "sell" or "wait" to create new position.

        Returns:
            open_trades (deque[StockTrade]):
                Updated deque list of 'StockTrade' objects.
        """

        # Create StockTrade object to record new long/short position
        # based on 'ent_sig'
        stock_trade = StockTrade(
            ticker=ticker,
            entry_date=dt,
            entry_action=ent_sig,
            entry_lots=Decimal(str(self.num_lots)),
            entry_price=Decimal(str(entry_price)),
        )
        open_trades.append(stock_trade)

        self._validate_open_trades(open_trades)

        return open_trades


class MultiHalfEntry(EntryStruct):
    """Allows multiple positions of same ticker i.e. new 'StockTrade' pydantic
    objects with entry lots decreasing by half with each multiple entry.

    - Add new long/short positions even if existing long/short positions.
    - Number of lots entered decreases from 'self.num_lots' by half e.g.
    10 -> 5 -> 2 -> 1
    - Only if 'self.num_lots' > 1; and no fractional lots allowed.

    Usage:
        >>> ticker = "AAPL"
        >>> dt = date(2025, 4, 11)
        >>> entry_price = 200.0
        >>> ent_sig = "buy"
        >>> multi_entry = MultiHalfEntry(num_lots=1)
        >>> open_trades = multi_entry.open_new_pos(ticker, dt, entry_price, ent_sig)

    Args:
        num_lots (int):
            Default number of lots to enter each time (Default: 1).

    Attributes:
        num_lots (int):
            Default number of lots to enter each time (Default: 1).
    """

    def __init__(self, num_lots: int = 1) -> None:
        super().__init__(num_lots)

    def open_new_pos(
        self,
        open_trades: deque[StockTrade],
        ticker: str,
        dt: date,
        entry_price: float,
        ent_sig: PriceAction,
    ) -> deque[StockTrade]:
        """Generate new 'StockTrade' object populating 'ticker', 'entry_date',
        'entry_lots' and 'entry_price'.

        Args:
            open_trades (deque[StockTrade]):
                Deque list of StockTrade pydantic object to record open trades.
            ticker (str):
                Stock ticker to be traded.
            dt (date):
                Trade date object.
            entry_price (float):
                Entry price for stock ticker.
            ent_sig (PriceAction):
                Entry signal i.e. "buy", "sell" or "wait" to create new position.

        Returns:
            open_trades (deque[StockTrade]):
                Updated deque list of 'StockTrade' objects.
        """

        # Get number of lots to enter for new position
        entry_lots = self.get_half_lots(open_trades)

        # Create StockTrade object to record new long/short position
        # based on 'ent_sig'
        stock_trade = StockTrade(
            ticker=ticker,
            entry_date=dt,
            entry_action=ent_sig,
            entry_lots=Decimal(str(entry_lots)),
            entry_price=Decimal(str(entry_price)),
        )
        open_trades.append(stock_trade)

        self._validate_open_trades(open_trades)

        return open_trades

    def get_half_lots(self, open_trades: deque[StockTrade]) -> Decimal:
        """Get half of latest StockTrade object in 'open_trades' till
        minimum 1 lot."""

        latest_entry_lots = (
            open_trades[-1].entry_lots if len(open_trades) > 0 else self.num_lots
        )

        return math.ceil(latest_entry_lots / 2)
