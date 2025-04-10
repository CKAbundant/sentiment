"""Abstract class and concrete implementation of various
entry stuctures."""

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
            Default number of lots to enter each time (Default: 1).

    Attributes:
        num_lots (int):
            Default number of lots to enter each time (Default: 1).
    """

    def __init__(self, num_lots: int = 1) -> None:
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

    def _validate_open_trades(self) -> bool:
        """Validate whether 'entry_action' field is the same for all StockTrade
        objects in 'self.open_trades'."""

        if len(self.open_trades) == 0:
            # No open trades available
            return False

        # Get 'entry_action' from 1st item in 'self.open_trades'
        first_action = self.open_trades[0].entry_action

        return all(
            [open_trade.entry_action == first_action for open_trade in self.open_trades]
        )


class MultiEntry(EntryStruct):
    """Allows multiple positions of same ticker i.e. new 'StockTrade' pydantic
    objects with same 'entry_action'.

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

    def __init__(self, num_lots: int) -> None:
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
        self.net_pos += self.num_lots if ent_sig == "buy" else -self.num_lots

        if not self._validate_open_trades():
            raise ValueError(
                f"'self.open_trades' is still empty or 'entry_action' is not consistent."
            )
