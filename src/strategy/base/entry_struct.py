"""Abstract class and concrete implementation of various
entry stuctures."""

import math
from abc import ABC, abstractmethod
from collections import Counter, deque
from datetime import date, datetime
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
        dt: date | str,
        ent_sig: PriceAction,
        entry_price: float,
    ) -> deque[StockTrade]:
        """Generate new 'StockTrade' object populating 'ticker', 'entry_date',
        'entry_lots' and 'entry_price'.

        Args:
            open_trades (deque[StockTrade]):
                Deque list of StockTrade pydantic object to record open trades.
            ticker (str):
                Stock ticker to be traded.
            dt (date | str):
                Trade date object or string in "YYYY-MM-DD" format.
            ent_sig (PriceAction):
                Entry signal i.e. "buy", "sell" or "wait" to create new position.
            entry_price (float):
                Entry price for stock ticker.

        Returns:
            open_trades (deque[StockTrade]):
                Updated deque list of 'StockTrade' objects.
        """
        pass

    def _create_new(
        self,
        open_trades: deque[StockTrade],
        ticker: str,
        dt: date | str,
        ent_sig: PriceAction,
        entry_price: float,
        entry_lots: int | None = None,
    ) -> StockTrade:
        """Generate new 'StockTrade' object populating 'ticker', 'entry_date',
        'entry_lots' and 'entry_price'.

        Args:
            open_trades (deque[StockTrade]):
                Deque list of StockTrade pydantic object to record open trades.
            ticker (str):
                Stock ticker to be traded.
            dt (date | str):
                Trade date object or string in "YYYY-MM-DD" format.
            ent_sig (PriceAction):
                Entry signal i.e. "buy", "sell" or "wait" to create new position.
            entry_price (float):
                Entry price for stock ticker.
            entry_lots (int | None):
                If provided, entry lots to enter for 'MultiHalfEntry' entry structure.

        Returns:
            (StockTrade): Newly created StockTrade object.
        """

        entry_lots = entry_lots or self.num_lots

        return StockTrade(
            ticker=self._validate_ticker(open_trades, ticker),
            entry_date=self._validate_entry_date(open_trades, dt),
            entry_action=self._validate_entry_action(open_trades, ent_sig),
            entry_lots=Decimal(str(entry_lots)),
            entry_price=Decimal(str(entry_price)),
        )

    def _validate_ticker(self, open_trades: deque[StockTrade], ticker: str) -> str:
        """Validate ticker is the same for all StockTrade objects in 'open_trades.

        Args:
            open_trades (deque[StockTrade]):
                Deque collection of open trades.
            ticker (str):
                Stock ticker used to create new open trade.

        Return:
            ticker (str):
                Validated stock ticker used to create new open trade.
        """

        if len(open_trades) == 0:
            return ticker

        # Check if ticker used is the same as latest ticker in 'open_trades'
        latest_ticker = open_trades[-1].ticker
        if ticker != latest_ticker:
            raise ValueError(
                f"'{ticker}' is different from ticker used in 'open_trades' ({latest_ticker})"
            )

        return ticker

    def _validate_entry_action(
        self, open_trades: deque[StockTrade], entry_action: PriceAction
    ) -> PriceAction:
        """Validate entry action is the same for all StockTrade objects in 'open_trades.

        Args:
            open_trades (deque[StockTrade]):
                Deque collection of open trades.
            entry_action (PriceAction):
                Entry action used to create new open trade.

        Return:
            entry_action (PriceAction):
                Entry action used to create new open trade.
        """

        if len(open_trades) == 0:
            return entry_action

        # Check if 'entry_action' is the same used latest entry in 'open_trades'
        latest_action = open_trades[-1].entry_action
        if entry_action != latest_action:
            raise ValueError(
                f"'{entry_action}' is different from entry action used in 'open_trades' ({latest_action})"
            )

        return entry_action

    def _validate_entry_date(
        self, open_trades: deque[StockTrade], entry_date: date | str
    ) -> str:
        """Validate entry date is the same for all StockTrade objects in 'open_trades.

        Args:
            open_trades (deque[StockTrade]):
                Deque collection of open trades.
            entry_date (date | str):
                Entry date used to create new open trade ("YYYY-MM-DD" if string).

        Return:
            entry_date (str):
                Validated entry date used to create new open trade.
        """

        if isinstance(entry_date, str):
            entry_date = datetime.strptime(entry_date, "%Y-%m-%d").date()

        if len(open_trades) == 0:
            return entry_date

        # Check if entry date is less than latest entry date in 'open_trades'
        latest_date = open_trades[-1].entry_date
        if entry_date < latest_date:
            raise ValueError(
                f"Entry date '{entry_date}' is earlier than latest entry date '{latest_date}'."
            )

        return entry_date

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

        # Get Counter for ticker and entry_action in 'open_trades'
        ticker_counter = Counter([trade.ticker for trade in open_trades])
        action_counter = Counter([trade.entry_action for trade in open_trades])

        if len(action_counter) > 1:
            raise ValueError(
                "'entry_action' field is not the same for all open trades."
            )

        if len(ticker_counter) > 1:
            raise ValueError("'ticker' field is not the same for all open trades.")

        entry_dates = [trade.entry_date for trade in open_trades]

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
        >>> open_trades = deque()
        >>> ticker = "AAPL"
        >>> dt = date(2025, 4, 11)
        >>> entry_price = 200.0
        >>> ent_sig = "buy"
        >>> multi_entry = MultiEntry(num_lots=1)
        >>> open_trades = multi_entry.open_new_pos(open_trades, ticker, dt, entry_price, ent_sig)

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
        dt: date | str,
        ent_sig: PriceAction,
        entry_price: float,
    ):
        """Generate new 'StockTrade' object populating 'ticker', 'entry_date',
        'entry_lots' and 'entry_price'.

        Args:
            open_trades (deque[StockTrade]):
                Deque list of StockTrade pydantic object to record open trades.
            ticker (str):
                Stock ticker to be traded.
            dt (date | str):
                Trade date object or string in "YYYY-MM-DD" format.
            ent_sig (PriceAction):
                Entry signal i.e. "buy", "sell" or "wait" to create new position.
            entry_price (float):
                Entry price for stock ticker.

        Returns:
            open_trades (deque[StockTrade]):
                Updated deque list of 'StockTrade' objects.
        """

        # Create StockTrade object to record new long/short position
        # based on 'ent_sig'
        stock_trade = self._create_new(open_trades, ticker, dt, ent_sig, entry_price)
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
        >>> open_trades = deque()
        >>> ticker = "AAPL"
        >>> dt = date(2025, 4, 11)
        >>> entry_price = 200.0
        >>> ent_sig = "buy"
        >>> multi_entry = MultiHalfEntry(num_lots=1)
        >>> open_trades = multi_entry.open_new_pos(open_trades, ticker, dt, entry_price, ent_sig)

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
        dt: date | str,
        ent_sig: PriceAction,
        entry_price: float,
    ) -> deque[StockTrade]:
        """Generate new 'StockTrade' object populating 'ticker', 'entry_date',
        'entry_lots' and 'entry_price'.

        Args:
            open_trades (deque[StockTrade]):
                Deque list of StockTrade pydantic object to record open trades.
            ticker (str):
                Stock ticker to be traded.
            dt (date | str):
                Trade date object or string in "YYYY-MM-DD" format.
            ent_sig (PriceAction):
                Entry signal i.e. "buy", "sell" or "wait" to create new position.
            entry_price (float):
                Entry price for stock ticker.

        Returns:
            open_trades (deque[StockTrade]):
                Updated deque list of 'StockTrade' objects.
        """

        # Get number of lots to enter for new position
        entry_lots = self.get_half_lots(open_trades)

        # Create StockTrade object to record new long/short position
        # based on 'ent_sig'
        stock_trade = self._create_new(
            open_trades, ticker, dt, ent_sig, entry_price, entry_lots
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


class SingleEntry(EntryStruct):
    """Allows only 1 position of same ticker i.e. no new open position created
    if there is existing open position.

    - Number of lots entered are fixed to 'self.num_lots'.

    Usage:
        >>> open_trades = deque()
        >>> ticker = "AAPL"
        >>> dt = date(2025, 4, 11)
        >>> entry_price = 200.0
        >>> ent_sig = "buy"
        >>> single_entry = SingleEntry(num_lots=1)
        >>> open_trades = single_entry.open_new_pos(open_trades, ticker, dt, entry_price, ent_sig)

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
        dt: date | str,
        ent_sig: PriceAction,
        entry_price: float,
    ):
        """Generate new 'StockTrade' object populating 'ticker', 'entry_date',
        'entry_lots' and 'entry_price'.

        Args:
            open_trades (deque[StockTrade]):
                Deque list of StockTrade pydantic object to record open trades.
            ticker (str):
                Stock ticker to be traded.
            dt (date | str):
                Trade date object or string in "YYYY-MM-DD" format.
            ent_sig (PriceAction):
                Entry signal i.e. "buy", "sell" or "wait" to create new position.
            entry_price (float):
                Entry price for stock ticker.

        Returns:
            open_trades (deque[StockTrade]):
                Updated deque list of 'StockTrade' objects.
        """

        if len(open_trades) > 0:
            # No new position added since there is existing position
            return open_trades

        # Create StockTrade object to record new long/short position
        # based on 'ent_sig'
        stock_trade = self._create_new(open_trades, ticker, dt, ent_sig, entry_price)
        open_trades.append(stock_trade)
        self._validate_open_trades(open_trades)

        return open_trades
