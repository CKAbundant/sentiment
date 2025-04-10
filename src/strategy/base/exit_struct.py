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
