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
