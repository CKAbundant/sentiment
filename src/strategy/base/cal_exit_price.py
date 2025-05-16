"""Class to calculate single stop price for all open positions based
on price movements."""

import math
from abc import ABC, abstractmethod
from collections import Counter, deque
from decimal import Decimal

from src.strategy.base.stock_trade import StockTrade
from src.utils.strategy_utils import get_std_field


class CalExitPrice(ABC):
    """Abstract class to generate exit price (i.e. either profit or stop loss)
    for multiple open positions based on price movement.

    Args:
        percent_loss (float):
            Percentage loss allowed for investment (Default: 0.2).

    Attributes:
        percent_loss (Decimal):
            Percentage loss allowed for investment (Default: 0.2).
    """

    def __init__(self, percent_loss: float = 0.2) -> None:
        self.percent_loss = Decimal(str(percent_loss))

    @abstractmethod
    def cal_exit_price(self, open_trades: deque[StockTrade]) -> Decimal:
        """Calculate a single exit price for multiple open positions.

        Args:
            open_trades (deque[StockTrade]):
                Deque list of StockTrade containing open trades info.

        Returns:
            (Decimal): Exit price for all multiple open positions.
        """

        pass


class PercentLoss(CalExitPrice):
    """Compute stop price such that total loss for all open positions
    is within the accepted percent loss. For example:

    - Long 50 (5 lots), 60 (3 lots), 70 (2 lots).
    - If percentage loss is 20%, then stop price = 57
    - 57 * 8 (total lots) = 352 = 80% of total investment of 570
    """

    def cal_exit_price(self, open_trades: deque[StockTrade]) -> Decimal:
        """Calculate stop price to meet percent loss for total investment.

        Args:
            open_trades (deque[StockTrade]):
                Deque list of StockTrade containing open trades info.

        Returns:
            (Decimal): Exit price for all multiple open positions.
        """

        entry_action = get_std_field(open_trades, "entry_action")

        # Current investment = investment value of current open positions
        cur_invest = sum(
            trade.entry_price * (trade.entry_lots - trade.exit_lots)
            for trade in open_trades
        )

        # Total number of open position
        total_open = self.get_net_pos(open_trades)

        # Compute stop price to meet stipulated percent loss
        stop_price = (
            cur_invest * (1 - self.percent_loss) / total_open
            if entry_action == "buy"
            else cur_invest * (1 + self.percent_loss) / total_open
        )

        return round(stop_price, 2)

    def get_net_pos(self, open_trades: deque[StockTrade]) -> int:
        """Get net positions from 'self.open_trades'."""

        return sum(trade.entry_lots - trade.exit_lots for trade in open_trades)


class LatestLoss(CalExitPrice):
    """Compute stop price based on the percentage stop loss set
    for latest open trade. For example:

    - Long 50 (5 lots), 60 (3 lots), 70 (2 lots)
    - latest trade = 70 (2 lots)
    - If percentage loss is 20%, then stop price = 0.8 * 70 = 56
    """

    def cal_exit_price(self, open_trades: deque[StockTrade]) -> Decimal:
        """Calculate stop price based on latest open position.

        Args:
            open_trades (deque[StockTrade]):
                Deque list of StockTrade containing open trades info.

        Returns:
            (Decimal): Exit price for all multiple open positions.
        """

        if len(open_trades) == 0:
            raise ValueError(f"'open_trades' cannot be empty.")

        # Get entry action and latest entry price
        entry_action = get_std_field(open_trades, "entry_action")
        latest_price = open_trades[-1].entry_price

        # Compute stop price to meet stipulated percent loss
        stop_price = (
            latest_price * (1 - self.percent_loss)
            if entry_action == "buy"
            else latest_price * (1 + self.percent_loss)
        )

        return Decimal(round(stop_price, 2))


class NearestLoss(CalExitPrice):
    """Use stop price (based on the percentage loss) that is nearest to
    current monitor price. For example:

    - Long 50 (5 lots),  70 (2 lots), 60 (3 lots)
    - Closest stop price for each position = [40, 56, 48]
    - Nearest stop price to current close = 56
    """

    def cal_exit_price(self, open_trades: deque[StockTrade]) -> Decimal:
        """Use highest stop price for long position and lowest stop price
        for short position.

        Args:
            open_trades (deque[StockTrade]):
                Deque list of StockTrade containing open trades info.

        Returns:
            (Decimal): Exit price for all multiple open positions.
        """

        if len(open_trades) == 0:
            raise ValueError(f"'open_trades' cannot be empty.")

        # Get entry action
        entry_action = get_std_field(open_trades, "entry_action")

        # Generate list of stop price for each open position
        stop_list = [
            (
                trade.entry_price * (1 - self.percent_loss)
                if entry_action == "buy"
                else trade.entry_price * (1 + self.percent_loss)
            )
            for trade in open_trades
        ]

        # Use highest stop price for long position and lowest stop price
        # for short position
        stop_price = max(stop_list) if entry_action == "buy" else min(stop_list)

        return Decimal(round(stop_price, 2))
