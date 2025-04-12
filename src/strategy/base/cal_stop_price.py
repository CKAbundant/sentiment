"""Class to calculate single stop price for all open positions based
on price movements."""

from abc import ABC, abstractmethod
from collections import deque
from decimal import Decimal

from src.strategy.base.stock_trade import StockTrade


class CalStopPrice(ABC):
    """Abstract class to generate stop price for multiple open positions
    based on price movement.

    Args:
        None.

    Attributes:
        None.
    """

    @abstractmethod
    def cal_stop_price(self, open_trades: deque[StockTrade]) -> Decimal:
        """Calculate a single stop price for multiple open positions.

        Args:
            open_trades (deque[StockTrade]):
                Deque list of StockTrade containing open trades info.

        Returns:
            (Decimal): Stop price for all multiple open positions.
        """

        pass


class PercentLoss(CalStopPrice):
    """Compute stop price such that maximum losses for all open positions
    is within the accepted percent loss.

    Args:


    """

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
