"""Class to compute the stop price for multiple position such that
losses will not exceed stipulated percentage drawdown. For example:

```
long stock A at 50 (5 lots), 60 (2 lots), 70 (1 lot)
Total investment = 50 * 5 + 60 * 2 + 70 * 1 = 440

if percentage drawdown = 0.2, we have:
Minimum acceptable investment value = total investment * (1-0.2) = 352

Therefore stop price = 352 / (5 + 2 + 1) = 44

Let's verify: If stock A price drop in value to 44 and we exit all 8 lots
at 44, total investment value will be:

44 * 5 + 44 * 2 + 44 * 1 = 352 (i.e. 80% of initial investment of 440)
```
"""

from collections import deque
from decimal import Decimal

from src.strategy.base import StockTrade


class CalStopPrice:
    """Calculate stop price to exit all open position if investment value drops by
    stipulated percentage loss.

    Usage:
        # open_trades = List of StockTrade pydantic object tracking open positions
        >>> cal_stop_price = PercentStopLoss(percent_loss=0.2)
        >>> stop_price = cal_stop_price(open_trades)

    Args:
        percent_loss (float):
            Percentage loss in investment value (Default: 0.2).

    Attributes:
        percent_loss (float):
            Percentage loss in investment value (Default: 0.2).
    """

    def __init__(self, percent_loss: float = 0.2) -> None:
        self.percent_loss = percent_loss

    def __call__(self, open_trades: deque[StockTrade]) -> Decimal:
        """Compute required stop price to keep investment losses within stipulated
        percentage loss.

        Args:
            open_trades (deque[StockTrade]):
                Deque collection of StockTrade pydantic object containing trade info
                for open positions.

        Returns:
            (Decimal): Required stop price to monitor.
        """
