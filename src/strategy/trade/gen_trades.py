"""Concrete implementation of 'GenTrades' abstract classes.

- fixed_percent -> Use maximum, mean or median percentage drawdown
(based on entry price) for stop loss; and percentage gain for profit
- trailing -> Use percentage from previous day high.
- fibo -> Use nearest fibonannci level as profit and stop loss level.

Note that:

1. For each trade generation, user can choose whether to enter multiple
positions or only maintain a single open position
- For example, if only long, then new long position can only be initiated
after the initial long position is closed.

2. All open positions will be closed if the closest stop loss is triggered.
- For example, we have 3 open positions have stop loss 95, 98 and 100; and stock
is trading at 120.
- If stock traded below 100, then all 3 open positions will be closed.

3. Profit can be taken on per trade basis or taken together if exit signal is present.
- For example, we have 3 open positions, 95, 98 and 100.
- If exit signal is triggered at 150, then we can choose to close the first trade (FIFO)
i.e. 95, and leave the 2 to run till the next exit signal ('fifo' profit).
- Or we can choose to close off all position at profit ('take_all' profit).
- Or we can choose to take 50% of all position at profit; and repeat till all profits
taken ('half_life' profit).

4. Profit or stop loss taken at the closing price of the same day as signal generated
unless specified otherwise.
"""

import pandas as pd

from config.variables import EntryStruct, ProfitStruct
from src.strategy.base import GetTrades, StockTrade


class SentiTrades(GetTrades):
    """Generate completed trades using sentiment rating strategy.

    - Get daily median sentiment rating (excluding rating 3) for stock ticker.
    - Perform buy on cointegrated/correlated ticker if median rating >= 4.
    - Perform sell on cointegrated/correlated ticker if median rating <= 2.

    Usage:
        # df = DataFrame containg sentiment rating and OHLCV prices
        >>> trades = SentiTrades()
        >>> df_results = trades.gen_trades(df)

    Args:
        entry_struct (EntryStruct):
            Whether to allow multiple open position ("mulitple") or single
            open position at a time ("single").
        profit_struct (ProfitStruct):
            Whether to apply first-in-first-out ("fifo"), last-in-first-out ("lifo"),
            take profit for half open positions repeatedly ("half_life") or
            take profit for all open positions ("take_all").
        num_lots (int):
            Number of lots to initiate new position each time (Default: 1).

    Attributes:
        entry_struct (EntryStruct):
            Whether to allow multiple open position ("mulitple") or single
            open position at a time ("single").
        profit_struct (ProfitStruct):
            Whether to apply first-in-first-out ("fifo"), last-in-first-out ("lifo"),
            take profit for half open positions repeatedly ("half_life") or
            take profit for all open positions ("take_all").
        num_lots (int):
            Number of lots to initiate new position each time (Default: 1).
        open_trades (list[StockTrade]):
            List of open trades containing StockTrade pydantic object.
    """

    def __init__(
        self,
        entry_struct: EntryStruct = "multiple",
        profit_struct: ProfitStruct = "take_all",
        num_lots: int = 1,
    ) -> None:
        super().__init__(entry_struct, profit_struct, num_lots)
        self.open_trades = []

    def gen_trades(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate DataFrame containing completed trades for trading strategy."""
