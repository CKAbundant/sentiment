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

from collections import deque
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import pandas as pd

from config.variables import EntryStruct, ExitStruct
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
        profit_struct (ExitStruct):
            Whether to apply first-in-first-out ("fifo"), last-in-first-out ("lifo"),
            take profit for half open positions repeatedly ("half_life") or
            take profit for all open positions ("take_all").
        num_lots (int):
            Number of lots to initiate new position each time (Default: 1).
        req_cols (list[str]):
            List of required columns to generate trades.

    Attributes:
        entry_struct (EntryStruct):
            Whether to allow multiple open position ("mulitple") or single
            open position at a time ("single") (Default: "multiple").
        profit_struct (ExitStruct):
            Whether to apply first-in-first-out ("fifo"), last-in-first-out ("lifo"),
            take profit for half open positions repeatedly ("half_life") or
            take profit for all open positions ("take_all") (Default: "take_all").
        num_lots (int):
            Number of lots to initiate new position each time (Default: 1).
        net_pos (int):
            Net open position. Positive value = net long while negative value =
            net short.
        open_trades (deque[StockTrade]):
            List of open trades containing StockTrade pydantic object.
        req_cols (list[str]):
            List of required columns to generate trades.
        no_trades (list[str]):
            List containing stock tickers with no completed trades.
    """

    def __init__(
        self,
        entry_struct: EntryStruct = "multiple",
        profit_struct: ExitStruct = "take_all",
        num_lots: int = 1,
        req_cols: list[str] = [
            "date",
            "ticker",
            "coint_corr_ticker",
            "close",
            "entry_signal",
            "exit_signal",
        ],
    ) -> None:
        super().__init__(entry_struct, profit_struct, num_lots)
        self.req_cols = req_cols
        self.no_trades = []

    def gen_trades(self, df_news: pd.DataFrame) -> pd.DataFrame:
        """Generate DataFrame containing completed trades for trading strategy."""

        # FIlter out null values for OHLC due to weekends and holiday
        df = df_news.loc[~df_news["close"].isna(), self.req_cols].copy()

        completed_trades = []

        for idx, record in df.itertuples(index=True, name=None):
            dt, _, c_ticker, close, ent_sig, ex_sig = record

            # Close off all open positions at end of trading period
            if idx >= len(df) - 1 and self.net_pos != 0:
                completed_trades.extend(self.update_via_take_all(dt, ex_sig, close))

            # Signal to close existing open positions i.e. net position not equal to 0
            elif (ex_sig == "sell" or ex_sig == "buy") and self.net_pos != 0:
                # Determine if exit signal is taking profit or stop loss
                if self._is_loss(close, ex_sig):
                    completed_trades.extend(self.update_via_take_all)

                else:
                    # Current closing price is higher than latest entry price
                    # i.e. running at a profit

                    match self.exit_struct:
                        case "fifo":
                            completed_trades.extend(
                                self.update_via_fifo_or_lifo(dt, ex_sig, close, "fifo")
                            )
                        case "lifo":
                            completed_trades.extend(
                                self.update_via_fifo_or_lifo(dt, ex_sig, close, "lifo")
                            )
                        case "half_life":
                            completed_trades.extend(
                                self.update_via_half_life(dt, ex_sig, close)
                            )
                        case _:
                            completed_trades.extend(
                                self.update_via_take_all(dt, ex_sig, close)
                            )

            # Signal to enter new position
            elif ent_sig == "buy" or ent_sig == "sell":
                self.open_new_pos(dt, c_ticker, close, ent_sig)

            else:
                # No signal to initate new open position; or no open positions to close
                continue

        # No completed trades recorded
        if not completed_trades:
            news_ticker = df["ticker"].iloc[-1]
            self.no_trades.append(news_ticker)

        # Convert to DataFrame
        df = pd.DataFrame(completed_trades)
        df
