"""Concrete implementation of 'GetTrades' abstract classes.

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

from collections import Counter
from datetime import datetime
from decimal import Decimal
from pprint import pformat
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from config.variables import EntryMethod, ExitMethod
from src.strategy.base import GenTrades


class SentiTrades(GenTrades):
    """Generate completed trades using sentiment rating strategy.

    - Get daily median sentiment rating (excluding rating 3) for stock ticker.
    - Perform buy on cointegrated/correlated ticker if median rating >= 4.
    - Perform sell on cointegrated/correlated ticker if median rating <= 2.

    Usage:
        # df = DataFrame containing sentiment rating and OHLCV prices
        >>> trades = SentiTrades()
        >>> df_results = trades.gen_trades(df)
    """

    def __init__(
        self,
        entry_struct: EntryMethod = "MultiEntry",
        exit_struct: ExitMethod = "TakeAllExit",
        num_lots: int = 1,
        monitor_close: bool = True,
        percent_loss: float = 0.05,
        stop_method: ExitMethod = "no_stop",
        trigger_trail: float | None = None,
        step: float | None = None,
        entry_struct_path: str = "./src/strategy/base/entry_struct.py",
        exit_struct_path: str = "./src/strategy/base/exit_struct.py",
        stop_method_path: str = "./src/strategy/base/cal_exit_price.py",
    ) -> None:
        super().__init__(
            entry_struct,
            exit_struct,
            num_lots,
            monitor_close,
            percent_loss,
            stop_method,
            trigger_trail,
            step,
            entry_struct_path,
            exit_struct_path,
            stop_method_path,
        )

    def gen_trades(self, df_senti: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Generate DataFrame containing completed trades for trading strategy.

        Args:
            df_senti (pd.DataFrame):
                DataFrame containing entry and exit signals based on sentiment rating.

        Returns:
            df_trades (pd.DataFrame):
                DataFrame containing completed trades.
            df_signals (pd.DataFrame):
                DataFrame containing updated exit signals based on price-related stops.
        """

        # Get news ticker and cointegrated/correlated ticker
        ticker = self.get_ticker(df_senti, "ticker")
        coint_corr_ticker = self.get_ticker(df_senti, "coint_corr_ticker")

        # Generate completed trades and updated signal DataFrame
        df_trades, df_senti = self.iterate_df(coint_corr_ticker, df_senti)
        df_trades.insert(0, "news_ticker", ticker)

        # Assume positions are opened or closed at market closing (1600 hrs New York)
        df_senti = self.set_mkt_cls_dt(df_senti)

        return df_trades, df_senti

    def get_ticker(self, df_senti: pd.DataFrame, ticker_col: str) -> str:
        """Get news ticker or cointegrated/correlated stock ticker with news ticker.

        Args:
            df_senti (pd.DataFrame): DataFrame containing sentiment rating.
            ticker_col (str): Name of column either 'ticker' or 'coint_corr_ticker'.

        Returns:
            (str): Name of stock ticker or cointegrated/correlated ticker.
        """

        ticker_counter = Counter(df_senti[ticker_col])

        if len(ticker_counter) > 1:
            raise ValueError(
                f"More than 1 {ticker_col} found : {ticker_counter.keys()}"
            )

        return list(ticker_counter.keys())[0]

    def set_mkt_cls_dt(self, df_signal: pd.DataFrame) -> pd.DataFrame:
        """Set datetime to NYSE closing time i.e. assume position open or closed
        only at 1600 hrs (New York Time)."""

        df = df_signal.copy()

        if "date" not in df.columns:
            raise ValueError("'date' column is not found in DataFrame.")

        df["date"] = pd.to_datetime(df["date"])
        df["date"] = df["date"].map(
            lambda dt: dt.replace(
                hour=16, minute=0, tzinfo=ZoneInfo("America/New_York")
            )
        )

        return df
