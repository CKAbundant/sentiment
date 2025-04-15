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
from typing import Any

import pandas as pd

from config.variables import EXIT_PRICE_MAPPING, EntryMethod, ExitMethod, PriceAction
from src.strategy.base import GenTrades
from src.utils.utils import get_class_instance, get_std_field


class SentiTrades(GenTrades):
    """Generate completed trades using sentiment rating strategy.

    - Get daily median sentiment rating (excluding rating 3) for stock ticker.
    - Perform buy on cointegrated/correlated ticker if median rating >= 4.
    - Perform sell on cointegrated/correlated ticker if median rating <= 2.

    Usage:
        # df = DataFrame containg sentiment rating and OHLCV prices
        >>> trades = SentiTrades()
        >>> df_results = trades.gen_trades(df)

    Args:
        entry_struct (EntryMethod):
            Whether to allow multiple open position ("mulitple") or single
            open position at a time ("single").
        exit_struct (ExitMethod):
            Whether to apply first-in-first-out ("fifo"), last-in-first-out ("lifo"),
            take profit for half open positions repeatedly ("half_life") or
            take profit for all open positions ("take_all").
        num_lots (int):
            Number of lots to initiate new position each time (Default: 1).
        req_cols (list[str]):
            List of required columns to generate trades.
        monitor_close (bool):
            Whether to monitor close price ("close") or both high and low price
            (Default: True).
        strategy_dir (str):
            Relative path to strategy folder containing subfolders for implementing
            trading strategy (Default: "./src/strategy").
        percent_loss (float):
            If provided, percentage loss allowed for investment.
        exit_method (ExitMethod):
            Exit method to generate stop price.

    Attributes:
        no_trades (list[str]):
            List containing stock tickers with no completed trades.
        percent_loss (float):
            If provided, percentage loss allowed for investment.
        exit_method (ExitMethod):
            Exit method to generate stop price.
    """

    def __init__(
        self,
        entry_struct: EntryMethod = "multiple",
        exit_struct: ExitMethod = "take_all",
        num_lots: int = 1,
        req_cols: list[str] = [
            "date",
            "high",
            "low",
            "close",
            "entry_signal",
            "exit_signal",
        ],
        monitor_close: str = "close",
        strategy_dir: str = "./src/strategy",
        percent_loss: float | None = None,
        exit_method: ExitMethod | None = None,
    ) -> None:
        super().__init__(
            entry_struct, exit_struct, num_lots, req_cols, monitor_close, strategy_dir
        )

        # Price-related stops
        self.percent_loss = percent_loss
        self.exit_method = exit_method
        self.no_trades = []

    def gen_trades(self, df_senti: pd.DataFrame) -> pd.DataFrame:
        """Generate DataFrame containing completed trades for trading strategy."""

        completed_list = []

        # Filter out null values for OHLC due to weekends and holiday
        df = df_senti.loc[:, self.req_cols].copy()

        # Get news ticker and cointegrated/correlated ticker
        ticker = self.get_ticker(df_senti, "ticker")
        coint_corr_ticker = self.get_ticker(df_senti, "coint_corr_ticker")

        for idx, dt, high, low, close, ent_sig, ex_sig in df.itertuples(
            index=True, name=None
        ):
            print(f"idx : {idx}")
            print(f"dt : {dt}")
            print(f"close : {close}")
            print(f"ent_sig : {ent_sig}")
            print(f"ex_sig : {ex_sig}")

            # Get net position
            net_pos = self.get_net_pos()
            print(f"net_pos : {net_pos}")

            # Close off all open positions at end of trading period
            if idx >= len(df) - 1 and net_pos != 0:
                completed_list.extend(self.exit_all(dt, close))

                # Skip creating new open positions after all open positions closed
                continue

            # Check to cut loss
            if net_pos != 0 and all(
                item is not None for item in [self.percent_loss, self.exit_method]
            ):
                completed_list.extend(self.stop_loss(dt, high, low, close))

            # Check to take profit
            if (ex_sig == "sell" or ex_sig == "buy") and net_pos != 0:
                completed_list.extend(self.take_profit(dt, ex_sig, close))

            # Check to enter new position
            if ent_sig == "buy" or ent_sig == "sell":
                self.open_pos(coint_corr_ticker, dt, ent_sig, close)

            print(f"net_pos after update : {self.get_net_pos()}")
            print(f"len(self.open_trades) : {len(self.open_trades)}")
            print(f"self.open_trades : {self.open_trades}\n")

        # No completed trades recorded
        if not completed_list:
            self.no_trades.append(ticker)

        # Append 'news_ticker' column to DataFrame generated from completed trades
        df_trades = pd.DataFrame(completed_list)
        df_trades.insert(0, "news_ticker", ticker)

        return df_trades, df_senti

    def stop_loss(
        self,
        dt: datetime,
        high: float,
        low: float,
        close: float,
    ) -> list[dict[str, Any]]:
        """Close all open positions if computed stop price is triggered.

        - Stop price is computed via concrete implementation of 'CalExitPrice'.

        Args:
            dt (datetime):
                Trade datetime object.
            high (float):
                Current day high of cointegrated/correlated stock ticker.
            low (float):
                Current day low of cointegrated/correlated stock ticker.
            close (float):
                Current day open of cointegrated/correlated stock ticker.

        Returns:
            completed_trades (list[dict[str, Any]]):
                List of dictionary containing required fields to generate DataFrame.
        """

        completed_trades = []

        # Compute stop loss price based on 'self.exit_method'
        stop_price = self.cal_stop_price()

        entry_action = get_std_field(self.open_trades, "entry_action")
        exit_action = "sell" if entry_action == "buy" else "buy"

        cond_list = [
            self.monitor_close and entry_action == "buy" and close < stop_price,
            self.monitor_close and entry_action == "sell" and close > stop_price,
            not self.monitor_close and entry_action == "buy" and low < stop_price,
            not self.monitor_close and entry_action == "sell" and high > stop_price,
        ]

        # Exit all open positions if any condition in 'cond_list' is true
        if any(cond_list):
            completed_trades.extend(self.exit_all(dt, exit_action, close))

    def cal_stop_price(self) -> Decimal:
        """Compute stop price via concrete implementation of 'CalExitPrice'.

        Args:
            None.

        Returns:
            (Decimal): Stop price to monitor.
        """

        # Name of concrete class implemenation of 'CalExitPrice'
        class_name = EXIT_PRICE_MAPPING.get(self.exit_method)

        # File path to concrete class implementation of 'CalExitPrice'
        calexitprice_path = f"{self.strategy_dir}/base/cal_exit_price.py"

        # Get initialized instance of concrete class implementation
        class_inst = get_class_instance(
            class_name, calexitprice_path, percent_loss=self.percent_loss
        )

        return class_inst.cal_exit_price(self.open_trades)

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
