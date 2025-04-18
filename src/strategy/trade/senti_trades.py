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

from config.variables import EXIT_PRICE_MAPPING, EntryMethod, ExitMethod, PriceAction
from src.strategy.base import GenTrades
from src.utils.utils import (
    display_open_trades,
    display_stop_price,
    get_class_instance,
    get_std_field,
)


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
            List of required columns to generate trades
            (Default: ["date", "high", "low", "close", "entry_signal", "exit_signal"]).
        monitor_close (bool):
            Whether to monitor close price ("close") or both high and low price
            (Default: True).
        percent_loss (float):
            Percentage loss allowed for investment (Default: 0.05).
        stop_method (ExitMethod):
            Exit method to generate stop price (Default: "no_stop").
        entry_struct_path (str):
            Relative path to 'entry_struct.py'
            (Default: "./src/strategy/base/entry_struct.py").
        exit_struct_path (str):
            Relative path to 'exit_struct.py'
            (Default: "./src/strategy/base/exit_struct.py").
        stop_method_path (str):
            Relative path to 'cal_exit_price.py'
            (Default: "./src/strategy/base/cal_exit_price.py").

    Attributes:
        percent_loss (float):
            If provided, percentage loss allowed for investment.
        stop_method (ExitMethod):
            Exit method to generate stop price.
        no_trades (list[str]):
            List containing stock tickers with no completed trades.
        stop_list (list[dict[str, datetime | Decimal]]):
            List to record datetime, stop price and whether stop price is triggered.
        entry_struct_path (str):
            Relative path to 'entry_struct.py'
            (Default: "./src/strategy/base/entry_struct.py").
        exit_struct_path (str):
            Relative path to 'exit_struct.py'
            (Default: "./src/strategy/base/exit_struct.py").
        stop_method_path (str):
            Relative path to 'cal_exit_price.py'
            (Default: "./src/strategy/base/cal_exit_price.py").
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
        monitor_close: bool = True,
        percent_loss: float = 0.05,
        stop_method: ExitMethod = "no_stop",
        entry_struct_path: str = "./src/strategy/base/entry_struct.py",
        exit_struct_path: str = "./src/strategy/base/exit_struct.py",
        stop_method_path: str = "./src/strategy/base/cal_exit_price.py",
    ) -> None:
        super().__init__(
            entry_struct,
            exit_struct,
            num_lots,
            req_cols,
            monitor_close,
            entry_struct_path,
            exit_struct_path,
        )
        self.stop_method_path = stop_method_path

        # Price-related stops
        self.percent_loss = percent_loss
        self.stop_method = stop_method
        self.no_trades = []
        self.stop_info_list = []

    def gen_trades(self, df_senti: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Generate DataFrame containing completed trades for trading strategy.

        Args:
            df_senti (pd.DataFrame):
                DataFrame containing entry and exit signals based on sentimet rating.

        Returns:
            df_trades (pd.DataFrame):
                DataFrame containing completed trades.
            df_signals (pd.DataFrame):
                DataFrame containing updated exit signals price-related stops.
        """

        completed_list = []

        # Filter out null values for OHLC due to weekends and holiday
        df = df_senti.copy()
        df = df.loc[:, self.req_cols]

        # Assume positions are opened or closed at market closing (1600 hrs New York)
        df = self.set_mkt_cls_dt(df)

        # Get news ticker and cointegrated/correlated ticker
        ticker = self.get_ticker(df_senti, "ticker")
        coint_corr_ticker = self.get_ticker(df_senti, "coint_corr_ticker")

        for idx, dt, high, low, close, ent_sig, ex_sig in df.itertuples(
            index=True, name=None
        ):
            # print(f"idx : {idx}")
            # print(f"dt : {dt}")
            # print(f"close : {close}")
            # print(f"ent_sig : {ent_sig}")
            # print(f"ex_sig : {ex_sig}")

            # Get net position and whether end of DataFrame
            net_pos = self.get_net_pos()
            is_end = idx >= len(df) - 1
            # print(f"net_pos : {net_pos}")

            # Close off all open positions at end of trading period
            if is_end and net_pos != 0:
                completed_list.extend(self.exit_all(dt, close))

                # Skip creating new open positions after all open positions closed
                continue

            # Check to cut loss for all open position
            if net_pos != 0 and self.stop_method != "no_stop":
                completed_list.extend(self.stop_loss(dt, high, low, close))

            # Check to take profit
            if (ex_sig == "sell" or ex_sig == "buy") and net_pos != 0:
                # Get standard 'entry_action' from 'self.open_trades'
                entry_action = get_std_field(self.open_trades, "entry_action")

                # Exit all open position in order to flip position
                # If entry_action == 'buy', then ex_sig must be 'sell'
                # ex_sig != entry_action
                if ex_sig == ent_sig and ex_sig != entry_action:
                    completed_list.extend(self.exit_all(dt, close))
                else:
                    completed_list.extend(self.take_profit(dt, ex_sig, close))

            # Check to enter new position
            if ent_sig == "buy" or ent_sig == "sell":
                self.open_pos(coint_corr_ticker, dt, ent_sig, close)

            # print(f"net_pos after update : {self.get_net_pos()}")
            # print(f"len(self.open_trades) : {len(self.open_trades)}")
            # display_open_trades(self.open_trades)

        # No completed trades recorded
        if not completed_list:
            self.no_trades.append(ticker)

        # Convert 'stop_info_list' to DataFrame and append to 'df_senti'
        # 'self.stop_list and 'self.trigger_list' are not empty
        if self.stop_info_list:
            df_senti = self.append_stop_info(df_senti)

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

        # Compute stop loss price based on 'self.stop_method'
        stop_price = self.cal_stop_price()
        entry_action = get_std_field(self.open_trades, "entry_action")
        # display_stop_price(
        #     self.monitor_close, stop_price, entry_action, high, low, close
        # )

        cond_list = [
            self.monitor_close and entry_action == "buy" and close < stop_price,
            self.monitor_close and entry_action == "sell" and close > stop_price,
            not self.monitor_close and entry_action == "buy" and low < stop_price,
            not self.monitor_close and entry_action == "sell" and high > stop_price,
        ]

        # Exit all open positions if any condition in 'cond_list' is true
        if any(cond_list):
            exit_action = "sell" if entry_action == "buy" else "buy"
            # print(f"\nStop triggered -> {exit_action} @ stop price {stop_price}\n")

            completed_trades.extend(self.exit_all(dt, stop_price))
            self.stop_info_list.append(
                {"date": dt, "stop_price": stop_price, "triggered": Decimal("1")}
            )

        else:
            self.stop_info_list.append(
                {"date": dt, "stop_price": stop_price, "triggered": Decimal("0")}
            )

        return completed_trades

    def cal_stop_price(self) -> Decimal:
        """Compute stop price via concrete implementation of 'CalExitPrice'.

        Args:
            None.

        Returns:
            (Decimal): Stop price to monitor.
        """

        # Name of concrete class implemenation of 'CalExitPrice'
        class_name = EXIT_PRICE_MAPPING.get(self.stop_method)

        # Get initialized instance of concrete class implementation
        class_inst = get_class_instance(
            class_name, self.stop_method_path, percent_loss=self.percent_loss
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

    def append_stop_info(self, df_senti: pd.DataFrame) -> pd.DataFrame:
        """Append stop info (i.e. stop price and whether triggered) to
        'df_senti' DataFrame."""

        df = df_senti.copy()

        # Convert 'self.stop_info_list' to DataFrame
        df_stop = pd.DataFrame(self.stop_info_list)

        # Set time to 0000 hrs for 'dt' column in order to perform join
        df_stop["date"] = df_stop["date"].map(
            lambda dt: dt.replace(hour=0, minute=0, tzinfo=None)
        )
        df_stop = df_stop.set_index("date")

        # Set 'date' column as index
        if "date" not in df.columns:
            raise ValueError("'date' column is not found.")

        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

        # Perform join via index
        df = df.join(df_stop)

        return df
