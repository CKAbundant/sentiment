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

from collections import Counter, deque
from datetime import datetime
from typing import Any

import pandas as pd

from config.variables import EntryMethod, ExitMethod, PriceAction
from src.strategy.base import GenTrades, StockTrade


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
        strategy_dir (str):
            Relative path to strategy folder containing subfolders for implementing
            trading strategy (Default: "./src/strategy").
        price_to_monitor (str):
            Whether to monitor close price ("close") or both high and low price
            ("high_low") (Default: "close").
        percent_loss (float):
            If provided, percentage loss allowed for investment.
        percent_profit (float):
            If provided, target percentage gain for investment.
        percent_profit_trade (float):
            If provided, target percentage gain for each trade.

    Attributes:
        no_trades (list[str]):
            List containing stock tickers with no completed trades.
        price_to_monitor (str):
            Whether to monitor close price ("close") or both high and low price
            ("high_low") (Default: "close").
        percent_loss (float):
            If provided, percentage loss allowed for investment.
        percent_profit (float):
            If provided, target percentage gain for investment.
        percent_profit_trade (float):
            If provided, target percentage gain for each trade.
    """

    def __init__(
        self,
        ticker: str,
        coint_corr_ticker: str,
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
        strategy_dir: str = "./src/strategy",
        price_to_monitor: str = "close",
        percent_loss: float | None = None,
        percent_loss_nearest: float | None = None,
        percent_profit: float | None = None,
        percent_profit_trade: float | None = None,
    ) -> None:
        super().__init__(entry_struct, exit_struct, num_lots, req_cols, strategy_dir)
        self.entry_struct = entry_struct
        self.exit_struct = exit_struct
        self.num_lots = num_lots
        self.req_cols = req_cols
        self.no_trades = []

        # Price-related stops
        self.price_to_monitor = price_to_monitor
        self.percent_loss = percent_loss
        self.percent_loss_nearest = percent_loss_nearest
        self.percent_profit = percent_profit
        self.percent_profit_trade = percent_profit_trade

    def gen_trades(self, df_senti: pd.DataFrame) -> pd.DataFrame:
        """Generate DataFrame containing completed trades for trading strategy."""

        completed_list = []

        # Filter out null values for OHLC due to weekends and holiday
        df = df_senti.loc[~df_senti["close"].isna(), self.req_cols].copy()

        # Get news ticker and cointegrated/correlated ticker
        ticker = self.get_ticker(df_senti, "ticker")
        coint_corr_ticker = self.get_ticker(df_senti, "coint_corr_ticker")

        for idx, dt, high, low, close, ent_sig, ex_sig in df.itertuples(
            index=True, name=None
        ):
            # Get net position
            net_pos = self.get_net_pos()

            # Close off all open positions at end of trading period
            if idx >= len(df) - 1 and net_pos != 0:
                completed_list.extend(self.stop_all(open_trades, dt, ex_sig, close))
                continue

            # Stop loss
            # elif self.percent_loss:
            #     # Compute stop price to ensure total investment loss is limited to 'percent_loss'
            #     stop_price = self.cal_stop_price()

            #     if self.price_to_monitor == "close":
            #         if (ent_sig == "buy" and close <= stop_price) or (
            #             ent_sig == "sell" and close >= stop_price
            #         ):
            #             self.completed_trades.extend(
            #                 self.update_via_take_all(dt, ex_sig, close)
            #             )
            #     else:
            #         if (ent_sig == "buy" and low <= stop_price) or (
            #             ent_sig == "sell" and high >= stop_price
            #         ):
            #             self.completed_trades.extend(
            #                 self.update_via_take_all(dt, ex_sig, close)
            #             )

            # Signal to close existing open positions
            if (ex_sig == "sell" or ex_sig == "buy") and net_pos != 0:
                completed_list.extend(self.close_pos(open_trades, dt, ex_sig, close))

            # Signal to enter new position after closing open positions (if any)
            if ent_sig == "buy" or ent_sig == "sell":
                open_trades = self.open_pos(
                    open_trades, coint_corr_ticker, dt, ent_sig, close
                )

        # No completed trades recorded
        if not completed_list:
            self.no_trades.append(ticker)

        # Append 'news_ticker' column to DataFrame generated from completed trades
        df = pd.DataFrame(completed_list)
        df.insert(0, "news_ticker", ticker)

        return df

    def close_pos(
        self,
        open_trades: deque[StockTrade],
        dt: datetime,
        ex_sig: PriceAction,
        close: float,
    ) -> tuple[deque[StockTrade], list[dict[str, Any]]]:
        """Close open position either by taking profit or cutting losses.

        Args:
            open_trades (deque[StockTrade]):
                Deque list of StockTrade pydantic object to record open trades.
            dt (datetime):
                Trade datetime object.
            ex_sig (PriceAction):
                Action to close open position either "buy" or "sell".
            close (float):
                Current day open for cointegrated/correlated stock ticker.

        Returns:
            open_trades (deque[StockTrade]):
                Updated deque list of 'StockTrade' objects.
            completed_trades (list[dict[str, Any]]):
                List of dictionary containing required fields to generate DataFrame.
        """

        # Determine if exit signal is taking profit or stop loss
        if self._is_latest_loss(close, ex_sig):
            # Close all open position if latest trade incurs loss
            open_trades, completed_trades = self.stop_all(
                open_trades, dt, ex_sig, close
            )
        else:
            # Current closing price is higher than latest entry price
            # i.e. running at a profit
            open_trades, completed_trades = self.take_profit(
                open_trades, dt, ex_sig, close
            )

        return open_trades, completed_trades

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
