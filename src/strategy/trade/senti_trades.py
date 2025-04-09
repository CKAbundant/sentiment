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

from collections import deque

import pandas as pd

from config.variables import EntryStruct, ExitStruct
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
        ticker (str):
            Stock ticker whose news are sentiment-rated i.e. news ticker.
        coint_corr_ticker (str):
            Stock ticker that is cointegrated/correlated with news ticker.
        entry_struct (EntryStruct):
            Whether to allow multiple open position ("mulitple") or single
            open position at a time ("single").
        exit_struct (ExitStruct):
            Whether to apply first-in-first-out ("fifo"), last-in-first-out ("lifo"),
            take profit for half open positions repeatedly ("half_life") or
            take profit for all open positions ("take_all").
        num_lots (int):
            Number of lots to initiate new position each time (Default: 1).
        req_cols (list[str]):
            List of required columns to generate trades.
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
        ticker (str):
            Stock ticker whose news are sentiment-rated i.e. news ticker.
        coint_corr_ticker (str):
            Stock ticker that is cointegrated/correlated with news ticker.
        entry_struct (EntryStruct):
            Whether to allow multiple open position ("mulitple") or single
            open position at a time ("single") (Default: "multiple").
        exit_struct (ExitStruct):
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
        price_to_monitor (str):
            Whether to monitor close price ("close") or both high and low price
            ("high_low") (Default: "close").
        percent_loss (float):
            If provided, percentage loss allowed for investment.
        percent_profit (float):
            If provided, target percentage gain for investment.
        percent_profit_trade (float):
            If provided, target percentage gain for each trade.
        no_trades (list[str]):
            List containing stock tickers with no completed trades.
    """

    def __init__(
        self,
        ticker: str,
        coint_corr_ticker: str,
        entry_struct: EntryStruct = "multiple",
        exit_struct: ExitStruct = "take_all",
        num_lots: int = 1,
        req_cols: list[str] = [
            "date",
            "high",
            "low",
            "close",
            "entry_signal",
            "exit_signal",
        ],
        price_to_monitor: str = "close",
        percent_loss: float | None = None,
        percent_loss_nearest: float | None = None,
        percent_profit: float | None = None,
        percent_profit_trade: float | None = None,
    ) -> None:
        super().__init__(entry_struct, exit_struct, num_lots)
        self.ticker = ticker
        self.coint_corr_ticker = coint_corr_ticker
        self.req_cols = req_cols
        self.price_to_monitor = price_to_monitor
        self.percent_loss = percent_loss
        self.percent_loss_nearest = percent_loss_nearest
        self.percent_profit = percent_profit
        self.percent_profit_trade = percent_profit_trade
        self.no_trades = []

    def gen_trades(self, df_news: pd.DataFrame) -> pd.DataFrame:
        """Generate DataFrame containing completed trades for trading strategy."""

        # Filter out null values for OHLC due to weekends and holiday
        df = df_news.loc[~df_news["close"].isna(), self.req_cols].copy()

        for idx, dt, high, low, close, ent_sig, ex_sig in df.itertuples(
            index=True, name=None
        ):
            # Close off all open positions at end of trading period
            if idx >= len(df) - 1 and self.net_pos != 0:
                self.completed_trades.extend(
                    self.update_via_take_all(dt, ex_sig, close)
                )

            elif self.percent_loss:
                # Compute stop price to ensure total investment loss is limited to 'percent_loss'
                stop_price = self.cal_stop_price()

                if self.price_to_monitor == "close":
                    if (ent_sig == "buy" and close <= stop_price) or (
                        ent_sig == "sell" and close >= stop_price
                    ):
                        self.completed_trades.extend(
                            self.update_via_take_all(dt, ex_sig, close)
                        )
                else:
                    if (ent_sig == "buy" and low <= stop_price) or (
                        ent_sig == "sell" and high >= stop_price
                    ):
                        self.completed_trades.extend(
                            self.update_via_take_all(dt, ex_sig, close)
                        )

            # Signal to close existing open positions i.e. net position not equal to 0
            elif (ex_sig == "sell" or ex_sig == "buy") and self.net_pos != 0:
                # Determine if exit signal is taking profit or stop loss
                if self._is_loss(close, ex_sig):
                    # Close all open position if latest trade incurs loss
                    self.completed_trades.extend(self.update_via_take_all)

                else:
                    # Current closing price is higher than latest entry price
                    # i.e. running at a profit
                    self.close_pos_with_profit(dt, close, ex_sig)

            # Signal to enter new position
            elif ent_sig == "buy" or ent_sig == "sell":
                self.open_new_pos(dt, self.coint_corr_ticker, close, ent_sig)

            else:
                # No signal to initate new open position or
                # no open positions to close
                continue

        # No completed trades recorded
        if not self.completed_trades:
            self.no_trades.append(self.ticker)

        # Append 'news_ticker' column to DataFrame generated from completed trades
        df = pd.DataFrame(self.completed_trades)
        df.insert(0, "news_ticker", self.ticker)

        return df
