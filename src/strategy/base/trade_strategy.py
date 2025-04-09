"""Class to combine concrete implementation of 'EntrySignal', 'ExitSignal'
and 'GenTrades' into a strategy"""

import numpy as np
import pandas as pd

from config.variables import EntryType, FixedPL
from src.strategy.base import EntrySignal, ExitSignal, GenTrades


class TradingStrategy:
    """Combine entry, profit and stop loss strategy as a complete trading strategy.

    Usage:
        >>> strategy = TradingStrategy(
                entry_type="long_only",
                entry=SentiEntry,
                exit=SentiExit,
                trades=SentiTrades,
            )
        >>> strategy.run()

    Args:
        entry_type (EntryType):
            Types of open positions allowed either 'long_only', 'short_only' or
            'long_or_short'.
        entry (EntrySignal):
            Class instance of concrete implementation of 'EntrySignal' abstract class.
        exit (ExitSignal):
            If provided, Class instance of concrete implementation of 'ExitSignal'
            abstract class. If None, standard profit and stop loss will be applied via
            'gen_trades'.
        trades (GetTrades):
            Class instance of concrete implementation of 'GetTrades' abstract class.
        fixed_pl (FixedPL | None):
            If provided, "mean_drawdown", or "max_drawdown" to stop loss based on price
            movement.

    Attributes:
        entry_type (EntryType):
            Types of open positions allowed either 'long_only', 'short_only' or
            'long_or_short'.
        entry (EntrySignal):
            Class instance of concrete implementation of 'EntrySignal' abstract class.
        exit (ExitSignal):
            Class instance of concrete implementation of 'ExitSignal' abstract class.
        trades (GetTrades):
            Instance of concrete implementation of 'GetTrades' abstract class.
        fixed_pl (FixedPL | None):
            If provided, "mean_drawdown", or "max_drawdown" to stop loss based on price
            movement.
    """

    def __init__(
        self,
        entry_type: EntryType,
        entry: type[EntrySignal],
        exit: type[ExitSignal],
        trades: GenTrades,
        fixed_pl: FixedPL | None = None,
        percent_drawdown: float = 0.2,
    ) -> None:
        self.entry_type = entry_type
        self.entry = entry(entry_type)
        self.exit = exit(entry_type)
        self.trades = trades
        self.fixed_pl = fixed_pl
        self.percent_drawdown = percent_drawdown

    def __call__(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Generate completed trades based on trading strategy i.e.
        combination of entry, profit exit and stop exit.

        Args:
            df (pd.DataFrame): DataFrame without price action.

        Returns:
            df_trades (pd.DataFrame):
                DataFrame containing completed trades info.
            df_pa (pd.DataFrame):
                DataFrame with price action (i.e. 'entry_signal', 'exit_signal').
        """

        # Append 'entry_signal' column
        df_pa = self.entry.gen_entry_signal(df)

        if self.exit is not None:
            # Append 'exit_signal' if 'self.exit' exist
            df_pa = self.exit.gen_exit_signal(df_pa)

        match self.fixed_pl:
            case "all_drawdown":
                df_pa = self.update_drawdown_mean(df_pa)
            case "max_drawdown":
                df_pa = self.update_drawdown_max(df_pa)
            case _:
                pass

        # Generate trades
        df_trades, df_pa = self.trades.gen_trades(df_pa)

        return df_trades, df_pa

    def update_drawdown(self, df_pa: pd.DataFrame) -> pd.DataFrame:
        """Update 'exit_signal' based on drawdown computed from open posiitons.

        - Get total investment value based on all open positions.
        - Calculate minimum accepted investment value based on percentage drawdown.
        - Calculate stop price to achieve minimum accepted investment value.

        Args:
            df_pa (pd.DataFrame): DataFrame containing both entry and exit signals.

        Returns:
            df_pa (pd.DataFrame): DataFrame updated with drawdown info.
        """

        # Filter out null values for OHLC due to weekends and holiday
        df = df_pa.loc[~df_pa["close"].isna(), ["close", "ent_sig"]].copy()

        long_stop_prices = []
        short_stop_prices = []
        long_entry_price_list = []
        short_entry_price_list = []
        exit_action = []

        for close, ent_sig in df.itertuples(index=False, name=None):
            # For long position
            if ent_sig == "buy":
                # Compute stop price
                long_stop_prices.append(close * (1 - self.percent_drawdown))
                long_entry_price_list.append(close)

            # For short position
            elif ent_sig == "sell":
                # Compute stop price
                short_stop_prices.append(close * (1 + self.percent_drawdown))
                short_entry_price_list.append(close)

            max_long_stop_price = np.max(long_stop_prices) if long_stop_prices else 0
            max_short_stop_price = (
                np.max(short_stop_prices) if av_short_stop_prices else 0
            )

            # stop loss hit or rating <= 2
            if long_stop_prices and close <= max_long_stop_price:
                exit_action.append("sell")

                # Reset stop prices
                long_stop_prices = []
                av_long_stop_price = 0

            # stop loss hit or rating >= 4
            elif short_stop_prices and close >= max_short_stop_price:
                exit_action.append("buy")

                # Reset stop prices
                short_stop_prices = []
                av_short_stop_prices = 0

            else:
                exit_action.append("wait")

        df["exit_signal"] = exit_action

        return df

    def update_drawdown_max(self, df_pa: pd.DataFrame) -> pd.DataFrame:
        """Update 'exit_signal' based on highest drawdown for open positions.

        - Long position -> highest stop price among all open positions.
        - Short position -> lowest stop price among all open positions.

        Args:
            df_pa (pd.DataFrame): DataFrame containing both entry and exit signals.

        Returns:
            df_pa (pd.DataFrame): DataFrame updated with drawdown info.
        """

        pass
