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
    """

    def __init__(
        self,
        entry_type: EntryType,
        entry: type[EntrySignal],
        exit: type[ExitSignal],
        trades: GenTrades,
        percent_drawdown: float = 0.2,
    ) -> None:
        self.entry_type = entry_type
        self.entry = entry(entry_type)
        self.exit = exit(entry_type)
        self.trades = trades
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

        # Generate trades
        df_trades, df_pa = self.trades.gen_trades(df_pa)

        return df_trades, df_pa
