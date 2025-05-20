"""Class to combine concrete implementation of 'EntrySignal', 'ExitSignal'
and 'GenTrades' into a strategy"""

from typing import Type

import numpy as np
import pandas as pd

from config.variables import EntryType
from src.strategy.base import EntrySignal, ExitSignal, GenTrades


class TradingStrategy:
    """Combine entry, profit and stop loss strategy as a complete trading strategy.

    Usage:
        >>> strategy = TradingStrategy(
                entry_type="long",
                entry=SentiEntry,
                exit=SentiExit,
                trades=SentiTrades,
            )
        >>> strategy.run()

    Args:
        entry (EntrySignal):
            Class instance of concrete implementation of 'EntrySignal' abstract class.
        exit (ExitSignal):
            If provided, Class instance of concrete implementation of 'ExitSignal'
            abstract class. If None, standard profit and stop loss will be applied via
            'gen_trades'.
        trades (GetTrades):
            Class instance of concrete implementation of 'GetTrades' abstract class.

    Attributes:
        entry (EntrySignal):
            Class instance of concrete implementation of 'EntrySignal' abstract class.
        exit (ExitSignal):
            Class instance of concrete implementation of 'ExitSignal' abstract class.
        trades (GetTrades):
            Instance of concrete implementation of 'GetTrades' abstract class.
    """

    def __init__(
        self,
        entry: EntrySignal,
        exit: ExitSignal,
        trades: GenTrades,
    ) -> None:
        self.entry = entry
        self.exit = exit
        self.trades = trades

    def __call__(self, df_ohlcv: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Generate completed trades based on trading strategy i.e.
        combination of entry, profit exit and stop exit.

        Args:
            df_ohlcv (pd.DataFrame):
                DataFrame containing OHLCV data and TA (if any) for
                specific stock ticker.

        Returns:
            df_trades (pd.DataFrame):
                DataFrame containing completed trades.
            df_signals (pd.DataFrame):
                DataFrame containing updated exit signals price-related stops.
        """

        # Append entry and exit signal
        df_pa = self.entry.gen_entry_signal(df_ohlcv)
        df_pa = self.exit.gen_exit_signal(df_pa)

        # Generate trades
        df_trades, df_signals = self.trades.gen_trades(df_pa)

        return df_trades, df_signals
