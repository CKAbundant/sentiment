"""Absolute import from 'src.strategy.base' instead of transversing down to
python script. For example, we can import GenTrades directly via:

'from src.strategy.base import GenTrades'

instead of:

'from src.strategy.base.gen_trades import GenTrades'
"""

from .entry_struct import MultiEntry, MultiHalfEntry, SingleEntry
from .exit_struct import FIFOExit, HalfFIFOExit, HalfLIFOExit, LIFOExit, TakeAllExit
from .gen_trades import GenTrades
from .stock_trade import StockTrade
from .trade_signal import EntrySignal, ExitSignal
from .trade_strategy import TradingStrategy

# Import following items when using 'from src.strategy.base import *'
__all__ = [
    "GenTrades",
    "StockTrade",
    "EntrySignal",
    "ExitSignal",
    "TradingStrategy",
    "MultiEntry",
    "MultiHalfEntry",
    "SingleEntry",
    "FIFOExit",
    "LIFOExit",
    "HalfFIFOExit",
    "HalfLIFOExit",
    "TakeAllExit",
]
