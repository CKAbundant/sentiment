"""Absolute import from 'src.strategy.trade' instead of transversing down to
python script. For example, we can import 'SentiTrades' directly via:

'from src.strategy.trade import SentiTrades'

instead of:

'from src.strategy.trade.senti_trades import SentiTrades'
"""

from .senti_trades import SentiTrades

# Import following items when using 'from src.strategy.base import *'
__all__ = ["SentiTrades"]
