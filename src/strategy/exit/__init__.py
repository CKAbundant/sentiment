"""Absolute import from 'src.strategy.exit' instead of transversing down to
python script. For example, we can import 'SentiExit' directly via:

'from src.strategy.exit import SentiExit'

instead of:

'from src.strategy.exit.senti_exit import SentiExit'
"""

from .senti_exit import SentiExit

# Import following items when using 'from src.strategy.exit import *'
__all__ = ["SentiExit"]
