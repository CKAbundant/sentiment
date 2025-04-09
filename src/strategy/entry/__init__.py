"""Absolute import from 'src.strategy.entry' instead of transversing down to
python script. For example, we can import 'SentiEntry' directly via:

'from src.strategy.entry import SentiEntry'

instead of:

'from src.strategy.entry.senti_entry import SentiEntry'
"""

from .senti_entry import SentiEntry

# Import following items when using 'from src.strategy.entry import *'
__all__ = ["SentiEntry"]
