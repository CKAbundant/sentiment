"""Abstract class and concrete implementation of various
exit stuctures."""

import math
from abc import ABC, abstractmethod
from collections import deque
from datetime import date
from decimal import Decimal

from config.variables import PriceAction

from .stock_trade import StockTrade


class ExitStruct(ABC):
    """Abstract class to populate 'StockTrade' pydantic object to close
    existing open positions fully or partially.

    - Exit open position with either profit or loss.
    - Incorporates fixed percentage gain and percentage loss.

    Args:
        ABC (_type_): _description_
    """
