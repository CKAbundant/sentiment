"""Helper functions for strategy implementation"""

import datetime
import importlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Type, TypeVar

from src.strategy.base import TakeAllExit

if TYPE_CHECKING:
    from src.strategy.base.stock_trade import StockTrade

# Create generic type variable 'T'
T = TypeVar("T")


def get_class_instance(
    class_name: str, script_path: str, **params: dict[str, Any]
) -> T:
    """Return instance of a class that is initialized with 'params'.

    Args:
        class_name (str):
            Name of class in python script.
        script_path (str):
            Relative file path to python script that contains the required class.
        **params (dict[str, Any]):
            Arbitrary Keyword input arguments to initialize class instance.

    Returns:
        (T): Initialized instance of class.
    """

    # Convert script path to package path
    module_path = convert_path_to_pkg(script_path)

    try:
        # Import python script at class path as python module
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(f"Module not found in '{script_path}' : {e}")

    try:
        # Get class from module
        req_class: Type[T] = getattr(module, class_name)
    except AttributeError as e:
        raise AttributeError(f"'{class_name}' class is not found in module.")

    # Intialize instance of class
    return req_class(**params)


def convert_path_to_pkg(script_path: str) -> str:
    """Convert file path to package path that can be used as input to importlib."""

    # Remove suffix ".py"
    script_path = Path(script_path).with_suffix("").as_posix()

    # Convert to package format for use in 'importlib.import_module'
    return script_path.replace("/", ".")


def get_net_pos(open_trades: list["StockTrade"]) -> int:
    """Get net positions from 'self.open_trades'."""

    return sum(
        (
            trade.entry_lots - trade.exit_lots
            if trade.entry_action == "buy"
            else -(trade.entry_lots - trade.exit_lots)
        )
        for trade in open_trades
    )


def exit_all(
    open_trades: list["StockTrade"],
    dt: datetime,
    exit_price: float,
) -> list[dict[str, Any]]:
    """Close all open positions via 'TakeAllExit.close_pos' method.

    Args:
        open
        dt (datetime):
            Trade datetime object.
        exit_price (float):
            Exit price of stock ticker.

    Returns:
        completed_trades (list[dict[str, Any]]):
            List of dictionary containing required fields to generate DataFrame.
    """

    # Get initialized instance of concrete class implementation
    take_all_exit = TakeAllExit()

    # Update open trades and generate completed trades
    open_trades, completed_trades = take_all_exit.close_pos(open_trades, dt, exit_price)

    return completed_trades
