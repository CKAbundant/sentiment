"""Class to compute profit and loss based on sentiment strategy.

Considerations
- Only long i.e. no short positions taken.
- 1 share purchased each time.
- Allow multiple open long positions but all positions will be closed upon
'sell' action.
- No market slippage and no commission to simplify proof of concept.
- Capture each trade in DataFrame:
    - 'ticker' -> Cointegrated stock ticker.
    - 'entry_date' -> Date when opening long position.
    - 'entry_price' -> Price when opening long position.
    - 'exit_date' -> Date when exiting long position.
    - 'exit_price' -> Price when exiting long position.
    - 'profit_loss' -> Profit loss made.
- All trades closed at end of simulation.
"""

from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

from config.variables import (
    CointCorrFn,
    EntryMethod,
    EntryType,
    ExitMethod,
    HfModel,
    StopMethod,
)
from src.strategy.base.stock_trade import StockTrade
from src.utils import utils


class CalProfitLoss:
    """Compute profit and loss based on sentiment strategy.

    - Iterate all csv files containing price action for cointegrated stocks in folder.
    - Compute P&L for each file and record each completed trade in DataFrame.

    Usage:
        >>> cal_pl = CalProfitLoss()
        >>> df_results = cal_pl.run()

    Args:
        path (DictConfig):
            OmegaConf DictConfig containing required folder and file paths.
        date (str):
            If provided, date when news are scraped.
        entry_type (EntryType):
            Whether to allow long ("long"), short ("short") or
            both long and short position ("longshort").
        entry_signal (str):
            Name of python script containing concrete implemenation of 'EntrySignal'.
        exit_signal (str):
            Name of python script containing concrete implementation of 'ExitSignal'.
        trades_method (str):
            Name of python script containing concrete implementation of 'GenTrades'.
        entry_struct (EntryMethod):
            Whether to allow multiple open position ("mulitple") or single
            open position at a time ("single") (Default: "multiple").
        exit_struct (ExitMethod):
            Whether to apply first-in-first-out ("fifo"), last-in-first-out ("lifo"),
            take profit for half open positions repeatedly ("half_life") or
            take profit for all open positions ("take_all") (Default: "take_all").
        stop_method (ExitMethod):
            Exit method to generate stop price.
        hf_model (HfModel):
            Name of FinBERT model in Huggi[ngFace (Default: "ziweichen").
        coint_corr_fn (CointCorrFn):
            Name of function to perform either cointegration or correlation.
        period (int):
            Time period used to compute cointegration (Default: 5).

    Attributes:
        path (DictConfig):
            OmegaConf DictConfig containing required folder and file paths.
        date (str):
            If provided, date when news are scraped.
        entry_type (EntryType):
            Whether to allow long ("long"), short ("short") or
            both long and short position ("longshort").
        entry_signal (str):
            Name of python script containing concrete implemenation of 'EntrySignal'.
        exit_signal (str):
            Name of python script containing concrete implementation of 'ExitSignal'.
        trades_method (str):
            Name of python script containing concrete implementation of 'GenTrades'.
        entry_struct (EntryMethod):
            Whether to allow multiple open position ("mulitple") or single
            open position at a time ("single") (Default: "multiple").
        exit_struct (ExitMethod):
            Whether to apply first-in-first-out ("fifo"), last-in-first-out ("lifo"),
            take profit for half open positions repeatedly ("half_life") or
            take profit for all open positions ("take_all") (Default: "take_all").
        stop_method (ExitMethod):
            Exit method to generate stop price.
        hf_model (HfModel):
            Name of FinBERT model in Huggi[ngFace (Default: "ziweichen").
        coint_corr_fn (CointCorrFn):
            Name of function to perform either cointegration or correlation.
        period (int):
            Time period used to compute cointegration (Default: 5).
        open_trades (list[StockTrade]):
            List containing only open trades
        num_open (int):
            Counter for number of existing open trades.
        no_trades (list[str]):
            List containing stock tickers with no completed trades.
        model_dir (str):
            Relative path of folder containing summary reports for specific
            model and cointegration period.
        price_action_dir (str):
            Relative path of folder containing price action of ticker pairs for specific
            model and cointegration period.
    """

    def __init__(
        self,
        path: DictConfig,
        date: str | None = None,
        entry_type: EntryType = "long",
        entry_signal: str = "SentiEntry",
        exit_signal: str = "SentiExit",
        trades_method: str = "SentiTrades",
        entry_struct: EntryMethod = "multiple",
        exit_struct: ExitMethod = "take_all",
        stop_method: StopMethod = "no_stop",
        hf_model: HfModel = "ziweichen",
        coint_corr_fn: CointCorrFn = "coint",
        period: int = 5,
    ) -> None:
        self.path = path
        self.date = date or utils.get_current_dt(fmt="%Y-%m-%d")
        self.entry_type = entry_type
        self.entry_signal = entry_signal
        self.exit_signal = exit_signal
        self.trades_method = trades_method
        self.entry_struct = entry_struct
        self.exit_struct = exit_struct
        self.stop_method = stop_method
        self.hf_model = hf_model
        self.coint_corr_fn = coint_corr_fn
        self.period = period
        self.open_trades = []
        self.num_open = 0
        self.no_trades = []

        # Generate required file paths
        self.gen_paths()

    def gen_paths(self) -> None:
        """Generate required file paths i.e. 'coint_corr_path', 'senti_path',
        'model_dir' and 'price_action_dir'."""

        # Generate required folder and file paths
        date_dir = f"{self.path.data_dir}/{self.date}"
        self.model_dir = (
            f"{date_dir}/"
            f"{self.entry_type}_{self.entry_struct}_{self.exit_struct}_{self.stop_method}/"
            f"{self.hf_model}_{self.coint_corr_fn}_{self.period}"
        )
        self.price_action_dir = f"{self.model_dir}/price_actions"

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
        """Generate and saved completed trades and summary statistics as
        DataFrames.

        Args:
            None.

        Returns:
            df_results (pd.DataFrame):
                DataFrame containing completed trades for all price action
                csv files in selected folder.
            df_summary (pd.DataFrame):
                DataFrame containing overall statistics for sentiment strategy.
            df_breakdown (pd.DataFrame):
                DataFrame containing breakdown of statistics for each
                ticker-cointegrated ticker pair.
            df_top_ret_paris (pd.DataFrame):
                DataFrame containing ticker pairs with highest annualized returns
                for specific stock ticker.
        """

        if not Path(self.price_action_dir).is_dir():
            raise FileNotFoundError(
                f"'{self.price_action_dir}' folder doesn't exist i.e. no price-actions "
                "csv files available for profit loss computation."
            )

        # Generate overall and breakdown summary
        df_overall = self.gen_overall_summary()
        df_breakdown = self.gen_breakdown_summary()
        df_top_ret_pair = self.gen_top_ret_pair(df_breakdown)

        return df_overall, df_breakdown, df_top_ret_pair

    def gen_overall_summary(self) -> pd.DataFrame:
        """Generate summary info from 'trade_results.csv'."""

        df = utils.load_csv(f"{self.model_dir}/trade_results.csv")

        # Get info on stock tickers used to generate news articles
        no_trades = sorted(list(set(self.no_trades)))
        trades = df["ticker"].unique().tolist()
        num_tickers_with_no_trades = len(no_trades)
        num_tickers_with_trades = len(trades)
        total_num_tickers = num_tickers_with_no_trades + num_tickers_with_trades

        # trade info
        total_trades = len(df)
        total_wins = df["win"].sum()
        total_loss = total_trades - total_wins
        first_entry_date = df["entry_date"].min()
        last_exit_date = df["exit_date"].max()
        trading_period = Decimal((last_exit_date - first_entry_date).days)

        # Profit/loss info
        total_profit = df["profit_loss"].sum()
        total_investment = df["entry_price"].sum()
        percent_ret = (total_profit / total_investment).quantize(Decimal("1.000000"))
        annual_ret = ((1 + percent_ret) ** (365 / trading_period) - 1).quantize(
            Decimal("1.000000")
        )

        overall = {
            "strategy": Path(self.model_dir).name,
            "total_num_stock_tickers": total_num_tickers,
            "num_tickers_with_no_trades": num_tickers_with_no_trades,
            "stock_ticker_without_trades": ", ".join(no_trades),
            "num_tickers_with_trades": num_tickers_with_trades,
            "stock_ticker_with_trades": ", ".join(trades),
            "total_num_trades": total_trades,
            "total_num_wins": total_wins,
            "total_num_loss": total_loss,
            "total_profit": total_profit,
            "max_loss": df["profit_loss"].min(),
            "max_profit": df["profit_loss"].max(),
            "mean_profit": df["profit_loss"].mean(),
            "median_profit": df["profit_loss"].median(),
            "first_trade": first_entry_date,
            "last_trade": df["entry_date"].max(),
            "min_days_held": df["days_held"].min(),
            "max_days_held": df["days_held"].max(),
            "mean_days_held": df["days_held"].mean(),
            "median_days_held": df["days_held"].median(),
            "trading_period": trading_period,
            "total_investment": total_investment,
            "percent_return": percent_ret,
            "annualized_return": annual_ret,
        }

        # Convert to DataFrame
        df_overall = pd.DataFrame.from_dict(
            overall, orient="index", columns=["Overall Statistics"]
        )
        utils.save_csv(
            df_overall, f"{self.model_dir}/overall_summary.csv", save_index=True
        )

        return df_overall

    def gen_breakdown_summary(self) -> pd.DataFrame:
        """Generate breakdown summary for each ticker-cointegrated ticker pair."""

        df = utils.load_csv(f"{self.model_dir}/trade_results.csv")

        # Compute aggregated values for each ticker-coint_ticker pair
        df_breakdown = df.groupby(by=["ticker", "coint_ticker"]).agg(
            {
                "entry_date": ["min", "max"],
                "exit_date": "max",
                "days_held": ["min", "max", "mean", "median"],
                "entry_price": "sum",
                "profit_loss": ["min", "max", "mean", "median", "sum"],
                "daily_ret": ["min", "max", "mean", "median"],
                "win": ["count", "sum"],
            }
        )

        # Flatten multi-level columns and index for ease of processing
        df.columns = ["_".join(col_tuple) for col_tuple in df.columns]

        # 'mean' and 'median' operation generates float output
        # Round to 6 decimal places and convert to decimal type
        df_breakdown = utils.set_decimal_type(df_breakdown, to_round=True)

        # Append 'trading_period' column
        days_held = pd.to_datetime(df_breakdown["exit_date_max"]) - pd.to_datetime(
            df_breakdown["entry_date_min"]
        )
        df_breakdown["trading_period"] = days_held.map(
            lambda delta: Decimal(str(delta.days))
        )

        # Append 'percent_ret' column rounded to nearest 6 significant figures
        percent_ret = (
            df_breakdown[("profit_loss", "sum")] / df_breakdown[("entry_price", "sum")]
        )
        df_breakdown["overall_percent_ret"] = percent_ret.map(
            lambda num: num.quantize(Decimal("1.000000"))
        )

        # Append 'daily_returns' column
        daily_ret = (1 + df_breakdown["overall_percent_ret"]) ** (
            1 / df_breakdown["trading_period"]
        ) - 1
        df_breakdown["overall_daily_ret"] = daily_ret.map(
            lambda num: num.quantize(Decimal("1.000000"))
        )

        # Append negative returns statistic
        df_breakdown = self.append_neg_rets(df_breakdown)

        # Save as csv file
        utils.save_csv(
            df_breakdown, f"{self.model_dir}/breakdown_summary.csv", save_index=False
        )

        return df_breakdown

    def append_neg_rets(self, df_breakdown: pd.DataFrame) -> pd.DataFrame:
        """Append the min, max, mean and median negative returns to
        breakdown summary DataFrame."""

        # Extract only negative returns from 'trade_results.csv'
        df = utils.load_csv(f"{self.model_dir}/trade_results.csv")
        df = df.loc[df["win"] == 0, ["percent_ret"]]

        # Get min, max, mean and median 'profit_loss'
        df_neg = df.groupby(by=["ticker", "coint_corr_ticker"]).agg(
            {"percent_ret": ["min", "max", "mean", "median"]}
        )

        # Collapse multi-index columns and rename columns
        df_neg.columns = [
            "neg_ret_min",
            "neg_ret_max",
            "neg_ret_mean",
            "neg_ret_median",
        ]

        # Ensure both DataFrame index are the same before performing join
        if df.index.names == df_breakdown.index.names:
            # Append negative columns to 'df_breakdown'
            df_breakdown = df_breakdown.join(df_neg)

        return df_breakdown

    def gen_top_ret_pair(self, df_breakdown: pd.DataFrame) -> pd.DataFrame:
        """Filter ticker-cointegrated ticker pair with highest annualized return
        for each ticker.

        Args:
            df_summary (pd.DataFrame):
                Multi-level DataFrame containing aggregated values of completed trades.

        Returns:
            df_highest (pd.DataFrame):
                Multi-level DataFrame containing ticker pair with highest annualized
                return for each ticker.
        """

        df = df_breakdown.copy()

        # Group by ticker i.e. level 0
        grouped = df.groupby(level=0)

        # Get ticker pair with highest annualized returns for each ticker
        highest_pairs = grouped.apply(
            lambda group: group.loc[group["overall_daily_ret"].idxmax()].name
        )

        # DataFrame filtered by 'highest_pairs'
        df_highest = df.loc[highest_pairs, :].sort_values(
            by=["overall_daily_ret"], ascending=False
        )

        # Save as csv file
        utils.save_csv(
            df_highest, f"{self.model_dir}/top_ticker_pairs.csv", save_index=True
        )

        return df_highest
