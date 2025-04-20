"""Class to compute profit and loss based on sentiment strategy.

Considerations
- Only long i.e. no short positions taken.
- 1 share purchased each time.
- Allow multiple open long positions but all positions will be closed upon
'sell' action.
- No market slippage and no commission to simplify proof of concept.
- Capture each trade in DataFrame:
    - 'ticker' -> Cointegrated stock ticker.
    - 'entry_datetime' -> Date when opening long position.
    - 'entry_price' -> Price when opening long position.
    - 'exit_datetime' -> Date when exiting long position.
    - 'exit_price' -> Price when exiting long position.
    - 'profit_loss' -> Profit loss made.
- All trades closed at end of simulation.
"""

from collections import deque
from decimal import Decimal
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

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
    """Compute profit and loss for specific combination of 'entry_type',
    'entry_signal', 'exit_signal' and 'trades_method'.

    - Iterate all csv files containing price action for cointegrated stocks in folder.
    - Compute P&L for each file and record each completed trade in DataFrame.

    Usage:
        >>> cal_pl = CalProfitLoss()
        >>> df_overall, df_breakdown, df_top_ret_pair = cal_pl.run()

    Args:
        path (DictConfig):
            OmegaConf DictConfig containing required folder and file paths.
        no_trades (list[str]):
            List of news tickers with no completed trades.
        date (str):
            If provided, date when news are scraped.
        entry_type (EntryType):
            Whether to allow long ("long"), short ("short") or
            both long and short position ("longshort").
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
        no_trades (list[str]):
            List of news tickers with no completed trades.
        date (str):
            If provided, date when news are scraped.
        entry_type (EntryType):
            Whether to allow long ("long"), short ("short") or
            both long and short position ("longshort").
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
        no_trades: list[str],
        date: str | None = None,
        entry_type: EntryType = "long",
        entry_struct: EntryMethod = "multiple",
        exit_struct: ExitMethod = "take_all",
        stop_method: StopMethod = "no_stop",
        hf_model: HfModel = "ziweichen",
        coint_corr_fn: CointCorrFn = "coint",
        period: int = 5,
    ) -> None:
        self.path = path
        self.no_trades = no_trades
        self.date = date or utils.get_current_dt(fmt="%Y-%m-%d")
        self.entry_type = entry_type
        self.entry_struct = entry_struct
        self.exit_struct = exit_struct
        self.stop_method = stop_method
        self.hf_model = hf_model
        self.coint_corr_fn = coint_corr_fn
        self.period = period

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

        df = utils.load_csv(
            f"{self.model_dir}/trade_results.csv", tz="America/New_York"
        )

        # Get info on stock tickers used to generate news articles
        no_trades = self.no_trades
        trades = df["ticker"].unique().tolist()
        num_tickers_with_no_trades = len(no_trades)
        num_tickers_with_trades = len(trades)
        total_num_tickers = num_tickers_with_no_trades + num_tickers_with_trades

        # trade info
        total_trades = len(df)
        total_wins = df["win"].sum()
        total_loss = total_trades - total_wins
        first_entry_date = df["entry_datetime"].min()
        last_exit_date = df["exit_datetime"].max()
        trading_period = Decimal((last_exit_date - first_entry_date).days)
        win_rate = Decimal(str(total_wins / total_trades)).quantize(Decimal("1.000000"))

        # Profit/loss info
        total_profit = df["profit_loss"].sum()
        total_investment = df["entry_price"].sum()
        percent_ret = (total_profit / total_investment).quantize(Decimal("1.000000"))
        annual_ret = ((1 + percent_ret) ** (365 / trading_period) - 1).quantize(
            Decimal("1.000000")
        )

        # Separate losing and winning trades
        df_neg = df.loc[df["win"] == 0, :]
        df_pos = df.loc[df["win"] == 1, :]

        overall = {
            "entry_type": self.entry_type,
            "entry_struct": self.entry_struct,
            "exit_struct": self.exit_struct,
            "stop_method": self.stop_method,
            "hf_model": self.hf_model,
            "coint_corr_fn": self.coint_corr_fn,
            "period": self.period,
            "total_num_stock_tickers": total_num_tickers,
            "num_tickers_with_no_trades": num_tickers_with_no_trades,
            "stock_ticker_without_trades": ", ".join(no_trades),
            "num_tickers_with_trades": num_tickers_with_trades,
            "stock_ticker_with_trades": ", ".join(trades),
            "total_num_trades": total_trades,
            "total_num_wins": total_wins,
            "total_num_loss": total_loss,
            "win_rate": win_rate,
            "min_percent_return": df["percent_ret"].min(),
            "max_percent_return": df["percent_ret"].max(),
            "mean_percent_return": df["percent_ret"].mean(),
            "median_percent_return": df["percent_ret"].median(),
            "min_pos_percent_return": df_pos["percent_ret"].min(),
            "lowest_neg_percent_return": df_neg["percent_ret"].max(),
            "highest_neg_percent_return": df_neg["percent_ret"].min(),
            "mean_neg_percent_return": df_neg["percent_ret"].mean(),
            "median_neg_percent_return": df_neg["percent_ret"].median(),
            "first_trade": first_entry_date,
            "last_trade": df["entry_datetime"].max(),
            "min_days_held": df["days_held"].min(),
            "max_days_held": df["days_held"].max(),
            "mean_days_held": df["days_held"].mean(),
            "median_days_held": df["days_held"].median(),
            "trading_period": trading_period,
            "total_investment": total_investment,
            "total_profit": total_profit,
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

        df = utils.load_csv(
            f"{self.model_dir}/trade_results.csv", tz="America/New_York"
        )

        # Compute aggregated values for each ticker-coint_corr_ticker pair
        df_breakdown = df.groupby(by=["news_ticker", "ticker"]).agg(
            {
                "entry_datetime": ["min", "max"],
                "exit_datetime": "max",
                "days_held": ["min", "max", "mean", "median"],
                "entry_price": "sum",
                "profit_loss": ["sum"],
                "percent_ret": ["min", "max", "mean", "median"],
                "win": ["count", "sum"],
            }
        )

        # print(f"df_breakdown : \n\n{df_breakdown}\n")

        # Flatten multi-level columns and index for ease of processing
        df_breakdown.columns = [
            "_".join(col_tuple) if col_tuple[1] != "" else col_tuple[0]
            for col_tuple in df_breakdown.columns
        ]

        # print(f"df_breakdown after flattening multi-columns : \n\n{df_breakdown}\n")

        # 'mean' and 'median' operation generates float output
        # Round to 6 decimal places and convert to decimal type
        df_breakdown = utils.set_decimal_type(df_breakdown, to_round=True)

        # Append 'trading_period' column
        days_held = pd.to_datetime(df_breakdown["exit_datetime_max"]) - pd.to_datetime(
            df_breakdown["entry_datetime_min"]
        )
        df_breakdown["trading_period"] = days_held.map(
            lambda delta: Decimal(str(delta.days))
        )

        # Append 'percent_ret' column rounded to nearest 6 significant figures
        percent_ret = df_breakdown["profit_loss_sum"] / df_breakdown["entry_price_sum"]
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

        # Reset multi-index if present
        if df_breakdown.index.names[0] == "news_ticker":
            df_breakdown = df_breakdown.reset_index()

        # Save as csv file
        utils.save_csv(
            df_breakdown, f"{self.model_dir}/breakdown_summary.csv", save_index=False
        )

        return df_breakdown

    def append_neg_rets(self, df_breakdown: pd.DataFrame) -> pd.DataFrame:
        """Append the min, max, mean and median negative returns to
        breakdown summary DataFrame."""

        # Extract only negative returns from 'trade_results.csv'
        df = utils.load_csv(
            f"{self.model_dir}/trade_results.csv", tz="America/New_York"
        )
        df = df.loc[df["win"] == 0, ["news_ticker", "ticker", "percent_ret"]]

        # Get min, max, mean and median 'profit_loss'
        df_neg = df.groupby(by=["news_ticker", "ticker"]).agg(
            {"percent_ret": ["min", "max", "mean", "median"]}
        )

        # Collapse multi-index columns and rename columns
        df_neg.columns = [
            "neg_ret_max",
            "neg_ret_min",
            "neg_ret_mean",
            "neg_ret_median",
        ]

        # Ensure both DataFrame index are the same before performing join
        if df_neg.index.names == df_breakdown.index.names:
            # Append negative columns to 'df_breakdown'
            df_breakdown = df_breakdown.join(df_neg)

        return df_breakdown

    def gen_top_ret_pair(self, df_breakdown: pd.DataFrame) -> pd.DataFrame:
        """Filter ticker-cointegrated ticker pair with highest annualized return
        for each ticker.

        Args:
            df_summary (pd.DataFrame):
                DataFrame containing aggregated values of completed trades.

        Returns:
            df_highest (pd.DataFrame):
                DataFrame containing ticker pair with highest annualized
                return for each ticker.
        """

        df = df_breakdown.copy()

        # Get ticker pair with highest daily return for each unique ticker
        max_ret_idx = df.groupby(by=["news_ticker"])["overall_daily_ret"].idxmax()
        df_highest = (
            df.loc[max_ret_idx, :]
            .sort_values(by=["overall_daily_ret"], ascending=False)
            .reset_index(drop=True)
        )

        # Save as csv file
        utils.save_csv(
            df_highest, f"{self.model_dir}/top_ticker_pairs.csv", save_index=False
        )

        return df_highest
