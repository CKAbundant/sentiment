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
from typing import Any, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, computed_field, field_validator
from tqdm import tqdm

from config.variables import COINT_CORR_FN, HF_MODEL
from src.utils import utils


class StockTrade(BaseModel):
    ticker: str = Field(description="Stock ticker")
    coint_ticker: str = Field(description="Cointegrated stock ticker.")
    action: str = Field(description="Either 'buy' or 'sell'", default="buy")
    entry_date: date = Field(description="Date when opening long position")
    entry_price: Decimal = Field(description="Price when opening long position")
    exit_date: Optional[date] = Field(
        description="Date when exiting long position", default=None
    )
    exit_price: Optional[Decimal] = Field(
        description="Price when exiting long position", default=None
    )

    @computed_field(description="Number of days held for trade")
    def days_held(self) -> Optional[int]:
        if self.exit_date is not None and self.entry_date is not None:
            days_held = self.exit_date - self.entry_date
            return days_held.days
        return

    @computed_field(description="Profit/loss when trade completed")
    def profit_loss(self) -> Optional[Decimal]:
        if self.exit_price is not None and self.entry_price is not None:
            profit_loss = self.exit_price - self.entry_price
            return profit_loss
        return

    @computed_field(description="Percentage return of trade")
    def percent_ret(self) -> Optional[Decimal]:
        if self.exit_price is not None and self.entry_price is not None:
            percent_ret = (self.exit_price - self.entry_price) / self.entry_price
            return percent_ret.quantize(Decimal("1.000000"))
        return

    @computed_field(description="daily percentage return of trade")
    def daily_ret(self) -> Optional[Decimal]:
        if self.percent_ret is not None and self.days_held is not None:
            daily_ret = (1 + self.percent_ret) ** (1 / Decimal(str(self.days_held))) - 1
            return daily_ret.quantize(Decimal("1.000000"))
        return

    @computed_field(description="Whether trade is profitable")
    def win(self) -> Optional[int]:
        if (pl := self.percent_ret) is not None:
            return int(pl > 0)
        return

    model_config = {"validate_assignment": True}

    @field_validator("exit_date")
    def validate_exit_date(
        cls, exit_date: Optional[date], info: dict[str, Any]
    ) -> Optional[date]:
        if exit_date and (entry_date := info.data.get("entry_date")):
            if exit_date < entry_date:
                raise ValueError("Exit date must be after entry date!")
        return exit_date


class CalProfitLoss:
    """Compute profit and loss based on sentiment strategy.

    - Iterate all csv files containing price action for cointegrated stocks in folder.
    - Compute P&L for each file and record each completed trade in DataFrame.

    Usage:
        >>> cal_pl = CalProfitLoss()
        >>> df_results = cal_pl.run()

    Args:
        date (str):
            If provided, date when cointegration is performed.
        hf_model (str):
            Name of FinBERT model in HuggingFace (Default: "ziweichen").
        coint_corr_fn (COINT_CORR_FN):
            Name of function to perform either cointegration or correlation
            (Default: "coint").
        period (int):
            Time period used to compute cointegration (Default: 5).
        results_dir (str):
            Relative path of folder containing price action for ticker pairs (i.e.
            stock ticker and its cointegrated ticker) (Default: "./data/results").

    Attributes:
        date (str):
            If provided, date when news are scraped.
        model_dir (str):
            Relative path of folder containing summary reports for specific
            model and cointegration period.
        price_action_dir (str):
            Relative path of folder containing price action of ticker pairs for specific
            model and cointegration period.
        open_trades (list[StockTrade]):
            List containing only open trades
        num_open (int):
            Counter for number of existing open trades.
        no_trades (list[str]):
            List containing stock tickers with no completed trades.
    """

    def __init__(
        self,
        date: str | None = None,
        hf_model: HF_MODEL = "ziweichen",
        coint_corr_fn: COINT_CORR_FN = "coint",
        period: int = 5,
        results_dir: str = "./data/results",
    ) -> None:
        self.date = date or utils.get_current_dt(fmt="%Y-%m-%d")
        self.open_trades = []
        self.num_open = 0
        self.no_trades = []
        self.model_dir = (
            f"{results_dir}/{self.date}/{hf_model}_{coint_corr_fn}_{period}"
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

        # List to store DataFrames containing completed trades info
        results_list = []

        # Iterate through all csv files in 'price_action_dir'
        for file_path in tqdm(Path(self.price_action_dir).rglob("*.csv")):
            # Load price action csv file
            df = utils.load_csv(file_path)

            # Get ticker and cointegrated ticker from file name
            ticker, coint_ticker = self.get_tickers(file_path)

            # Append DataFrame containining completed trades for cointegrated ticker
            results_list.append(self.record_trades(df, ticker, coint_ticker))

        # Combine all DataFrames
        df_results = pd.concat(results_list, axis=0).reset_index(drop=True)

        # Create folder if not present
        utils.create_folder(self.model_dir)

        # Save combined DataFrame
        utils.save_csv(
            df_results, f"{self.model_dir}/trade_results.csv", save_index=False
        )

        # Generate overall and breakdown summary
        df_overall = self.gen_overall_summary()
        df_breakdown = self.gen_breakdown_summary()
        df_top_ret_pair = self.gen_top_ret_pair(df_breakdown)

        return df_results, df_overall, df_breakdown, df_top_ret_pair

    def get_tickers(self, file_path: Path) -> list[str, str]:
        """Get stock ticker and its cointegrated stock tick from file path."""

        return file_path.stem.split("_")

    def record_trades(
        self, data: pd.DataFrame, ticker: str, coint_ticker: str
    ) -> pd.DataFrame:
        """Iterate through each row to record completed trades.

        Args:
            data (pd.DataFrame):
                DataFrame containing price action generated from sentiment
                ratings for specific ticker.
            ticker (str):
                Stock ticker whose news articles are sentiment rated.
            coint_ticker (str):
                Stock ticker which is among the top 10 stocks with lowest
                cointegration pvalue.

        Returns:
            (pd.DataFrame): DataFrame containing completed trades info.
        """

        coint_close = f"{coint_ticker}_close"

        # Filter out null values for cointegrated stock close price
        df = data.loc[
            ~data[coint_close].isna(), ["date", coint_close, "action"]
        ].reset_index(drop=True)

        completed_trades = []

        for idx, dt, price, action in df.itertuples(index=True, name=None):
            if idx >= len(df) - 1:
                if self.num_open > 0:
                    completed_trades.extend(self.update_sell_info(dt, price))

            elif action == "buy":
                stock_trade = StockTrade(
                    ticker=ticker,
                    coint_ticker=coint_ticker,
                    action=action,
                    entry_date=dt,
                    entry_price=Decimal(str(price)),
                )
                self.open_trades.append(stock_trade)
                self.num_open += 1

            elif action == "sell" and self.num_open > 0:
                # Update existing StockTrade objects and remove from self.open_trades
                completed_trades.extend(self.update_sell_info(dt, price))

            else:
                # No short position. Sell to close long position
                continue

        # No completed trades recorded
        if not completed_trades:
            self.no_trades.append(ticker)

        return pd.DataFrame(completed_trades)

    def update_sell_info(self, dt: date, price: float) -> list[dict[str, Any]]:
        """Update existing StockTrade objects and remove from self.open_trades.

        Args:
            dt (date):
                datetime.date object of trade date.
            price (float):
                Closing price of cointegrated stock.

        Returns:
            (list[dict[str, Any]]):
                List of dictionary containing required fields to generate DataFrame.
        """

        updated_trades = []

        for trade in self.open_trades:
            # Update StockTrade objects with exit info
            trade.exit_date = dt
            trade.exit_price = Decimal(str(price))

            # Convert StockTrade to dictionary only if all fields are populated
            # i.e. trade completed.
            if self.validate_completed_trades(trade):
                updated_trades.append(trade.model_dump())

        # Reset self.open_trade and self.num_open
        if len(updated_trades) == len(self.open_trades):
            self.open_trades = []
            self.num_open = 0

        return updated_trades

    def validate_completed_trades(self, stock_trade: StockTrade) -> bool:
        """Check if all the fields in StockTrade object are not null i.e.
        all trade info (both entry and exit) are available."""

        return all(field is not None for field in stock_trade.model_dump().values())

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

        # 'mean' and 'median' operation generates float output
        # Round to 6 decimal places and convert to decimal type
        df_breakdown = utils.set_decimal_type(df_breakdown, to_round=True)

        # Append 'trading_period' column
        days_held = pd.to_datetime(df_breakdown[("exit_date", "max")]) - pd.to_datetime(
            df_breakdown[("entry_date", "min")]
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

        # Append 'annualized_returns' column
        daily_ret = (1 + df_breakdown["overall_percent_ret"]) ** (
            1 / df_breakdown["trading_period"]
        ) - 1
        df_breakdown["overall_daily_ret"] = daily_ret.map(
            lambda num: num.quantize(Decimal("1.000000"))
        )

        # Save as csv file
        utils.save_csv(
            df_breakdown, f"{self.model_dir}/breakdown_summary.csv", save_index=True
        )

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
