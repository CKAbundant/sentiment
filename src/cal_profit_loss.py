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
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from pydantic import BaseModel, Field, computed_field, field_validator
from tqdm import tqdm

from src.utils import utils


class StockTrade(BaseModel):
    ticker: str = Field(description="Stock ticker")
    coint_ticker: str = Field(description="Cointegrated stock ticker.")
    action: str = Field(description="Either 'buy' or 'sell'", default="buy")
    entry_date: date = Field(description="Date when opening long position")
    entry_price: float = Field(description="Price when opening long position")
    exit_date: Optional[date] = Field(
        description="Date when exiting long position", default=None
    )
    exit_price: Optional[float] = Field(
        description="Price when exiting long position", default=None
    )

    @computed_field(description="Profit & loss once trade completed")
    def profit_loss(self) -> Optional[float]:
        if self.exit_price is not None and self.entry_price is not None:
            return self.exit_price - self.entry_price
        return

    model_config = {"validate_assignment": True}

    @field_validator("exit_date")
    def validate_exit_date(
        cls, exit_date: Optional[date], info: dict[str, Any]
    ) -> Optional[date]:
        if exit_date and exit_date < info.data["entry_date"]:
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

    Attributes:
        date (str):
            If provided, date when cointegration is performed.
        results_dir (str):
            Relative path containing price action csv files for computing profit & loss.
        open_trades (list[StockTrade]):
            List containing only open trades
        num_open (int):
            Counter for number of existing open trades.
        no_trades (list[str]):
            List containing stock tickers with no completed trades.
    """

    def __init__(self, date: str | None = None) -> None:
        self.date = date or utils.get_current_dt(fmt="%Y-%m-%d")
        self.results_dir = f"./data/results/{self.date}"
        self.open_trades = []
        self.num_open = 0
        self.no_trades = []

    def run(self) -> pd.DataFrame | None:
        """Generate and save DataFrame containing completed trades for all price action
        csv files in selected folder."""

        if not Path(self.results_dir).is_dir():
            print(f"'{self.results_dir}' folder doesn't exist.")
            return

        # List to store DataFrames containing completed trades info
        results_list = []
        files_list = [
            file_path
            for file_path in Path(self.results_dir).rglob("*.csv")
            if file_path.name != "trade_results.csv"
        ]

        for file_path in tqdm(files_list):
            # Load price action csv file
            df = pd.read_csv(file_path)

            # Set 'date' as date type
            df["date"] = pd.to_datetime(df["date"]).dt.date

            # Get ticker and cointegrated ticker from file name
            ticker, coint_ticker = self.get_tickers(file_path)

            # Update completed trades for cointegrated ticker
            results_list.append(self.record_trades(df, ticker, coint_ticker))

        # Combine all DataFrames and saved as csv file
        df_results = pd.concat(results_list, axis=0)
        df_results.to_csv(f"{self.results_dir}/trade_results.csv", index=False)

        # Compute summary
        df_summary = self.gen_summary(df_results)

        return df_results

    def get_tickers(self, file_path: Path) -> list[str, str]:
        """Get stock ticker and its cointegrated stock tick from file path."""

        return file_path.stem.split("_")

    def record_trades(
        self, data: pd.DataFrame, ticker: str, coint_ticker: str
    ) -> pd.DataFrame:
        """Iterate through each row to record completed trades.

        Args:
            data (pd.DataFrame):
                DataFrame containing price action generated from sentiment ratings.
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
                    entry_price=price,
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

        # Convert to DataFrame
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
            trade.exit_price = price

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

    def gen_summary(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate summary info given the compiled completed trades."""

        # List of stock tickers without
