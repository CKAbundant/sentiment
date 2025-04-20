"""Generate price action based on sentiment rating for the day.

- Compute average sentiment rating for the day.
- Buy 1 co-integrated stock if average rating is more than 4.
- Sell 1 co-integrated stock if average rating is less than 2.
- Generate DataFrame for stock:
    1. num_news -> Number of news articles for the day.
    2. av_rating -> Average sentiment rating for all news articles.
    3. av_bull_bear -> Average sentiment rating ignoring news with ranking 3.
    4. <ticker>_close -> Closing price of stock
    4. <ticker>_close -> Closing price of co-integrated stock.
    5. action -> "buy", "sell", "nothing"

Considerations:

1. Actual time of news publishing is not known and estimated from time period lapsed
e.g. '23 minutes ago'. Therefore, we take the average sentiment rating for the stock
for the day.
2. We will trade the stock that has the lowest p-value for co-integration test.
3. We assume decision to buy/sell stock occurs before market closing. Hence we observe
only the closing price of co-integrated stock.
4. We take past 10 days data from 13 Mar 2025 to observe the price changes.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

from config.variables import (
    EXIT_PRICE_MAPPING,
    SIGNAL_MAPPING,
    STRUCT_MAPPING,
    TRADES_MAPPING,
    CointCorrFn,
    EntryMethod,
    EntryType,
    ExitMethod,
    HfModel,
    StopMethod,
)
from src.cal_coint_corr import CalCointCorr
from src.strategy.base import GenTrades, TradingStrategy
from src.utils import utils


class GenPriceAction:
    """Generate sentiment rating, closing price and price action for
    specific date, model, period, and correlation/cointegration combination.

    Usage:
        >>> gen_pa = GenPriceAction()
        >>> df = gen_pa.run()

    Args:
        path (DictConfig):
            OmegaConf DictConfig containing required folder and file paths.
        snp500_list (list[str]):
            List of S&P 500 list.
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
        num_lots (int):
            Number of lots to intitate new open positions (Default: 1).
        req_cols (list[str]):
            List of required columns to generate trades
            (Default: ["date", "high", "low", "close", "entry_signal", "exit_signal"]).
        monitor_close (bool):
            Whether to monitor close price ("close") or both high and low price
            (Default: True).
        percent_loss (float):
            Percentage loss allowed for investment (Default: 0.05).
        top_n (int):
            Top N number of stocks with lowest pvalue.

    Attributes:
        path (DictConfig):
            OmegaConf DictConfig containing required folder and file paths.
        snp500_list (list[str]):
            List of S&P 500 list.
        date (str):
            If provided, date when news are scraped.
        entry_type (EntryType):
            Whether to allow long ("long"), short ("short") or
            both long and short position ("longshort").
        entry_signal (str):
            Name of concrete implementation of 'EntrySignal' abstract class.
        exit_signal (str):
            Name of concrete implementation of 'ExitSignal' abstract class.
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
        hf_model (str):
            Name of FinBERT model in HuggingFace (Default: "ziweichen").
        coint_corr_fn (CointCorrFn):
            Name of function to perform either cointegration or correlation
            (Default: "coint").
        period (int):
            Time period used to compute cointegration (Default: 5).
        num_lots (int):
            Number of lots to intitate new open positions (Default: 1).
        req_cols (list[str]):
            List of required columns to generate trades
            (Default: ["date", "high", "low", "close", "entry_signal", "exit_signal"]).
        monitor_close (bool):
            Whether to monitor close price ("close") or both high and low price
            (Default: True).
        percent_loss (float):
            Percentage loss allowed for investment (Default: 0.05).
        top_n (int):
            Top N number of stocks with lowest pvalue.
        coint_corr_path (str):
            If provided, relative path to CSV cointegration information (Default: None).
        senti_path (str):
            Relative path to CSV file containing news sentiment rating
            (Default: "./data/sentiment.csv").
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
        snp500_list: list[str],
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
        num_lots: int = 1,
        req_cols: list[str] = [
            "date",
            "high",
            "low",
            "close",
            "entry_signal",
            "exit_signal",
        ],
        monitor_close: bool = True,
        percent_loss: float = 0.05,
        top_n: int = 10,
    ) -> None:
        self.path = path
        self.snp500_list = snp500_list
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
        self.num_lots = num_lots
        self.req_cols = req_cols
        self.monitor_close = monitor_close
        self.percent_loss = percent_loss
        self.top_n = top_n

        # Generate required file paths
        self.gen_paths()

    def gen_paths(self) -> None:
        """Generate required file paths i.e. 'coint_corr_path', 'senti_path',
        'model_dir' and 'price_action_dir'."""

        # Generate required folder and file paths
        coint_corr_date_dir = f"{self.path.coint_corr_dir}/{self.date}"
        date_dir = f"{self.path.data_dir}/{self.date}"

        self.coint_corr_path = f"{coint_corr_date_dir}/coint_corr_{self.period}y.csv"
        self.senti_path = f"{date_dir}/sentiment.csv"
        self.model_dir = (
            f"{date_dir}/"
            f"{self.entry_type}_{self.entry_struct}_{self.exit_struct}_{self.stop_method}/"
            f"{self.hf_model}_{self.coint_corr_fn}_{self.period}"
        )
        self.price_action_dir = f"{self.model_dir}/price_actions"

    def run(self) -> list[str]:
        """Generate and save DataFrame including average sentiment rating and
        closing price of stock and co-integrated stock.

        Args:
            None.

        Returns:
            no_trades_list (list[str]): List of news ticker without completed trades.
        """

        # Load 'sentiment.csv' and 'coint_5y.csv'
        df_senti = pd.read_csv(self.senti_path)
        df_coint_corr = self.load_coint_corr()

        results_list = []
        no_trades_list = []

        for ticker in df_senti["ticker"].unique():
            # Filter specific ticker
            df_ticker = df_senti.loc[df_senti["ticker"] == ticker, :].reset_index(
                drop=True
            )

            # Group by publication date and compute mean sentiment rating
            df_av = self.cal_mean_sentiment(df_ticker)

            # Append 'ticker' column
            df_av.insert(0, "ticker", ticker)

            # Append closing price of top N co-integrated stocks with lowest pvalue
            df_trades, no_trades = self.gen_topn_close(df_av, df_coint_corr, ticker)
            results_list.append(df_trades)
            no_trades_list.extend(no_trades)

        # Combine list of DataFrame to a single DataFrame
        df_results = pd.concat(results_list, axis=0).reset_index(drop=True)

        # Create folder if not exist
        utils.create_folder(self.model_dir)

        # Save combined DataFrame
        utils.save_csv(
            df_results, f"{self.model_dir}/trade_results.csv", save_index=False
        )

        return sorted(no_trades_list, reverse=False)

    def load_coint_corr(self) -> pd.DataFrame:
        """Load csv file containing cointegration and correlation info."""

        csv_path = Path(self.coint_corr_path)

        if not csv_path.is_file():
            print(
                f"'{csv_path.name}' is not available at '{csv_path}'. "
                f"Proceed to generate '{csv_path.name}'..."
            )
            cal_coint_corr = CalCointCorr(snp500_list=self.snp500_list, date=self.date)
            cal_coint_corr.run()

        return pd.read_csv(self.coint_corr_path)

    def cal_mean_sentiment(self, df_ticker: pd.DataFrame) -> pd.DataFrame:
        """Compute the average sentiment and average sentiment (excluding rating 3)
        for each trading day"""

        df = df_ticker.copy()

        # Generate 'date' column from 'pub_date' by extracting out the date
        # i.e. exclude time component
        df["pub_date"] = pd.to_datetime(df["pub_date"])
        df["date"] = df["pub_date"].dt.date

        # Exclude news with rating 3
        df_exclude = df.loc[df[self.hf_model] != 3, :]

        # Get Pandas Series of mean ratings (with and without rating 3)
        series_incl = df.groupby(by=["date"])[self.hf_model].mean()
        series_excl = df_exclude.groupby(by=["date"])[self.hf_model].median()

        # Generate DataFrame by concatenating 'series_incl' and 'series_excl'
        df_av = pd.concat([series_incl, series_excl], axis=1)
        df_av.columns = ["av_rating", "median_rating_excl"]

        # Replace null value in 'median_rating_excl' with 3.0 since all the news
        # articles have rating of 3 on the same day are excluded. Hence median will
        # return null.
        df_av = df_av.fillna(3)

        return df_av

    def append_dayname(self, data: pd.DataFrame) -> pd.DataFrame:
        """Append 'day_name' column i.e. from 'Monday' to 'Sunday' based on
        'date' index."""

        df = data.copy()

        # Set 'date' index as column; and set as datetime type
        df = df.reset_index()
        df["date"] = pd.to_datetime(df["date"])

        # Insert 'day' column and set 'date' as index
        ticker_index = df.columns.get_loc("ticker")
        day_name = df["date"].dt.day_name()
        df.insert(ticker_index + 1, "day_name", day_name)
        df = df.set_index("date")

        return df

    def gen_topn_close(
        self,
        df_av: pd.DataFrame,
        df_coint_corr: pd.DataFrame,
        ticker: str,
    ) -> tuple[pd.DataFrame, list[str]]:
        """Generate and save Dataframe for each 'top_n' cointegrated stocks with
        lowest pvalue.

        Args:
            df_av (pd.DataFrame):
                DataFrame containing average sentiment rating and closing price
                for ticker.
            df_coint_corr (pd.DataFrame):
                DataFrame containing cointegration and correlation info for
                stock ticker pairs.
            ticker (str):
                Stock ticker whose news are sentiment-rated.

        Returns:
            df_trades (pd.DataFrame):
                List of DataFrames containing completed trades for specific news ticker.
            no_trades_list (list[str]):
                List of news tickers with no completed trades.
        """

        df = df_av.copy()
        trades_list = []

        # Get list of cointegrated stocks with lowest pvalue
        coint_corr_list = self.get_topn_tickers(ticker, df_coint_corr)

        if coint_corr_list is None:
            return pd.DataFrame(), []

        no_trades_list = []

        for coint_corr_ticker in coint_corr_list:
            # Generate and save DataFrame for each cointegrated stock
            df_coint_corr_ticker = self.append_coint_corr_ohlc(df, coint_corr_ticker)

            # Load Sentiment Strategy
            trading_strategy = self.gen_strategy()
            df_trades, df_signals = trading_strategy(df_coint_corr_ticker)
            trades_list.append(df_trades)

            # Update no_trades_list for ticker pair
            file_name = f"{ticker}_{coint_corr_ticker}.csv"
            no_trades_list.extend(trading_strategy.no_trades)

            # Create folder if not exist
            utils.create_folder(self.price_action_dir)

            # Save signals DataFrame as csv file
            file_path = f"{self.price_action_dir}/{file_name}"
            utils.save_csv(df_signals, file_path, save_index=False)

        # Combine all trades DataFrame
        df_trades = pd.concat(trades_list, axis=0).reset_index(drop=True)

        # Ensure no duplicates in 'no_trades_list
        no_trades_list = list(set(no_trades_list))

        return df_trades, no_trades_list

    def gen_strategy(self) -> TradingStrategy:
        """Generate strategy based on 'entry_type', 'entry_struct', 'exit_struct',
        'stop_method', 'hf_model', 'coint_corr_fn' and 'period'."""

        # Get instance of concrete implementation of 'EntrySignal' abstract class
        class_name = SIGNAL_MAPPING.get(self.entry_signal)
        entry_inst = utils.get_class_instance(
            class_name, self.path.entry_signal_path, entry_type=self.entry_type
        )

        # Get instance of concrete implementation of 'ExitSignal' abstract class
        class_name = SIGNAL_MAPPING.get(self.exit_signal)
        exit_inst = utils.get_class_instance(
            class_name, self.path.exit_signal_path, entry_type=self.entry_type
        )

        # Get instance of concrete implementation of 'GenTrades' abstract class
        class_name = TRADES_MAPPING.get(self.trades_method)
        trades_inst = utils.get_class_instance(
            class_name,
            self.path.trades_path,
            entry_struct=self.entry_struct,
            exit_struct=self.exit_struct,
            num_lots=self.num_lots,
            req_cols=self.req_cols,
            monitor_close=self.monitor_close,
            percent_loss=self.percent_loss,
            stop_method=self.stop_method,
            entry_struct_path=self.path.entry_struct_path,
            exit_struct_path=self.path.exit_struct_path,
            stop_method_path=self.path.stop_method_path,
        )

        # Create trading strategy from 'EntrySignal', 'ExitSignal' and 'GenTrades'
        return TradingStrategy(entry_inst, exit_inst, trades_inst)

    def get_topn_tickers(
        self, ticker: str, df_coint_corr: pd.DataFrame
    ) -> list[str] | None:
        """Get list of top N stock with lowest pvalue for cointegration test with 'ticker'.

        Args:
            ticker (str):
                Stock ticker whose news are sentiment-rated.
            df_coint_corr (pd.DataFrame):
                DataFrame containing cointegration and correlation info for
                stock ticker pairs.

        Returns:
            (list[str]) | None:
                List of top N stocks with lowest cointegration pvalue or highest
                correlation if available.
        """

        # Filter records containing 'ticker' in 'ticker1' or 'ticker2' columns
        cond = (df_coint_corr["ticker1"] == ticker) | (
            df_coint_corr["ticker2"] == ticker
        )
        df = df_coint_corr.loc[cond, :].copy()

        if self.coint_corr_fn == "coint":
            # Ensure cointegration pvalue is less than 0.05
            df = df.loc[df[self.coint_corr_fn] < 0.05, :]
            sort_order = True

        else:
            # Ensure correlation is at more than 0.5
            df = df.loc[df[self.coint_corr_fn] > 0.5, :]
            sort_order = False

        if df.empty:
            print(
                f"Records doesn't meet minimum requirement for '{self.coint_corr_fn}' method [ticker: {ticker}, period: {self.period} years]."
            )
            return None

        # Get top N stocks based on 'self.coint_corr_fn'
        df_topn = df.sort_values(by=self.coint_corr_fn, ascending=sort_order).head(
            self.top_n
        )

        # Get set of unique tickers from 'ticker1' and 'ticker2' columns
        coint_set = set(df_topn["ticker1"].to_list()) | set(
            df_topn["ticker2"].to_list()
        )

        # Convert set to sorted list excluding 'ticker'
        return [symb for symb in sorted(list(coint_set)) if symb != ticker]

    def append_coint_corr_ohlc(
        self, df_av: pd.DataFrame, coint_corr_ticker: str
    ) -> pd.DataFrame:
        """Append OHLC data of cointegrated stocks."""

        # Load OHLCV prices for ticker
        ohlcv_path = f"{self.path.stock_dir}/{coint_corr_ticker}.parquet"
        df_ohlcv = pd.read_parquet(ohlcv_path)

        # Ensure both df.index and df_ohlcv are datetime objects
        df_av.index = pd.to_datetime(df_av.index)
        df_ohlcv.index = pd.to_datetime(df_ohlcv.index)
        df_ohlcv.index.name = df_ohlcv.index.name.lower()

        # Get date of earliest and latest date in df_av
        earliest_date = df_av.index.min()

        # # Get 1st day of the earliest month and last day of the latest month
        # start_date = earliest_date.replace(day=1)

        # Filter df_ohlcv based on start and end date
        df_ohlcv = df_ohlcv.loc[
            (df_ohlcv.index >= earliest_date) & (df_ohlcv.index <= self.date), :
        ]

        # Perform left join of 'df_av' on 'df_ohlcv'
        df = df_ohlcv.join(df_av)

        # Rename column
        df = df.rename(columns={"Ticker": "coint_corr_ticker"})
        df.columns = [col.lower() for col in df.columns]

        # Ensure no missing values at 'ticker' column
        df["ticker"] = df["ticker"].ffill().bfill()

        # Insert 'day_name' column just after 'ticker' column
        df = self.append_dayname(df)

        # Reset index and ensure 'date' column is on datetime type
        df = df.reset_index()
        df["date"] = pd.to_datetime(df["date"])

        return df
