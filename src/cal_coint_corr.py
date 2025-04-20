"""Class to perform cointegration for S&P 500 stocks"""

from itertools import combinations
from pathlib import Path
from typing import get_args

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from matplotlib.figure import Figure
from omegaconf import DictConfig
from scipy import stats
from statsmodels.tsa.stattools import coint
from tqdm import tqdm

from config.variables import CorrFn
from src.utils import utils


class CalCointCorr:
    """Calculate correlation and cointegration between all ticker pairs
    combination in S&P500 stocks.

    - Correlation (Pearson, Spearman and Kendall)
    - Cointegration (Engle-Granger)

    Usage:
        # Use default setting i.e. date = <current date>
        >>> get_rel = GetRel()
        >>> get_rel.run()

    Args:
        path (DictConfig):
            OmegaConf DictConfig containing required file and directory paths.
        snp500_list (list[str]):
            List of S&P 500 list.
        date (str):
            If provided, date when news are scraped.
        periods (list[int]):
            list of time periods to compute correlation and cointegration
            (Default: [1, 3, 5]).

    Attributes
        path (DictConfig):
            OmegaConf DictConfig containing required file and directory paths.
        snp500_list (list[str]):
            List of S&P 500 list.
        date (str):
            If provided, date when news are scraped.
        periods (list[int]):
            list of time periods to compute correlation and cointegration
            (Default: [1, 3, 5]).
        corr_fn_list (tuple[str]):
            List of correlation functions (Default: ("pearsonr", "spearmanr", "kendalltau")).
        stock_dir (str):
            Relative path to folder containing stocks OHLCV data
            (Default: "./data/stock").
        coint_corr_date_dir (str):
            Relative path to subfolder under 'coint_corr_dir'.
    """

    def __init__(
        self,
        path: DictConfig,
        snp500_list: list[str],
        date: str | None = None,
        periods: list[int] = [1, 3, 5],
        corr_fn_list: tuple[str] | None = None,
    ) -> None:
        self.snp500_list = snp500_list
        self.date = date or utils.get_current_dt(fmt="%Y-%m-%d")
        self.periods = periods
        self.corr_fn_list = corr_fn_list or get_args(CorrFn)

        # Get directory paths
        self.stock_dir = path.stock_dir
        self.coint_corr_date_dir = f"{path.coint_corr_dir}/{self.date}"

    def run(self) -> None:
        """Generate DataFrame containing correlation and cointegration between
        all pair combinations of S&P500 stocks."""

        # Perform correlation and cointegration analysis for 5, 3 and 1 year
        for num_years in self.periods:
            # Path to csv file containing required cointegration and correlation data
            coint_corr_path = Path(
                f"{self.coint_corr_date_dir}/coint_corr_{num_years}y.csv"
            )

            # If file exist, skip cointegration and correlation computation
            if coint_corr_path.is_file():
                print(
                    f"'{coint_corr_path.name}' exist at '{coint_corr_path.as_posix()}'. No further computation required."
                )
                return

            # Ensure subfolder exist
            utils.create_folder(self.coint_corr_date_dir)

            # Generate DataFrame containing cointegration and correlation info
            df_coint_corr = self.gen_coint_corr_df(num_years)
            df_coint_corr.to_csv(coint_corr_path, index=False)

    def gen_coint_corr_df(self, num_years: int) -> pd.DataFrame:
        """Generate DataFrame containing cointegration and correlation
        between closing price of all S&P500 ticker pairs.

        Args:
            num_years (int):
                Number of years required to compute cointegration.

        Returns:
            (pd.DataFrame):
                DataFrame containing cointegration and correlation info.
        """

        # Get list of tickers with enough data for cointegration computation
        updated_list = self.get_enough_period(num_years)

        records = []

        for ticker1, ticker2 in combinations(updated_list, 2):
            print(f"{ticker1}-{ticker2} [num_years : {num_years}]")

            try:
                # Get equal length 'Close' price for both tickers
                df_close = self.gen_timeseries(ticker1, ticker2, num_years)

                # Compute various correlation and cointegration values
                results_dict = self.cal_coint_corr(df_close, ticker1, ticker2)
                records.append(results_dict)

            except Exception as e:
                print(f"Error encountered for {ticker1}-{ticker2} : {e}")

        # Convert to DataFrame; and save as csv file
        return pd.DataFrame(records)

    def cal_coint_corr(
        self, df_close: pd.DataFrame, ticker1: str, ticker2
    ) -> dict[str, float]:
        """Compute various cointegration and correlation between ticker1 and ticker2.

        Args:
            df_close (pd.DataFrame):
                DataFrame containing closing price of ticker1 and ticker2.
            ticker1 (str):
                Stock ticker whose news are sentiment-rated.
            ticker2 (str):
                Cointegrated stock ticker to 'ticker1'.

        Returns:
            record (dict[str, float]):
                Dictionary containing cointegration and correlation values
                for ticker pair.
        """

        # Compute cointegration values based on Engle-Granger test
        _, coint_pvalue, _ = coint(df_close[ticker1], df_close[ticker2])

        # Create dictionary to store cointegration and correlation results
        record = {
            "ticker1": ticker1,
            "ticker2": ticker2,
            "coint": coint_pvalue,
        }

        for corr_fn_name in self.corr_fn_list:
            if not hasattr(stats, corr_fn_name):
                raise ModuleNotFoundError(f"{corr_fn_name} is an invalid Scipy model.")

            # Compute correlation and update 'record' dictionary
            corr_fn = getattr(stats, corr_fn_name)
            corr_results = corr_fn(df_close[ticker1], df_close[ticker2])
            record[corr_fn_name] = corr_results.statistic

        return record

    def get_enough_period(self, num_years: int) -> list[str]:
        """Identify list of tickers that have enough data for 'num_years'
        number of years.

        Args:
            num_years (int): Number of years required to compute cointegration.

        Returns:
            (list[str]): List of tickers with enough data points.
        """

        return [
            ticker for ticker in self.snp500_list if self.is_enough(ticker, num_years)
        ]

    def gen_timeseries(
        self, ticker1: str, ticker2: str, num_years: int
    ) -> pd.DataFrame | None:
        """Generate equal length timeseries of Pandas Series Type based on given
        tickers and number of years.

        Args:
            ticker1 (str): First stock ticker.
            ticker2 (str): Second stock ticker.
            num_years (int): Number of years required to compute cointegration.

        Returns:
            df (pd.DataFrame): DataFrame containing closing price for both tickers.
        """

        # Load DataFrame for both tickers
        df1 = pd.read_parquet(f"{self.stock_dir}/{ticker1}.parquet")
        df2 = pd.read_parquet(f"{self.stock_dir}/{ticker2}.parquet")

        # Filter DataFrame to contain past 'num_years' years of records
        df1 = self.filter_df(df1, num_years)
        df2 = self.filter_df(df2, num_years)

        # Ensure timeseries for both tickers are of equal length by concatenating
        # "Close" Pandas Series and removing null values
        df = pd.concat([df1["Close"], df2["Close"]], axis=1)
        df.columns = [ticker1, ticker2]
        df = df.dropna()

        return df

    def filter_df(self, df: pd.DataFrame, num_years: int) -> pd.DataFrame:
        """Filter DataFrame to contain past 'num_years' years of records."""

        # Get the earliest date based on 'num_years' period
        req_earliest = self.cal_req_earliest(df, num_years)

        return df.loc[df.index >= req_earliest, :].reset_index(drop=True)

    def cal_req_earliest(self, df: pd.DataFrame, num_years: int) -> pd.Timestamp:
        """Calculate earliest date for past 'num_years' records in DataFrame."""

        # latest date in DataFrame
        latest_date = df.index.max()

        return latest_date - pd.DateOffset(years=num_years)

    def is_enough(self, ticker: str, num_years: int) -> bool:
        """Check whether there is enough data for cointegration computation.

        Args:
            ticker (str): Stock ticker
            num_years (int): Number of years required to compute cointegration.

        Returns:
            (bool): True if there is enough data.
        """

        # Load DataFrame from parquet file
        df = pd.read_parquet(f"{self.stock_dir}/{ticker}.parquet")

        # Compute required earliest date
        earliest = df.index.min()
        req_earliest = self.cal_req_earliest(df, num_years)

        return earliest <= req_earliest
