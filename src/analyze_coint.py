"""Class to analyze cointegration data of specific ticker:

- Iterate through folder containing cointegration csv files of
different dates and time period.
- Identify if cointegrated tickers remain the same for different dates and period.
"""

import itertools
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


class AnalyzeCoint:
    """Analyze cointegration of specific ticker for different dates and time period:

    - Display common tickers for same period (i.e. 1, 5 and 10 years).
    - Generate DataFrame for different date and period combination.

    Usage:
        >>> analyze_coint = AnalyzeCoint()
        >>> df_coint_analysis = analyze_coint("AAPL")

    Args:
        coint_dir (str):
            Relative path to folder containing cointegration CSV files
            (Default: "./data/coint").
        top_n (int):
            Top N cointegrated tickers with lowest pvalue (Default: 20).

    Attributes:
        coint_dir (str):
            Relative path to folder containing cointegration CSV files.
        coint_paths (list[str]):
            Path glob iterator for csv files in 'coint_dir'.
        top_n (int):
            Top N cointegrated tickers with lowest pvalue (Default: 10).
    """

    def __init__(
        self,
        coint_dir: str = "./data/coint",
        top_n: int = 20,
    ) -> None:
        self.coint_dir = coint_dir
        self.coint_paths = list(Path(self.coint_dir).rglob("*.csv"))
        self.top_n = top_n

    def __call__(self, ticker: str) -> pd.DataFrame:
        """Determine if top N tickers having lowest conintegrated pvalue with 'ticker'
        are consistent across different dates and periods."""

        if not Path(self.coint_dir).is_dir():
            raise FileExistsError(f"'{self.coint_dir}' does not exist!")

        records = []

        # Generate dictionary mapping date and period to list of top N cointegrated tickers
        coint_dict = self.gen_coint_dict(ticker)

        for path1, path2 in itertools.combinations(self.coint_paths, 2):
            date1 = self.get_date(path1)
            date2 = self.get_date(path2)
            period1 = self.get_period(path1)
            period2 = self.get_period(path2)
            top_n_tickers1 = coint_dict[date1][period1]
            top_n_tickers2 = coint_dict[date2][period2]
            common_tickers = self.get_common_tickers(top_n_tickers1, top_n_tickers2)

            record = {
                "date1": date1,
                "period1": period1,
                f"top_{self.top_n}_tickers1": ", ".join(top_n_tickers1),
                "date2": date2,
                "period2": period2,
                f"top_{self.top_n}_tickers2": ", ".join(top_n_tickers2),
                "num_common": len(common_tickers),
                "common_tickers": ", ".join(common_tickers),
            }
            records.append(record)

        # Convert to DataFrame
        return pd.DataFrame(records)

    def get_coint_ticker(self, ticker: str, file_path: Path) -> pd.DataFrame:
        """Get DataFrame containing stocks which are co-integrated with 'ticker'
        having pvalue < 0.05.

        Args:
            ticker (str): Stock ticker.
            file_path (str): Path object to cointegration csv files.

        Returns:
            df_ticker (pd.DataFrame):
                DataFrame containing cointegrated stocks with ticker
                sorted by pvalue in ascending order.
        """

        df_coint = pd.read_csv(file_path)

        cond = ((df_coint["cointegrate"] == 1) & (df_coint["ticker1"] == ticker)) | (
            (df_coint["cointegrate"] == 1) & (df_coint["ticker2"] == ticker)
        )

        df_ticker = df_coint.loc[cond, :].sort_values(by=["pvalue"], ascending=True)

        return df_ticker

    def gen_coint_dict(self, ticker: str) -> defaultdict[dict]:
        """Generate dictionary mapping date and period to list of top N tickers
        having lowest cointegration pvalue with 'ticker'."""

        coint_dict = defaultdict(dict)

        for file_path in self.coint_paths:
            # Get date and period from 'file_path'
            date = self.get_date(file_path)
            period = self.get_period(file_path)

            # Get cointegration tickers for ticker from cointegrated csv file
            df_coint_ticker = self.get_coint_ticker(ticker, file_path)
            coint_dict[date][period] = self.get_coint_list(ticker, df_coint_ticker)

        return coint_dict

    def get_coint_list(self, ticker: str, coint_ticker: pd.DataFrame) -> list[str]:
        """Get list of stocks which are cointegrated with 'ticker' with pvalue
        in ascending order.

        Args:
            ticker (str):
                Stock ticker whose news are sentiment-rated.
            coint_ticker (pd.DataFrame):
                DataFrame containing cointegrated stocks with 'ticker'.

        Returns:
            (list[str]): List of sorted cointegrated stocks with ticker
        """

        coint_list = []
        # Extract cointegrated stocks by iterating through DataFrame
        for ticker1, ticker2 in coint_ticker.loc[:, ["ticker1", "ticker2"]].itertuples(
            index=False, name=None
        ):
            if ticker1 == ticker:
                coint_list.append(ticker2)
            else:
                coint_list.append(ticker1)

        return coint_list[: self.top_n] if self.top_n else coint_list

    def get_date(self, file_path: Path) -> str:
        """Get date from file path of cointegrated csv file."""

        # e.g. './data/coint/2025-03-26/coint_3y.csv'
        return file_path.parent.name

    def get_period(self, file_path: Path) -> int:
        """Get period file path of cointegrated csv file."""

        # e.g. './data/coint/2025-03-26/coint_3y.csv'
        return int(re.findall(r"(?<=coint_)\d+", file_path.stem)[0])

    def get_common_tickers(self, list1: list[str], list2: list[str]) -> list[str]:
        """Get common tickers in both 'list1' and 'list2' while preserving order
        in 'list1'."""

        return [ticker for ticker in list1 if ticker in list2]
