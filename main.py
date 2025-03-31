"""Main function to execute."""

import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from pprint import pformat

import pandas as pd

# Add repo directory to sys.path if not exist
repo_dir = Path(__file__).parent.as_posix()
if repo_dir not in sys.path:
    sys.path.append(repo_dir)

from src.cal_profit_loss import CalProfitLoss
from src.download_ohlcv import DownloadOHLCV
from src.gen_data import GenData
from src.gen_price_action import GenPriceAction
from src.utils import utils


def main() -> None:
    """Generate proof-of-concept by running instance of Poc class."""

    # Get arguments from command line
    args = parse_arguments()
    date = args.date
    no_sentiment = args.no_sentiment

    if not no_sentiment:
        # Generate DataFrame containing news and sentiment scores for different
        # FinBERT variant.
        gen_data = GenData(date=date)
        df_senti = gen_data.run()
        print(f"\nsentiment : \n\n{df_senti}\n")

    # Download OHLCV data for S&P500 stocks
    download_ohlcv = DownloadOHLCV(end_date=date)
    download_ohlcv.run()

    # Generate price action of top 10 stocks with lowest cointegration pvalue
    # with selected stocks
    gen_pa = GenPriceAction(date=date)
    gen_pa.run()

    # # Compile profit and loss; and generate reports
    # cal_pl = CalProfitLoss(date=date)
    # df_results, df_overall, df_breakdown, df_top_ret_pairs = cal_pl.run()
    # print(f"df_results : \n\n{pformat(df_results)}\n")
    # print(f"df_overall : \n\n{pformat(df_overall)}\n")
    # print(f"df_breakdown : \n\n{pformat(df_breakdown)}\n")
    # print(f"df_top_ret_pairs : \n\n{pformat(df_top_ret_pairs)}\n")

    # Test out different strategies


def parse_arguments() -> Namespace:
    """Parse optional 'date' and 'no-sentiment' flags from command line."""

    parser = ArgumentParser(description="Parse optional date")

    # Add optional date' argument (default to today's date)
    parser.add_argument(
        "--date",
        type=str,
        default=utils.get_current_dt(fmt="%Y-%m-%d"),  # Default to today's date
        help="Date in 'YYYY-MM-DD' format (default: today's date).",
    )

    # Add '--no-sentiment' flag
    parser.add_argument(
        "--no-sentiment",
        action="store_true",  # Set to True if flag is present
        help="Skip web scraping and sentiment analysis.",
    )

    return parser.parse_args()


def test_strategies() -> pd.DataFrame:
    """"""


if __name__ == "__main__":
    main()
