"""Main function to execute."""

import sys
from argparse import ArgumentParser, Namespace
from itertools import product
from pathlib import Path
from pprint import pformat
from typing import get_args

import pandas as pd

# Add repo directory to sys.path if not exist
repo_dir = Path(__file__).parent.as_posix()
if repo_dir not in sys.path:
    sys.path.append(repo_dir)

from config.variables import COINT_CORR_FN, HF_MODEL, IGNORE_LIST, URL
from src.cal_coint_corr import CalCointCorr
from src.cal_profit_loss import CalProfitLoss
from src.download_ohlcv import DownloadOHLCV
from src.gen_data import GenData
from src.gen_price_action import GenPriceAction
from src.plot_news import PlotNews
from src.utils import utils


def main() -> None:
    """Generate proof-of-concept by running instance of Poc class."""

    # Get arguments from command line
    args = parse_arguments()
    date = args.date
    no_sentiment = args.no_sentiment

    # # Generate list of S&P500 stocks
    # snp500_list = utils.gen_snp500_list(URL, IGNORE_LIST)

    # if not no_sentiment:
    #     # Generate DataFrame containing news and sentiment scores for different
    #     # FinBERT variant.
    #     gen_data = GenData(date=date)
    #     df_senti = gen_data.run()
    #     print(f"\nsentiment : \n\n{df_senti}\n")

    # # Download OHLCV data for S&P500 stocks
    # download_ohlcv = DownloadOHLCV(snp500_list=snp500_list, end_date=date)
    # download_ohlcv.run()

    # # Perform cointegration and correlation analysis and save results as csv file
    # cal_coint_corr = CalCointCorr(snp500_list=snp500_list, date=date)
    # cal_coint_corr.run()

    # # Test out different strategies
    # hf_models = get_args(HF_MODEL)
    # coint_corr_fns = get_args(COINT_CORR_FN)
    # periods = (1, 3, 5)

    # for hf_model, coint_corr_fn, period in product(hf_models, coint_corr_fns, periods):
    #     # Generate price action of top 10 cointegrated/correlated stocks
    #     gen_pa = GenPriceAction(date, hf_model, coint_corr_fn, period)
    #     gen_pa.run()

    #     # Compile profit and loss; and generate reports
    #     cal_pl = CalProfitLoss(date, hf_model, coint_corr_fn, period)
    #     _, _, _, _ = cal_pl.run()

    # Perform analysis on different strategies

    # print(f"df_results : \n\n{pformat(df_results)}\n")
    # print(f"df_overall : \n\n{pformat(df_overall)}\n")
    # print(f"df_breakdown : \n\n{pformat(df_breakdown)}\n")
    # print(f"df_top_ret_pairs : \n\n{pformat(df_top_ret_pairs)}\n")

    plot_news = PlotNews(date=date)
    plot_news.run()


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


if __name__ == "__main__":
    main()
