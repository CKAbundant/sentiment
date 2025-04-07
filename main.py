"""Main function to execute."""

import sys
from itertools import product
from pathlib import Path

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
from src.plot_coint_corr import PlotCointCorr
from src.plot_news import PlotNews
from src.plot_strategies import PlotStrategies
from src.utils import main_utils, utils


def main() -> None:
    """Generate proof-of-concept by running instance of Poc class."""

    # Get arguments from command line
    args = main_utils.parse_arguments()
    date = args.date
    strategy = args.strategy
    no_sentiment = args.no_sentiment

    print(f"date : {date}")

    # Generate list of S&P500 stocks
    snp500_list = utils.gen_snp500_list(URL, IGNORE_LIST)

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
    # main_utils.run_strategies(date)

    # plot_news = PlotNews(date=date)
    # plot_news.run()

    # plot_coint_corr = PlotCointCorr(date=date)
    # plot_coint_corr.run()

    # # Perform analysis on different strategies
    # plot_strategies = PlotStrategies(date=date)
    # plot_strategies.run()


if __name__ == "__main__":
    main()
