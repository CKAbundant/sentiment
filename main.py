"""Main function to execute."""

import sys
from itertools import product
from pathlib import Path

import pandas as pd

# Add repo directory to sys.path if not exist
repo_dir = Path(__file__).parent.as_posix()
if repo_dir not in sys.path:
    sys.path.append(repo_dir)

from src.cal_coint_corr import CalCointCorr
from src.download_ohlcv import DownloadOHLCV
from src.gen_data import GenData
from src.plot_coint_corr import PlotCointCorr
from src.plot_news import PlotNews
from src.plot_strategies import PlotStrategies
from src.utils import main_utils, utils


def main() -> None:
    """Generate proof-of-concept by running instance of Poc class."""

    # load configuration from 'config.yaml'
    cfg = utils.load_config()
    date = cfg.date or utils.get_current_dt(fmt="%Y-%m-%d")

    # Generate list of S&P500 stocks
    snp500_list = utils.gen_snp500_list(cfg.snp500_url, cfg.ignore_list)

    if cfg.download_ohlcv:
        # Download OHLCV data for S&P500 stocks
        download_ohlcv = DownloadOHLCV(
            snp500_list=snp500_list, end_date=date, **cfg.ohlcv
        )
        download_ohlcv.run()

    if cfg.cal_coint_corr:
        # Perform cointegration and correlation analysis and save results as csv file
        cal_coint_corr = CalCointCorr(
            path=cfg.path, snp500_list=snp500_list, date=date, **cfg.coint_corr
        )
        cal_coint_corr.run()

    if cfg.sentiment:
        # Generate DataFrame containing news and sentiment scores for different
        # FinBERT variant.
        gen_data = GenData(date=date, **cfg.senti)
        gen_data.run()

    if cfg.test_all is not None:
        # Generate price action signals
        main_utils.gen_signals(date=date, snp500_list=snp500_list, cfg=cfg)

    if cfg.plot_graph:
        # plot_news = PlotNews(path=cfg.path, date=date)
        # plot_news.run()

        # plot_coint_corr = PlotCointCorr(path=cfg.path, date=date, periods=cfg.periods)
        # plot_coint_corr.run()

        # Perform analysis on different strategies
        plot_strategies = PlotStrategies(path=cfg.path, date=date, **cfg.plot_strat)
        plot_strategies.run()


if __name__ == "__main__":
    main()
