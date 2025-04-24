"""Helper functions used in 'main.py'"""

from collections import defaultdict
from decimal import Decimal
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig, ListConfig
from tqdm import tqdm

from config.variables import CointCorrFn, HfModel
from src.cal_profit_loss import CalProfitLoss
from src.gen_price_action import GenPriceAction
from src.utils import utils


def gen_signals(date: str, snp500_list: str, cfg: DictConfig) -> None:
    """Generate price signals for single strategy or all strategies combinations.

    Args:
        date (str): Date when news are sentiment rated.
        snp500_list (list[str]): List of S&P500 stock tickers.
        cfg (DictConfig): OmegaConf DictConfig containing required parameters.

    Returns:
        None.
    """

    if cfg.test_all:
        # Test out different strategies
        run_strategies(date, snp500_list, cfg)

    else:
        # Test specific strategy
        gen_pa = GenPriceAction(
            date=date, snp500_list=snp500_list, **cfg.single, **cfg.std
        )
        _ = gen_pa.run()


def run_strategies(date: str, snp500_list: list[str], cfg: DictConfig) -> None:
    """Run different combinations of HuggingFace FinBERT sentiment rater,
    cointegration/correlation analysis and time periods for selected 'date'.

    Args:
        date (str):
            Date when news is sentiment rated.
        snp500_list (list[str]):
            List of S&P500 stock tickers.
        cfg (DictConfig):
            OmegaConf DictConfig containing required parameters.

    Returns:
        None.
    """

    # Get list of combinations for long, short and long-short strategies
    combi_list = [list(product(*strat)) for strat in cfg.full]
    combi_list = [combi for sub_list in combi_list for combi in sub_list]

    for (
        ent_type,
        ent_struct,
        ex_struct,
        stop_method,
        hf_model,
        coint_corr_fn,
        period,
    ) in tqdm(combi_list):
        # Generate price actions of top 10 cointegrated/correlated stocks
        gen_pa = GenPriceAction(
            date=date,
            snp500_list=snp500_list,
            entry_type=ent_type,
            entry_struct=ent_struct,
            exit_struct=ex_struct,
            stop_method=stop_method,
            hf_model=hf_model,
            coint_corr_fn=coint_corr_fn,
            period=period,
            **cfg.std,
        )
        gen_pa.run()

        # Calculate overall summary, breakdown summary and top ticker pairs
        # with highest daily return for each news ticker
        cal_pl = CalProfitLoss(
            path=cfg.path,
            news_ticker_list=cfg.news_ticker_list,
            date=date,
            entry_type=ent_type,
            entry_struct=ent_struct,
            exit_struct=ex_struct,
            stop_method=stop_method,
            hf_model=hf_model,
            coint_corr_fn=coint_corr_fn,
            period=period,
        )
        _, _, _ = cal_pl.run()


def gen_stats_df(
    combined_path: str,
    analysis_list: list[str] = [
        "annualized_return",
        "win_rate",
        "max_days_held",
        "highest_neg_percent_return",
    ],
) -> pd.DataFrame:
    """Generate statistics of combined overall summary for all strategies.

    Args:
        combined_path (str):
            Relative path to 'combined_overall.csv' for specific date.
        strat_comp_list (list[str]):
            List of component for strategies
            (Default: ["entry_type", "entry_struct", "exit_struct",
            "stop_method", "hf_model", "coint_corr_fn", "period"]).
        analysis_list: (list[str]):
            List of columns for analysis (Default: ["annualized_return",
            "win_rate", "max_days_held", "highest_neg_percent_return"])

    Returns:
        (pd.DataFrame):
            DataFrame containing stats for different strategy components.
    """

    # Load csv file if exist
    combined_path = Path(combined_path)
    date_dir = combined_path.parent

    if not combined_path.is_file():
        raise FileNotFoundError(
            f"'combined_overall.csv' is not found at '{combined_path}'."
        )

    # Filter columns to 'analysis_list'
    df = utils.load_csv(combined_path, tz="America/New_York")
    df = df.loc[:, analysis_list]

    # Convert to float type in order for 'pandas.describe'
    df = df.astype(float)
    df = df.describe().T

    # Convert float back to Decimal
    df = df.map(lambda val: Decimal(str(val)).quantize(Decimal("1.000000")))

    # Save DataFrame as csv file
    utils.save_csv(df, date_dir.joinpath("combined_overall_stats.csv"), save_index=True)

    return df


def get_top_pairs_by_entry_type(
    top_n_path: str,
    top_n: int = 10,
    req_cols: list[str] = [
        "ticker_pair",
        "overall_daily_ret",
        "days_held_max",
        "trading_period",
        "win_rate",
        "win_count",
        "neg_ret_mean",
        "neg_ret_max",
    ],
) -> dict[str, pd.DataFrame]:
    """Get top N ticker pairs with highest daily return for each category
    of entry type."""

    # load 'top_15_tickers.csv' if available
    top_n_path = Path(top_n_path)

    if not top_n_path.is_file():
        raise FileNotFoundError(
            f"{top_n_path.name} is not present in '{top_n_path.as_posix()}'."
        )

    df = utils.load_csv(top_n_path)
    df_list = []

    for category in df["entry_type"].unique():
        # Filter by category and required columns; sort by overall_daily_ret
        df_cat = df.loc[df["entry_type"] == category, req_cols]
        df_cat = df_cat.sort_values(by=["overall_daily_ret"], ascending=False)

        # Drop duplicates by keeping the highest overall_daily_ret
        # Get top N ticker pairs with highest overall daily returns
        df_cat = df_cat.drop_duplicates(subset="ticker_pair").reset_index(drop=True)
        df_cat = df_cat.head(top_n)

        # Insert "category" column
        df_cat.insert(0, "category", category)

        df_list.append(df_cat)

    return pd.concat(df_list, axis=0).reset_index(drop=True)
