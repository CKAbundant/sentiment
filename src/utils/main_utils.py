"""Helper functions used in 'main.py'"""

from decimal import Decimal
from itertools import product
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


def gen_stats(
    combined: pd.DataFrame,
    strat_comp: str,
    category: str,
    analysis_cols=[
        "annualized_return",
        "win_rate",
        "max_days_held",
        "mean_neg_percent_return",
        "highest_neg_percent_return",
    ],
) -> pd.DataFrame:
    # Generate Panda Series containing mean value for each analysis column
    results = combined.loc[:, analysis_cols].mean()
    results = results.map(lambda val: Decimal(str(round(val, 6))))

    # Convert to DataFrame and transpose
    df = pd.DataFrame(results).T

    # Append 'strat_comp' and 'comp'
    df.insert(0, "category", category)
    df.insert(0, "strat_comp", strat_comp)

    return df


def gen_stats_df(
    df_combined_overall: pd.DataFrame,
    strat_comp_list: list[str] = [
        "entry_type",
        "entry_struct",
        "exit_struct",
        "stop_method",
        "hf_model",
        "coint_corr_fn",
        "period",
    ],
    analysis_list: list[str] = [
        "annualized_return",
        "win_rate",
        "max_days_held",
        "mean_neg_percent_return",
        "highest_neg_percent_return",
    ],
) -> pd.DataFrame:
    """Generate statistics of combined overall summary for all strategies.

    Args:
        df_combined_overall (pd.DataFrame):
            DataFrame containing overall summary for all strategies.
        strat_comp_list (list[str]):
            List of component for strategies
            (Default: ["entry_type", "entry_struct", "exit_struct",
            "stop_method", "hf_model", "coint_corr_fn", "period"]).

    Returns:
        (pd.DataFrame):
            DataFrame containing stats for different strategy components.
    """

    df = df_combined_overall.copy()
    df_list = []

    for strat_comp in strat_comp_list:
        # Get unique component of strategy component
        for category in df[strat_comp].unique():
            df_list.append(
                gen_stats(
                    df.loc[df[strat_comp] == category, :],
                    strat_comp,
                    category,
                    analysis_list,
                )
            )

    # Concatenate row-wise
    return pd.concat(df_list, axis=0).reset_index(drop=True)
