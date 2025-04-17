"""Helper functions used in 'main.py'"""

from decimal import Decimal
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from config.variables import COINT_CORR_FN, HF_MODEL
from src.cal_profit_loss import CalProfitLoss
from src.gen_price_action import GenPriceAction
from src.utils import utils


def gen_signals(date: str, cfg: DictConfig) -> None:
    """Generate price signals for single strategy or all strategies combinations."""

    params = {
        "date": date,
        "top_n": cfg.top_n_tickers,
        "data_dir": cfg.data_dir,
        "stock_dir": cfg.stock_dir,
        "coint_corr_dir": cfg.coint_corr_dir,
    }

    if cfg.test_all:
        # Test out different strategies
        run_strategies(params, cfg.full)

    else:
        # Test specific strategy
        gen_pa = GenPriceAction(params, cfg.single)
        gen_pa.run()


def run_strategies(params: dict[str, Any], full: DictConfig) -> None:
    """Run different combinations of HuggingFace FinBERT sentiment rater,
    cointegration/correlation analysis and time periods for selected 'date'.

    Args:
        params (dict[str, Any]):
            Dictionary containing additional parameters i.e. 'date', 'top_n',
            and file paths required to initialze 'GenPriceAction' class.
        full (DictConfig):
            OmegaConf DictConfig object containing parameters for running all
            strategies.

    Returns:
        None.
    """

    # # Get FinBERT models, cointegration/correlation functions and time periods
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

    # Get list of combinations for long, short and long-short strategies
    combi_list = [list(product(*strat)) for strat in full]
    combi_list = [combi for sub_list in combi_list for combi in sub_list]

    for (
        ent_type,
        ent_struct,
        ex_struct,
        stop_method,
        hf_model,
        coint_corr_fn,
        period,
    ) in combi_list:
        # Generate price actions of top 10 cointegrated/correlated stocks
        gen_pa = GenPriceAction(
            date=params["date"],
            entry_type=ent_type,
            entry_struct=ent_struct,
            exit_struct=ex_struct,
            stop_method=stop_method,
            hf_model=hf_model,
            coint_corr_fn=coint_corr_fn,
            period=period,
            top_n=params["top_n"],
            data_dir=params["data_dir"],
            stock_dir=params["stock_dir"],
            coint_corr_dir=params["coint_corr_dir"],
        )
        gen_pa.run()


def convert_to_decimal(val: np.number) -> Decimal:
    return Decimal(str(round(val, 6)))


def gen_stats(
    combined: pd.DataFrame,
    strat_comp: str,
    comp: str,
    cols=["annualized_return", "win_rate", "max_days_held"],
) -> pd.DataFrame:
    results = combined.loc[:, cols].mean()
    results = results.map(convert_to_decimal)

    # Convert to DataFrame and transpose
    df = pd.DataFrame(results).T

    # Append 'strat_comp' and 'comp'
    df.insert(0, "comp", comp)
    df.insert(0, "strat_comp", strat_comp)

    return df


def gen_stats_df(
    df_combined_overall: pd.DataFrame,
    strat_comp_list: list[str] = ["hf_model", "coint_corr_fn", "period"],
    analysis_list: list[str] = ["annualized_return", "win_rate", "max_days_held"],
) -> pd.DataFrame:
    """Generate DataFrame containing

    Args:
        df_combined_overall (pd.DataFrame):
            DataFrame containing overall summary for all 48 strategies.
        strat_comp_list (list[str]):
            List of component for strategies
            (Default: ["hf_model", "coint_corr_fn", "period"]).

    Returns:
        (pd.DataFrame):
            DataFrame containing stats for different strategy components.
    """

    df = df_combined_overall.copy()
    df_list = []

    for strat_comp in strat_comp_list:
        # Get unique component of strategy component
        for comp in df[strat_comp].unique():
            df_list.append(
                gen_stats(
                    df.loc[df[strat_comp] == comp, :], strat_comp, comp, analysis_list
                )
            )

    # Concatenate row-wise
    return pd.concat(df_list, axis=0).reset_index(drop=True)
