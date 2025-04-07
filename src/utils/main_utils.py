"""Helper functions used in 'main.py'"""

from argparse import ArgumentParser, Namespace
from decimal import Decimal
from itertools import product
from typing import get_args

import numpy as np
import pandas as pd

from config.variables import COINT_CORR_FN, HF_MODEL
from src.cal_profit_loss import CalProfitLoss
from src.gen_price_action import GenPriceAction
from src.utils import utils


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

    # Add '--strategy' flag
    parser.add_argument(
        "--strategy",
        type=str,
        default="Senti",
        help="Name of entry signal strategy (Default: 'Senti')",
    )

    # Add '--trade' flag
    parser.add_argument(
        "--trade",
        type=str,
        default="multiple_entry",
        help="Name of  (Default: 'multiple_entry')",
    )

    return parser.parse_args()


def run_strategies(date: str) -> None:
    """Run different combinations of HuggingFace FinBERT sentiment rater,
    cointegration/correlation analysis and time periods for selected 'date'.
    """

    # Get FinBERT models, cointegration/correlation functions and time periods
    hf_models = get_args(HF_MODEL)
    coint_corr_fns = get_args(COINT_CORR_FN)
    periods = (1, 3, 5)

    for hf_model, coint_corr_fn, period in product(hf_models, coint_corr_fns, periods):
        # Generate price action of top 10 cointegrated/correlated stocks
        gen_pa = GenPriceAction(date, hf_model, coint_corr_fn, period)
        gen_pa.run()

        # Compile profit and loss; and generate reports
        cal_pl = CalProfitLoss(date, hf_model, coint_corr_fn, period)
        _, _, _, _ = cal_pl.run()


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
