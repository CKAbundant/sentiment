"""Helper functions used in 'main.py'"""

from argparse import ArgumentParser, Namespace
from itertools import product
from typing import get_args

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
