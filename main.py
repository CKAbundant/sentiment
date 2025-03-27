"""Main function to execute."""

import sys
from pathlib import Path

import pandas as pd

# Add repo directory to sys.path if not exist
repo_dir = Path(__file__).parent.as_posix()
if repo_dir not in sys.path:
    sys.path.append(repo_dir)

from src.cal_profit_loss import CalProfitLoss
from src.gen_data import GenData
from src.gen_price_action import GenPriceAction


def main() -> None:
    """Generate proof-of-concept by running instance of Poc class."""

    # # Generate DataFrame containing news and sentiment scores for different
    # # FinBERT variant.
    # gen_data = GenData()
    # df_senti = gen_data.run()
    # print(f"\nsentiment : \n\n{df_senti}\n")

    # Generate price action of top 10 stocks with lowest cointegration pvalue
    # with selected stocks
    gen_pa = GenPriceAction(date="2025-03-26")
    gen_pa.run()

    results_dir = f"./data/results/{gen_pa.date}"
    print(f"\nlist of csv files in '{results_dir}' :")
    for idx, fpath in enumerate(Path(results_dir).glob("*.csv")):
        print(f"{idx+1:>3}) {fpath}")

    # Compile profit and loss
    cal_pl = CalProfitLoss(date="2025-03-26")
    df_results, df_overall, df_breakdown = cal_pl.run()
    print(f"df_results : \n\n{df_results}\n")
    print(f"df_overall : \n\n{df_overall}\n")
    print(f"df_breakdown : \n\n{df_breakdown}\n")


if __name__ == "__main__":
    main()
