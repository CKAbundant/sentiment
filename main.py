"""Main function to execute."""

import sys
from pathlib import Path

import pandas as pd

# Add repo directory to sys.path if not exist
repo_dir = Path(__file__).parent.as_posix()
if repo_dir not in sys.path:
    sys.path.append(repo_dir)

from src.gen_data import GenData
from src.sentiment import SentimentRater


def main() -> None:
    """Generate proof-of-concept by running instance of Poc class."""

    # Generate DataFrame containing news and sentiment scores for different
    # FinBERT variant.
    gen_data = GenData()
    result_df = gen_data.run()
    print(result_df)


if __name__ == "__main__":
    main()
