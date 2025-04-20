"""Class to plot graphs relating to news web-scrapped from yfinance:

1. Publication Date Distribution by ticker
2. Publication time Distribution by ticker
3. Publisher Distribution
4. Word Distribution by news title, news content and combined
5. Punctuation Distribution by news title, news content and combined
6. Special Characters Distribution by news title, news content and combined
"""

from collections import Counter, defaultdict
from datetime import timedelta
from typing import get_args

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from omegaconf import DictConfig

from config.variables import Component
from src.utils import plot_utils, utils

# Set default fontsize for labels and ticks
plt.rcParams["font.size"] = 12
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14


class PlotNews:
    """Plot and save graphs relating to news web-scrapped from yfinance.

    Usage:
        >>> plot_news = PlotNews()
        >>> plot_news.run() # Graphs are saved in models subfolder e.g. 'ziweichen_coint_3'

    Args:
        path (DictConfig):
            OmegaConf DictConfig containing required file and directory paths.
        date (str):
            If provided, date when news are scraped.

    Attributes:
        date (str):
            If provided, date when news are scraped.
        news_path (str):
            Relative path to DataFrame containing news scraped on specific date.
        graph_date_dir (str):
            Relative path to folder to save generated graphs.
        pickle_path (str):
            Relative path to save dictionary containing counters object.
    """

    def __init__(
        self,
        path: DictConfig,
        date: str | None = None,
    ) -> None:
        self.date = date or utils.get_current_dt(fmt="%Y-%m-%d")
        self.news_path = f"{path.data_dir}/{self.date}/news.csv"
        self.graph_date_dir = f"{path.graph_dir}/{self.date}"
        self.pickle_path = f"{self.graph_date_dir}/counters.pkl"

    def run(self) -> None:
        """Generate and save graphs for news analysis:

        1. Publication Date Distribution by ticker
        2. Publication time Distribution by ticker
        3. Publisher Distribution
        4. Word Distribution by news title, news content and combined
        5. Punctuation Distribution by news title, news content and combined
        6. Special Characters Distribution by news title, news content and combined
        """

        # Load 'news.csv' as DataFrame
        df_news = pd.read_csv(self.news_path)

        self.plot_date(df_news, value_name="date")
        self.plot_time(
            df_news,
            value_name="time",
            value_vars=["min_minutes", "max_minutes"],
        )
        self.plot_publisher(df_news)
        self.plot_word_count(df_news)
        self.plot_top_n(df_news, top_n=30)

    def plot_date(
        self,
        df_news: pd.DataFrame,
        value_name: str,
        id_vars: list[str] = ["ticker"],
        value_vars: list[str] = ["min", "max"],
    ) -> None:
        """Plot Date distribution (i.e earliest published to latest published date)
        for each ticker.

        Args:
            df_news (pd.DataFrame):
                DataFrame containing web-scraped news.
            value_name (str):
                Set name for 'value' column.
            id_vars (list[str]):
                List of col to be used as id (Default: ["ticker"]).
            value_vars (list[str]):
                List of col to be used as values (Default: ["min", "max"]).

        Returns:
            None.
        """

        # Generate DataFrame containing min and max datetime for each ticker
        df_dt = plot_utils.gen_date_dist_df(df_news)

        # Convert 'df_dt' to long form
        df_long = pd.melt(
            df_dt,
            id_vars=id_vars,
            value_vars=value_vars,
            value_name=value_name,
        )

        # Create custom palette with same color
        n_tickers = df_news["ticker"].nunique()
        palette = sns.color_palette(["green"] * n_tickers)

        fig, ax = plt.subplots(figsize=(15, 8))
        sns.lineplot(
            data=df_long,
            x=value_name,
            y=id_vars[0],
            hue=id_vars[0],
            palette=palette,
            marker="o",
            legend=False,
            ax=ax,
        )

        ax.set_xlabel(value_name)
        ax.set_ylabel(id_vars[0])
        ax.set_title("News Publication Period Distribution")
        ax.grid(True)

        # Annotate duration for each line plot
        for index, _, min_date, max_date in df_dt.itertuples(index=True, name=None):
            duration = f"{(max_date - min_date).days} days"

            # Offset text by 2 days for proper formatting
            ax.text(min_date + timedelta(days=2), index, duration)

        # Create folder if not exist
        utils.create_folder(self.graph_date_dir)

        # Ensure date labels do not overlap
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.savefig(f"{self.graph_date_dir}/news_date_period.png")

    def plot_time(
        self,
        df_news: pd.DataFrame,
        value_name: str,
        id_vars: list[str] = ["ticker"],
        value_vars: list[str] = ["min", "max"],
    ) -> None:
        """Plot time distribution (i.e earliest published to latest published time)
        for each ticker.

        Args:
            df_news (pd.DataFrame):
                DataFrame containing web-scraped news.
            value_name (str):
                Set name for 'value' column.
            id_vars (list[str]):
                List of col to be used as id (Default: ["ticker"]).
            value_vars (list[str]):
                List of col to be used as values (Default: ["min", "max"]).

        Returns:
            None.
        """

        # Generate DataFrame containing min and max time for each ticker
        df_time = plot_utils.gen_time_dist_df(df_news)

        # Convert 'df_dt' to long form
        df_long = pd.melt(
            df_time,
            id_vars=id_vars,
            value_vars=value_vars,
            value_name=value_name,
        )

        # Create custom palette with same color
        n_tickers = df_news["ticker"].nunique()
        palette = sns.color_palette(["green"] * n_tickers)

        fig, ax = plt.subplots(figsize=(15, 8))
        sns.lineplot(
            data=df_long,
            x=value_name,
            y=id_vars[0],
            hue=id_vars[0],
            palette=palette,
            marker="o",
            legend=False,
            ax=ax,
        )

        ax.set_xlabel(value_name)
        ax.set_ylabel(id_vars[0])
        ax.set_title("News Publication Time Period Distribution")
        ax.set_xlim(0, 1550)
        ax.grid(True)

        # Annotate duration for each line plot
        for (
            index,
            _,
            min_time,
            max_time,
            min_minutes,
            max_minutes,
        ) in df_time.itertuples(index=True, name=None):
            msg = f"({min_time.strftime("%H%M hrs")}, {max_time.strftime("%H%M hrs")})"
            ax.text(min_minutes, index, msg)

        # Create folder if not exist
        utils.create_folder(self.graph_date_dir)

        # Ensure date labels do not overlap
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.savefig(f"{self.graph_date_dir}/news_time_period.png")

    def plot_publisher(self, df_news: pd.DataFrame, top_n: int = 20) -> None:
        """Plot histogram of publisher distribution."""

        df = df_news.copy()

        # Get top N publisher with highest count
        top_n_publisher = plot_utils.get_top_n(df, "publisher", top_n)

        fig, ax = plt.subplots(figsize=(15, 8))
        sns.barplot(x=top_n_publisher.index, y=top_n_publisher, ax=ax)

        ax.set_title("Publisher Distribution")
        ax.set_xlabel("Publisher")
        ax.set_ylabel("Counts")

        # Create folder if not exist
        utils.create_folder(self.graph_date_dir)

        fig.autofmt_xdate()
        plt.tight_layout()
        plt.savefig(f"{self.graph_date_dir}/publisher.png")

    def plot_word_count(self, df_news: pd.DataFrame) -> None:
        """Plot word count for news title, news content and combined news title
        and content."""

        df_word_count = plot_utils.append_word_count(df_news)

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 8))

        # Iterate through each news type to plot histogram
        for ax, col in zip(axes, ["title_wc", "content_wc", "combined_wc"]):
            sns.histplot(data=df_word_count, x=col, ax=ax, kde=True)
            news_type = col.split("_")[0].title()

            ax.set_title(f"Frequency of Word Counts in News {news_type}")
            ax.set_xlabel(f"Words Count in {news_type}")
            ax.set_ylabel("Frequency")

        # Create folder if not exist
        utils.create_folder(self.graph_date_dir)

        plt.tight_layout()
        plt.savefig(f"{self.graph_date_dir}/word_count.png")

    def plot_top_n(self, df_news: pd.DataFrame, top_n: int = 50) -> None:
        """Plot top N most frequent words, punctuations and special characters
        for news title, news content and combined news title and content.

        Args:
            df_news (pd.DataFrame): DataFrame containing web-scraped news.
            top_n (int): Top N number of items with highest count.

        Returns:
            None.
        """

        # Generate dictionary mapping top N words, punctuations and special words to
        # news title, news content; and combined news title and content
        df_dict = self.gen_topn_dict(df_news, top_n)

        for component, news_dict in df_dict.items():
            for news_type, df_count in news_dict.items():
                fig, ax = plt.subplots(figsize=(20, 4))
                sns.barplot(x=df_count.index, y=df_count["count"], ax=ax)

                ax.set_title(f"Top {top_n} '{component}' for News {news_type.title()}")
                ax.set_xlabel(component)
                ax.set_ylabel("Counts")
                ax.tick_params(axis="x")
                ax.tick_params(axis="y")

                if component == "word":
                    ax.tick_params(axis="x", labelrotation=30)

                plt.tight_layout()
                plt.savefig(
                    f"{self.graph_date_dir}/top_{top_n}_{component}_{news_type}.png"
                )

    def gen_counters(self, df_news: pd.DataFrame) -> dict[str, defaultdict[Counter]]:
        """Generate Counter objects for words, punctuation and special characters
        in news title and content.

        Args:
            df_news (pd.DataFrame): DataFrame containing news info scrapped from yfinance.
            pickle_path (str): Relative path of folder to save Counter object.

        Returns:
            (dict[str, dict[str, Counter]]):
                Dictionary mapping word, punctuation and special characters.
        """

        counters = {
            component: defaultdict(Counter) for component in get_args(Component)
        }

        fn_mapping = {
            "word": plot_utils.gen_word_counter,
            "punct": plot_utils.gen_punct_counter,
            "special": plot_utils.gen_special_counter,
        }

        df = df_news.loc[:, ["title", "content"]].copy()

        # Iterate through word, followed by punctuation and special characters
        for component in counters.keys():
            for title, content in df.itertuples(index=False, name=None):
                # Iterate through each record to perform count on title and content
                counters[component]["title"].update(fn_mapping[component](title))
                counters[component]["content"].update(fn_mapping[component](content))

            # Combine title and content counter to form overall counter
            counters[component]["combined"].update(counters[component]["title"])
            counters[component]["combined"].update(counters[component]["content"])

        utils.save_pickle(counters, self.pickle_path)

        return counters

    def gen_topn_dict(
        self, df_news: pd.DataFrame, top_n: int = 50
    ) -> dict[str, dict[str, pd.DataFrame]]:
        """Generate dictionary mapping top N words, punctuations and special words to
        news title, news content; and combined news title and content."""

        # Generate dictionary containing counters for words, punctuations, and
        # special characters for news title, news content and combined news
        # title and content
        counters_dict = self.gen_counters(df_news)
        df_dict = {}

        for component, news_dict in counters_dict.items():
            df_count_dict = {}

            for news_type, counter in news_dict.items():
                # Convert counter to DataFrame
                df_count = pd.DataFrame.from_dict(
                    counter, orient="index", columns=["count"]
                )
                df_count_dict[news_type] = df_count.sort_values(
                    by="count", ascending=False
                ).head(top_n)

            df_dict[component] = df_count_dict

        return df_dict
