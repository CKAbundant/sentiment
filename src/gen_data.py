"""Class to generate Proof-of-Concept for sentiment-based strategy:

1. Get 'publisher', 'period' (time lapsed since news published), 'title',
and 'content' for "AAPL", "NVDA", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "JPM",
"JNJ", "V", "XOM", "UNH", "WMT", "PG", "HD", "NFLX", "CRM", "BAC", and "BA" for
past 10 days.
2. Get overall sentiment for each stock for each day via FinBERT.
3. Determine 'co_integ' stock (i.e. stock that has highest cointegration
value with selected stock)
4. Buy 'co_integ' stock if overall sentiment (for day) is positive;
Sell 'co_integ' stock if overall sentiment (for day) is negative

Note:
1. Overall sentiment for the day is taken since time of news release is not available.
There is possibility similiar news are released concurrently by different publisher.
Hence we are not able to estimate the stock price.
2. Stocks selected are supposed to have news published on a frequent basis.
3. We assume news sentiment may vary within 10 days for the selected stocks.
4. We use Playwright to perform webscrolling on Yahoo Finance to capture the overall
HTML content first. Then we use BeautifulSoup to extract relevant news and publisher
info. Reason being Scrapling is not able to perform scrolling.
"""

import re
import time
from functools import partial
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from tqdm import tqdm

from src.sentiment import SentimentRater
from src.utils import utils, yahoo_utils


class GenData:
    """Generate DataFrame containing news scrapped from Yahoo Finance:

    - Extract news for past 10 days for 'AAPL', 'NVDA', 'PG'.
    - Perform sentiment analysis via FinBert.
    - Save DataFrame as 'sentiment.csv'

    Usage:
        >>> gen_data = GenData() # Use default values
        >>> result_df = poc.run()

    Args:
        date (str):
            If provided, date when news are scraped.
        base_url (str):
            URL to Yahoo Finance to specifics stock news by replacing 'ticker' with
            stock symbol (Default: "https://finance.yahoo.com/quote/ticker/news").
        stock_list (list[str]):
            List of stocks for POC studies (Default: ["AAPL", "NVDA", "MSFT", "AMZN",
            "GOOGL", "META", "TSLA", "JPM", "JNJ", "V", "XOM", "UNH", "WMT", "PG",
            "HD", "NFLX", "CRM", "BAC", "BA"]).
        max_scrolls (int):
            Maximum number of scrolls to extract news article from Yahoo Finance
            (Default: 20).
        model_list (list[str]):
            List of Hugging Face FinBERT models (Default: ["ProsusAI/finbert",
            "yiyanghkust/finbert-tone", "ZiweiChen/FinBERT-FOMC",
            "AventIQ-AI/finbert-sentiment-analysis"]).
        data_dir (str):
            Relative path of folder containing all sentiment, price actions and
            trade results (Default: "./data").

    Attributes:
        date (str):
            If provided, date when news are scraped.
        base_url (str):
            URL to Yahoo Finance to specifics stock news by replacing 'ticker' with
            stock symbol (Default: "https://finance.yahoo.com/quote/ticker/news).
        stock_list (list[str]):
            List of stocks for POC studies (Default: ["AAPL", "NVDA", "MSFT", "AMZN",
            "GOOGL", "META", "TSLA", "JPM", "JNJ", "V", "XOM", "UNH", "WMT", "PG",
            "HD", "NFLX", "CRM", "BAC", "BA"]).
        max_scrolls (int):
            Maximum number of scrolls to extract news article from Yahoo Finance (Default: 20).
        model_list (list[str]):
            List of Hugging Face FinBERT models (Default: ["ProsusAI/finbert",
            "yiyanghkust/finbert-tone", "ZiweiChen/FinBERT-FOMC",
            "AventIQ-AI/finbert-sentiment-analysis"]).
        date_dir (str):
            Relative path of folder containing subfolders for different strategies.
        news_path (str):
            Relative path to downloaded stock-related news.
        senti_path (str):
            Relative path to download stock-related news that are sentiment-rated.
    """

    def __init__(
        self,
        date: str | None = None,
        base_url: str = "https://finance.yahoo.com/quote/{ticker}/news",
        stock_list: list[str] = [
            "AAPL",
            "NVDA",
            "MSFT",
            "AMZN",
            "GOOGL",
            "META",
            "TSLA",
            "JPM",
            "JNJ",
            "V",
            "XOM",
            "UNH",
            "WMT",
            "PG",
            "HD",
            "NFLX",
            "CRM",
            "BAC",
            "BA",
        ],
        max_scrolls: int = 15,
        model_list: str = [
            "ProsusAI/finbert",  # Financial PhraseBank
            "yiyanghkust/finbert-tone",  # Analyst reports
            "ZiweiChen/FinBERT-FOMC",  # FOMC reports
            "AventIQ-AI/finbert-sentiment-analysis",  # General English quotes
        ],
        data_dir: str = "./data",
    ):
        self.date = date or utils.get_current_dt(fmt="%Y-%m-%d")
        self.base_url = base_url
        self.stock_list = stock_list
        self.max_scrolls = max_scrolls
        self.model_list = model_list
        self.date_dir = f"{data_dir}/{self.date}"
        self.news_path = f"{self.date_dir}/news.csv"
        self.senti_path = f"{self.date_dir}/sentiment.csv"

    def run(self) -> pd.DataFrame:
        """Generate DataFrame containing news extracted from Yahoo Finance; and
        generate sentiment score."""

        df_list = []

        # Create 'results' folder if not exist
        if not Path(self.date_dir).is_dir():
            utils.create_folder(self.date_dir)

        for ticker in self.stock_list:
            print(f"\nticker : {ticker}")
            # Get current datetime as "YYYYMM-DD_HHMM" string
            scrape_dt = utils.get_current_dt()

            # Generate DataFrame containing news info for each ticker
            html_content = self.extract_html(ticker, scrape_dt)
            filtered_content = self.filter_html(html_content)
            df_news = self.extract_news_info(filtered_content)

            # Append 'ticker', 'pub_date', and 'score' column to DataFrame
            df_news = GenData.append_ticker(df_news, ticker)
            df_news = GenData.append_pub_date(df_news, scrape_dt)
            df_list.append(df_news)

            # Wait 2 seconds for browser to close completely
            time.sleep(2)

        # Combine list of DataFrames row-wise and Append sentiment scores
        # for different rater. Save DataFrame as csv file
        df_combine = pd.concat(df_list, axis=0).reset_index(drop=True)
        df_combine.to_csv(self.news_path, index=False)

        # Append sentiment scores for various FinBERT models
        df_combine = self.append_sentiment_scores(df_combine)
        df_combine.to_csv(self.senti_path, index=False)

        return df_combine

    def extract_html(self, ticker: str, current_dt: str) -> str:
        """Use Playwright to launch Yahoo Finance news website for selected ticker
        and scroll down 5 times to load all required div elements.
        """

        # Replace 'ticker' with actual stock symbol
        url = self.base_url.format(ticker=ticker)

        with sync_playwright() as p:
            # Playwright to launch google chrome and load url
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="domcontentloaded")

            previous_height = 0
            scroll_count = 0

            while scroll_count < self.max_scrolls:
                # Scroll down and wait for 2 seconds for webpage load
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                page.wait_for_timeout(2000)

                # Get new height after scrolling down
                new_height = page.evaluate("document.body.scrollHeight")
                print(f"new_height ({scroll_count+1} scroll): {new_height}")

                # Break while loop if new height doesn't change because no more
                # new items to load i.e. new height = previous height
                if new_height == previous_height:
                    break

                previous_height = new_height
                scroll_count += 1

            # Extract html_content as string
            html_content = page.content()

            # Save HTML content and close browser
            file_name = f"{ticker}_{current_dt}.html"
            utils.save_html(html_content, file_name)
            browser.close()

        return html_content

    def filter_html(self, html_content: str) -> BeautifulSoup:
        """Filter out unwanted tags i.e. script, style, link, meta and iframe
        from html content."""

        soup = BeautifulSoup(html_content, "html.parser")
        for unwanted in soup(["script", "style", "link", "meta", "iframe"]):
            # Remove tag
            unwanted.decompose()

        return soup

    def extract_news_info(self, html_content: BeautifulSoup) -> pd.DataFrame:
        """Generate DataFrame by extracting publisher info and news content from
        filtered HTML content."""

        div_elements = html_content.find_all("div", class_="content")

        news_info = []
        for div in div_elements:
            title = self.get_text_info(div, "h3", "clamp")
            content = self.get_text_info(div, "p", "clamp")
            publisher_info = self.get_text_info(div, "div", "publishing")
            publisher, period = self.get_publisher_info(publisher_info)

            news_info.append(
                {
                    "publisher": publisher,
                    "period": period,
                    "title": title,
                    "content": content,
                }
            )

        return pd.DataFrame(news_info)

    def get_text_info(self, div_element: str, tag: str, class_name: str) -> str:
        """Get text infomation from respective HTML tags and class name."""

        try:
            return div_element.find(tag, class_=class_name).text
        except:
            return "Not available"

    def get_publisher_info(self, pub_str: str) -> list[str]:
        """Get publisher and period lapsed since news published from
        text extracted via BeautifulSoup."""

        if pub_str is None:
            return ["Not available", "Not available"]

        if not re.search(r"\u2022", pub_str):
            return [pub_str, "Not available"]

        # Publisher is separated from period by "•" i.e. bullet point represented
        # by unicode \u2022
        return [item.strip() for item in re.split(r"\u2022", pub_str)]

    @staticmethod
    def append_ticker(df_news: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Append 'ticker' column to DataFrame."""

        df = df_news.copy()
        df.insert(0, "ticker", [ticker] * len(df_news))

        return df

    @staticmethod
    def append_pub_date(df_news: pd.DataFrame, scrape_dt: str) -> pd.DataFrame:
        """Append published date for each news to DataFrame.

        Args:
            df_news (pd.DataFrame): DataFrame containing news title and content.
            scrape_dt (str): Date and time when Yahoo Finance webpage is scrapped.

        Returns:
            (pd.DataFrame): DataFrame with appended 'pub_date' column.
        """

        df = df_news.copy()

        # Create partial function with fixed scrape_dt
        cal_pub_date_fixed_scrape_dt = partial(
            yahoo_utils.cal_pub_date, scrape_dt=scrape_dt
        )

        # Append 'pub_date' column to DataFrame
        df.insert(0, "pub_date", df["period"].map(cal_pub_date_fixed_scrape_dt))

        return df

    @staticmethod
    def append_finbert_score(
        df_news: pd.DataFrame, rater: SentimentRater, col_name: str = "sentiment"
    ) -> pd.DataFrame:
        """Append sentiment score (1 to 5) using FinBERT based on:
        - news title & content
        - news title only
        - news content only

        Args:
            df_news (pd.DataFrame):
                DataFrame containing news info.
            rater (SentimentRater):
                Instance of SentimentRater to rate news sentiment using a variant
                of FinBERT.
            col_name (str):
                Name of column containing sentiment scores (Default: "sentiment").

        Returns:
            df (pd.DataFrame): DataFrame with appended sentiment score.
        """

        df = df_news.copy()

        # Combine title and content; and format combined text string
        df["news"] = df["title"] + "\n\n" + df["content"]
        df["news"] = df["news"].map(GenData.format_news)

        # Classify sentiment based on news title & content; news title only and
        # news content only
        df[col_name] = rater.classify_sentiment(df["news"].to_list())
        df[f"{col_name}_title"] = rater.classify_sentiment(df["title"].to_list())
        df[f"{col_name}_content"] = rater.classify_sentiment(df["content"].to_list())

        # Drop 'news' column
        df = df.drop(columns=["news"])

        return df

    @staticmethod
    def format_news(news_str: str) -> str:
        """Return 'Not available' if duplicates of 'Not available' exists; Remove 'Not available if only 1 copy of 'Not available' is found in text string."""

        count = len(re.findall(r"Not available", news_str))

        if count > 1:
            return "News is not available"

        if count == 1:
            return re.sub(r"Not available", "", news_str).strip()

        return news_str

    def append_sentiment_scores(self, df_news: pd.DataFrame) -> pd.DataFrame:
        """Append sentiment scores to DataFrame for different FinBERT variant."""

        df = df_news.copy()

        for model_name in tqdm(self.model_list):
            # Get column name for sentiment rating
            col_name = self.get_col_name(model_name)

            rater = SentimentRater(model_name=model_name)
            print(f"\nmodel_name : {model_name}\n")

            # Append sentiment score to DataFrame
            df = GenData.append_finbert_score(df, rater, col_name)

        return df

    def get_col_name(self, model_name: str) -> str:
        """Get column name from hugging face 'model_name'."""

        # Get first part of 'model_name' separated by '/'
        first_part = model_name.split("/")[0]

        # replace '-' with "_" and lowercase name
        col_name = first_part.replace("-", "_").lower()

        return col_name
