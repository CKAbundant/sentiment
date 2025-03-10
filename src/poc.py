"""Class to generate Proof-of-Concept for sentiment-based strategy:

1. Get 'publisher', 'period' (time lapsed since news published), 'title',
and 'content' for AAPL, NVDA and PG for past 10 days.
2. Get overall sentiment for each stock for each day via FinBERT.
3. Determine 'co_integ' stock (i.e. stock that has highest cointegration
value with selected stock)
4. Buy 'co_integ' stock if overall sentiment (for day) is positive;
Sell 'co_integ' stock if overall sentiment (for day) is negative

Note:
1. Overall sentiment for the day is taken since time of news release is not available.
There is possibility similiar news are released concurrently by different publisher.
Hence we are not able to estimate the stock price.
2. We selected 'AAPL', 'NVDA' and 'PG' for a start. Ideally we should be getting 1
stock from each of GICS Sector.
3. We assume news sentiment may vary within 10 days for the selected stocks.
4. We use Playwright to perform webscrolling on Yahoo Finance to capture the overall
HTML content first. Then we use BeautifulSoup to extract relevant news and publisher
info. Reason being Scrapling is not able to perform scrolling.
"""

import re
import sys
import time
from datetime import datetime
from functools import partial
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

from src.sentiment import SentimentRater
from src.utils import utils, yahoo_utils

# # Add repo directory to sys.path if not exist
# repo_dir = Path(__file__).parent.as_posix()
# if repo_dir not in sys.path:
#     sys.path.append(repo_dir)


class Poc:
    """Generate Proof-of-Concept for sentiment-based strategy:

    - Extract news for past 10 days for 'AAPL', 'NVDA', 'PG'.
    - Perform sentiment analysis via FinBert.
    - Get stock with highest cointegration values with selected stock.
    - Generate DataFrame indicating 'Buy', 'Sell', 'No Action' for past 10 days

    Usage:
        >>> poc = Poc() # Use default values
        >>> result_df = poc.run()

    Args:
        base_url (str):
            URL to Yahoo Finance to specifics stock news by replacing 'ticker' with
            stock symbol (Default: "https://finance.yahoo.com/quote/ticker/news").
        stock_list (list[str]):
            List of stocks for POC studies (Default: ["AAPL", "NVDA", "PG"]).
        max_scroll (int):
            Maximum number of scrolls to extract news article from Yahoo Finance.


    Attributes:
        base_url (str):
            URL to Yahoo Finance to specifics stock news by replacing 'ticker' with
            stock symbol (Default: "https://finance.yahoo.com/quote/ticker/news).
        stock_list (list[str]):
            List of stocks for POC studies (Default: ["AAPL", "NVDA", "PG"]).
        max_scroll (int):
            Maximum number of scrolls to extract news article from Yahoo Finance.
    """

    def __init__(
        self,
        base_url: str = "https://finance.yahoo.com/quote/ticker/news",
        stock_list: list[str] = ["AAPL", "NVDA", "PG"],
        max_scroll: int = 8,
    ):
        self.base_url = base_url
        self.stock_list = stock_list
        self.max_scroll = max_scroll

    def run(self) -> pd.DataFrame:
        """Extract news from Yahoo Finance; and generate sentiment score."""

        rater = SentimentRater()
        df_list = []

        for ticker in self.stock_list:
            # Get current datetime as "YYYYMM-DD_HHMM" string
            scrape_dt = utils.get_current_dt()

            # Generate DataFrame containing news info for each ticker
            html_content = self.extract_html(ticker, scrape_dt)
            filtered_content = self.filter_html(html_content)
            df_news = self.extract_news_info(filtered_content)

            # Append 'ticker', 'pub_date', and 'score' column to DataFrame
            df_news = self.append_ticker(df_news, ticker)
            df_news = self.append_pub_date(df_news, scrape_dt)
            df_news = self.append_finbert_score(df_news, rater)
            df_list.append(df_news)

            # Wait 2 seconds for browser to close completely
            time.sleep(2)

        # Combine list of DataFrames row-wise
        return pd.concat(df_list, axis=0).reset_index(drop=True)

    def extract_html(self, ticker: str, current_dt: str) -> str:
        """Use Playwright to launch Yahoo Finance news website for selected ticker
        and scroll down 5 times to load all required div elements.
        """

        # Replace 'ticker' with actual stock symbol
        url = self.base_url.replace("ticker", ticker)

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
            title = div.find("h3", class_="clamp").text
            content = div.find("p", class_="clamp").text
            publisher_info = div.find("div", class_="publishing").text
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

    def get_publisher_info(self, pub_str: str) -> list[str]:
        """Get publisher and period lapsed since news published from
        text extracted via BeautifulSoup."""

        # Publisher is separated from period by "•" i.e. bullet point represented
        # by unicode \u2022
        return [item.strip() for item in re.split(r"\u2022", pub_str)]

    def append_ticker(self, df_news: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Append 'ticker' column to DataFrame."""

        df = df_news.copy()
        df.insert(0, "ticker", [ticker] * len(df_news))

        return df

    def append_pub_date(self, df_news: pd.DataFrame, scrape_dt: str) -> pd.DataFrame:
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

    def append_finbert_score(
        self, df_news: pd.DataFrame, rater: SentimentRater
    ) -> pd.DataFrame:
        """Append sentiment score (1 to 5) based on news title and content using
        FinBERT."""

        df = df_news.copy()

        # Combine 'title' and 'content' column with 2 cartridge returns
        df["news"] = df["title"] + "\n\n" + df["content"]

        # Rate sentiment of combined news
        df["sentiment"] = df["news"].map(rater.classify_sentiment)

        # Drop 'news' column
        df = df.drop(columns=["news"])

        return df
