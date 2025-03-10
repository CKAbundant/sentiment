"""Helper functions for usuage with"""

from datetime import datetime, timedelta

import pytimeparse
from scrapling.defaults import Fetcher


def extract_all_text(url: str) -> str:
    """Extract all text content from all HTML tags except script and style
    from given url."""

    # Create an Adaptor instance via GET request to url
    page = Fetcher.get(url)

    # Get all text content from all HTML tags except 'script' and 'style'
    all_text = page.get_all_text(ignore_tags=["script", "style"])

    return str(all_text)


def extract_news_briefs(url: str) -> list[dict[str, str]]:
    """Extract news brief including news heading and publishing info for given url."""

    # Create an Adaptor instance via GET request to url
    page = Fetcher.get(url)

    # Get list of CSS elements containing news brief
    results = page.find_all("div", class_="content yf-82qtw3")

    news_info = []
    for result in results:
        # TextHandlers with 2 elements i.e. news headings and news brief
        news = result.css(".clamp::text")

        # TextHandlers with 2 elements i.e. publisher and published period
        publisher_info = result.css(".publishing::text")
        period_str = str(publisher_info[1].clean())

        # Remove extra whitespace and save publisher, period (how long ago) and news
        # content as dictionary
        news_info.append(
            {
                "publisher": str(publisher_info[0].clean()),
                "period": convert_to_timedelta(period_str),
                "title": news[0].clean(),
                "content": news[1].clean(),
            }
        )

    return news_info


def convert_to_timedelta(period_str: str) -> timedelta | None:
    """Convert time period string into timedelta object."""

    if period_str == "yesterday":
        period_str = "1 day"
    else:
        # Remove 'ago' in 'period_str'
        period_str = period_str.replace("ago", "").strip()

    # Get time period in seconds
    period_in_seconds = pytimeparse.parse(period_str)

    if period_in_seconds is None:
        print(f"Invalid time period string : {period_str}")
        return

    return timedelta(seconds=period_in_seconds)


def cal_pub_date(period: str, scrape_dt: str) -> datetime:
    """Compute published date based on 'period' and 'current_dt'.

    Args:
        period (str):
            Amount of time lapsed since news is published.
        scrape_dt (str):
            Date and time ('YYYYMMDD_HHMM' format) when Yahoo Finance
            is webscrapped.

    Returns:
        (datetime): Published date in "YYYY-MM-DD' format.
    """

    # Convert 'scrape_dt' to datetime object
    scrape_dt = datetime.strptime(scrape_dt, "%Y%m%d_%H%M")

    # Convert 'period' string to timedelta object
    delta = convert_to_timedelta(period)

    # Published date = scrape_dt - delta
    pub_dt = scrape_dt - delta

    # Omit time
    return pub_dt
