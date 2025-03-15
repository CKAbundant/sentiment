"""Scrape data via Playwright."""

import os
import re
import sys
from pathlib import Path

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from playwright.sync_api import Playwright, sync_playwright

repo_dir = Path(__file__).parents[1].as_posix()
if repo_dir not in sys.path:
    sys.path.append(repo_dir)

from src.utils import utils


def run(
    playwright: Playwright,
    url: str = "https://www.bloomberg.com/latest?utm_source=homepage&utm_medium=web&utm_campaign=latest",
) -> None:
    # Playwright to launch google chrome; load url and persist webpage for 20 seconds
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto(url)

    # # Attempt to login by clicking on 'Sign In' button
    # page.get_by_role("button", name=re.compile("most popular", re.IGNORECASE)).click()

    html_content = page.content()
    browser.close()

    return html_content


def filter_html(html_content: str) -> BeautifulSoup:
    """Filter out unwanted tags i.e. script, style, link, meta and iframe
    from html content."""

    soup = BeautifulSoup(html_content, "html.parser")
    for unwanted in soup(["script", "style", "link", "meta", "iframe"]):
        # Remove tag
        unwanted.decompose()

    return soup


if __name__ == "__main__":
    # Load environment variables from '.env' file
    load_dotenv()

    with sync_playwright() as playwright:
        html_content = run(playwright)

    filtered_content = filter_html(html_content)
    print(filtered_content)
