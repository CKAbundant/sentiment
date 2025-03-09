"""Functions to perform web-scrapping via Playwright."""

from pathlib import Path

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright


def extract_html(
    url: str = "https://finance.yahoo.com/quote/AAPL/news/", max_scrolls: int = 5
) -> str:
    """Use Playwright to launch Yahoo Finance website and scroll down 5 times to
    load all required div elements."""

    # Load HTML code if 'page_content.html' is present
    if Path("page_content.html").is_file():
        with open("page_content.html", "r") as file:
            print(f"page_content.html exist!")
            html_content = file.read()

    else:
        # Use Playwright to scroll and save html content
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url)

            previous_height = page.evaluate("document.body.scrollheight")
            print(f"Initial previous_height : {previous_height}")

            scroll_count = 0

            while scroll_count < max_scrolls:
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                page.wait_for_timeout(2000)
                new_height = page.evaluate("document.body.scrollHeight")
                print(f"new_height : {new_height}")

                if new_height == previous_height:
                    break

                previous_height = new_height
                scroll_count += 1

            html_content = page.content()
            with open("page_content.html", "w") as file:
                file.write(html_content)

            print(type(html_content))

            browser.close()

    return html_content


def filter_html(html_content: str) -> BeautifulSoup:
    """Filter out 'script' and 'style' tags from html content."""

    soup = BeautifulSoup(html_content, "html.parser")
    for unwanted in soup(["script", "style", "link", "meta", "iframe"]):
        # Remove tag
        unwanted.decompose()

    # Saved filtered content
    with open("filtered_content.html", "w") as file:
        file.write(soup.prettify())

    return soup


def extract_div(soup: BeautifulSoup) -> list[str]:
    """Extract required 'div' elements from filtered HTML content."""

    div_elements = soup.find_all("div", class_="content")

    extracted_divs = [div.prettify() for div in div_elements]

    print(f"len(extracted_divs) : {len(extracted_divs)}")

    for idx, div in enumerate(extracted_divs):
        print(f"\nidx : {idx}\n")
        print(div)

    return extracted_divs


if __name__ == "__main__":
    html_content = extract_html()
    filtered_content = filter_html(html_content)
    extracted_divs = extract_div(filtered_content)
