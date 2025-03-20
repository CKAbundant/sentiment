"""Scrape data via Playwright."""

import os
import random
import re
import sys
import time
from pathlib import Path
from pprint import pformat

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from playwright.sync_api import Locator, Page, Playwright, sync_playwright
from playwright_stealth import stealth_sync

repo_dir = Path(__file__).parents[1].as_posix()
if repo_dir not in sys.path:
    sys.path.append(repo_dir)

from src.utils import init_script_utils, scraper_utils


def human_delay(min_ms: int = 800, max_ms: int = 2500):
    """Simulate human-like delays between 'min_ms' and 'max_ms' milliseconds."""
    time.sleep(random.uniform(min_ms / 1000, max_ms / 1000))


def fill_textbox(
    page: Page, label: str, text: str, delay_range: tuple[int] = (800, 2500)
) -> None:
    """Fill either 'email' or 'password' textbox by moving mouse to hover over
    textbox before pressing text sequentially.

    Args:
        page (Page):
            Instance of Playwright Play object with loaded url.
        label (str):
            Either "email" or "password".
        text (str):
            Text to enter in textbox.
        delay_range (list[int]):
            List containing minimum and maximum duration in milliseconds
            (Default: (800, 2500)).

    Returns:
        None.
    """

    # Get relevant textbox i.e. 'email' or 'password'
    textbox = page.get_by_label(re.compile(label, re.IGNORECASE))
    textbox.wait_for()

    # Get bounding box for selected textbox
    box = textbox.bounding_box()

    # Move mouse to area within textbox (random location around center of box)
    x_coord = box["x"] + box["width"] * random.uniform(0.33, 0.66)
    y_coord = box["y"] + box["height"] * random.uniform(0.33, 0.66)
    page.mouse.move(x_coord, y_coord)

    # Pause for a while before clicking on textbox
    human_delay(700, 900)
    page.mouse.click()

    # Fill in the textbox
    textbox.press_sequentially(text, delay=random.uniform(100, 300))
    human_delay(*delay_range)


def submit_login(
    page: Page,
    button: Locator,
    request_url: str = "https://seekingalpha.com/api/v3/login_tokens",
) -> None:
    """Submit login form i.e. email and password by clicking on 'Sign in' button.

    Args:
        page (Page):
            Instance of Playwright Play object with loaded url.
        button (Locator):
            Playwright Locator object for submit button in login form.
        request_url (str):
            Actual url that is send to web server
            (Default: "https://seekingalpha.com/api/v3/login_tokens").

    Returns:
        None.
    """

    try:
        with page.expect_response(request_url) as response_info:
            button.click()

            response = response_info.value
            print(f"response.status : {response.status}")
            print(f"response.url : {response.url}")
            print(f"response.body : {response.body()}")

    except Exception as e:
        print(f"Unable to submit login : {e}")
        page.wait_for_timeout(20000)


def run_test(playwright: Playwright, url: str = "https://httpbin.org/headers") -> None:
    """Run test on bot detection website."""

    # Generate custom header and randomly selected user agent
    user_agent = rotate_user_agent()
    headers = get_headers(user_agent)

    print(f"\nuser_agent : {user_agent}")
    print(f"headers : {pformat(headers, sort_dicts=False)}\n")

    browser = playwright.chromium.launch(
        headless=False,
        args=["--disable-blink-features=AutomationControlled", "--start-maximized"],
    )

    # Create new context instead of using browser directly
    context = browser.new_context(
        user_agent=user_agent,
        locale="en-US",
        color_scheme=random.choice(["light", "dark"]),
        viewport={"width": 1920, "height": 1080},
        extra_http_headers=headers,
    )
    page = context.new_page()

    # Override browser properties to mimic read user
    # set_webdriver_undefined(page)
    init_script_utils.remove_webdriver(page)
    init_script_utils.set_fake_plugins(page)
    init_script_utils.set_languages(page)
    init_script_utils.set_window_navigator_chrome(page, user_agent)

    # Launch bot detection website
    page.goto(url)
    if url == "https://httpbin.org/headers":
        print(page.content())
    else:
        human_delay(2000, 5000)
        page.screenshot(path="stealth_test.png", full_page=True)

    browser.close()


def run(
    playwright: Playwright,
    # url: str = "https://www.bloomberg.com/latest?utm_source=homepage&utm_medium=web&utm_campaign=latest",
) -> None:
    login_url = "https://seekingalpha.com/alpha-picks/subscribe"
    cur_url = "https://seekingalpha.com/alpha-picks/picks/current"
    xpath = (
        "//div[@data-test-id='modal-content']//button[@data-test-id='sign-in-button']"
    )

    # Playwright to launch google chrome; load url and persist webpage for 20 seconds
    browser = playwright.chromium.launch(headless=False)
    page = browser.new_page()
    page.set_extra_http_headers(get_headers())

    # Go to subscribe page instead of 'cur_url' directly
    page.goto(login_url)
    page.wait_for_url(login_url)

    # Access login window by clicking on 'LOG IN' button
    page.locator("button:has-text('LOG IN')").click()

    # Wait for login form to popup i.e. 'Sign in' button to be visible
    signin_button = page.locator(f"xpath={xpath}")
    signin_button.wait_for()

    # Fill in email and password; and click "Sign in"
    fill_textbox(page, "email", os.getenv("ALPHA_EMAIL"))
    fill_textbox(page, "password", os.getenv("ALPHA_PASSWORD"))
    submit_login(page, signin_button)

    # Check if the page has redirected to the expected URL
    if page.url == cur_url:
        print("Login successful and redirected")
    else:
        print(f"Login did not redirect as expected. Current URL: {page.url}")

    html_content = page.content()
    page.wait_for_timeout(20000)
    browser.close()

    return html_content


def rotate_user_agent() -> str:
    """Rotate user agent (limited to Linux and Chromium) to avoid bot detection"""

    user_agents = [
        # "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
        # "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.6422.112 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36",
    ]

    return random.choice(user_agents)


def get_chrome_version(user_agent: str) -> str:
    """Get Chrome version from user agent."""

    # Chrome version is after 'Chrome/' string e.g. 'Chrome/134.0.0.0'
    results = re.findall(r"(?<=Chrome/)\d+", user_agent)

    # There should only 1 item in 'results'
    return results[0]


def get_platform(user_agent: str) -> str:
    """Get operating system from user-agent for desktop device only i.e. 'Chrome OS',
    'Chromium OS', 'Linux', 'macOS', and 'Windows'."""

    mapping = {
        "Linux": "Linux",
        "Macintosh": "macOS",
        "Windows": "Windows",
        "CrOS x86_64": "Chrome OS",
    }

    # Return "Unknown" if no keys are found in 'user_agent'
    if all(keyword not in user_agent for keyword in mapping.keys()):
        raise ValueError(
            "No suitable platform i.e. Linux, Macintosh, Windows or Chrome OS found."
        )

    for keyword, platform in mapping.items():
        if keyword in user_agent:
            # break loop once platform is found
            return platform


def get_headers(
    user_agent: str = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36",
) -> dict[str, str]:
    """Generate header for Playwright based on given user agent."""

    version = get_chrome_version(user_agent)
    platform = get_platform(user_agent)

    return {
        "Accept-Language": "en-US,en;q=0.9",
        # "Referer": "https://seekingalpha.com/alpha-picks/subscribe",
        "Sec-CH-UA": f'"Chromium";v="{version}", "Not:A-Brand";v="24", "Google Chrome";v="{version}"',
        "Sec-CH-UA-Platform": f'"{platform}"',
        "Sec-CH-UA-Mobile": "?0",
    }


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

    url = "https://bot.sannysoft.com/"
    # url = "https://abrahamjuliot.github.io/creepjs/"
    # url = "https://pixelscan.net/"

    with sync_playwright() as playwright:
        run_test(playwright, url=url)

    # filtered_content = filter_html(html_content)
    # print(filtered_content)
