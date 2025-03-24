"""Functions to call Playwright 'page.add_init_script'
including helper functions."""

import json
import platform
import random
import re
from pprint import pformat
from typing import Any

from playwright.sync_api import Page, Route

from src.utils import scraper_utils


def set_webdriver_undefined(page: Page) -> None:
    """Set navigator.webdriver to be undefined."""

    if not hasattr(page, "add_init_script"):
        raise ValueError("Invalid 'page' object. Expect a Playwright Page instance.")

    page.add_init_script(
        """
        Object.defineProperty(navigator, 'webdriver', {
            get: () => undefined,
            configurable: true
        });
        """
    )


def remove_webdriver(page: Page) -> None:
    """Remove navigator.webdriver."""

    if not hasattr(page, "add_init_script"):
        raise ValueError("Invalid 'page' object. Expect a Playwright Page instance.")

    page.add_init_script(
        """
        delete Object.getPrototypeOf(navigator).webdriver;
        """
    )


def set_languages(page: Page) -> None:
    """Set navigator.languages to standard ['en-US', 'en']."""

    if not hasattr(page, "add_init_script"):
        raise ValueError("Invalid 'page' object. Expect a Playwright Page instance.")

    page.add_init_script(
        """
        Object.defineProperty(navigator, 'languages', {
            get: () => ['en-US', 'en']
        });
        """
    )


def set_window_navigator_chrome(page: Page, user_agent: str) -> None:
    """Set window.navigator.chrome to default chrome values."""

    if not hasattr(page, "add_init_script"):
        raise ValueError("Invalid 'page' object. Expect a Playwright Page instance.")

    # Get runtime dictionary based on 'user_agent'
    runtime = get_runtime(user_agent)
    print(f"runtime : {pformat(runtime, sort_dicts=False)}\n")

    page.add_init_script(
        f"""
        window.navigator.chrome = {{
            app: {{ isInstalled: false }},
            webstore: {{ onInstallStageChanged: {{}}, onDownloadProgress: {{}} }},
            runtime: {{{runtime}}}
        }};
        """
    )


def get_runtime(user_agent: str) -> str:
    """Get operating system from user-agent for desktop device only i.e. 'Chrome OS',
    'Chromium OS', 'Linux', 'macOS', and 'Windows'."""

    mapping = {
        "Linux": {"PlatformOs": "linux"},
        "Macintosh": {"PlatformOs": "mac"},
        "Windows": {"PlatformOs": "win"},
        "CrOS x86_64": {"PlatformOs": "cros"},
    }

    # Return "Unknown" if no keys are found in 'user_agent'
    if all(keyword not in user_agent for keyword in mapping.keys()):
        raise ValueError(
            "No suitable platform i.e. Linux, Macintosh, Windows or Chrome OS found."
        )

    # Get processor architecture
    proc_arch = scraper_utils.get_processor_architecture()

    # Update processor architecture to 'runtime' dictionary
    arch_dict = {
        "PlatformArch": proc_arch,
        "PlatformNaclArch": proc_arch,
    }
    mapping = {k: dict(**v, **arch_dict) for k, v in mapping.items()}

    for keyword, runtime in mapping.items():
        if keyword in user_agent:
            # break loop once runtime is found
            return runtime


def rotate_webgl(page: Page) -> None:
    """Rotate software web GL renderer for non-hardware acceleration."""

    # List of webGL renderers for non hardware acceleration
    renderers = scraper_utils.gen_webgl_list(proc_family="amd")

    # Randomly select one software webGL renderers from available list
    webgl = random.choice(renderers)
    print(f"webgl : {webgl}")

    page.add_init_script(
        f"""
        const getParameter = WebGLRenderingContext.prototype.getParameter;
        WebGLRenderingContext.prototype.getParameter = function (parameter) {{
            if (parameter === 37445) return {webgl};
            return getParameter.call(this, parameter);
        }}
        """
    )


def set_stealth_plugins(page: Page) -> None:
    """Plugin configuration optimized for Chromium browsers only.

    - Include 'Chromium PDF Viewer' - Always present in Chromium installations.
    - 'Native Client' - 90% of real-world Chromium setups.
    - Append 'PDF Viewer' - 1-5% chance users install additional PDF-handling extensions.
    """

    # Random float number between between 0 and 1 to simulate probabiltiy
    random_num = random.random()

    # Core plugins i.e. Chromium PDF Viewer
    plugins = {
        "0": {
            "name": "Chromium PDF Viewer",
            "description": "Portable Document Format",
            "filename": "mhjfbmdgcfjbbpaeojofohoefgiehjai",
        }
    }

    # Add Native Client (~90% probability)
    if random_num < 0.9:
        plugins["1"] = {
            "name": "Native Client",
            "description": "native Client Execution",
            "filename": "internal-nacl-plugin",
        }

    # Add PDF Viewer (1-5% probability)
    if random_num < random.uniform(0.01, 0.05):
        plugins["2"] = {
            "name": "PDF Viewer",
            "description": "Portable Document Format",
            "filename": "oemmndcbldboiebfnladdacbdfmadadm",
        }

    # Update 'mimeTypes' for 'Chrome PDF Viewer' and 'PDF Viewer'; and convert to
    # json string
    plugins = append_mimetypes(plugins)
    plugins_js = json.dumps(plugins)
    print(f"\nplugins : \n\n{pformat(plugins, sort_dicts=False)}\n")

    page.add_init_script(
        f"""
        Object.defineProperty(navigator, 'plugins', {{
            get: () => {plugins_js},
            configurable: true
        }});
        """
    )


def throttle_tcp(page: Page, min_ms: int = 500, max_ms: int = 800) -> None:
    """Intercept HTTP requests and introduce human delays before continuing them."""

    def delay_request(route: Route) -> None:
        delay = random.randint(min_ms, max_ms)
        page.wait_for_timeout(delay)
        route.continue_()

    page.route("**/*", delay_request)


def append_mimetypes(plugins: dict[str, dict[str, str]]) -> dict[str, dict[str, Any]]:
    """Update plugins with 'mimeTypes' dictionary for 'Chrome PDF Viewer'
    and 'PDF Viewer'.

    Args:
        plugins (dict[str, dict[str, str]]):
            List containing 'Chrome PDF Viewer', 'Native Client' and 'PDF Viewer'
            dictionary containing 'name', 'description' and 'filename' keys.

    Returns:
        updated_plugins (dict[str, dict[str, Any]]):
            Plugins updated with 'mimeTypes' info.
    """

    updated_plugins = {}

    for idx, plugin in plugins.items():
        # Update only for Chromium PDF Viewer and PDF Viewer
        if plugin["name"] in ["Chromium PDF Viewer", "PDF Viewer"]:
            plugin["mimeTypes"] = [
                {
                    "type": "application/pdf",
                    "suffixes": "pdf",
                    "description": "Portable Document Format",
                },
                {
                    "type": "text/pdf",
                    "suffixes": "pdf",
                    "description": "Portable Document Format",
                },
            ]

        updated_plugins[idx] = plugin

    return updated_plugins
