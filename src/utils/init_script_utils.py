"""Functions to call Playwright 'page.add_init_script'
including helper functions."""

import json
import platform
import random
import re
from pprint import pformat
from typing import Any

from playwright.sync_api import Page


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


def set_fake_plugins(page: Page) -> None:
    """Set navigator.plugins to fake plugins list."""

    if not hasattr(page, "add_init_script"):
        raise ValueError("Invalid 'page' object. Expect a Playwright Page instance.")

    plugins = {
        "0": {"0": {}, "1": {}},
        "1": {"0": {}, "1": {}},
        "2": {"0": {}, "1": {}},
        "3": {"0": {}, "1": {}},
        "4": {"0": {}, "1": {}},
    }

    # Serialize Python dictionary to JSON for injection into JavaScript
    plugins_js = json.dumps(plugins)

    page.add_init_script(
        f"""
        Object.defineProperty(navigator, 'plugins', {{
            get: () => {plugins_js}
        }});
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
    print(f"\nuser_agent : {user_agent}")
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


def get_processor_architecture() -> str:
    """Get processor architecture of device that will be running the Bot test."""

    arch = platform.machine().lower()

    match arch:
        case "amd64" | "x86_64":
            return "x86-64"
        case "aarch64":
            return "arm64"
        case "arm":
            return "arm"
        case "i386" | "i686":
            return "x86-32"
        case _:
            return arch  # Unknown architecture


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
    proc_arch = get_processor_architecture()

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


def rotate_software_webgl(page: Page) -> None:
    """Rotate software web GL renderer for non-hardware acceleration."""

    # List of webGL renderers for non hardware acceleration
    renderers = [
        "Google SwiftShader",
        "ANGLE (Google, Vulkan 1.3.0 (SwiftShader Device (Subzero) (0x0000C0DE)), SwiftShader driver)",
    ]

    # Randomly select one software webGL renderers from available list
    webgl = random.choice(renderers)

    page.add_init_script(
        f"""
        const getParameter = WebGLRenderingContext.prototype.getParameter;
        WebGLRenderingContext.prototype.get
        """
    )


def rotate_webgl(page: Page, user_agent: str) -> None:
    """Rotate web GL renderer based on 'user_agent'."""


def set_stealth_plugins(page: Page) -> None:
    """Plugin configuration optimized for Chromium browsers only.

    - Include 'Chromium PDF Viewer' - Always present in Chromium installations.
    - 'Native Client' - 90% of real-world Chromium setups.
    - Append 'PDF Viewer' - 1-5% chance users install additional PDF-handling extensions.
    """

    # Random float number between between 0 and 1 to simulate probabiltiy
    random_num = random.random()

    # Core plugins i.e. Chromium PDF Viewer
    plugins = [
        {
            "name": "Chromium PDF Viewer",
            "description": "Portable Document Format",
            "filename": "mhjfbmdgcfjbbpaeojofohoefgiehjai",
        }
    ]

    # Add Native Client (~90% probability)
    if random_num < 0.9:
        plugins.append(
            {
                "name": "Native Client",
                "description": "native Client Execution",
                "filename": "internal-nacl-plugin",
            }
        )

    # Add PDF Viewer (1-5% probability)
    if random_num < random.uniform(0.01, 0.05):
        plugins.append(
            {
                "name": "PDF Viewer",
                "description": "Portable Document Format",
                "filename": "oemmndcbldboiebfnladdacbdfmadadm",
            }
        )

    # Update 'mimeTypes' for 'Chrome PDF Viewer' and 'PDF Viewer'; and convert to
    # json string
    plugins = append_mimetypes(plugins)
    plugins_js = json.dumps(plugins)

    page.add_init_script(
        f"""
        Object.defineProperty(navigator, 'plugins', {{
            get: () => {plugins_js},
            configurable: true
        }});
        """
    )


def append_mimetypes(plugins: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Update plugins with 'mimeTypes' dictionary for 'Chrome PDF Viewer'
    and 'PDF Viewer'.

    Args:
        plugins (list[dict[str, str]]):
            List containing 'Chrome PDF Viewer', 'Native Client' and 'PDF Viewer'
            dictionary containing 'name', 'description' and 'filename' keys.

    Returns:
        updated_plugins (list[dict[str, Any]]):
            Plugins updated with 'mimeTypes' info.
    """

    updated_plugins = []

    for plugin in plugins:
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

        updated_plugins.append(plugin)

    return updated_plugins
