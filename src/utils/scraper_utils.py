"""Helper functions for scraping regardless of scraper."""

import platform
import random
import re
import time
from pathlib import Path
from typing import Literal

ProcFamily = Literal["intel", "amd", "apple", "nvidia"]


def human_delay(min_ms: int = 800, max_ms: int = 2500):
    """Simulate human-like delays between 'min_ms' and 'max_ms' milliseconds."""
    time.sleep(random.uniform(min_ms / 1000, max_ms / 1000))


def get_ua_chrome_version(user_agent: str) -> str:
    """Get Chrome version from user agent."""

    # Chrome version is after 'Chrome/' string e.g. 'Chrome/134.0.0.0'
    results = re.findall(r"(?<=Chrome/)\d+", user_agent)

    # There should only 1 item in 'results'
    return results[0]


def get_ua_os(user_agent: str) -> str:
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


def get_processor_architecture() -> str:
    """Get processor architecture of the device running bot test."""

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


def get_os() -> str:
    """Get Operating System of the device running bot test."""

    os_name = platform.system()

    match os_name:
        case "Windows":
            return "Windows"
        case "Darwin":
            return "macOS"
        case "Linux":
            return "Chrome OS" if is_chrome_os() else "Linux"
        case _:
            return "Unknown OS"


def is_chrome_os() -> bool:
    """Check if operating system is Chrome OS."""

    lsb_path = "/etc/lsb-release"

    # Return True if "CHROMEOS" exist in 'lsb-release' file
    if Path(lsb_path).exists() and "CHROMEOS" in open(lsb_path).read():
        return True

    return False


def gen_webgl_list(proc_family: ProcFamily = "intel") -> list[str] | None:
    """Generate list of WebGL renderers that are compatible to device OS and
    graphics processor family.

    Args:
        proc_family (ProcFamily):
            Either "intel", "amd", "apple" or "nvidia" (Default: "intel").

    Returns:
        (list[str]): List of compatible WebGL renderers.
    """

    proc_family = proc_family.lower()

    # Get operating system
    os_name = get_os().lower()

    # Set os_name to be 'linux' if Chrome OS as WebGL renderers is same for Linux
    # and Chrome OS
    if os_name == "chrome os":
        os_name = "linux"

    mapping = {
        "amd": {
            "windows": [
                "ANGLE (AMD, AMD Radeon(TM) Vega 8 Graphics (0x000015D8) Direct3D11 vs_5_0 ps_5_0, D3D11)",
                "ANGLE (AMD, AMD Radeon (TM) Graphics (0x000015E7) Direct3D11 vs_5_0 ps_5_0, D3D11)",
                "ANGLE (AMD, AMD Radeon RX 580 2048SP Direct3D11 vs_5_0 ps_5_0, D3D11)",
            ],
            "linux": [
                "Mesa Radeon RX 5700 XT (RADV)",
                "Mesa Radeon RX Vega 8 (RADV)",
                "Mesa Radeon Vega 3 Graphics (RADV)",
            ],
        },
        "intel": {
            "windows": [
                "ANGLE (Intel, Intel(R) HD Graphics 5500 (0x00001616) Direct3D11 vs_5_0 ps_5_0, D3D11)",
                "ANGLE (Intel, Intel(R) HD Graphics 530 (0x00001912) Direct3D11 vs_5_0 ps_5_0, D3D11)",
                "ANGLE (Intel, Intel(R) HD Graphics 4600 (0x00000412) Direct3D11 vs_5_0 ps_5_0, D3D11)",
            ],
            "linux": [
                "Mesa Intel(R) Graphics (RPL-U)",
                "Mesa Intel(R) UHD Graphics (CML GT2)",
                "Mesa Intel(R) Iris Graphics 6100",
            ],
        },
        "apple": {
            "macos": [
                "ANGLE (Apple, ANGLE Metal Renderer: Apple M1 Pro, Unspecified Version)",
                "ANGLE (Apple, ANGLE Metal Renderer: Apple M1, Unspecified Version)",
                "ANGLE (Apple, ANGLE Metal Renderer: Apple M2, Unspecified Version)",
                "ANGLE (Apple, ANGLE Metal Renderer: Apple M3, Unspecified Version)",
            ],
        },
        "nvidia": {
            "windows": [
                "ANGLE (NVIDIA, NVIDIA GeForce RTX 3050 Laptop GPU Direct3D11 vs_5_0 ps_5_0, D3D11)",
                "ANGLE (NVIDIA, NVIDIA GeForce RTX 3060 Direct3D11 vs_5_0 ps_5_0, D3D11)",
                "ANGLE (NVIDIA, NVIDIA GeForce RTX 3070 Ti Direct3D11 vs_5_0 ps_5_0, D3D11)",
            ],
            "linux": [
                "Mesa NVIDIA GeForce GTX 1050 Ti (NVIDIA 560.35.3.0)",
                "Mesa NVIDIA GeForce RTX 3080 (NVIDIA 515.65.01)",
                "Mesa NVIDIA GeForce GTX 1660 Super (NVIDIA 470.129.06)",
            ],
        },
    }

    if compatible_list := mapping[proc_family].get(os_name, None):
        return compatible_list

    print(
        f"No compatible WebGL renderers for {proc_family} processor in {os_name} environment."
    )
