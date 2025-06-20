#!/usr/bin/env python3
"""Build standalone executables for the Subwoofer Simulator."""
import argparse
import subprocess
from pathlib import Path


def run_pyinstaller(windowed: bool) -> None:
    args = [
        "pyinstaller",
        "--onefile",
    ]
    if windowed:
        args.append("--windowed")
    args.append("subwoofer_simulation.py")
    subprocess.check_call(args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create standalone executable")
    parser.add_argument(
        "--console", action="store_true", help="Show console window (default hides)"
    )
    args = parser.parse_args()
    run_pyinstaller(not args.console)


if __name__ == "__main__":
    main()
