#!/usr/bin/env python3
"""Build standalone executables and a macOS installer for the Subwoofer Simulator."""
import argparse
import subprocess
import sys
import shutil
from pathlib import Path


def run_pyinstaller(name: str, windowed: bool, dest: Path) -> None:
    """Invoke PyInstaller to create a single executable."""
    args = [
        "pyinstaller",
        "--onefile",
        "--distpath",
        str(dest),
        "--name",
        name,
    ]
    if windowed:
        args.append("--windowed")
    args.append("subwoofer_simulation.py")
    subprocess.check_call(args)


def build_pkg(binary: Path, pkg_path: Path) -> None:
    """Create a simple macOS ``.pkg`` installer using ``pkgbuild``."""
    if shutil.which("pkgbuild") is None:
        print("pkgbuild not found - skipping macOS package creation", file=sys.stderr)
        return

    root = binary.parent / "pkgroot" / "usr" / "local" / "bin"
    root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(binary, root / binary.name)

    subprocess.check_call(
        [
            "pkgbuild",
            "--root",
            str(root.parent.parent),
            "--identifier",
            "com.example.subwoofer-sim",
            "--version",
            "0.1.0",
            str(pkg_path),
        ]
    )
    shutil.rmtree(binary.parent / "pkgroot")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create Windows executables and a macOS package if possible"
    )
    parser.add_argument(
        "--dist",
        type=Path,
        default=Path("dist"),
        help="Output directory for generated files",
    )
    args = parser.parse_args()

    dist = args.dist
    dist.mkdir(exist_ok=True)

    # Build console and GUI executables
    run_pyinstaller("subwoofer_simulation_console", False, dist)
    run_pyinstaller("subwoofer_simulation", True, dist)

    # On macOS also create a .pkg using the GUI executable
    if sys.platform == "darwin":
        build_pkg(dist / "subwoofer_simulation", dist / "subwoofer_simulation.pkg")


if __name__ == "__main__":
    main()
