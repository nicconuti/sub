# Subwoofer Simulator Packaging

This project uses PyInstaller to create standalone executables.

## Requirements
- Python 3.9 or newer
- Dependencies listed in `requirements.txt`

## Building
Run the helper script on the target OS:

```bash
pip install -r requirements.txt
python build_installer.py
```

The script now creates two executables and, on macOS, a `.pkg` installer:

- `dist/subwoofer_simulation.exe` – GUI application
- `dist/subwoofer_simulation_console.exe` – console-enabled version
- `dist/subwoofer_simulation.pkg` on macOS if `pkgbuild` is available

After building, you can use platform-specific tools such as Inno Setup (Windows)
or `productbuild` (macOS) to further customise the installer packages.
