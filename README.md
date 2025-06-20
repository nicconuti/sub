# Subwoofer Simulator Packaging

This project uses PyInstaller to create standalone executables.

## Requirements
- Python 3.9 or newer
- Dependencies listed in `requirements.txt`

## Building
Install the runtime requirements, then (only if you need to build an
executable) install the optional `build` extras and run the helper script:

```bash
pip install -r requirements.txt
pip install .[build]
python build_installer.py
```

- On Windows this produces `dist/subwoofer_simulation.exe`.
- On macOS running the script creates `dist/subwoofer_simulation` which can be
  packaged with `pkgbuild` or similar to create a `.pkg` installer.

After building, you can use platform-specific tools such as Inno Setup (Windows)
or `pkgbuild`/`productbuild` (macOS) to wrap the executable in a standard
installer package.
