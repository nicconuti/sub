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

- On Windows this produces `dist/subwoofer_simulation.exe`.
- On macOS running the script creates `dist/subwoofer_simulation` which can be
  packaged with `pkgbuild` or similar to create a `.pkg` installer.

After building, you can use platform-specific tools such as Inno Setup (Windows)
or `pkgbuild`/`productbuild` (macOS) to wrap the executable in a standard
installer package.
