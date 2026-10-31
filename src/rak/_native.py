"""Loader for the native RAK extension module."""

import importlib

def load_native():
    try:
        return importlib.import_module("_rak")
    except ImportError as exc:
        raise RuntimeError(
            "Native module '_rak' is not available. Build RAK with CMake "
            "(see scripts/build/build.sh) and install the package before use."
        ) from exc

native = load_native()
