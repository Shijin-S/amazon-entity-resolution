"""Smoke test to verify environment setup and core dependencies."""
import importlib
import os
import sys


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def test_environment():
    """Verify Python version, virtual environment activation, and package imports."""
    # 1. Python version check (require >= 3.10)
    version_info = sys.version_info
    version_str = sys.version.split()[0]
    print(f"Python version: {version_str}")
    assert version_info >= (3, 10), f"Expected Python >= 3.10, found {version_str}"

    # 2. Confirmation the virtual environment is active
    in_venv = (sys.prefix != sys.base_prefix) or bool(os.environ.get("VIRTUAL_ENV"))
    print(f"Virtual environment active: {in_venv} (sys.prefix: {sys.prefix})")
    assert in_venv, "Virtual environment is not active or sys.prefix equals sys.base_prefix"

    # 3. Successful import of core packages
    packages = [
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("scikit-learn", "sklearn"),
        ("rapidfuzz", "rapidfuzz"),
        ("xgboost", "xgboost"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("joblib", "joblib"),
    ]

    imported_modules = {}
    for pkg_name, module_name in packages:
        try:
            mod = importlib.import_module(module_name)
            imported_modules[pkg_name] = mod
            print(f"Successfully imported {pkg_name} (module: {module_name})")
        except ImportError as exc:
            assert False, f"Failed to import {pkg_name} ({module_name}): {exc}"

    # 4. Installed version numbers of: pandas, numpy, scikit-learn, rapidfuzz, xgboost
    version_keys = ["pandas", "numpy", "scikit-learn", "rapidfuzz", "xgboost"]
    print("\nKey package versions:")
    for key in version_keys:
        mod = imported_modules[key]
        version = getattr(mod, "__version__", "unknown")
        print(f"  {key}: {version}")
        assert version != "unknown", f"Could not determine version for {key}"


if __name__ == "__main__":
    test_environment()
