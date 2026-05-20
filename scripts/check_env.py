from __future__ import annotations

import argparse
import importlib
import platform
import sys


CORE_IMPORTS = ["fasb", "numpy", "omegaconf", "hydra", "stable_baselines3", "gymnasium"]
METADRIVE_IMPORT = "metadrive"


def check_import(module: str) -> tuple[bool, str]:
    try:
        imported = importlib.import_module(module)
    except Exception as exc:  # noqa: BLE001 - show import-time dependency failures clearly.
        return False, f"{type(exc).__name__}: {exc}"
    version = getattr(imported, "__version__", "unknown")
    return True, str(version)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check local FASB/MetaDrive environment readiness.")
    parser.add_argument(
        "--require-metadrive",
        action="store_true",
        help="Exit nonzero if MetaDrive or runtime RL dependencies are unavailable.",
    )
    args = parser.parse_args()

    print(f"Python executable: {sys.executable}")
    print(f"Python version: {platform.python_version()}")
    print()

    failed_core: list[str] = []
    metadrive_failed = False

    print("Core imports:")
    for module in CORE_IMPORTS:
        ok, detail = check_import(module)
        status = "OK" if ok else "FAIL"
        print(f"  {status:4} {module}: {detail}")
        if not ok:
            failed_core.append(module)

    print("\nMetaDrive import:")
    for module in [METADRIVE_IMPORT]:
        ok, detail = check_import(module)
        status = "OK" if ok else "WARN"
        print(f"  {status:4} {module}: {detail}")
        if not ok:
            metadrive_failed = True

    if metadrive_failed:
        print(
            "\nMetaDrive is not importable. Use an existing MetaDrive-compatible venv if "
            "available. Python 3.10/3.11 may be safer. Do not assume Python 3.12 works "
            "with metadrive-simulator because it can pull old gym dependency versions."
        )

    if failed_core:
        print(f"\nMissing core dependencies: {', '.join(failed_core)}")
        return 1
    if args.require_metadrive and metadrive_failed:
        print("\nMetaDrive is required but not importable.")
        return 1

    print("\nEnvironment check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
