"""Validate analyses.json manifest against experiment directories.

Usage:
    python -m analysis.validate
"""

import sys

from analysis.data import validate_manifest


def main() -> None:
    warnings = validate_manifest()
    if not warnings:
        print("All manifest entries OK.")
    else:
        for w in warnings:
            print(f"  WARNING: {w}")
        sys.exit(1)


if __name__ == "__main__":
    main()
