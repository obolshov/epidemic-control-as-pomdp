"""Analyze Optuna sweep CSV exports — parameter ranges in top trials.

Usage:
    python -m support.optuna.analyze results.csv
    python -m support.optuna.analyze results.csv --top 10
    python -m support.optuna.analyze results.csv --within-abs 0.15
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXPORTS_DIR = Path(__file__).parent / "exports"


def load_trials(path: str) -> pd.DataFrame:
    """Load CSV, keep COMPLETE and PRUNED trials, drop all-NaN param columns."""
    df = pd.read_csv(path)
    df.columns = [c.replace("Param ", "") if c.startswith("Param ") else c for c in df.columns]
    trials = df[df["State"].isin(("COMPLETE", "PRUNED"))].copy()
    trials = trials.dropna(axis=1, how="all")
    return trials


def select_top(df: pd.DataFrame, method: str, value: float) -> pd.DataFrame:
    """Select top trials by the chosen criterion."""
    df_sorted = df.sort_values("Value", ascending=False)
    if method == "top":
        return df_sorted.head(int(value))
    elif method == "within-abs":
        threshold = df_sorted["Value"].max() - value
        return df_sorted[df_sorted["Value"] >= threshold]
    raise ValueError(f"Unknown method: {method}")


def _param_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("Number", "State", "Value")]


def _fmt(v: float) -> str:
    if abs(v) < 0.01 or abs(v) >= 1000:
        return f"{v:.2e}"
    return f"{v:.4f}"


def _fmt_val(v) -> str:
    """Format a single parameter value from CSV (float, int, or string)."""
    if isinstance(v, str):
        return v
    if isinstance(v, float) and not v.is_integer():
        return _fmt(v)
    return str(int(v))


def _is_log_scale(values: pd.Series) -> bool:
    vmin, vmax = values.min(), values.max()
    return vmin > 0 and vmax / vmin > 10


def _check_edge(top_vals: pd.Series, full_vals: pd.Series) -> str:
    """Detect if top set median clusters near search range boundaries."""
    vmed = top_vals.median()
    fmin, fmax = full_vals.min(), full_vals.max()

    if _is_log_scale(full_vals):
        log_med = math.log(vmed)
        log_min = math.log(fmin)
        log_max = math.log(fmax)
        log_range = log_max - log_min
        if log_range <= 0:
            return ""
        if (log_med - log_min) / log_range < 0.15:
            return "<< LOW"
        if (log_max - log_med) / log_range < 0.15:
            return ">> HIGH"
    else:
        r = fmax - fmin
        if r <= 0:
            return ""
        if (vmed - fmin) / r < 0.15:
            return "<< LOW"
        if (fmax - vmed) / r < 0.15:
            return ">> HIGH"
    return ""


def print_param_table(top: pd.DataFrame, full: pd.DataFrame) -> None:
    """Print parameter ranges for selected trials with edge detection."""
    params = _param_cols(top)

    for col in params:
        top_vals = top[col].dropna()
        full_vals = full[col].dropna()
        if len(top_vals) == 0:
            continue

        is_categorical = len(full_vals.unique()) <= 10

        if is_categorical:
            counts = top_vals.value_counts().sort_index()
            dist = "  ".join(
                f"{_fmt_val(v)}({int(c)})"
                for v, c in counts.items()
            )
            mode = top_vals.mode().iloc[0]
            full_sorted = sorted(full_vals.unique())
            edge = ""
            if mode == full_sorted[0]:
                edge = "<< MIN"
            elif mode == full_sorted[-1]:
                edge = ">> MAX"
            print(f"  {col:<22} {dist}  {edge}")
        else:
            vmin, vmed, vmax = top_vals.min(), top_vals.median(), top_vals.max()
            edge = _check_edge(top_vals, full_vals)
            print(f"  {col:<22} [{_fmt(vmin)}, {_fmt(vmax)}]  median={_fmt(vmed)}  {edge}")


def print_top_trials(top: pd.DataFrame) -> None:
    """Print individual top trials as a formatted table."""
    params = _param_cols(top)

    formatted: dict[str, list[str]] = {"#": [], "Value": []}
    for c in params:
        formatted[c] = []

    for _, row in top.iterrows():
        formatted["#"].append(str(int(row["Number"])))
        formatted["Value"].append(f"{row['Value']:.4f}")
        for c in params:
            v = row[c]
            if pd.isna(v):
                formatted[c].append("")
            else:
                formatted[c].append(_fmt_val(v))

    widths = {col: max(len(col), max((len(v) for v in vals), default=0)) for col, vals in formatted.items()}

    header = "  ".join(f"{col:>{widths[col]}}" for col in formatted)
    print(f"\n{header}")
    print("=" * len(header))
    for i in range(len(top)):
        row_str = "  ".join(f"{formatted[col][i]:>{widths[col]}}" for col in formatted)
        print(row_str)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Optuna sweep CSV results")
    parser.add_argument("csv", help="Path to Optuna CSV export")
    parser.add_argument("--exclude", type=int, nargs="+", default=[], help="Trial numbers to exclude")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--top", type=int, default=None, help="Select top N trials (default: 10)")
    group.add_argument("--within-abs", type=float, default=None, help="Trials within N absolute of best")
    args = parser.parse_args()

    if args.top is not None:
        method, value, label = "top", args.top, f"top {args.top}"
    elif args.within_abs is not None:
        method, value, label = "within-abs", args.within_abs, f"within {args.within_abs} of best"
    else:
        method, value, label = "top", 10, "top 10"

    csv_path = Path(args.csv)
    if not csv_path.exists():
        csv_path = EXPORTS_DIR / args.csv
    df = load_trials(str(csv_path))
    if args.exclude:
        df = df[~df["Number"].isin(args.exclude)]
    if len(df) == 0:
        print("No completed trials found.")
        sys.exit(1)

    top = select_top(df, method, value)

    print(f"\nCompleted trials: {len(df)}")
    print(f"Selected ({label}): {len(top)}")
    print(f"Reward range (selected): [{top['Value'].min():.4f}, {top['Value'].max():.4f}]")
    print(f"Reward range (all):      [{df['Value'].min():.4f}, {df['Value'].max():.4f}]")
    print(f"\nParameter ranges:")
    print("  Edge flags: << LOW / >> HIGH = median near search boundary (continuous)")
    print("              << MIN / >> MAX  = mode at boundary (categorical)\n")
    print_param_table(top, df)
    print_top_trials(top)


if __name__ == "__main__":
    main()
