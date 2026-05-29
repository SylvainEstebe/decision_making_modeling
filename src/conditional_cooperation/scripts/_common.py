"""Shared CLI helpers for the fit scripts."""

from __future__ import annotations

import argparse
from pathlib import Path


def add_common_args(parser: argparse.ArgumentParser, *, default_label: str) -> None:
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/public_good/HerrmannThoeniGaechterDATA.csv"),
        help="Path to the Herrmann/Thöni/Gächter CSV.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(f"results/{default_label}"),
        help="Where to write traces and figures.",
    )
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1983)
