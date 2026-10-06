"""Create a reproducible inventory and plot for the Panasonic dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.panasonic import (
    build_inventory,
    plot_drive_cycle,
    write_inventory,
    write_summary,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARCHIVE = PROJECT_ROOT / "data/raw/panasonic-18650pf-v1.zip"
DEFAULT_INVENTORY = PROJECT_ROOT / "results/tables/panasonic_file_inventory.csv"
DEFAULT_SUMMARY = PROJECT_ROOT / "results/metrics/panasonic_dataset_summary.json"
DEFAULT_PLOT = PROJECT_ROOT / "results/plots/panasonic_25c_cycle1_overview.png"
SAMPLE_MEMBER_SUFFIX = "25degC_Cycle_1_Pan18650PF.mat"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect the approved public Panasonic archive. This command does "
            "not train a model."
        )
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=DEFAULT_ARCHIVE,
        help="Path to the downloaded Panasonic ZIP archive.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the named inventory, summary, and plot if they exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.archive.is_file():
        raise FileNotFoundError(f"Dataset archive not found: {args.archive}")

    rows, summary = build_inventory(args.archive)
    write_inventory(rows, DEFAULT_INVENTORY, overwrite=args.overwrite)
    write_summary(summary, DEFAULT_SUMMARY, overwrite=args.overwrite)
    plot_drive_cycle(
        args.archive,
        SAMPLE_MEMBER_SUFFIX,
        DEFAULT_PLOT,
        overwrite=args.overwrite,
    )

    print(f"Inspected {summary['matlab_file_count']} MATLAB files.")
    print(
        "Summed file-row observations (duplicates may be present): "
        f"{summary['file_row_observation_count']:,}"
    )
    print(f"Inventory: {DEFAULT_INVENTORY.relative_to(PROJECT_ROOT)}")
    print(f"Summary:   {DEFAULT_SUMMARY.relative_to(PROJECT_ROOT)}")
    print(f"Plot:      {DEFAULT_PLOT.relative_to(PROJECT_ROOT)}")
    print("Model training: not performed.")


if __name__ == "__main__":
    main()
