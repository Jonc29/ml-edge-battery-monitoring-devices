"""Create descriptive EDA figures for the Panasonic SoC proxy dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from src.data.prepare_panasonic_soc_proxy import (
    DEFAULT_DATASET,
    MODEL_FEATURES,
)
from src.preprocessing.prepare_panasonic_model_data import (
    TARGET,
    load_and_validate_dataset,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DISTRIBUTIONS = (
    PROJECT_ROOT / "results/figures/panasonic_feature_distributions.png"
)
DEFAULT_CORRELATION = PROJECT_ROOT / "results/figures/panasonic_feature_correlation.png"
DEFAULT_SUMMARY = PROJECT_ROOT / "results/metrics/panasonic_eda_summary.json"
EDA_COLUMNS = (*MODEL_FEATURES, TARGET)
EDA_LABELS = {
    "voltage_v": "Voltage (V)",
    "current_a": "Current (A)",
    "battery_temp_c": "Battery temperature (°C)",
    "elapsed_time_s": "Elapsed time (s)",
    TARGET: "Approximate SoC proxy (%)",
}


def build_eda_summary(
    rows: list[dict[str, str]],
    features: np.ndarray,
    targets: np.ndarray,
) -> dict[str, Any]:
    """Summarize pooled sample distributions and feature correlations."""
    if not rows or features.shape[0] != len(rows) or targets.shape[0] != len(rows):
        raise ValueError("EDA input rows, features, and targets must be non-empty and aligned")
    values = np.column_stack((features, targets))
    if values.shape[1] != len(EDA_COLUMNS) or not np.isfinite(values).all():
        raise ValueError("EDA columns have an unexpected shape or non-finite values")

    statistics = {
        column: {
            "count": int(values.shape[0]),
            "mean": float(np.mean(values[:, index])),
            "std": float(np.std(values[:, index])),
            "median": float(np.median(values[:, index])),
            "minimum": float(np.min(values[:, index])),
            "maximum": float(np.max(values[:, index])),
        }
        for index, column in enumerate(EDA_COLUMNS)
    }
    correlation = np.corrcoef(values, rowvar=False)
    if not np.isfinite(correlation).all():
        raise ValueError("Cannot compute finite correlations for constant EDA columns")

    cycle_counts: dict[str, int] = {}
    for row in rows:
        cycle_id = row["cycle_id"]
        cycle_counts[cycle_id] = cycle_counts.get(cycle_id, 0) + 1

    return {
        "dataset": "Panasonic 18650PF 25 °C separate drive-cycle proxy table",
        "row_count": len(rows),
        "cycle_count": len(cycle_counts),
        "rows_per_cycle": cycle_counts,
        "columns": list(EDA_COLUMNS),
        "statistics": statistics,
        "pearson_correlation": {
            column: {
                other: float(correlation[row_index, column_index])
                for column_index, other in enumerate(EDA_COLUMNS)
            }
            for row_index, column in enumerate(EDA_COLUMNS)
        },
        "interpretation": (
            "Descriptive pooled-row analysis of one cell at 25 °C. Time-series "
            "rows are correlated and cycle sizes differ, so these values are "
            "not independent-sample inference or evidence of causation. The "
            "target is an approximate derived proxy, not measured SoC."
        ),
        "physical_hardware_tested": False,
        "physical_power_measured": False,
    }


def plot_distributions(
    features: np.ndarray,
    targets: np.ndarray,
    output_path: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Save histograms for four inputs and the explicitly labelled proxy."""
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    values = np.column_stack((features, targets))
    if values.shape[1] != len(EDA_COLUMNS) or not np.isfinite(values).all():
        raise ValueError("Distribution plot input has an unexpected shape or values")

    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for index, column in enumerate(EDA_COLUMNS):
        axis = axes.flat[index]
        axis.hist(values[:, index], bins=50, color="#2878B5", edgecolor="white")
        axis.set_title(EDA_LABELS[column])
        axis.set_ylabel("Sample count")
        axis.grid(axis="y", alpha=0.2)
        if column == TARGET:
            axis.axvline(0, color="#B22222", linestyle="--", linewidth=1)
            axis.axvline(100, color="#B22222", linestyle="--", linewidth=1)
    axes.flat[-1].set_visible(False)
    figure.suptitle(
        "Panasonic 25 °C input and approximate SoC proxy distributions",
        fontsize=14,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(str(output_path), dpi=160)
    plt.close(figure)


def plot_correlations(
    features: np.ndarray,
    targets: np.ndarray,
    output_path: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Save a pooled Pearson correlation heatmap for inputs and proxy."""
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    values = np.column_stack((features, targets))
    if values.shape[1] != len(EDA_COLUMNS) or not np.isfinite(values).all():
        raise ValueError("Correlation plot input has an unexpected shape or values")
    correlation = np.corrcoef(values, rowvar=False)
    if not np.isfinite(correlation).all():
        raise ValueError("Cannot plot correlations for constant EDA columns")

    labels = [EDA_LABELS[column] for column in EDA_COLUMNS]
    figure, axis = plt.subplots(figsize=(9, 7), constrained_layout=True)
    image = axis.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
    axis.set_xticks(np.arange(len(labels)), labels=labels, rotation=35, ha="right")
    axis.set_yticks(np.arange(len(labels)), labels=labels)
    axis.set_title(
        "Pooled-row Pearson correlations\n"
        "(target is an approximate SoC proxy, not ground truth)"
    )
    for row_index in range(correlation.shape[0]):
        for column_index in range(correlation.shape[1]):
            axis.text(
                column_index,
                row_index,
                f"{correlation[row_index, column_index]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=8,
            )
    figure.colorbar(image, ax=axis, label="Pearson correlation")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(str(output_path), dpi=160)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate distribution and correlation figures for the Panasonic "
            "proxy dataset; figures are descriptive, not validation."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--distributions", type=Path, default=DEFAULT_DISTRIBUTIONS)
    parser.add_argument("--correlation", type=Path, default=DEFAULT_CORRELATION)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing figures and summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dataset.is_file():
        raise FileNotFoundError(f"Prepared proxy dataset not found: {args.dataset}")
    outputs = (args.distributions, args.correlation, args.summary)
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing EDA output(s): "
            + ", ".join(str(path) for path in existing)
            + "; pass --overwrite to replace them."
        )

    rows, features, targets, _ = load_and_validate_dataset(args.dataset)
    summary = build_eda_summary(rows, features, targets)
    plot_distributions(
        features,
        targets,
        args.distributions,
        overwrite=args.overwrite,
    )
    plot_correlations(
        features,
        targets,
        args.correlation,
        overwrite=args.overwrite,
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"EDA complete: {summary['row_count']:,} pooled samples across "
        f"{summary['cycle_count']} cycles."
    )
    print(f"Distributions: {args.distributions}")
    print(f"Correlations: {args.correlation}")
    print(f"Summary: {args.summary}")
    print("Interpretation: descriptive pooled rows; proxy is not physical SoC.")


if __name__ == "__main__":
    main()
