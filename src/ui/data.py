"""Validated access to the project's existing dashboard data and artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROXY_DATASET = PROJECT_ROOT / "data/processed/panasonic_soc_proxy_25c_1hz.csv"
MODEL_COMPARISON = PROJECT_ROOT / "results/comparisons/model_comparison.csv"
OPTIMIZATION_COMPARISON = (
    PROJECT_ROOT / "results/comparisons/panasonic_rf_optimization.csv"
)
RESOURCE_BENCHMARKS = (
    PROJECT_ROOT / "results/metrics/panasonic_resource_benchmarks.json"
)
MODEL_SELECTION = PROJECT_ROOT / "results/metrics/panasonic_model_selection.json"
PROXY_METADATA = PROJECT_ROOT / "results/metrics/panasonic_soc_proxy_metadata.json"
PREPROCESSING_LOG = (
    PROJECT_ROOT / "results/metrics/panasonic_preprocessing_log.json"
)
DATASET_SUMMARY = PROJECT_ROOT / "results/metrics/panasonic_dataset_summary.json"
C_EXPORT_VALIDATION = (
    PROJECT_ROOT / "results/metrics/panasonic_c_export_validation.json"
)
LEAVE_ONE_CYCLE_OUT = (
    PROJECT_ROOT / "results/metrics/panasonic_leave_one_cycle_out.json"
)
FEATURE_CORRELATION_FIGURE = (
    PROJECT_ROOT / "results/figures/panasonic_feature_correlation.png"
)
FEATURE_DISTRIBUTIONS_FIGURE = (
    PROJECT_ROOT / "results/figures/panasonic_feature_distributions.png"
)

BASELINE_MODEL = PROJECT_ROOT / "models/baseline/panasonic_soc_proxy_selected_rf.joblib"
OPTIMIZED_MODEL = PROJECT_ROOT / "models/optimized/panasonic_rf_50_trees.joblib"

FEATURE_COLUMNS = (
    "voltage_v",
    "current_a",
    "battery_temp_c",
    "elapsed_time_s",
)
TARGET_COLUMN = "soc_proxy_percent"
DATASET_COLUMNS = ("cycle_id", *FEATURE_COLUMNS, TARGET_COLUMN)


def load_json(path: Path) -> dict[str, Any]:
    """Read a JSON object, rejecting malformed or unexpected top-level data."""
    with path.open(encoding="utf-8") as input_file:
        value = json.load(input_file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def load_csv(path: Path, required_columns: tuple[str, ...]) -> pd.DataFrame:
    """Read a CSV and validate its required columns and numeric data."""
    frame = pd.read_csv(path)
    missing = set(required_columns).difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{path} contains no rows")
    return frame


def load_proxy_dataset(path: Path = PROXY_DATASET) -> pd.DataFrame:
    """Read finite, correctly shaped feature and proxy samples."""
    frame = load_csv(path, DATASET_COLUMNS)
    for column in (*FEATURE_COLUMNS, TARGET_COLUMN):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if (
        frame["cycle_id"].isna().any()
        or frame["cycle_id"].astype(str).str.len().eq(0).any()
    ):
        raise ValueError(f"{path} contains an empty cycle ID")
    numeric = frame.loc[:, (*FEATURE_COLUMNS, TARGET_COLUMN)].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{path} contains non-finite feature or proxy values")
    return frame


def load_model_comparison(path: Path = MODEL_COMPARISON) -> pd.DataFrame:
    """Read the fixed whole-cycle model-comparison results."""
    return load_csv(
        path,
        (
            "model",
            "proxy_test_mae_percentage_points",
            "proxy_test_rmse_percentage_points",
            "proxy_test_r2",
        ),
    )


def load_optimization_comparison(
    path: Path = OPTIMIZATION_COMPARISON,
) -> pd.DataFrame:
    """Read the baseline/optimized model trade-off results."""
    return load_csv(
        path,
        (
            "variant",
            "n_estimators",
            "proxy_test_mae_percentage_points",
            "proxy_test_rmse_percentage_points",
            "proxy_test_r2",
            "artifact_size_bytes",
            "artifact_size_reduction_percent_vs_baseline",
            "mae_change_percentage_points_vs_baseline",
        ),
    )
