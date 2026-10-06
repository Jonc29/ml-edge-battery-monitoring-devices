"""Validate, scale, and document the Panasonic proxy model dataset."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from src.data.prepare_panasonic_soc_proxy import DEFAULT_DATASET, MODEL_FEATURES
from src.models.train_panasonic_soc_proxy_baseline import GROUP, SEED, TARGET

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLEANED_DATASET = (
    PROJECT_ROOT / "data/processed/panasonic_soc_proxy_25c_1hz_clean.csv"
)
DEFAULT_SCALED_DATASET = (
    PROJECT_ROOT / "data/processed/panasonic_soc_proxy_25c_1hz_scaled.csv"
)
DEFAULT_SCALER = PROJECT_ROOT / "results/metrics/panasonic_feature_scaler.json"
DEFAULT_CLEANING_LOG = (
    PROJECT_ROOT / "results/metrics/panasonic_preprocessing_log.json"
)
REQUIRED_COLUMNS = (GROUP, "elapsed_time_s", *MODEL_FEATURES, TARGET)
CLEANED_COLUMNS = REQUIRED_COLUMNS
SCALED_COLUMNS = (
    GROUP,
    "elapsed_time_s",
    *(f"{feature}_scaled" for feature in MODEL_FEATURES),
    TARGET,
)


def load_and_validate_dataset(
    dataset_path: Path,
) -> tuple[list[dict[str, str]], np.ndarray, np.ndarray, np.ndarray]:
    """Load data and reject malformed rows instead of silently dropping them."""
    with dataset_path.open(newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        missing = set(REQUIRED_COLUMNS).difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
        rows = list(reader)

    if not rows:
        raise ValueError(f"Dataset contains no samples: {dataset_path}")

    features: list[list[float]] = []
    targets: list[float] = []
    groups: list[str] = []
    seen_time_keys: set[tuple[str, float]] = set()
    previous_time_by_group: dict[str, float] = {}

    for row_number, row in enumerate(rows, start=2):
        group = row[GROUP]
        if not group:
            raise ValueError(f"Dataset contains an empty {GROUP} on row {row_number}")
        try:
            elapsed_time = float(row["elapsed_time_s"])
            feature_values = [float(row[name]) for name in MODEL_FEATURES]
            target = float(row[TARGET])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Dataset contains a non-numeric value on row {row_number}"
            ) from error
        values = [elapsed_time, *feature_values, target]
        if not np.isfinite(values).all():
            raise ValueError(f"Dataset contains non-finite values on row {row_number}")
        if elapsed_time < previous_time_by_group.get(group, float("-inf")):
            raise ValueError(
                f"{group}: elapsed time moves backwards on row {row_number}"
            )
        previous_time_by_group[group] = elapsed_time

        time_key = (group, elapsed_time)
        if time_key in seen_time_keys:
            raise ValueError(
                f"Duplicate ({GROUP}, elapsed_time_s) key on row {row_number}"
            )
        seen_time_keys.add(time_key)
        groups.append(group)
        features.append(feature_values)
        targets.append(target)

    return (
        rows,
        np.asarray(features, dtype=np.float64),
        np.asarray(targets, dtype=np.float64),
        np.asarray(groups, dtype=str),
    )


def prepare_model_data(
    dataset_path: Path,
    *,
    test_size: float = 0.2,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Validate rows and fit a scaler on complete-cycle training rows only."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between zero and one")
    rows, features, targets, groups = load_and_validate_dataset(dataset_path)
    unique_groups = np.unique(groups)
    if unique_groups.size < 3:
        raise ValueError("At least three cycle groups are required for grouped holdout")

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=SEED)
    train_indices, test_indices = next(splitter.split(features, targets, groups))
    train_groups = np.unique(groups[train_indices])
    test_groups = np.unique(groups[test_indices])
    overlap = set(train_groups).intersection(test_groups)
    if overlap:
        raise RuntimeError(f"Cycle groups leaked across the split: {sorted(overlap)}")

    scaler = StandardScaler()
    scaler.fit(features[train_indices])
    scaler_mean = scaler.mean_
    scaler_scale = scaler.scale_
    scaler_variance = scaler.var_
    if scaler_mean is None or scaler_scale is None or scaler_variance is None:
        raise RuntimeError("Fitted StandardScaler did not expose its statistics")
    scaled_features = scaler.transform(features)
    if not np.isfinite(scaled_features).all():
        raise ValueError("Scaled model features contain non-finite values")

    cleaned_rows: list[dict[str, Any]] = []
    scaled_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        cleaned_rows.append({column: row[column] for column in CLEANED_COLUMNS})
        scaled_row: dict[str, Any] = {
            GROUP: row[GROUP],
            "elapsed_time_s": float(row["elapsed_time_s"]),
        }
        scaled_row.update(
            {
                f"{feature}_scaled": float(scaled_features[index, feature_index])
                for feature_index, feature in enumerate(MODEL_FEATURES)
            }
        )
        scaled_row[TARGET] = float(targets[index])
        scaled_rows.append(scaled_row)

    time_steps: list[float] = []
    previous: dict[str, float] = {}
    for row in rows:
        group = row[GROUP]
        elapsed_time = float(row["elapsed_time_s"])
        if group in previous:
            time_steps.append(elapsed_time - previous[group])
        previous[group] = elapsed_time

    out_of_range_mask = (targets < 0.0) | (targets > 100.0)
    scaler_metadata = {
        "scaler": "sklearn.preprocessing.StandardScaler",
        "feature_order": list(MODEL_FEATURES),
        "mean": {
            feature: float(value)
            for feature, value in zip(MODEL_FEATURES, scaler_mean, strict=True)
        },
        "scale": {
            feature: float(value)
            for feature, value in zip(MODEL_FEATURES, scaler_scale, strict=True)
        },
        "variance": {
            feature: float(value)
            for feature, value in zip(MODEL_FEATURES, scaler_variance, strict=True)
        },
        "transform": "(value - mean) / scale",
        "fit_scope": "training rows from complete drive-cycle groups only",
        "split_method": "GroupShuffleSplit by complete drive-cycle file",
        "random_state": SEED,
        "test_size_fraction": test_size,
        "train_cycle_ids": sorted(train_groups.tolist()),
        "test_cycle_ids": sorted(test_groups.tolist()),
        "fit_sample_count": int(train_indices.size),
        "Ah_used_as_feature": False,
    }
    cleaning_log = {
        "source_dataset": str(dataset_path),
        "cleaned_dataset": str(DEFAULT_CLEANED_DATASET),
        "scaled_dataset": str(DEFAULT_SCALED_DATASET),
        "scaler_artifact": str(DEFAULT_SCALER),
        "rows_read": len(rows),
        "rows_written_cleaned": len(cleaned_rows),
        "rows_written_scaled": len(scaled_rows),
        "rows_removed": 0,
        "cleaning_policy": (
            "Malformed, non-finite, non-monotonic, or duplicate-time rows are "
            "rejected with an explicit error; no sample is silently dropped."
        ),
        "checks": {
            "required_columns_present": True,
            "all_model_features_and_targets_finite": True,
            "elapsed_time_monotonic_within_cycles": True,
            "duplicate_cycle_time_keys": 0,
            "input_feature_columns": list(MODEL_FEATURES),
            "target_column": TARGET,
            "target_values_clipped": False,
            "out_of_range_proxy_rows_preserved": int(
                np.count_nonzero(out_of_range_mask)
            ),
        },
        "cycle_count": int(unique_groups.size),
        "train_cycle_ids": sorted(train_groups.tolist()),
        "test_cycle_ids": sorted(test_groups.tolist()),
        "train_rows": int(train_indices.size),
        "test_rows": int(test_indices.size),
        "scaler_fit_on_train_only": True,
        "scaler_fit_rows": int(train_indices.size),
        "sampling_interval_seconds": {
            "minimum": float(min(time_steps)) if time_steps else None,
            "median": float(np.median(time_steps)) if time_steps else None,
            "maximum": float(max(time_steps)) if time_steps else None,
            "intended_period": 1.0,
            "note": (
                "The current builder selects logged observations near regular "
                "one-second targets; this is not exact interpolation, and "
                "gaps reflect the source logging schedule."
            ),
        },
        "scaler": "StandardScaler fitted on training rows only",
        "target_status": (
            "Approximate capacity-referenced coulomb-counting proxy; not "
            "validated physical SoC."
        ),
        "physical_hardware_tested": False,
        "physical_power_measured": False,
    }
    return cleaned_rows, scaled_rows, scaler_metadata, cleaning_log


def write_outputs(
    cleaned_rows: list[dict[str, Any]],
    scaled_rows: list[dict[str, Any]],
    scaler_metadata: dict[str, Any],
    cleaning_log: dict[str, Any],
    cleaned_path: Path,
    scaled_path: Path,
    scaler_path: Path,
    log_path: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Write all preprocessing artifacts, refusing accidental replacement."""
    output_paths = (cleaned_path, scaled_path, scaler_path, log_path)
    existing = [path for path in output_paths if path.exists()]
    if existing and not overwrite:
        paths = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Refusing to overwrite existing output(s): {paths}; "
            "pass --overwrite to replace them."
        )

    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    for output_rows, output_path, fieldnames in (
        (cleaned_rows, cleaned_path, CLEANED_COLUMNS),
        (scaled_rows, scaled_path, SCALED_COLUMNS),
    ):
        with output_path.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(output_rows)
    scaler_path.write_text(
        json.dumps(scaler_metadata, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    cleaning_log["cleaned_dataset"] = str(cleaned_path)
    cleaning_log["scaled_dataset"] = str(scaled_path)
    cleaning_log["scaler_artifact"] = str(scaler_path)
    log_path.write_text(
        json.dumps(cleaning_log, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the Panasonic proxy table, preserve its labels, and fit "
            "a feature scaler on complete-cycle training rows only."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--cleaned", type=Path, default=DEFAULT_CLEANED_DATASET)
    parser.add_argument("--scaled", type=Path, default=DEFAULT_SCALED_DATASET)
    parser.add_argument("--scaler", type=Path, default=DEFAULT_SCALER)
    parser.add_argument("--log", type=Path, default=DEFAULT_CLEANING_LOG)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing preprocessing outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dataset.is_file():
        raise FileNotFoundError(f"Prepared proxy dataset not found: {args.dataset}")
    cleaned, scaled, scaler, log = prepare_model_data(
        args.dataset,
        test_size=args.test_size,
    )
    write_outputs(
        cleaned,
        scaled,
        scaler,
        log,
        args.cleaned,
        args.scaled,
        args.scaler,
        args.log,
        overwrite=args.overwrite,
    )
    print(
        f"Validated {len(cleaned):,} rows across {log['cycle_count']} cycles; "
        f"removed {log['rows_removed']} rows."
    )
    print(
        f"Scaler fitted on {log['scaler_fit_rows']:,} training rows from "
        f"{len(log['train_cycle_ids'])} complete cycles."
    )
    print(f"Cleaned data: {args.cleaned}")
    print(f"Scaled data: {args.scaled}")
    print(f"Cleaning log: {args.log}")
    print("Labels remain approximate proxies; no clipping or ground-truth claim.")


if __name__ == "__main__":
    main()
