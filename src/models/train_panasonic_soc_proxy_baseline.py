"""Train a grouped-holdout baseline against the explicitly approximate proxy."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

from src.data.prepare_panasonic_soc_proxy import (
    DEFAULT_DATASET,
    MODEL_FEATURES,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = PROJECT_ROOT / "models/baseline/panasonic_soc_proxy_rf.joblib"
DEFAULT_METRICS = (
    PROJECT_ROOT / "results/metrics/panasonic_soc_proxy_baseline.json"
)
TARGET = "soc_proxy_percent"
GROUP = "cycle_id"
SEED = 42


def load_proxy_csv(dataset_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read model features, proxy target, and cycle groups from CSV."""
    features: list[list[float]] = []
    targets: list[float] = []
    groups: list[str] = []
    with dataset_path.open(newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        required = {*MODEL_FEATURES, TARGET, GROUP}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Proxy dataset is missing columns: {sorted(missing)}")
        for row in reader:
            features.append([float(row[name]) for name in MODEL_FEATURES])
            targets.append(float(row[TARGET]))
            groups.append(row[GROUP])

    if not features:
        raise ValueError(f"Proxy dataset contains no samples: {dataset_path}")
    X = np.asarray(features, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32)
    group_array = np.asarray(groups, dtype=str)
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Proxy dataset contains non-finite features or target values")
    if np.unique(group_array).size < 3:
        raise ValueError("At least three cycle groups are required for grouped holdout")
    return X, y, group_array


def fit_grouped_baseline(
    dataset_path: Path,
    *,
    test_size: float = 0.2,
) -> tuple[RandomForestRegressor, dict[str, Any]]:
    """Fit a seeded random-forest baseline with whole cycles held out."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between zero and one")
    X, y, groups = load_proxy_csv(dataset_path)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=SEED)
    train_indices, test_indices = next(splitter.split(X, y, groups))
    train_groups = np.unique(groups[train_indices])
    test_groups = np.unique(groups[test_indices])
    overlap = set(train_groups).intersection(test_groups)
    if overlap:
        raise RuntimeError(f"Cycle groups leaked across the split: {sorted(overlap)}")

    model = RandomForestRegressor(
        n_estimators=100,
        min_samples_leaf=2,
        max_leaf_nodes=128,
        max_features=1.0,
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(X[train_indices], y[train_indices])
    predictions = model.predict(X[test_indices])
    metrics = {
        "dataset": str(dataset_path),
        "target": TARGET,
        "target_status": "approximate_capacity_referenced_coulomb_counting_proxy",
        "model": "RandomForestRegressor",
        "model_parameters": {
            "n_estimators": 100,
            "min_samples_leaf": 2,
            "max_leaf_nodes": 128,
            "max_features": 1.0,
            "random_state": SEED,
        },
        "input_features": list(MODEL_FEATURES),
        "excluded_target_source_channel": "Ah",
        "split_method": "GroupShuffleSplit by complete drive-cycle file",
        "test_size_fraction": test_size,
        "train_cycle_ids": sorted(train_groups.tolist()),
        "test_cycle_ids": sorted(test_groups.tolist()),
        "train_sample_count": int(train_indices.size),
        "test_sample_count": int(test_indices.size),
        "proxy_test_mae_percentage_points": float(
            mean_absolute_error(y[test_indices], predictions)
        ),
        "proxy_test_rmse_percentage_points": float(
            np.sqrt(mean_squared_error(y[test_indices], predictions))
        ),
        "proxy_test_r2": float(r2_score(y[test_indices], predictions)),
        "interpretation": (
            "These metrics describe held-out-cycle agreement with the derived "
            "proxy only. They are not validated SoC accuracy or multi-cell "
            "generalization."
        ),
        "physical_hardware_tested": False,
        "physical_power_measured": False,
    }
    return model, metrics


def write_model_outputs(
    model: RandomForestRegressor,
    metrics: dict[str, Any],
    model_path: Path,
    metrics_path: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Persist the estimator and its proxy-only evaluation metadata."""
    existing = [path for path in (model_path, metrics_path) if path.exists()]
    if existing and not overwrite:
        paths = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Refusing to overwrite existing output(s): {paths}; "
            "pass --overwrite to replace them."
        )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    metrics["model_artifact"] = str(model_path)
    metrics_path.write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train an experimental baseline against Panasonic-derived proxy "
            "labels; reported scores are not validated SoC metrics."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of cycle groups held out (default: 0.2).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing model and metrics outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dataset.is_file():
        raise FileNotFoundError(
            f"Prepared proxy dataset not found: {args.dataset}. "
            "Run src.data.prepare_panasonic_soc_proxy first."
        )
    model, metrics = fit_grouped_baseline(
        args.dataset,
        test_size=args.test_size,
    )
    write_model_outputs(
        model,
        metrics,
        args.model,
        args.metrics,
        overwrite=args.overwrite,
    )
    print("Trained grouped-holdout RandomForest baseline.")
    print(
        "Held-out-cycle proxy RMSE: "
        f"{metrics['proxy_test_rmse_percentage_points']:.3f} percentage points"
    )
    print(f"Test cycle IDs: {', '.join(metrics['test_cycle_ids'])}")
    print("Interpretation: proxy agreement only; not validated SoC accuracy.")


if __name__ == "__main__":
    main()
