"""Evaluate smaller Random Forest variants against the selected baseline."""

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

from src.data.prepare_panasonic_soc_proxy import DEFAULT_DATASET
from src.models.compare_panasonic_models import (
    DEFAULT_METADATA as DEFAULT_COMPARISON_METADATA,
)
from src.models.train_panasonic_soc_proxy_baseline import (
    SEED,
    load_proxy_csv,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASELINE = PROJECT_ROOT / "models/baseline/panasonic_soc_proxy_selected_rf.joblib"
DEFAULT_OPTIMIZED_DIR = PROJECT_ROOT / "models/optimized"
DEFAULT_COMPARISON = PROJECT_ROOT / "results/comparisons/panasonic_rf_optimization.csv"
DEFAULT_METADATA = PROJECT_ROOT / "results/metrics/panasonic_rf_optimization.json"
VARIANTS: dict[str, dict[str, int]] = {
    "rf_50_trees": {
        "n_estimators": 50,
        "min_samples_leaf": 2,
        "max_leaf_nodes": 128,
    },
    "rf_100_64_leaves": {
        "n_estimators": 100,
        "min_samples_leaf": 2,
        "max_leaf_nodes": 64,
    },
    "rf_50_64": {
        "n_estimators": 50,
        "min_samples_leaf": 2,
        "max_leaf_nodes": 64,
    },
    "rf_25_32": {
        "n_estimators": 25,
        "min_samples_leaf": 2,
        "max_leaf_nodes": 32,
    },
}
CSV_COLUMNS = [
    "variant",
    "n_estimators",
    "max_leaf_nodes",
    "min_samples_leaf",
    "train_cycle_ids",
    "test_cycle_ids",
    "train_sample_count",
    "test_sample_count",
    "proxy_test_mae_percentage_points",
    "proxy_test_rmse_percentage_points",
    "proxy_test_r2",
    "artifact_size_bytes",
    "artifact_size_reduction_percent_vs_baseline",
    "mae_change_percentage_points_vs_baseline",
    "per_cycle_metrics_json",
    "artifact",
]


def _load_split(
    dataset_path: Path,
    comparison_metadata: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    """Load arrays and verify metadata defines distinct, present cycle groups."""
    features, targets, groups = load_proxy_csv(dataset_path)
    train_groups = comparison_metadata.get("train_cycle_ids")
    test_groups = comparison_metadata.get("test_cycle_ids")
    if not isinstance(train_groups, list) or not isinstance(test_groups, list):
        raise ValueError("Comparison metadata must list training and test cycles")
    if not train_groups or not test_groups:
        raise ValueError("Training and test cycle lists must both be non-empty")
    if set(train_groups).intersection(test_groups):
        raise ValueError("Comparison metadata leaks cycle groups across the split")
    present_groups = set(np.unique(groups).tolist())
    missing_groups = (set(train_groups) | set(test_groups)).difference(present_groups)
    if missing_groups:
        raise ValueError(f"Dataset is missing split cycles: {sorted(missing_groups)}")
    unexpected_groups = present_groups.difference(set(train_groups) | set(test_groups))
    if unexpected_groups:
        raise ValueError(
            f"Comparison split omits dataset cycles: {sorted(unexpected_groups)}"
        )

    train_mask = np.isin(groups, train_groups)
    test_mask = np.isin(groups, test_groups)
    if not train_mask.any() or not test_mask.any():
        raise ValueError("Training and test groups must contain samples")
    return (
        features[train_mask],
        targets[train_mask],
        features[test_mask],
        targets[test_mask],
        groups[test_mask],
        sorted(train_groups),
        sorted(test_groups),
    )


def _evaluate_predictions(
    truth: np.ndarray,
    predictions: np.ndarray,
    test_groups: np.ndarray,
    cycle_ids: list[str],
) -> tuple[dict[str, float], dict[str, dict[str, float | int]]]:
    if predictions.shape != truth.shape or not np.isfinite(predictions).all():
        raise ValueError("Model predictions must be finite and match test targets")
    per_cycle: dict[str, dict[str, float | int]] = {}
    for cycle_id in cycle_ids:
        mask = test_groups == cycle_id
        if not mask.any():
            raise ValueError(f"Test split has no samples for {cycle_id}")
        per_cycle[cycle_id] = {
            "sample_count": int(mask.sum()),
            "mae_percentage_points": float(
                mean_absolute_error(truth[mask], predictions[mask])
            ),
            "rmse_percentage_points": float(
                np.sqrt(mean_squared_error(truth[mask], predictions[mask]))
            ),
            "r2": float(r2_score(truth[mask], predictions[mask])),
        }
    return (
        {
            "mae": float(mean_absolute_error(truth, predictions)),
            "rmse": float(np.sqrt(mean_squared_error(truth, predictions))),
            "r2": float(r2_score(truth, predictions)),
        },
        per_cycle,
    )


def optimize_models(
    dataset_path: Path,
    comparison_metadata: dict[str, Any],
    baseline_path: Path,
    optimized_dir: Path,
    *,
    overwrite: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fit constrained variants and measure proxy error and serialized size."""
    if not baseline_path.is_file():
        raise FileNotFoundError(f"Selected baseline artifact not found: {baseline_path}")
    (
        X_train,
        y_train,
        X_test,
        y_test,
        test_group_rows,
        train_groups,
        test_groups,
    ) = _load_split(dataset_path, comparison_metadata)

    baseline = joblib.load(baseline_path)
    if not isinstance(baseline, RandomForestRegressor):
        raise TypeError("Selected baseline artifact is not a RandomForestRegressor")
    if baseline.n_estimators != 100 or baseline.max_leaf_nodes != 128:
        raise ValueError("Baseline artifact does not match the recorded 100/128 config")
    baseline_predictions = np.asarray(baseline.predict(X_test), dtype=float)
    baseline_metrics, baseline_per_cycle = _evaluate_predictions(
        y_test,
        baseline_predictions,
        test_group_rows,
        test_groups,
    )
    baseline_size = baseline_path.stat().st_size
    results: list[dict[str, Any]] = [
        {
            "variant": "selected_baseline",
            "n_estimators": int(baseline.n_estimators),
            "max_leaf_nodes": int(baseline.max_leaf_nodes),
            "min_samples_leaf": int(baseline.min_samples_leaf),
            "train_cycle_ids": ";".join(train_groups),
            "test_cycle_ids": ";".join(test_groups),
            "train_sample_count": int(X_train.shape[0]),
            "test_sample_count": int(X_test.shape[0]),
            "proxy_test_mae_percentage_points": baseline_metrics["mae"],
            "proxy_test_rmse_percentage_points": baseline_metrics["rmse"],
            "proxy_test_r2": baseline_metrics["r2"],
            "artifact_size_bytes": int(baseline_size),
            "artifact_size_reduction_percent_vs_baseline": 0.0,
            "mae_change_percentage_points_vs_baseline": 0.0,
            "per_cycle_metrics_json": json.dumps(
                baseline_per_cycle,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
            "artifact": str(baseline_path),
        }
    ]

    optimized_dir.mkdir(parents=True, exist_ok=True)
    existing_variants = [
        optimized_dir / f"panasonic_{variant}.joblib"
        for variant in VARIANTS
        if (optimized_dir / f"panasonic_{variant}.joblib").exists()
    ]
    if existing_variants and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing variant(s): "
            + ", ".join(str(path) for path in existing_variants)
            + "; pass --overwrite to replace them."
        )
    for variant, parameters in VARIANTS.items():
        artifact_path = optimized_dir / f"panasonic_{variant}.joblib"
        model = RandomForestRegressor(
            n_estimators=parameters["n_estimators"],
            min_samples_leaf=parameters["min_samples_leaf"],
            max_leaf_nodes=parameters["max_leaf_nodes"],
            max_features=1.0,
            random_state=SEED,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        predictions = np.asarray(model.predict(X_test), dtype=float)
        metrics, per_cycle = _evaluate_predictions(
            y_test,
            predictions,
            test_group_rows,
            test_groups,
        )
        joblib.dump(model, artifact_path)
        artifact_size = artifact_path.stat().st_size
        results.append(
            {
                "variant": variant,
                "n_estimators": int(model.n_estimators),
                "max_leaf_nodes": int(model.max_leaf_nodes),
                "min_samples_leaf": int(model.min_samples_leaf),
                "train_cycle_ids": ";".join(train_groups),
                "test_cycle_ids": ";".join(test_groups),
                "train_sample_count": int(X_train.shape[0]),
                "test_sample_count": int(X_test.shape[0]),
                "proxy_test_mae_percentage_points": metrics["mae"],
                "proxy_test_rmse_percentage_points": metrics["rmse"],
                "proxy_test_r2": metrics["r2"],
                "artifact_size_bytes": int(artifact_size),
                "artifact_size_reduction_percent_vs_baseline": float(
                    100.0 * (baseline_size - artifact_size) / baseline_size
                ),
                "mae_change_percentage_points_vs_baseline": float(
                    metrics["mae"] - baseline_metrics["mae"]
                ),
                "per_cycle_metrics_json": json.dumps(
                    per_cycle,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ),
                "artifact": str(artifact_path),
            }
        )

    metadata = {
        "dataset": str(dataset_path),
        "target_status": "approximate_capacity_referenced_coulomb_counting_proxy",
        "input_features": list(comparison_metadata["input_features"]),
        "excluded_target_source_channel": "Ah",
        "split_method": "same fixed complete-cycle split as model comparison",
        "train_cycle_ids": train_groups,
        "test_cycle_ids": test_groups,
        "train_sample_count": int(X_train.shape[0]),
        "test_sample_count": int(X_test.shape[0]),
        "baseline_artifact": str(baseline_path),
        "baseline_artifact_size_bytes": int(baseline_size),
        "serialization": "joblib.dump with default compression (uncompressed)",
        "optimization_strategy": (
            "Reduce tree count, maximum leaf nodes, or both while keeping "
            "random seed, minimum leaf size, and feature settings constant."
        ),
        "metrics": (
            "Proxy MAE/RMSE/R2 and exact serialized artifact file size. "
            "Latency and memory are reserved for roadmap phase 8."
        ),
        "interpretation": (
            "All scores compare predictions with an approximate proxy. The "
            "fixed test cycles informed model selection and are reused here "
            "for optimization, so these are exploratory trade-off results, "
            "not an independent final evaluation."
        ),
        "physical_hardware_tested": False,
        "physical_power_measured": False,
    }
    return results, metadata


def write_results(
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    comparison_path: Path,
    metadata_path: Path,
    *,
    overwrite: bool = False,
) -> None:
    existing = [path for path in (comparison_path, metadata_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing outputs: "
            + ", ".join(str(path) for path in existing)
            + "; pass --overwrite to replace them."
        )
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with comparison_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=CSV_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate smaller Random Forest variants against the selected "
            "model using the same held-out cycles."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--comparison-metadata",
        type=Path,
        default=DEFAULT_COMPARISON_METADATA,
    )
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--optimized-dir", type=Path, default=DEFAULT_OPTIMIZED_DIR)
    parser.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for path in (args.dataset, args.comparison_metadata, args.baseline):
        if not path.is_file():
            raise FileNotFoundError(f"Required input not found: {path}")
    result_paths = (args.comparison, args.metadata)
    existing = [path for path in result_paths if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing outputs: "
            + ", ".join(str(path) for path in existing)
            + "; pass --overwrite to replace them."
        )
    comparison_metadata = json.loads(
        args.comparison_metadata.read_text(encoding="utf-8")
    )
    rows, metadata = optimize_models(
        args.dataset,
        comparison_metadata,
        args.baseline,
        args.optimized_dir,
        overwrite=args.overwrite,
    )
    write_results(
        rows,
        metadata,
        args.comparison,
        args.metadata,
        overwrite=args.overwrite,
    )
    print("Random Forest size/accuracy trade-off experiment complete.")
    for row in rows:
        print(
            f"{row['variant']}: MAE {row['proxy_test_mae_percentage_points']:.3f}, "
            f"RMSE {row['proxy_test_rmse_percentage_points']:.3f}, "
            f"size {row['artifact_size_bytes']:,} bytes, "
            f"size reduction "
            f"{row['artifact_size_reduction_percent_vs_baseline']:.1f}%."
        )
    print(f"Comparison: {args.comparison}")
    print(f"Metadata: {args.metadata}")
    print("Interpretation: proxy agreement and serialized size only.")


if __name__ == "__main__":
    main()
