"""Compare baseline regressors on the same held-out Panasonic drive cycles."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data.prepare_panasonic_soc_proxy import DEFAULT_DATASET
from src.models.train_panasonic_soc_proxy_baseline import (
    MODEL_FEATURES,
    SEED,
    TARGET,
    load_proxy_csv,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COMPARISON = PROJECT_ROOT / "results/comparisons/model_comparison.csv"
DEFAULT_METADATA = (
    PROJECT_ROOT / "results/metrics/panasonic_model_comparison_metadata.json"
)
HOLDOUT_CYCLE_IDS = (
    "03-19-17_03.25 25degC_Cycle_2_Pan18650PF",
    "03-19-17_14.31 25degC_Cycle_4_Pan18650PF",
    "03-21-17_09.38 25degC_LA92_Pan18650PF",
)
MODEL_PARAMETERS: dict[str, dict[str, Any]] = {
    "RidgeRegression": {"alpha": 1.0, "scaler": "StandardScaler fitted on train only"},
    "RandomForestRegressor": {
        "n_estimators": 100,
        "min_samples_leaf": 2,
        "max_leaf_nodes": 128,
        "max_features": 1.0,
        "random_state": SEED,
        "n_jobs": -1,
    },
    "ExtraTreesRegressor": {
        "n_estimators": 100,
        "min_samples_leaf": 2,
        "max_leaf_nodes": 128,
        "max_features": 1.0,
        "random_state": SEED,
        "n_jobs": -1,
    },
}
CSV_COLUMNS: list[str] = [
    "model",
    "train_cycle_ids",
    "test_cycle_ids",
    "train_sample_count",
    "test_sample_count",
    "proxy_test_mae_percentage_points",
    "proxy_test_rmse_percentage_points",
    "proxy_test_r2",
    "per_cycle_metrics_json",
    "model_parameters_json",
    "target_status",
]


def make_models() -> dict[str, Any]:
    """Create fresh estimators for one controlled model comparison."""
    return {
        "RidgeRegression": make_pipeline(
            StandardScaler(),
            Ridge(alpha=1.0),
        ),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=100,
            min_samples_leaf=2,
            max_leaf_nodes=128,
            max_features=1.0,
            random_state=SEED,
            n_jobs=-1,
        ),
        "ExtraTreesRegressor": ExtraTreesRegressor(
            n_estimators=100,
            min_samples_leaf=2,
            max_leaf_nodes=128,
            max_features=1.0,
            random_state=SEED,
            n_jobs=-1,
        ),
    }


def compare_models(dataset_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fit candidates on seven cycles and evaluate on fixed whole-cycle holdouts."""
    features, targets, groups = load_proxy_csv(dataset_path)
    unique_groups = set(np.unique(groups).tolist())
    missing_holdouts = set(HOLDOUT_CYCLE_IDS).difference(unique_groups)
    if missing_holdouts:
        raise ValueError(
            f"Dataset is missing required holdout cycle(s): {sorted(missing_holdouts)}"
        )

    test_mask = np.isin(groups, HOLDOUT_CYCLE_IDS)
    train_indices = np.flatnonzero(~test_mask)
    test_indices = np.flatnonzero(test_mask)
    train_groups = sorted(np.unique(groups[train_indices]).tolist())
    test_groups = sorted(np.unique(groups[test_indices]).tolist())
    if not train_indices.size or not test_indices.size:
        raise ValueError("Both training and test sets must contain samples")
    if set(train_groups).intersection(test_groups):
        raise RuntimeError("A complete drive-cycle group leaked across the holdout")

    X_train, X_test = features[train_indices], features[test_indices]
    y_train, y_test = targets[train_indices], targets[test_indices]
    groups_test = groups[test_indices]
    rows: list[dict[str, Any]] = []
    for model_name, model in make_models().items():
        model.fit(X_train, y_train)
        predictions = np.asarray(model.predict(X_test), dtype=float)
        if not np.isfinite(predictions).all():
            raise ValueError(f"{model_name} produced non-finite predictions")

        per_cycle: dict[str, dict[str, float | int]] = {}
        for cycle_id in test_groups:
            mask = groups_test == cycle_id
            cycle_truth = y_test[mask]
            cycle_predictions = predictions[mask]
            per_cycle[cycle_id] = {
                "sample_count": int(mask.sum()),
                "mae_percentage_points": float(
                    mean_absolute_error(cycle_truth, cycle_predictions)
                ),
                "rmse_percentage_points": float(
                    np.sqrt(mean_squared_error(cycle_truth, cycle_predictions))
                ),
                "r2": float(r2_score(cycle_truth, cycle_predictions)),
            }

        rows.append(
            {
                "model": model_name,
                "train_cycle_ids": ";".join(train_groups),
                "test_cycle_ids": ";".join(test_groups),
                "train_sample_count": int(train_indices.size),
                "test_sample_count": int(test_indices.size),
                "proxy_test_mae_percentage_points": float(
                    mean_absolute_error(y_test, predictions)
                ),
                "proxy_test_rmse_percentage_points": float(
                    np.sqrt(mean_squared_error(y_test, predictions))
                ),
                "proxy_test_r2": float(r2_score(y_test, predictions)),
                "per_cycle_metrics_json": json.dumps(
                    per_cycle,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ),
                "model_parameters_json": json.dumps(
                    MODEL_PARAMETERS[model_name],
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ),
                "target_status": (
                    "approximate capacity-referenced coulomb-counting proxy; "
                    "not validated physical SoC"
                ),
            }
        )

    metadata = {
        "dataset": str(dataset_path),
        "target": TARGET,
        "target_status": "approximate_capacity_referenced_coulomb_counting_proxy",
        "input_features": list(MODEL_FEATURES),
        "excluded_target_source_channel": "Ah",
        "split_method": "fixed complete-cycle holdout",
        "random_state": SEED,
        "train_cycle_ids": train_groups,
        "test_cycle_ids": test_groups,
        "train_sample_count": int(train_indices.size),
        "test_sample_count": int(test_indices.size),
        "candidates": list(MODEL_PARAMETERS),
        "scaling": (
            "Ridge uses StandardScaler within its training-only pipeline; "
            "tree models use raw features."
        ),
        "selection_status": (
            "No model selected. Compare metrics and limitations before the "
            "separate model-selection phase."
        ),
        "evaluation_scope": (
            "Held-out-cycle agreement with a derived proxy only. The holdout "
            "includes Cycle 4, which has proxy values outside [0, 100]. This "
            "does not establish physical SoC accuracy or generalization to "
            "other cells or temperatures."
        ),
        "physical_hardware_tested": False,
        "physical_power_measured": False,
    }
    return rows, metadata


def write_comparison(
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    comparison_path: Path,
    metadata_path: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Write the comparison CSV and split/provenance metadata."""
    existing = [path for path in (comparison_path, metadata_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing output(s): "
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Ridge, Random Forest, and Extra Trees on the same held-out "
            "Panasonic drive cycles. Results are proxy-agreement metrics."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing comparison outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dataset.is_file():
        raise FileNotFoundError(f"Prepared proxy dataset not found: {args.dataset}")
    existing = [
        path for path in (args.comparison, args.metadata) if path.exists()
    ]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing output(s): "
            + ", ".join(str(path) for path in existing)
            + "; pass --overwrite to replace them."
        )

    rows, metadata = compare_models(args.dataset)
    write_comparison(
        rows,
        metadata,
        args.comparison,
        args.metadata,
        overwrite=args.overwrite,
    )
    print("Model comparison complete; no winner selected.")
    for row in rows:
        print(
            f"{row['model']}: MAE {row['proxy_test_mae_percentage_points']:.3f}, "
            f"RMSE {row['proxy_test_rmse_percentage_points']:.3f}, "
            f"R² {row['proxy_test_r2']:.4f} proxy percentage points."
        )
    print(f"Comparison: {args.comparison}")
    print(f"Metadata: {args.metadata}")
    print("Interpretation: held-out-cycle proxy agreement, not validated SoC.")


if __name__ == "__main__":
    main()
