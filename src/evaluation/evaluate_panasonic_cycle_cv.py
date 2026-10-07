"""Evaluate model agreement with the proxy using leave-one-cycle-out folds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data.prepare_panasonic_soc_proxy import DEFAULT_DATASET
from src.models.compare_panasonic_models import MODEL_PARAMETERS, make_models
from src.models.train_panasonic_soc_proxy_baseline import load_proxy_csv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = PROJECT_ROOT / "results/comparisons/panasonic_leave_one_cycle_out.csv"
DEFAULT_METRICS = (
    PROJECT_ROOT / "results/metrics/panasonic_leave_one_cycle_out.json"
)
CSV_COLUMNS = [
    "model",
    "held_out_cycle_id",
    "train_cycle_ids",
    "train_sample_count",
    "test_sample_count",
    "proxy_test_mae_percentage_points",
    "proxy_test_rmse_percentage_points",
    "proxy_test_r2",
    "target_status",
]
TARGET_STATUS = "approximate capacity-referenced coulomb-counting proxy; not validated physical SoC"


def evaluate_leave_one_cycle_out(
    dataset_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Produce one out-of-fold prediction per sample for each model."""
    features, targets, groups = load_proxy_csv(dataset_path)
    cycle_ids = sorted(np.unique(groups).tolist())
    if len(cycle_ids) < 3:
        raise ValueError(
            "Leave-one-cycle-out evaluation requires at least three cycle groups"
        )

    prediction_buffers = {
        model_name: np.full(targets.shape, np.nan, dtype=float)
        for model_name in MODEL_PARAMETERS
    }
    fold_rows: list[dict[str, Any]] = []

    for cycle_id in cycle_ids:
        test_mask = groups == cycle_id
        train_mask = ~test_mask
        if not test_mask.any() or not train_mask.any():
            raise ValueError(f"Cycle split has an empty partition for {cycle_id}")
        if set(groups[train_mask]).intersection(set(groups[test_mask])):
            raise RuntimeError(f"Cycle group leaked across fold: {cycle_id}")

        train_cycles = sorted(np.unique(groups[train_mask]).tolist())
        for model_name, model in make_models().items():
            model.fit(features[train_mask], targets[train_mask])
            predictions = np.asarray(model.predict(features[test_mask]), dtype=float)
            if (
                predictions.shape != targets[test_mask].shape
                or not np.isfinite(predictions).all()
            ):
                raise ValueError(
                    f"{model_name} produced invalid predictions for {cycle_id}"
                )
            prediction_buffers[model_name][test_mask] = predictions
            fold_rows.append(
                {
                    "model": model_name,
                    "held_out_cycle_id": cycle_id,
                    "train_cycle_ids": ";".join(train_cycles),
                    "train_sample_count": int(train_mask.sum()),
                    "test_sample_count": int(test_mask.sum()),
                    "proxy_test_mae_percentage_points": float(
                        mean_absolute_error(targets[test_mask], predictions)
                    ),
                    "proxy_test_rmse_percentage_points": float(
                        np.sqrt(mean_squared_error(targets[test_mask], predictions))
                    ),
                    "proxy_test_r2": float(
                        r2_score(targets[test_mask], predictions)
                    ),
                    "target_status": TARGET_STATUS,
                }
            )

    model_summaries: dict[str, Any] = {}
    for model_name, predictions in prediction_buffers.items():
        if not np.isfinite(predictions).all():
            raise RuntimeError(
                f"{model_name} did not generate exactly one prediction per sample"
            )
        cycle_metrics = [
            row
            for row in fold_rows
            if row["model"] == model_name
        ]
        model_summaries[model_name] = {
            "out_of_fold_sample_count": int(targets.size),
            "pooled_out_of_fold_metrics": {
                "mae_percentage_points": float(
                    mean_absolute_error(targets, predictions)
                ),
                "rmse_percentage_points": float(
                    np.sqrt(mean_squared_error(targets, predictions))
                ),
                "r2": float(r2_score(targets, predictions)),
            },
            "unweighted_cycle_mean_metrics": {
                "mae_percentage_points": float(
                    np.mean(
                        [
                            row["proxy_test_mae_percentage_points"]
                            for row in cycle_metrics
                        ]
                    )
                ),
                "rmse_percentage_points": float(
                    np.mean(
                        [
                            row["proxy_test_rmse_percentage_points"]
                            for row in cycle_metrics
                        ]
                    )
                ),
                "r2": float(
                    np.mean([row["proxy_test_r2"] for row in cycle_metrics])
                ),
            },
            "worst_cycle_by_mae": max(
                cycle_metrics,
                key=lambda row: row["proxy_test_mae_percentage_points"],
            )["held_out_cycle_id"],
            "model_parameters": MODEL_PARAMETERS[model_name],
        }

    summary = {
        "dataset": str(dataset_path),
        "target": "soc_proxy_percent",
        "target_status": "approximate_capacity_referenced_coulomb_counting_proxy",
        "input_features": [
            "voltage_v",
            "current_a",
            "battery_temp_c",
            "elapsed_time_s",
        ],
        "split_method": "leave-one-complete-cycle-out",
        "cycle_count": len(cycle_ids),
        "fold_count": len(cycle_ids),
        "sample_count": int(targets.size),
        "cycle_ids": cycle_ids,
        "aggregation": (
            "Pooled metrics combine every out-of-fold sample. Unweighted cycle "
            "means weight each cycle equally and are reported separately."
        ),
        "models": model_summaries,
        "interpretation": (
            "All metrics measure agreement with the derived proxy, not "
            "validated physical SoC. The cycles in this dataset have already "
            "informed earlier model selection and optimization; this repeated "
            "grouped evaluation is a robustness analysis, not an independent "
            "confirmatory test. It covers one cell at 25 degrees C only."
        ),
        "physical_hardware_tested": False,
        "physical_power_measured": False,
    }
    return fold_rows, summary


def write_results(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    csv_path: Path,
    metrics_path: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Write per-cycle metrics and evaluation provenance."""
    existing = [path for path in (csv_path, metrics_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing output(s): "
            + ", ".join(str(path) for path in existing)
            + "; pass --overwrite to replace them."
        )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=CSV_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    metrics_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate proxy agreement with each Panasonic cycle held out once; "
            "this is not validated physical SoC accuracy."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing per-cycle CSV and summary JSON outputs.",
    )
    args = parser.parse_args()

    if not args.dataset.is_file():
        raise FileNotFoundError(f"Prepared proxy dataset not found: {args.dataset}")
    existing = [path for path in (args.csv, args.metrics) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing output(s): "
            + ", ".join(str(path) for path in existing)
            + "; pass --overwrite to replace them."
        )

    rows, summary = evaluate_leave_one_cycle_out(args.dataset)
    write_results(rows, summary, args.csv, args.metrics, overwrite=args.overwrite)
    print("Leave-one-cycle-out evaluation complete.")
    for model_name, result in summary["models"].items():
        metrics = result["pooled_out_of_fold_metrics"]
        print(
            f"{model_name}: pooled MAE {metrics['mae_percentage_points']:.3f}, "
            f"RMSE {metrics['rmse_percentage_points']:.3f}, "
            f"R² {metrics['r2']:.4f} proxy percentage points."
        )
    print(f"Per-cycle results: {args.csv}")
    print(f"Summary: {args.metrics}")
    print("Interpretation: proxy agreement only; not independent confirmation.")


if __name__ == "__main__":
    main()
