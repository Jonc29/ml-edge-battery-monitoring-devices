"""Select a comparison winner for the next experimental phase."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.data.prepare_panasonic_soc_proxy import DEFAULT_DATASET
from src.models.compare_panasonic_models import (
    DEFAULT_COMPARISON,
    DEFAULT_METADATA as DEFAULT_COMPARISON_METADATA,
    MODEL_PARAMETERS,
    make_models,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = PROJECT_ROOT / "models/baseline/panasonic_soc_proxy_selected_rf.joblib"
DEFAULT_DECISION = PROJECT_ROOT / "results/metrics/panasonic_model_selection.json"


def select_candidate(comparison_rows: list[dict[str, str]]) -> dict[str, Any]:
    """Choose lowest aggregate MAE, then RMSE; require all comparison models."""
    if not comparison_rows:
        raise ValueError("Model comparison CSV contains no candidate rows")
    required = {
        "model",
        "proxy_test_mae_percentage_points",
        "proxy_test_rmse_percentage_points",
        "proxy_test_r2",
        "per_cycle_metrics_json",
    }
    missing_columns = required.difference(comparison_rows[0])
    if missing_columns:
        raise ValueError(f"Comparison CSV is missing columns: {sorted(missing_columns)}")

    expected = set(MODEL_PARAMETERS)
    actual = [row["model"] for row in comparison_rows]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError(
            f"Expected exactly one result for each model {sorted(expected)}; "
            f"found {actual}"
        )

    parsed_rows: list[dict[str, Any]] = []
    for row in comparison_rows:
        try:
            mae = float(row["proxy_test_mae_percentage_points"])
            rmse = float(row["proxy_test_rmse_percentage_points"])
            r2 = float(row["proxy_test_r2"])
            per_cycle = json.loads(row["per_cycle_metrics_json"])
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(
                f"Invalid comparison metrics for {row['model']}"
            ) from error
        if not np.isfinite([mae, rmse, r2]).all():
            raise ValueError(f"Non-finite comparison metrics for {row['model']}")
        if not isinstance(per_cycle, dict) or not per_cycle:
            raise ValueError(f"Missing per-cycle metrics for {row['model']}")
        parsed_rows.append(
            {
                **row,
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "per_cycle": per_cycle,
            }
        )

    ranked = sorted(
        parsed_rows,
        key=lambda row: (
            row["mae"],
            row["rmse"],
            -row["r2"],
            row["model"],
        ),
    )
    return ranked[0]


def train_selected_estimator(
    dataset_path: Path,
    train_cycle_ids: list[str],
    model_name: str,
) -> Any:
    """Refit the selected candidate using only comparison training cycles."""
    from src.models.train_panasonic_soc_proxy_baseline import load_proxy_csv

    features, targets, groups = load_proxy_csv(dataset_path)
    train_mask = np.isin(groups, train_cycle_ids)
    if not train_mask.any():
        raise ValueError("No samples found for the comparison training cycles")
    if set(np.unique(groups[~train_mask])).intersection(train_cycle_ids):
        raise RuntimeError("A holdout cycle appears in the selection training set")

    model = make_models().get(model_name)
    if model is None:
        raise ValueError(f"No estimator implementation found for {model_name}")
    model.fit(features[train_mask], targets[train_mask])
    return model


def build_decision(
    selected: dict[str, Any],
    candidates: list[dict[str, Any]],
    comparison_metadata: dict[str, Any],
    dataset_path: Path,
    model_path: Path,
) -> dict[str, Any]:
    """Record the evidence and guardrails behind this phase-6 choice."""
    return {
        "selected_model": selected["model"],
        "decision_status": "provisional_for_optimization_phase",
        "decision_criteria": [
            "Lowest aggregate MAE among candidates on identical held-out cycles.",
            "Use aggregate RMSE and R2 as supporting metrics.",
            "Inspect each held-out cycle, including Cycle 4.",
            "Keep the selected estimator trainable with the existing bounded tree settings.",
        ],
        "reason": (
            "Random Forest has the lowest aggregate MAE and RMSE and the "
            "highest R2 of the three candidates on this fixed holdout. It also "
            "has the lowest MAE on each of the three individual held-out cycles."
        ),
        "comparison_metrics": {
            "mae_percentage_points": selected["mae"],
            "rmse_percentage_points": selected["rmse"],
            "r2": selected["r2"],
            "per_cycle": selected["per_cycle"],
        },
        "other_candidates": [
            {
                "model": row["model"],
                "mae_percentage_points": row["mae"],
                "rmse_percentage_points": row["rmse"],
                "r2": row["r2"],
            }
            for row in sorted(candidates, key=lambda row: (row["mae"], row["rmse"]))
            if row["model"] != selected["model"]
        ],
        "model_parameters": MODEL_PARAMETERS[selected["model"]],
        "training_data": {
            "dataset": str(dataset_path),
            "train_cycle_ids": comparison_metadata["train_cycle_ids"],
            "test_cycle_ids": comparison_metadata["test_cycle_ids"],
            "train_sample_count": comparison_metadata["train_sample_count"],
            "test_sample_count": comparison_metadata["test_sample_count"],
            "estimator_refit_on_training_cycles_only": True,
        },
        "artifact": str(model_path),
        "target_status": "approximate_capacity_referenced_coulomb_counting_proxy",
        "limitations": [
            "This model choice is provisional and based on one fixed split of ten cycles from one cell at 25 °C.",
            "The choice uses the comparison holdout; its metrics are selection evidence and are not an independent final performance estimate.",
            "All reported scores measure agreement with the derived proxy, not validated physical SoC accuracy.",
            "No physical hardware, memory, latency, or power comparison has been done yet.",
        ],
        "next_phase": (
            "Optimize the selected Random Forest experimentally and compare "
            "accuracy against the same held-out cycles. Treat those results as "
            "exploratory because this holdout informed model selection."
        ),
        "physical_hardware_tested": False,
        "physical_power_measured": False,
    }


def read_comparison(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as input_file:
        rows = list(csv.DictReader(input_file))
    if not rows:
        raise ValueError(f"Model comparison contains no rows: {path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select the best candidate from the recorded fixed-holdout model "
            "comparison and refit it on comparison training cycles only."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument(
        "--comparison-metadata",
        type=Path,
        default=DEFAULT_COMPARISON_METADATA,
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--decision", type=Path, default=DEFAULT_DECISION)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for path in (args.dataset, args.comparison, args.comparison_metadata):
        if not path.is_file():
            raise FileNotFoundError(f"Required input not found: {path}")
    existing = [path for path in (args.model, args.decision) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing output(s): "
            + ", ".join(str(path) for path in existing)
            + "; pass --overwrite to replace them."
        )

    comparison_rows = read_comparison(args.comparison)
    selected = select_candidate(comparison_rows)
    candidates = [
        {
            "model": row["model"],
            "mae": float(row["proxy_test_mae_percentage_points"]),
            "rmse": float(row["proxy_test_rmse_percentage_points"]),
            "r2": float(row["proxy_test_r2"]),
        }
        for row in comparison_rows
    ]
    comparison_metadata = json.loads(
        args.comparison_metadata.read_text(encoding="utf-8")
    )
    if comparison_metadata.get("target") != "soc_proxy_percent":
        raise ValueError("Comparison metadata has an unexpected target")
    model = train_selected_estimator(
        args.dataset,
        comparison_metadata["train_cycle_ids"],
        selected["model"],
    )
    decision = build_decision(
        selected,
        candidates,
        comparison_metadata,
        args.dataset,
        args.model,
    )
    args.model.parent.mkdir(parents=True, exist_ok=True)
    args.decision.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model)
    args.decision.write_text(
        json.dumps(decision, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Selected {selected['model']} provisionally for the optimization phase."
    )
    print(
        f"Comparison MAE/RMSE: {selected['mae']:.3f}/"
        f"{selected['rmse']:.3f} proxy percentage points."
    )
    print(f"Training-only refit artifact: {args.model}")
    print(f"Decision record: {args.decision}")
    print("The comparison holdout informed selection; metrics are not independent.")


if __name__ == "__main__":
    main()
