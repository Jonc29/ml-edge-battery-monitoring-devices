"""Measure how the two observed capacity references affect the SoC proxy."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "data/processed/panasonic_soc_proxy_25c_1hz.csv"
DEFAULT_METADATA = PROJECT_ROOT / "results/metrics/panasonic_soc_proxy_metadata.json"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "results/metrics/panasonic_soc_proxy_sensitivity.json"
)
REQUIRED_COLUMNS = {"cycle_id", "elapsed_time_s", "soc_proxy_percent"}


def build_sensitivity_report(
    rows: list[dict[str, str]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Recalculate the proxy using each measured capacity and their mean."""
    if not rows:
        raise ValueError("Proxy dataset contains no samples")

    references = metadata.get("capacity_reference_files")
    if not isinstance(references, list) or len(references) < 2:
        raise ValueError("Metadata must contain at least two capacity references")

    capacities: list[dict[str, Any]] = []
    for index, reference in enumerate(references, start=1):
        try:
            capacity_ah = float(reference["discharge_throughput_ah"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Capacity reference {index} has no valid discharge throughput"
            ) from error
        if not np.isfinite(capacity_ah) or capacity_ah <= 0:
            raise ValueError(
                f"Capacity reference {index} must be finite and greater than zero"
            )
        capacities.append(
            {
                "name": f"measured_reference_{index}",
                "capacity_ah": capacity_ah,
                "source": reference.get("member"),
            }
        )

    mean_capacity = float(np.mean([item["capacity_ah"] for item in capacities]))
    try:
        declared_mean = float(metadata["reference_capacity_ah"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Metadata has no valid reference_capacity_ah") from error
    if not np.isclose(mean_capacity, declared_mean, rtol=0.0, atol=1e-8):
        raise ValueError(
            "Metadata reference_capacity_ah does not match the mean of its "
            "capacity_reference_files"
        )

    capacities.append(
        {
            "name": "mean_reference",
            "capacity_ah": mean_capacity,
            "source": "Arithmetic mean of measured references",
        }
    )

    missing_columns = REQUIRED_COLUMNS.difference(rows[0])
    if missing_columns:
        raise ValueError(
            f"Proxy dataset is missing columns: {sorted(missing_columns)}"
        )

    cycles: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row_number, row in enumerate(rows, start=2):
        try:
            cycle_id = row["cycle_id"]
            elapsed_time = float(row["elapsed_time_s"])
            soc_proxy = float(row["soc_proxy_percent"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Proxy dataset contains invalid values on row {row_number}"
            ) from error
        if not cycle_id:
            raise ValueError(f"Proxy dataset has an empty cycle_id on row {row_number}")
        if not np.isfinite(elapsed_time) or not np.isfinite(soc_proxy):
            raise ValueError(
                f"Proxy dataset contains non-finite values on row {row_number}"
            )
        cycles[cycle_id].append((elapsed_time, soc_proxy))

    baseline_proxy: list[np.ndarray] = []
    for cycle_id, samples in cycles.items():
        samples.sort(key=lambda sample: sample[0])
        times = np.asarray([sample[0] for sample in samples], dtype=float)
        if np.any(np.diff(times) < 0):
            raise ValueError(f"{cycle_id}: elapsed time must not move backwards")
        baseline_proxy.append(
            np.asarray([sample[1] for sample in samples], dtype=float)
        )

    baseline = np.concatenate(baseline_proxy)
    result_scenarios: list[dict[str, Any]] = []
    for scenario in capacities:
        capacity_ah = scenario["capacity_ah"]
        cycle_summaries: list[dict[str, Any]] = []
        scenario_proxy_all: list[np.ndarray] = []

        for cycle_id, samples in cycles.items():
            samples.sort(key=lambda sample: sample[0])
            times = np.asarray([sample[0] for sample in samples], dtype=float)
            baseline_cycle = np.asarray(
                [sample[1] for sample in samples],
                dtype=float,
            )
            relative_ah = (baseline_cycle / 100.0 - 1.0) * mean_capacity
            scenario_proxy = 100.0 * (1.0 + relative_ah / capacity_ah)
            scenario_proxy_all.append(scenario_proxy)

            difference = np.abs(scenario_proxy - baseline_cycle)
            out_of_range = (scenario_proxy < 0.0) | (scenario_proxy > 100.0)
            cycle_summaries.append(
                {
                    "cycle_id": cycle_id,
                    "sample_count": int(scenario_proxy.size),
                    "duration_seconds": float(times[-1] - times[0]),
                    "proxy_start_percent": float(scenario_proxy[0]),
                    "proxy_end_percent": float(scenario_proxy[-1]),
                    "proxy_min_percent": float(scenario_proxy.min()),
                    "proxy_max_percent": float(scenario_proxy.max()),
                    "out_of_range_samples": int(np.count_nonzero(out_of_range)),
                    "mean_absolute_change_from_mean_reference_percentage_points": (
                        float(difference.mean())
                    ),
                    "maximum_absolute_change_from_mean_reference_percentage_points": (
                        float(difference.max())
                    ),
                }
            )

        scenario_proxy_flat = np.concatenate(scenario_proxy_all)
        absolute_change = np.abs(scenario_proxy_flat - baseline)
        out_of_range_count = int(
            np.count_nonzero(
                (scenario_proxy_flat < 0.0) | (scenario_proxy_flat > 100.0)
            )
        )
        result_scenarios.append(
            {
                **scenario,
                "sample_count": int(scenario_proxy_flat.size),
                "proxy_min_percent": float(scenario_proxy_flat.min()),
                "proxy_max_percent": float(scenario_proxy_flat.max()),
                "out_of_range_samples": out_of_range_count,
                "out_of_range_fraction": float(
                    out_of_range_count / scenario_proxy_flat.size
                ),
                "mean_absolute_change_from_mean_reference_percentage_points": (
                    float(absolute_change.mean())
                ),
                "maximum_absolute_change_from_mean_reference_percentage_points": (
                    float(absolute_change.max())
                ),
                "cycles": cycle_summaries,
            }
        )

    return {
        "dataset": metadata.get("dataset"),
        "baseline_capacity_ah": mean_capacity,
        "baseline_sample_count": int(baseline.size),
        "capacity_scenarios": result_scenarios,
        "interpretation": (
            "Sensitivity analysis only: the alternative proxies rescale the "
            "relative Ah changes reconstructed from the existing mean-reference "
            "proxy. The measured capacity references are not per-cycle "
            "capacities or independent SoC labels. Differences are not a "
            "confidence interval and do not establish physical SoC accuracy."
        ),
        "physical_hardware_tested": False,
        "physical_power_measured": False,
    }


def read_inputs(
    dataset_path: Path,
    metadata_path: Path,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Load the prepared proxy CSV and its reference-capacity metadata."""
    with dataset_path.open(newline="", encoding="utf-8") as input_file:
        rows = list(csv.DictReader(input_file))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"Proxy metadata must be a JSON object: {metadata_path}")
    return rows, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the proxy created with each measured 1C capacity reference "
            "and their mean. This does not validate physical SoC."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the sensitivity report if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.dataset, args.metadata):
        if not path.is_file():
            raise FileNotFoundError(f"Required input not found: {path}")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite {args.output}; pass --overwrite to replace it."
        )

    rows, metadata = read_inputs(args.dataset, args.metadata)
    report = build_sensitivity_report(rows, metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("Capacity-reference sensitivity audit complete.")
    for scenario in report["capacity_scenarios"]:
        print(
            f"{scenario['name']}: {scenario['capacity_ah']:.5f} Ah, "
            f"max change {scenario['maximum_absolute_change_from_mean_reference_percentage_points']:.3f} "
            "percentage points from mean-reference proxy, "
            f"{scenario['out_of_range_samples']:,} samples outside [0, 100]."
        )
    print(f"Report: {args.output}")
    print("Interpretation: proxy sensitivity only; not validated SoC accuracy.")


if __name__ == "__main__":
    main()
