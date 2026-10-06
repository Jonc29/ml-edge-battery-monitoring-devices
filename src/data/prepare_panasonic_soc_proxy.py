"""Create an explicitly approximate SoC reference for Panasonic 25 °C cycles."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np

from src.data.panasonic import list_mat_members, load_measurements
from src.data.validate_panasonic_target import (
    audit_discharge_reference,
    select_25c_drive_cycles,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARCHIVE = PROJECT_ROOT / "data/raw/panasonic-18650pf-v1.zip"
DEFAULT_DATASET = PROJECT_ROOT / "data/processed/panasonic_soc_proxy_25c_1hz.csv"
DEFAULT_METADATA = PROJECT_ROOT / "results/metrics/panasonic_soc_proxy_metadata.json"
MODEL_FEATURES = (
    "voltage_v",
    "current_a",
    "battery_temp_c",
    "elapsed_time_s",
)


def select_pre_drive_capacity_references(members: list[str]) -> list[str]:
    """Select only the 25 °C start-of-tests 1C discharge capacity records."""
    selected = []
    for member in members:
        path = Path(member)
        parts = [part.casefold() for part in path.parts]
        if "25degc" not in parts:
            continue
        if not any("1c discharge tests_start_of_tests" in part for part in parts):
            continue
        if path.name.casefold().endswith(("dis1c_1.mat", "dis1c_2.mat")):
            selected.append(member)
    return sorted(selected)


def derive_soc_proxy(
    ah_values: np.ndarray,
    *,
    reference_capacity_ah: float,
) -> np.ndarray:
    """Estimate relative SoC from full-charge start and measured Ah throughput."""
    ah = np.asarray(ah_values, dtype=float)
    if ah.ndim != 1 or ah.size == 0:
        raise ValueError("Ah values must be a non-empty one-dimensional array")
    if not np.isfinite(ah).all():
        raise ValueError("Ah values must be finite")
    if not np.isfinite(reference_capacity_ah) or reference_capacity_ah <= 0:
        raise ValueError("Reference capacity must be finite and greater than zero")

    discharged_ah = ah - ah[0]
    return 100.0 * (1.0 + discharged_ah / reference_capacity_ah)


def _sample_indices(time_seconds: np.ndarray, sample_period_seconds: float) -> np.ndarray:
    """Choose the first logged sample at each regular elapsed-time interval."""
    if not np.isfinite(sample_period_seconds) or sample_period_seconds <= 0:
        raise ValueError("Sample period must be finite and greater than zero")
    if time_seconds.ndim != 1 or time_seconds.size < 2:
        raise ValueError("Time must contain at least two observations")
    if not np.isfinite(time_seconds).all() or np.any(np.diff(time_seconds) < 0):
        raise ValueError("Time must be finite and must not move backwards")

    targets = np.arange(
        time_seconds[0],
        time_seconds[-1] + sample_period_seconds,
        sample_period_seconds,
    )
    indices = np.searchsorted(time_seconds, targets, side="left")
    indices = indices[indices < time_seconds.size]
    return np.unique(indices)


def build_soc_proxy_dataset(
    archive_path: Path,
    *,
    sample_period_seconds: float = 1.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build a downsampled proxy dataset using measured pre-test capacities."""
    with ZipFile(archive_path) as archive:
        members = list_mat_members(archive)
        cycle_members = select_25c_drive_cycles(members)
        capacity_members = select_pre_drive_capacity_references(members)
        if not cycle_members:
            raise ValueError("No separate 25 °C drive-cycle files found")
        if len(capacity_members) < 2:
            raise ValueError(
                "At least two measured 25 °C start-of-tests capacity references "
                "are required"
            )

        capacity_references = [
            audit_discharge_reference(
                member,
                load_measurements(archive, member),
            )
            for member in capacity_members
        ]
        if any(row["test_period"] != "start_of_tests" for row in capacity_references):
            raise ValueError("Only start-of-tests capacity measurements are valid here")

        reference_capacity_ah = float(
            np.mean(
                [
                    row["discharge_throughput_ah"]
                    for row in capacity_references
                ]
            )
        )

        rows: list[dict[str, Any]] = []
        observed_soc_min = float("inf")
        observed_soc_max = float("-inf")
        out_of_range_count = 0
        source_observation_count = 0
        cycle_summaries: list[dict[str, Any]] = []
        for member in cycle_members:
            measurements = load_measurements(archive, member)
            time_seconds = np.asarray(measurements["Time"], dtype=float)
            current = np.asarray(measurements["Current"], dtype=float)
            voltage = np.asarray(measurements["Voltage"], dtype=float)
            temperature = np.asarray(
                measurements["Battery_Temp_degC"],
                dtype=float,
            )
            ah = np.asarray(measurements["Ah"], dtype=float)
            if not all(
                np.isfinite(values).all()
                for values in (time_seconds, current, voltage, temperature, ah)
            ):
                raise ValueError(f"{member}: required data contains non-finite values")
            if np.any(np.diff(time_seconds) < 0):
                raise ValueError(f"{member}: elapsed time must not move backwards")

            soc_proxy = derive_soc_proxy(
                ah,
                reference_capacity_ah=reference_capacity_ah,
            )
            source_observation_count += int(time_seconds.size)
            observed_soc_min = min(observed_soc_min, float(soc_proxy.min()))
            observed_soc_max = max(observed_soc_max, float(soc_proxy.max()))
            cycle_out_of_range = int(
                np.count_nonzero((soc_proxy < 0.0) | (soc_proxy > 100.0))
            )
            out_of_range_count += cycle_out_of_range

            indices = _sample_indices(time_seconds, sample_period_seconds)
            group = Path(member).stem
            cycle_summaries.append(
                {
                    "cycle_id": group,
                    "source_observations": int(time_seconds.size),
                    "prepared_observations": int(indices.size),
                    "proxy_min_percent": float(soc_proxy.min()),
                    "proxy_max_percent": float(soc_proxy.max()),
                    "out_of_range_proxy_samples": cycle_out_of_range,
                }
            )
            for index in indices:
                rows.append(
                    {
                        "cycle_id": group,
                        "elapsed_time_s": float(time_seconds[index]),
                        "voltage_v": float(voltage[index]),
                        "current_a": float(current[index]),
                        "battery_temp_c": float(temperature[index]),
                        "soc_proxy_percent": float(soc_proxy[index]),
                    }
                )

    if not rows:
        raise ValueError("No samples were selected for the proxy dataset")
    metadata = {
        "dataset": "Panasonic 18650PF 25 °C separate drive-cycle files",
        "label_name": "soc_proxy_percent",
        "label_status": "approximate_capacity_referenced_coulomb_counting_proxy",
        "label_definition": (
            "100 * (1 + (Ah_sample - Ah_first_sample) / Q_reference_Ah)"
        ),
        "assumptions": [
            "Each drive cycle begins at approximately 100% SoC after the "
            "documented full-charge procedure.",
            "A fixed Q_reference is used for all selected 25 °C cycles.",
            "Q_reference is the arithmetic mean of measured throughput in the "
            "two available 25 °C start-of-tests 1C discharge reference files.",
        ],
        "reference_capacity_ah": reference_capacity_ah,
        "reference_capacity_range_ah": [
            min(row["discharge_throughput_ah"] for row in capacity_references),
            max(row["discharge_throughput_ah"] for row in capacity_references),
        ],
        "capacity_reference_files": capacity_references,
        "input_features": list(MODEL_FEATURES),
        "excluded_source_channel": (
            "Ah is used only to derive the target and is not included in the "
            "prepared feature columns."
        ),
        "sample_period_seconds": sample_period_seconds,
        "drive_cycle_files": cycle_members,
        "drive_cycle_count": len(cycle_members),
        "drive_cycle_summaries": cycle_summaries,
        "source_observations": source_observation_count,
        "prepared_observations": len(rows),
        "proxy_min_percent_before_downsampling": observed_soc_min,
        "proxy_max_percent_before_downsampling": observed_soc_max,
        "out_of_range_proxy_samples_before_downsampling": out_of_range_count,
        "out_of_range_policy": (
            "Values are not clipped to [0, 100]; they remain visible as evidence "
            "of reference-capacity and initial-state uncertainty."
        ),
        "limitations": [
            "This is a derived proxy, not an independently measured SoC label.",
            "The full initial SoC is assumed from the charge protocol and is not "
            "measured for every drive-cycle file.",
            "A shared early capacity reference cannot capture capacity fade "
            "between individual drive-cycle runs.",
            "The two measured capacity references differ; proxy sensitivity to "
            "that difference is not represented as a statistical confidence "
            "interval.",
            "The data cover one physical cell and a single ambient temperature.",
        ],
        "evaluation_claim": (
            "Any model metrics measure agreement with this derived proxy, not "
            "validated physical SoC accuracy."
        ),
        "physical_hardware_tested": False,
        "physical_power_measured": False,
    }
    return rows, metadata


def write_proxy_dataset(
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    dataset_path: Path,
    metadata_path: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Write the prepared CSV and provenance metadata safely."""
    existing = [path for path in (dataset_path, metadata_path) if path.exists()]
    if existing and not overwrite:
        paths = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Refusing to overwrite existing output(s): {paths}; "
            "pass --overwrite to replace them."
        )
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with dataset_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a derived Panasonic SoC proxy. This is not ground truth "
            "and does not train a model."
        )
    )
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument(
        "--sample-period-seconds",
        type=float,
        default=1.0,
        help="Regular output interval in seconds (default: 1.0).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the prepared dataset and metadata if they exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.archive.is_file():
        raise FileNotFoundError(f"Dataset archive not found: {args.archive}")
    rows, metadata = build_soc_proxy_dataset(
        args.archive,
        sample_period_seconds=args.sample_period_seconds,
    )
    write_proxy_dataset(
        rows,
        metadata,
        args.dataset,
        args.metadata,
        overwrite=args.overwrite,
    )
    print(f"Prepared {len(rows):,} samples across {metadata['drive_cycle_count']} runs.")
    print(f"Measured capacity reference: {metadata['reference_capacity_ah']:.5f} Ah")
    print(f"Dataset: {args.dataset}")
    print(f"Metadata: {args.metadata}")
    print("Label status: approximate proxy; not validated SoC ground truth.")


if __name__ == "__main__":
    main()
