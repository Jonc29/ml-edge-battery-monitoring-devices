"""Audit whether Panasonic drive-cycle Ah measurements support an SoC target."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np

from src.data.panasonic import list_mat_members, load_measurements

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARCHIVE = PROJECT_ROOT / "data/raw/panasonic-18650pf-v1.zip"
DEFAULT_OUTPUT = PROJECT_ROOT / "results/metrics/panasonic_soc_target_audit.json"


def select_25c_drive_cycles(members: list[str]) -> list[str]:
    """Select the separate 25 °C drive-cycle files, not the combined log."""
    selected = []
    for member in members:
        path = Path(member)
        normalized_parts = {part.casefold() for part in path.parts}
        if "25degc" not in normalized_parts or "drive cycles" not in normalized_parts:
            continue
        if "us06_hwfet_udds_la92" in path.name.casefold():
            continue
        selected.append(member)
    return sorted(selected)


def select_25c_capacity_comparisons(members: list[str]) -> list[str]:
    """Select start/end 1C discharge files that may inform capacity drift."""
    selected = []
    for member in members:
        path = Path(member)
        normalized_parts = [part.casefold() for part in path.parts]
        if "25degc" not in normalized_parts:
            continue
        if not any("1c discharge tests_" in part for part in normalized_parts):
            continue
        if path.name.casefold().endswith(("dis1c_1.mat", "dis1c_2.mat")):
            selected.append(member)
    return sorted(selected)


def audit_ah_current_consistency(
    member: str,
    measurements: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Compare recorded Ah change with current integrated over elapsed time."""
    time_seconds = np.asarray(measurements["Time"], dtype=float)
    current_amperes = np.asarray(measurements["Current"], dtype=float)
    amp_hours = np.asarray(measurements["Ah"], dtype=float)
    voltage = np.asarray(measurements["Voltage"], dtype=float)

    if not (time_seconds.size == current_amperes.size == amp_hours.size == voltage.size):
        raise ValueError(f"{member}: time, current, Ah, and voltage lengths differ")
    if time_seconds.size < 2:
        raise ValueError(f"{member}: at least two observations are required")
    if not all(
        np.isfinite(values).all()
        for values in (time_seconds, current_amperes, amp_hours, voltage)
    ):
        raise ValueError(f"{member}: required channels contain non-finite values")
    if np.any(np.diff(time_seconds) < 0):
        raise ValueError(f"{member}: elapsed time must not move backwards")

    ah_change = float(amp_hours[-1] - amp_hours[0])
    integrated_current_ah = float(
        np.trapezoid(current_amperes, time_seconds) / 3600.0
    )
    difference = float(ah_change - integrated_current_ah)
    if ah_change < 0:
        direction = "net_discharge"
    elif ah_change > 0:
        direction = "net_charge"
    else:
        direction = "no_net_Ah_change"

    return {
        "member": member,
        "observations": int(time_seconds.size),
        "elapsed_time_seconds": float(time_seconds[-1] - time_seconds[0]),
        "ah_start": float(amp_hours[0]),
        "ah_end": float(amp_hours[-1]),
        "ah_change": ah_change,
        "integrated_current_ah": integrated_current_ah,
        "ah_minus_integrated_current_ah": difference,
        "absolute_difference_ah": abs(difference),
        "net_direction_from_ah": direction,
        "voltage_start_v": float(voltage[0]),
        "voltage_end_v": float(voltage[-1]),
    }


def audit_discharge_reference(member: str, measurements: dict[str, np.ndarray]) -> dict[str, Any]:
    """Summarize measured throughput in a 25 °C 1C discharge reference file."""
    amp_hours = np.asarray(measurements["Ah"], dtype=float)
    voltage = np.asarray(measurements["Voltage"], dtype=float)
    if amp_hours.size != voltage.size or amp_hours.size < 2:
        raise ValueError(f"{member}: Ah and voltage must have matching multi-row data")
    if not np.isfinite(amp_hours).all() or not np.isfinite(voltage).all():
        raise ValueError(f"{member}: Ah or voltage contains non-finite values")

    test_period = (
        "end_of_tests"
        if any("end_of_tests" in part.casefold() for part in Path(member).parts)
        else "start_of_tests"
    )
    return {
        "member": member,
        "test_period": test_period,
        "observations": int(amp_hours.size),
        "ah_start": float(amp_hours[0]),
        "ah_end": float(amp_hours[-1]),
        "discharge_throughput_ah": abs(float(amp_hours[-1] - amp_hours[0])),
        "minimum_voltage_v": float(voltage.min()),
        "per_drive_cycle_capacity_reference": False,
    }


def build_target_audit(archive_path: Path) -> dict[str, Any]:
    """Audit drive-cycle Ah/current consistency without asserting absolute SoC."""
    with ZipFile(archive_path) as archive:
        all_members = list_mat_members(archive)
        members = select_25c_drive_cycles(all_members)
        if not members:
            raise ValueError("No separate 25 °C drive-cycle MATLAB files found")
        rows = [
            audit_ah_current_consistency(
                member,
                load_measurements(archive, member),
            )
            for member in members
        ]
        capacity_members = select_25c_capacity_comparisons(all_members)
        capacity_rows = [
            audit_discharge_reference(
                member,
                load_measurements(archive, member),
            )
            for member in capacity_members
        ]

    total_observations = sum(row["observations"] for row in rows)
    maximum_difference = max(row["absolute_difference_ah"] for row in rows)
    return {
        "archive": archive_path.name,
        "audit_scope": (
            "Separate 25 °C drive-cycle files. Combined continuous log excluded "
            "to avoid auditing its duplicated drive-cycle observations."
        ),
        "files_audited": len(rows),
        "observations_audited": total_observations,
        "maximum_absolute_ah_current_difference_ah": maximum_difference,
        "files": rows,
        "capacity_comparison_scope": (
            "Measured throughput from named 25 °C 1C discharge records. "
            "These are comparison runs, not capacity measurements for each "
            "drive-cycle file."
        ),
        "capacity_comparison_runs": capacity_rows,
        "target_decision": {
            "status": "absolute_soc_not_validated",
            "ah_current_consistency_audit": "completed",
            "model_training_ready": False,
            "reason": (
                "Agreement between recorded Ah and integrated current checks "
                "internal charge accounting, not the initial SoC or the "
                "capacity denominator required for absolute SoC. Other 1C "
                "discharge files provide capacity comparisons but do not "
                "establish a per-drive-cycle denominator. The archive does not "
                "provide a validated per-row SoC label."
            ),
            "input_feature_constraint": (
                "Do not use Ah as an input if a future target is derived from Ah."
            ),
        },
        "physical_hardware_tested": False,
        "physical_power_measured": False,
    }


def write_audit(
    audit: dict[str, Any],
    output_path: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Write JSON audit results, refusing replacement unless requested."""
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite {output_path}; pass --overwrite to replace it."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(audit, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit Panasonic Ah/current agreement. This does not create an "
            "absolute SoC label or train a model."
        )
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=DEFAULT_ARCHIVE,
        help="Path to the downloaded Panasonic ZIP archive.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path for the JSON audit output.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output file if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.archive.is_file():
        raise FileNotFoundError(f"Dataset archive not found: {args.archive}")

    audit = build_target_audit(args.archive)
    write_audit(audit, args.output, overwrite=args.overwrite)
    print(f"Audited {audit['files_audited']} separate 25 °C drive-cycle files.")
    print(
        "Maximum absolute Ah/current integration difference: "
        f"{audit['maximum_absolute_ah_current_difference_ah']:.6f} Ah"
    )
    print(f"Audit: {args.output}")
    print("Absolute SoC target: not validated; model training: not performed.")


if __name__ == "__main__":
    main()
