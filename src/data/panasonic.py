"""Read and summarize the public Panasonic 18650PF MATLAB dataset."""

from __future__ import annotations

import csv
import json
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
from scipy.io import loadmat

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

MATLAB_SUFFIX = ".mat"
MEASUREMENT_KEY = "meas"
NUMERIC_CHANNELS = (
    "Voltage",
    "Current",
    "Ah",
    "Wh",
    "Power",
    "Battery_Temp_degC",
    "Time",
    "Chamber_Temp_degC",
)
EXPECTED_CHANNELS = (
    "TimeStamp",
    *NUMERIC_CHANNELS,
)


def list_mat_members(archive: ZipFile) -> list[str]:
    """Return MATLAB measurement file paths from the archive."""
    return sorted(
        name
        for name in archive.namelist()
        if name.lower().endswith(MATLAB_SUFFIX)
    )


def load_measurements(archive: ZipFile, member: str) -> dict[str, np.ndarray]:
    """Load one MATLAB file and validate that its measurement channels align."""
    matlab_file = loadmat(
        BytesIO(archive.read(member)),
        simplify_cells=True,
    )
    measurements: Any = matlab_file.get(MEASUREMENT_KEY)
    if not isinstance(measurements, dict):
        raise ValueError(f"{member}: expected a MATLAB '{MEASUREMENT_KEY}' struct")

    missing_channels = set(EXPECTED_CHANNELS).difference(measurements)
    if missing_channels:
        missing = ", ".join(sorted(missing_channels))
        raise ValueError(f"{member}: missing expected channels: {missing}")

    arrays = {
        channel: np.atleast_1d(np.asarray(values)).reshape(-1)
        for channel, values in measurements.items()
    }
    sample_counts = {values.size for values in arrays.values()}
    if len(sample_counts) != 1:
        counts = {channel: values.size for channel, values in arrays.items()}
        raise ValueError(f"{member}: measurement channels have unequal lengths: {counts}")

    return arrays


def _channel_statistics(values: np.ndarray) -> tuple[int, float | None, float | None]:
    numeric = np.asarray(values, dtype=float)
    finite = numeric[np.isfinite(numeric)]
    if finite.size == 0:
        return 0, None, None
    return int(finite.size), float(finite.min()), float(finite.max())


def _group_from_member(member: str) -> str:
    path = Path(member)
    if len(path.parts) < 2:
        return "unclassified"
    return "/".join(path.parts[1:-1]) or "root"


def build_inventory(archive_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Inspect all MATLAB files and return per-file and aggregate metadata."""
    rows: list[dict[str, Any]] = []
    channel_ranges: dict[str, dict[str, float | None]] = {
        channel: {"minimum": None, "maximum": None}
        for channel in NUMERIC_CHANNELS
    }
    total_observations = 0

    with ZipFile(archive_path) as archive:
        members = list_mat_members(archive)
        if not members:
            raise ValueError(f"No MATLAB files found in {archive_path}")

        for member in members:
            measurements = load_measurements(archive, member)
            time_values = np.asarray(measurements["Time"], dtype=float)
            finite_time = time_values[np.isfinite(time_values)]
            time_steps = np.diff(finite_time)
            positive_steps = time_steps[time_steps > 0]

            row: dict[str, Any] = {
                "member": member,
                "group": _group_from_member(member),
                "observations": int(time_values.size),
                "time_start_seconds": (
                    float(finite_time[0]) if finite_time.size else None
                ),
                "time_end_seconds": (
                    float(finite_time[-1]) if finite_time.size else None
                ),
                "median_positive_time_step_seconds": (
                    float(np.median(positive_steps)) if positive_steps.size else None
                ),
            }

            total_observations += time_values.size
            for channel in NUMERIC_CHANNELS:
                count, minimum, maximum = _channel_statistics(measurements[channel])
                row[f"{channel}_finite_count"] = count
                row[f"{channel}_minimum"] = minimum
                row[f"{channel}_maximum"] = maximum

                overall = channel_ranges[channel]
                if minimum is not None:
                    overall["minimum"] = (
                        minimum
                        if overall["minimum"] is None
                        else min(float(overall["minimum"]), minimum)
                    )
                    overall["maximum"] = (
                        maximum
                        if overall["maximum"] is None
                        else max(float(overall["maximum"]), float(maximum))
                    )
            rows.append(row)

        entry_counts = Counter(
            Path(name).suffix.lower() or "[no extension]"
            for name in archive.namelist()
            if not name.endswith("/")
        )

    summary = {
        "archive": str(archive_path),
        "matlab_file_count": len(rows),
        "file_row_observation_count": int(total_observations),
        "observation_count_note": (
            "Sum of per-file rows. The dataset readme documents duplicated "
            "drive-cycle records in contiguous and split files; this is not "
            "a unique-observation count."
        ),
        "archive_file_count_by_extension": dict(sorted(entry_counts.items())),
        "measurement_channels": list(EXPECTED_CHANNELS),
        "aggregate_numeric_channel_ranges": channel_ranges,
        "groups": dict(sorted(Counter(row["group"] for row in rows).items())),
        "dataset_target_status": (
            "No validated per-row SoC target has been established. Ah-based "
            "target reconstruction requires separate validation."
        ),
        "experiment_type": "public-dataset offline software exploration",
        "physical_hardware_tested": False,
        "physical_power_measured": False,
    }
    return rows, summary


def write_inventory(
    rows: list[dict[str, Any]],
    output_path: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Write per-file inventory CSV, refusing to replace results by default."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite {output_path}; pass --overwrite to replace it."
        )
    if not rows:
        raise ValueError("Cannot write an empty dataset inventory")

    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(
    summary: dict[str, Any],
    output_path: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Write JSON summary, refusing to replace results by default."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite {output_path}; pass --overwrite to replace it."
        )
    output_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def plot_drive_cycle(
    archive_path: Path,
    member_suffix: str,
    output_path: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Plot measured signals from one selected drive-cycle file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite {output_path}; pass --overwrite to replace it."
        )

    with ZipFile(archive_path) as archive:
        matches = [
            member
            for member in list_mat_members(archive)
            if member.endswith(member_suffix)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one archive member ending in {member_suffix!r}; "
                f"found {len(matches)}"
            )
        member = matches[0]
        measurements = load_measurements(archive, member)

    time_seconds = np.asarray(measurements["Time"], dtype=float)
    figure, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(time_seconds, measurements["Voltage"], linewidth=0.7)
    axes[0].set_ylabel("Voltage (V)")
    axes[1].plot(time_seconds, measurements["Current"], linewidth=0.7)
    axes[1].set_ylabel("Current (A)")
    axes[2].plot(
        time_seconds,
        measurements["Battery_Temp_degC"],
        linewidth=0.7,
        label="Battery case",
    )
    axes[2].plot(
        time_seconds,
        measurements["Chamber_Temp_degC"],
        linewidth=0.7,
        label="Chamber",
    )
    axes[2].set_ylabel("Temperature (°C)")
    axes[2].set_xlabel("Elapsed time within file (s)")
    axes[2].legend()
    figure.suptitle(f"Measured signals: {Path(member).name}\nOffline dataset exploration")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
