"""Benchmark host-side latency, serialized size, and process RSS for RF models."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.data.prepare_panasonic_soc_proxy import DEFAULT_DATASET, MODEL_FEATURES
from src.models.compare_panasonic_models import (
    DEFAULT_METADATA as DEFAULT_COMPARISON_METADATA,
)
from src.models.train_panasonic_soc_proxy_baseline import load_proxy_csv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASELINE = PROJECT_ROOT / "models/baseline/panasonic_soc_proxy_selected_rf.joblib"
DEFAULT_OPTIMIZED = PROJECT_ROOT / "models/optimized/panasonic_rf_50_trees.joblib"
DEFAULT_JSON = PROJECT_ROOT / "results/metrics/panasonic_resource_benchmarks.json"
DEFAULT_CSV = PROJECT_ROOT / "results/comparisons/panasonic_resource_benchmarks.csv"
BATCH_CONFIG = {
    1: {"warmup_calls": 50, "measured_calls": 500},
    32: {"warmup_calls": 20, "measured_calls": 200},
    256: {"warmup_calls": 10, "measured_calls": 100},
}
CSV_COLUMNS = [
    "model",
    "artifact",
    "artifact_size_bytes",
    "batch_size",
    "warmup_calls",
    "measured_calls",
    "latency_median_ms_per_batch",
    "latency_p95_ms_per_batch",
    "latency_p99_ms_per_batch",
    "latency_mean_ms_per_batch",
    "median_latency_us_per_sample",
    "median_throughput_samples_per_second",
    "rss_before_dataset_load_bytes",
    "rss_after_dataset_load_bytes",
    "rss_after_model_load_bytes",
    "rss_after_inference_bytes",
    "rss_change_during_dataset_load_bytes",
    "rss_change_during_model_load_bytes",
    "rss_high_water_before_dataset_load_bytes",
    "rss_high_water_after_dataset_load_bytes",
    "rss_high_water_after_model_load_bytes",
    "rss_high_water_after_inference_bytes",
    "rss_high_water_bytes",
    "rss_high_water_change_during_model_load_bytes",
]


def read_rss_bytes() -> tuple[int, int]:
    """Read current and high-water RSS from Linux procfs."""
    status_path = Path("/proc/self/status")
    if not status_path.is_file():
        raise OSError("RSS measurement requires Linux /proc/self/status")
    memory_kib: dict[str, int] = {}
    for line in status_path.read_text(encoding="utf-8").splitlines():
        if line.startswith(("VmRSS:", "VmHWM:")):
            label, value, unit = line.split()
            if unit != "kB":
                raise ValueError(f"Unexpected /proc memory unit: {unit}")
            memory_kib[label.rstrip(":")] = int(value)
    if "VmRSS" not in memory_kib or "VmHWM" not in memory_kib:
        raise ValueError("Could not read VmRSS and VmHWM from /proc/self/status")
    return memory_kib["VmRSS"] * 1024, memory_kib["VmHWM"] * 1024


def build_latency_batches(
    test_features: np.ndarray,
    batch_sizes: tuple[int, ...] = tuple(BATCH_CONFIG),
) -> dict[int, list[np.ndarray]]:
    """Prebuild representative batches so data slicing is outside timed calls."""
    if test_features.ndim != 2 or test_features.shape[0] == 0:
        raise ValueError("Test features must be a non-empty two-dimensional array")
    batches: dict[int, list[np.ndarray]] = {}
    for batch_size in batch_sizes:
        if batch_size < 1:
            raise ValueError("Batch sizes must be positive")
        if batch_size > test_features.shape[0]:
            raise ValueError(
                f"Batch size {batch_size} exceeds available held-out rows "
                f"({test_features.shape[0]})"
            )
        start_indices = np.linspace(
            0,
            test_features.shape[0] - batch_size,
            num=min(16, max(1, test_features.shape[0] - batch_size + 1)),
            dtype=int,
        )
        batches[batch_size] = [
            np.ascontiguousarray(test_features[start : start + batch_size])
            for start in np.unique(start_indices)
        ]
    return batches


def summarize_latencies(
    durations_ns: list[int],
    *,
    batch_size: int,
    warmup_calls: int,
) -> dict[str, float | int]:
    """Convert prediction durations to reproducible batch and per-sample stats."""
    if not durations_ns or batch_size < 1:
        raise ValueError("Durations and a positive batch size are required")
    durations_ms = np.asarray(durations_ns, dtype=np.float64) / 1_000_000.0
    if not np.isfinite(durations_ms).all() or np.any(durations_ms <= 0):
        raise ValueError("Latency durations must be finite and positive")
    median_ms = float(np.median(durations_ms))
    return {
        "batch_size": batch_size,
        "warmup_calls": warmup_calls,
        "measured_calls": int(durations_ms.size),
        "latency_median_ms_per_batch": median_ms,
        "latency_p95_ms_per_batch": float(np.percentile(durations_ms, 95)),
        "latency_p99_ms_per_batch": float(np.percentile(durations_ms, 99)),
        "latency_mean_ms_per_batch": float(np.mean(durations_ms)),
        "median_latency_us_per_sample": median_ms * 1000.0 / batch_size,
        "median_throughput_samples_per_second": 1000.0 * batch_size / median_ms,
    }


def _worker_benchmark(
    model_name: str,
    artifact_path: Path,
    dataset_path: Path,
    comparison_metadata_path: Path,
) -> dict[str, Any]:
    """Benchmark a single model in an isolated Python subprocess."""
    comparison_metadata = json.loads(
        comparison_metadata_path.read_text(encoding="utf-8")
    )
    if comparison_metadata.get("input_features") != list(MODEL_FEATURES):
        raise ValueError("Comparison metadata feature order does not match model inputs")
    if comparison_metadata.get("target") != "soc_proxy_percent":
        raise ValueError("Comparison metadata has an unexpected target")

    rss_before_dataset, hwm_before_dataset = read_rss_bytes()
    features, _, groups = load_proxy_csv(dataset_path)
    train_cycle_ids = comparison_metadata.get("train_cycle_ids")
    test_cycle_ids = comparison_metadata.get("test_cycle_ids")
    if (
        not isinstance(train_cycle_ids, list)
        or not train_cycle_ids
        or not isinstance(test_cycle_ids, list)
        or not test_cycle_ids
    ):
        raise ValueError("Comparison metadata must list training and held-out cycles")
    if set(train_cycle_ids).intersection(test_cycle_ids):
        raise ValueError("Comparison metadata leaks cycles across the split")
    test_mask = np.isin(groups, test_cycle_ids)
    if not test_mask.any():
        raise ValueError("No held-out rows were found in the dataset")
    present_groups = set(np.unique(groups).tolist())
    declared_groups = set(train_cycle_ids) | set(test_cycle_ids)
    missing_groups = declared_groups.difference(present_groups)
    if missing_groups:
        raise ValueError(f"Dataset is missing split cycles: {sorted(missing_groups)}")
    unexpected_groups = present_groups.difference(declared_groups)
    if unexpected_groups:
        raise ValueError(f"Comparison split omits cycles: {sorted(unexpected_groups)}")
    test_features = features[test_mask]
    batches = build_latency_batches(test_features)

    rss_after_dataset, hwm_after_dataset = read_rss_bytes()
    model = joblib.load(artifact_path)
    rss_after_model, hwm_after_model = read_rss_bytes()

    latency_results: list[dict[str, Any]] = []
    prediction_checks: list[dict[str, Any]] = []
    for batch_size, config in BATCH_CONFIG.items():
        batch_candidates = batches[batch_size]
        for warmup_index in range(config["warmup_calls"]):
            output = np.asarray(
                model.predict(batch_candidates[warmup_index % len(batch_candidates)]),
                dtype=float,
            )
            if not np.isfinite(output).all():
                raise ValueError(f"{model_name} emitted a non-finite prediction")

        durations_ns: list[int] = []
        for call_index in range(config["measured_calls"]):
            batch = batch_candidates[call_index % len(batch_candidates)]
            start_ns = time.perf_counter_ns()
            output = np.asarray(model.predict(batch), dtype=float)
            elapsed_ns = time.perf_counter_ns() - start_ns
            if not np.isfinite(output).all():
                raise ValueError(f"{model_name} emitted a non-finite prediction")
            durations_ns.append(elapsed_ns)

        summary = summarize_latencies(
            durations_ns,
            batch_size=batch_size,
            warmup_calls=config["warmup_calls"],
        )
        latency_results.append(summary)
        prediction_checks.append(
            {
                "batch_size": batch_size,
                "finite_predictions": True,
                "last_prediction_count": int(output.size),
            }
        )

    rss_after_inference, hwm_after_inference = read_rss_bytes()
    return {
        "model": model_name,
        "artifact": str(artifact_path),
        "artifact_size_bytes": artifact_path.stat().st_size,
        "held_out_sample_count": int(test_features.shape[0]),
        "test_cycle_ids": test_cycle_ids,
        "feature_order": list(MODEL_FEATURES),
        "latency": latency_results,
        "memory": {
            "method": (
                "Linux /proc/self/status current VmRSS and VmHWM, sampled in an "
                "isolated subprocess after loading the test data and again "
                "after model load/inference."
            ),
            "rss_before_dataset_load_bytes": rss_before_dataset,
            "rss_after_dataset_load_bytes": rss_after_dataset,
            "rss_after_model_load_bytes": rss_after_model,
            "rss_after_inference_bytes": rss_after_inference,
            "rss_change_during_dataset_load_bytes": (
                rss_after_dataset - rss_before_dataset
            ),
            "rss_change_during_model_load_bytes": rss_after_model - rss_after_dataset,
            "rss_high_water_before_dataset_load_bytes": hwm_before_dataset,
            "rss_high_water_after_dataset_load_bytes": hwm_after_dataset,
            "rss_high_water_after_model_load_bytes": hwm_after_model,
            "rss_high_water_after_inference_bytes": hwm_after_inference,
            "rss_high_water_bytes": max(
                hwm_before_dataset,
                hwm_after_dataset,
                hwm_after_model,
                hwm_after_inference,
            ),
            "rss_high_water_change_during_model_load_bytes": (
                hwm_after_model - hwm_after_dataset
            ),
            "interpretation": (
                "Process RSS is a host-side memory indicator, not exact model "
                "RAM usage. Dataset/model deltas include runtime allocations "
                "and may be negative as pages are released; the high-water "
                "change can miss short-lived peaks."
            ),
        },
        "prediction_checks": prediction_checks,
        "physical_hardware_tested": False,
        "physical_power_measured": False,
    }


def _cpu_description() -> str | None:
    cpuinfo_path = Path("/proc/cpuinfo")
    if not cpuinfo_path.is_file():
        return None
    for line in cpuinfo_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("model name"):
            return line.split(":", maxsplit=1)[1].strip()
    return None


def _run_worker(
    model_name: str,
    artifact_path: Path,
    dataset_path: Path,
    comparison_metadata_path: Path,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "src.evaluation.benchmark_panasonic_models",
        "--worker",
        "--model-name",
        model_name,
        "--artifact",
        str(artifact_path),
        "--dataset",
        str(dataset_path),
        "--comparison-metadata",
        str(comparison_metadata_path),
    ]
    process = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"Benchmark worker failed for {model_name} with exit code "
            f"{process.returncode}: {process.stderr.strip()}"
        )
    try:
        result = json.loads(process.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"Benchmark worker returned invalid JSON for {model_name}: "
            f"{process.stdout[:500]}"
        ) from error
    return result


def build_benchmark_report(
    worker_results: list[dict[str, Any]],
    comparison_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Combine isolated worker results with host and method provenance."""
    if not worker_results:
        raise ValueError("At least one model result is required")
    test_splits = {tuple(result["test_cycle_ids"]) for result in worker_results}
    if len(test_splits) != 1:
        raise ValueError("Benchmarked models did not use the same test cycles")
    return {
        "benchmark_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_type": "host-side software inference; not physical edge hardware",
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": _cpu_description() or platform.processor() or None,
            "logical_cpu_count": os.cpu_count(),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "scikit_learn_version": __import__("sklearn").__version__,
            "joblib_version": joblib.__version__,
        },
        "target_status": "approximate_capacity_referenced_coulomb_counting_proxy",
        "split_method": comparison_metadata["split_method"],
        "train_cycle_ids": comparison_metadata["train_cycle_ids"],
        "test_cycle_ids": comparison_metadata["test_cycle_ids"],
        "feature_order": list(MODEL_FEATURES),
        "latency_method": (
            "perf_counter_ns around model.predict only; data batches are prepared "
            "before timing. Each model runs in a fresh subprocess. Results "
            "include 50/20/10 warm-up calls and 500/200/100 measured calls for "
            "batch sizes 1/32/256; report median, p95, and p99."
        ),
        "memory_method": (
            "Linux process VmRSS and VmHWM from /proc/self/status in the isolated "
            "worker. Current RSS is sampled before/after reading the test data, "
            "after model load, and after prediction. Values include Python, "
            "NumPy, scikit-learn, test data, and runtime overhead and are not "
            "exact model-only memory."
        ),
        "artifact_size_method": (
            "Exact on-disk file size in bytes for the existing joblib files; "
            "no additional compression is applied."
        ),
        "selection_holdout_caveat": (
            "The held-out cycles were used during model selection and the "
            "optimization comparison; this is exploratory resource "
            "characterization, not an independent final evaluation."
        ),
        "results": worker_results,
        "physical_hardware_tested": False,
        "physical_power_measured": False,
    }


def write_report(
    report: dict[str, Any],
    json_path: Path,
    csv_path: Path,
    *,
    overwrite: bool = False,
) -> None:
    existing = [path for path in (json_path, csv_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing output(s): "
            + ", ".join(str(path) for path in existing)
            + "; pass --overwrite to replace them."
        )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=CSV_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        for result in report["results"]:
            memory = result["memory"]
            for latency in result["latency"]:
                writer.writerow(
                    {
                        "model": result["model"],
                        "artifact": result["artifact"],
                        "artifact_size_bytes": result["artifact_size_bytes"],
                        **latency,
                        "rss_before_dataset_load_bytes": memory[
                            "rss_before_dataset_load_bytes"
                        ],
                        "rss_after_dataset_load_bytes": memory[
                            "rss_after_dataset_load_bytes"
                        ],
                        "rss_after_model_load_bytes": memory[
                            "rss_after_model_load_bytes"
                        ],
                        "rss_after_inference_bytes": memory[
                            "rss_after_inference_bytes"
                        ],
                        "rss_change_during_dataset_load_bytes": memory[
                            "rss_change_during_dataset_load_bytes"
                        ],
                        "rss_change_during_model_load_bytes": memory[
                            "rss_change_during_model_load_bytes"
                        ],
                        "rss_high_water_before_dataset_load_bytes": memory[
                            "rss_high_water_before_dataset_load_bytes"
                        ],
                        "rss_high_water_after_dataset_load_bytes": memory[
                            "rss_high_water_after_dataset_load_bytes"
                        ],
                        "rss_high_water_after_model_load_bytes": memory[
                            "rss_high_water_after_model_load_bytes"
                        ],
                        "rss_high_water_after_inference_bytes": memory[
                            "rss_high_water_after_inference_bytes"
                        ],
                        "rss_high_water_bytes": memory["rss_high_water_bytes"],
                        "rss_high_water_change_during_model_load_bytes": memory[
                            "rss_high_water_change_during_model_load_bytes"
                        ],
                    }
                )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark selected and optimized Panasonic RF models for host-side "
            "latency, artifact size, and process RSS."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--comparison-metadata",
        type=Path,
        default=DEFAULT_COMPARISON_METADATA,
    )
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--optimized", type=Path, default=DEFAULT_OPTIMIZED)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model-name", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--artifact", type=Path, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.worker:
        result = _worker_benchmark(
            args.model_name,
            args.artifact,
            args.dataset,
            args.comparison_metadata,
        )
        print(json.dumps(result, allow_nan=False))
        return

    required_files = (
        args.dataset,
        args.comparison_metadata,
        args.baseline,
        args.optimized,
    )
    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(f"Required benchmark input not found: {path}")
    existing = [path for path in (args.json, args.csv) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing benchmark output(s): "
            + ", ".join(str(path) for path in existing)
            + "; pass --overwrite to replace them."
        )

    comparison_metadata = json.loads(
        args.comparison_metadata.read_text(encoding="utf-8")
    )
    worker_results = [
        _run_worker(
            "selected_baseline_100_trees",
            args.baseline,
            args.dataset,
            args.comparison_metadata,
        ),
        _run_worker(
            "optimized_candidate_50_trees",
            args.optimized,
            args.dataset,
            args.comparison_metadata,
        ),
    ]
    report = build_benchmark_report(worker_results, comparison_metadata)
    write_report(report, args.json, args.csv, overwrite=args.overwrite)
    print("Host-side resource benchmark complete.")
    for result in worker_results:
        print(
            f"{result['model']}: {result['artifact_size_bytes']:,} bytes; "
            f"single-row median "
            f"{result['latency'][0]['latency_median_ms_per_batch']:.6f} ms; "
            f"RSS change during model load "
            f"{result['memory']['rss_change_during_model_load_bytes']:,} bytes."
        )
    print(f"JSON: {args.json}")
    print(f"CSV: {args.csv}")
    print("Host software measurements only; not physical edge performance or power.")


if __name__ == "__main__":
    main()
