import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.evaluation.benchmark_panasonic_models import (
    build_benchmark_report,
    build_latency_batches,
    summarize_latencies,
    write_report,
)


class PanasonicResourceBenchmarkTests(unittest.TestCase):
    def test_latency_batches_are_deterministic_and_within_holdout(self) -> None:
        features = np.arange(400, dtype=np.float32).reshape(100, 4)

        first = build_latency_batches(features, (1, 8))
        second = build_latency_batches(features, (1, 8))

        self.assertEqual(len(first[1]), 16)
        self.assertEqual(len(first[8]), 16)
        for batch_size in first:
            for left, right in zip(first[batch_size], second[batch_size]):
                np.testing.assert_array_equal(left, right)
                self.assertEqual(left.shape, (batch_size, 4))

    def test_latency_batches_reject_oversized_batch(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceeds available"):
            build_latency_batches(np.ones((2, 4)), (3,))

    def test_latency_summary_reports_distribution_and_throughput(self) -> None:
        summary = summarize_latencies(
            [1_000_000, 2_000_000, 3_000_000],
            batch_size=2,
            warmup_calls=5,
        )

        self.assertEqual(summary["measured_calls"], 3)
        self.assertEqual(summary["latency_median_ms_per_batch"], 2.0)
        self.assertEqual(summary["latency_p95_ms_per_batch"], 2.9)
        self.assertEqual(summary["median_latency_us_per_sample"], 1000.0)
        self.assertEqual(summary["median_throughput_samples_per_second"], 1000.0)

    def test_latency_summary_rejects_zero_duration(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            summarize_latencies([0], batch_size=1, warmup_calls=0)

    def test_report_and_csv_preserve_memory_and_latency_fields(self) -> None:
        metadata = {
            "split_method": "fixed complete-cycle holdout",
            "train_cycle_ids": ["train-a"],
            "test_cycle_ids": ["test-a"],
        }
        result = {
            "model": "test-model",
            "artifact": "test.joblib",
            "artifact_size_bytes": 128,
            "test_cycle_ids": ["test-a"],
            "latency": [
                {
                    "batch_size": 1,
                    "warmup_calls": 1,
                    "measured_calls": 2,
                    "latency_median_ms_per_batch": 0.5,
                    "latency_p95_ms_per_batch": 0.6,
                    "latency_p99_ms_per_batch": 0.6,
                    "latency_mean_ms_per_batch": 0.5,
                    "median_latency_us_per_sample": 500.0,
                    "median_throughput_samples_per_second": 2000.0,
                }
            ],
            "memory": {
                "rss_before_dataset_load_bytes": 100,
                "rss_after_dataset_load_bytes": 150,
                "rss_after_model_load_bytes": 160,
                "rss_after_inference_bytes": 170,
                "rss_change_during_dataset_load_bytes": 50,
                "rss_change_during_model_load_bytes": 10,
                "rss_high_water_before_dataset_load_bytes": 110,
                "rss_high_water_after_dataset_load_bytes": 160,
                "rss_high_water_after_model_load_bytes": 170,
                "rss_high_water_after_inference_bytes": 180,
                "rss_high_water_bytes": 180,
                "rss_high_water_change_during_model_load_bytes": 10,
            },
        }
        report = build_benchmark_report([result], metadata)
        with tempfile.TemporaryDirectory() as temp_dir:
            json_path = Path(temp_dir) / "result.json"
            csv_path = Path(temp_dir) / "result.csv"

            write_report(report, json_path, csv_path)

            with csv_path.open(newline="", encoding="utf-8") as input_file:
                rows = list(csv.DictReader(input_file))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["artifact_size_bytes"], "128")
            self.assertEqual(rows[0]["latency_median_ms_per_batch"], "0.5")
            self.assertEqual(rows[0]["rss_change_during_model_load_bytes"], "10")

            with self.assertRaisesRegex(FileExistsError, "overwrite"):
                write_report(report, json_path, csv_path)


if __name__ == "__main__":
    unittest.main()
