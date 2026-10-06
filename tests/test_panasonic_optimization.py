import csv
import tempfile
import unittest
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestRegressor

from src.data.prepare_panasonic_soc_proxy import MODEL_FEATURES
from src.models.compare_panasonic_models import HOLDOUT_CYCLE_IDS
from src.models.optimize_panasonic_rf import (
    optimize_models,
    write_results,
)
from src.models.train_panasonic_soc_proxy_baseline import GROUP, TARGET, SEED


class PanasonicOptimizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.dataset_path = self.root / "proxy.csv"
        self.train_groups = ["train-cycle-a", "train-cycle-b"]
        self.test_groups = list(HOLDOUT_CYCLE_IDS)
        self.comparison_metadata = {
            "target": "soc_proxy_percent",
            "input_features": list(MODEL_FEATURES),
            "train_cycle_ids": self.train_groups,
            "test_cycle_ids": self.test_groups,
        }
        with self.dataset_path.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(
                output_file,
                fieldnames=[GROUP, "elapsed_time_s", *MODEL_FEATURES, TARGET],
            )
            writer.writeheader()
            for group_index, cycle_id in enumerate(
                self.train_groups + self.test_groups
            ):
                for sample_index in range(6):
                    writer.writerow(
                        {
                            GROUP: cycle_id,
                            "elapsed_time_s": sample_index,
                            "voltage_v": 4.2 - 0.1 * sample_index,
                            "current_a": -0.2 * sample_index,
                            "battery_temp_c": 25.0 + 0.1 * group_index,
                            TARGET: 100.0 - 15.0 * sample_index,
                        }
                    )

        self.baseline_path = self.root / "baseline.joblib"
        baseline = RandomForestRegressor(
            n_estimators=100,
            min_samples_leaf=2,
            max_leaf_nodes=128,
            max_features=1.0,
            random_state=SEED,
            n_jobs=-1,
        )
        with self.dataset_path.open(newline="", encoding="utf-8") as input_file:
            baseline_rows = list(csv.DictReader(input_file))
        train_rows = [row for row in baseline_rows if row[GROUP] in self.train_groups]
        X = [
            [float(row[feature]) for feature in MODEL_FEATURES]
            for row in train_rows
        ]
        y = [float(row[TARGET]) for row in train_rows]
        baseline.fit(X, y)
        joblib.dump(baseline, self.baseline_path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_smaller_variants_report_metrics_and_artifact_size(self) -> None:
        variants_dir = self.root / "optimized"
        rows, metadata = optimize_models(
            self.dataset_path,
            self.comparison_metadata,
            self.baseline_path,
            variants_dir,
        )

        self.assertEqual(len(rows), 5)
        self.assertEqual(rows[0]["variant"], "selected_baseline")
        self.assertEqual(set(metadata["test_cycle_ids"]), set(self.test_groups))
        self.assertTrue(all(row["test_sample_count"] == 18 for row in rows))
        self.assertTrue(all(row["artifact_size_bytes"] > 0 for row in rows))
        for row in rows[1:]:
            self.assertTrue(Path(row["artifact"]).is_file())
            self.assertEqual(
                row["artifact_size_bytes"],
                Path(row["artifact"]).stat().st_size,
            )
            self.assertGreaterEqual(
                row["artifact_size_reduction_percent_vs_baseline"],
                0.0,
            )

    def test_writes_csv_and_metadata(self) -> None:
        rows, metadata = optimize_models(
            self.dataset_path,
            self.comparison_metadata,
            self.baseline_path,
            self.root / "optimized",
        )
        comparison_path = self.root / "optimization.csv"
        metadata_path = self.root / "optimization.json"

        write_results(rows, metadata, comparison_path, metadata_path)

        with comparison_path.open(newline="", encoding="utf-8") as input_file:
            saved_rows = list(csv.DictReader(input_file))
        self.assertEqual(len(saved_rows), 5)
        self.assertTrue(metadata_path.is_file())

    def test_refuses_cycles_missing_from_declared_split(self) -> None:
        metadata = dict(self.comparison_metadata)
        metadata["test_cycle_ids"] = self.test_groups[:-1]

        with self.assertRaisesRegex(ValueError, "omits dataset cycles"):
            optimize_models(
                self.dataset_path,
                metadata,
                self.baseline_path,
                self.root / "optimized",
            )


if __name__ == "__main__":
    unittest.main()
