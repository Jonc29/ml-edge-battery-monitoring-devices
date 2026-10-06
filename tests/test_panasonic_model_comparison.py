import csv
import tempfile
import unittest
from pathlib import Path

from src.data.prepare_panasonic_soc_proxy import MODEL_FEATURES
from src.models.compare_panasonic_models import (
    HOLDOUT_CYCLE_IDS,
    compare_models,
    write_comparison,
)
from src.models.train_panasonic_soc_proxy_baseline import GROUP, TARGET


class PanasonicModelComparisonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dataset_path = Path(self.temp_dir.name) / "proxy.csv"
        cycle_ids = [
            *HOLDOUT_CYCLE_IDS,
            "train-cycle-a",
            "train-cycle-b",
        ]
        with self.dataset_path.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(
                output_file,
                fieldnames=[GROUP, "elapsed_time_s", *MODEL_FEATURES, TARGET],
            )
            writer.writeheader()
            for group_index, cycle_id in enumerate(cycle_ids):
                for sample_index in range(6):
                    writer.writerow(
                        {
                            GROUP: cycle_id,
                            "elapsed_time_s": sample_index,
                            "voltage_v": 4.2 - sample_index * 0.1 - group_index * 0.01,
                            "current_a": -sample_index * 0.2,
                            "battery_temp_c": 25.0 + group_index * 0.1,
                            TARGET: 100.0 - sample_index * 20.0,
                        }
                    )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_all_models_use_fixed_complete_cycle_holdout(self) -> None:
        rows, metadata = compare_models(self.dataset_path)

        self.assertEqual(
            [row["model"] for row in rows],
            ["RidgeRegression", "RandomForestRegressor", "ExtraTreesRegressor"],
        )
        self.assertEqual(set(metadata["test_cycle_ids"]), set(HOLDOUT_CYCLE_IDS))
        self.assertEqual(len(metadata["train_cycle_ids"]), 2)
        self.assertFalse(
            set(metadata["train_cycle_ids"]).intersection(metadata["test_cycle_ids"])
        )
        self.assertIn("No model selected", metadata["selection_status"])
        self.assertTrue(all("proxy" in row["target_status"] for row in rows))

    def test_writes_comparison_csv_and_metadata(self) -> None:
        rows, metadata = compare_models(self.dataset_path)
        with tempfile.TemporaryDirectory() as temp_dir:
            comparison_path = Path(temp_dir) / "model_comparison.csv"
            metadata_path = Path(temp_dir) / "metadata.json"

            write_comparison(rows, metadata, comparison_path, metadata_path)

            with comparison_path.open(newline="", encoding="utf-8") as input_file:
                saved_rows = list(csv.DictReader(input_file))
            self.assertEqual(len(saved_rows), 3)
            self.assertTrue(metadata_path.is_file())


if __name__ == "__main__":
    unittest.main()
