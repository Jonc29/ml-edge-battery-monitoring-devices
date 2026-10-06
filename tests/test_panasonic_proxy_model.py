import csv
import tempfile
import unittest
from pathlib import Path

from src.models.train_panasonic_soc_proxy_baseline import (
    MODEL_FEATURES,
    fit_grouped_baseline,
    load_proxy_csv,
)


class PanasonicProxyModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dataset_path = Path(self.temp_dir.name) / "proxy.csv"
        with self.dataset_path.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(
                output_file,
                fieldnames=[
                    "cycle_id",
                    "elapsed_time_s",
                    "voltage_v",
                    "current_a",
                    "battery_temp_c",
                    "Ah",
                    "soc_proxy_percent",
                ],
            )
            writer.writeheader()
            for group_index in range(5):
                for sample_index in range(6):
                    writer.writerow(
                        {
                            "cycle_id": f"cycle-{group_index}",
                            "elapsed_time_s": sample_index,
                            "voltage_v": 4.2 - sample_index * 0.1,
                            "current_a": -0.2,
                            "battery_temp_c": 25.0,
                            "Ah": -sample_index * 0.1,
                            "soc_proxy_percent": 100.0 - sample_index * 3.0,
                        }
                    )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_loads_only_declared_features_not_ah(self) -> None:
        features, _, _ = load_proxy_csv(self.dataset_path)

        self.assertEqual(features.shape, (30, len(MODEL_FEATURES)))
        self.assertNotIn("Ah", MODEL_FEATURES)

    def test_holdout_split_keeps_whole_cycles_separate(self) -> None:
        _, metrics = fit_grouped_baseline(self.dataset_path)

        self.assertTrue(metrics["train_cycle_ids"])
        self.assertTrue(metrics["test_cycle_ids"])
        self.assertFalse(
            set(metrics["train_cycle_ids"]).intersection(metrics["test_cycle_ids"])
        )
        self.assertIn("proxy", metrics["interpretation"])


if __name__ == "__main__":
    unittest.main()
