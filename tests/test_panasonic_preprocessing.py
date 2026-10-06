import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from src.preprocessing.prepare_panasonic_model_data import (
    MODEL_FEATURES,
    TARGET,
    GROUP,
    load_and_validate_dataset,
    prepare_model_data,
    write_outputs,
)


class PanasonicPreprocessingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.dataset_path = self.root / "proxy.csv"
        fieldnames = [GROUP, "elapsed_time_s", *MODEL_FEATURES, TARGET]
        with self.dataset_path.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            for group_index in range(5):
                for sample_index in range(4):
                    writer.writerow(
                        {
                            GROUP: f"cycle-{group_index}",
                            "elapsed_time_s": sample_index,
                            "voltage_v": 3.0 + group_index + sample_index * 0.1,
                            "current_a": -group_index - sample_index * 0.2,
                            "battery_temp_c": 20.0 + group_index,
                            "elapsed_time_s": sample_index,
                            TARGET: 105.0 if sample_index == 3 else 100.0 - sample_index,
                        }
                    )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_fits_scaler_on_grouped_training_rows_only(self) -> None:
        cleaned, scaled, scaler_metadata, log = prepare_model_data(self.dataset_path)
        rows, features, targets, groups = load_and_validate_dataset(self.dataset_path)
        train_indices, test_indices = next(
            GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(
                features, targets, groups
            )
        )
        self.assertFalse(
            set(groups[train_indices]).intersection(groups[test_indices])
        )
        expected_means = features[train_indices].mean(axis=0)

        self.assertEqual(len(cleaned), len(rows))
        self.assertEqual(len(scaled), len(rows))
        self.assertEqual(log["rows_removed"], 0)
        self.assertEqual(log["scaler_fit_rows"], len(train_indices))
        self.assertTrue(log["scaler_fit_on_train_only"])
        self.assertEqual(log["checks"]["out_of_range_proxy_rows_preserved"], 5)
        np.testing.assert_allclose(
            [scaler_metadata["mean"][name] for name in MODEL_FEATURES],
            expected_means,
        )
        self.assertEqual(
            set(log["train_cycle_ids"]).intersection(log["test_cycle_ids"]),
            set(),
        )

    def test_rejects_non_finite_values_and_duplicate_times(self) -> None:
        original = self.dataset_path.read_text(encoding="utf-8")
        self.dataset_path.write_text(
            original.replace("3.0", "nan", 1),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "non-finite"):
            load_and_validate_dataset(self.dataset_path)

        self.dataset_path.write_text(original, encoding="utf-8")
        with self.dataset_path.open(newline="", encoding="utf-8") as input_file:
            rows = list(csv.DictReader(input_file))
        rows[1]["elapsed_time_s"] = "0"
        with self.dataset_path.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaisesRegex(ValueError, "moves backwards|Duplicate"):
            load_and_validate_dataset(self.dataset_path)

    def test_writes_all_preprocessing_artifacts(self) -> None:
        outputs = prepare_model_data(self.dataset_path)
        cleaned, scaled, scaler, log = outputs
        paths = [self.root / name for name in ("clean.csv", "scaled.csv", "scaler.json", "log.json")]

        write_outputs(
            cleaned,
            scaled,
            scaler,
            log,
            *paths,
        )

        self.assertTrue(all(path.is_file() for path in paths))
        with paths[0].open(newline="", encoding="utf-8") as input_file:
            self.assertEqual(len(list(csv.DictReader(input_file))), len(cleaned))
        with paths[1].open(newline="", encoding="utf-8") as input_file:
            scaled_rows = list(csv.DictReader(input_file))
        self.assertIn("voltage_v_scaled", scaled_rows[0])
        self.assertEqual(float(scaled_rows[0][TARGET]), float(cleaned[0][TARGET]))


if __name__ == "__main__":
    unittest.main()
