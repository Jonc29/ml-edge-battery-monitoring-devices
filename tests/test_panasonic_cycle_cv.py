import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.data.prepare_panasonic_soc_proxy import MODEL_FEATURES
from src.evaluation.evaluate_panasonic_cycle_cv import (
    evaluate_leave_one_cycle_out,
    write_results,
)


class PanasonicCycleCrossValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.dataset_path = self.root / "proxy.csv"
        with self.dataset_path.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(
                output_file,
                fieldnames=[
                    "cycle_id",
                    *MODEL_FEATURES,
                    "soc_proxy_percent",
                ],
            )
            writer.writeheader()
            for cycle_index, cycle_id in enumerate(
                ("cycle-a", "cycle-b", "cycle-c")
            ):
                for sample_index in range(6):
                    writer.writerow(
                        {
                            "cycle_id": cycle_id,
                            "voltage_v": 4.2 - 0.1 * sample_index,
                            "current_a": -0.2 * sample_index,
                            "battery_temp_c": 25.0 + 0.1 * cycle_index,
                            "elapsed_time_s": float(sample_index),
                            "soc_proxy_percent": (
                                100.0 - 12.0 * sample_index - cycle_index
                            ),
                        }
                    )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_evaluates_each_complete_cycle_once_for_each_model(self) -> None:
        rows, summary = evaluate_leave_one_cycle_out(self.dataset_path)

        self.assertEqual(len(rows), 9)
        self.assertEqual(summary["fold_count"], 3)
        self.assertEqual(summary["sample_count"], 18)
        for model_name, result in summary["models"].items():
            model_rows = [row for row in rows if row["model"] == model_name]
            self.assertEqual(
                {row["held_out_cycle_id"] for row in model_rows},
                {"cycle-a", "cycle-b", "cycle-c"},
            )
            for row in model_rows:
                self.assertNotIn(
                    row["held_out_cycle_id"],
                    row["train_cycle_ids"].split(";"),
                )
            self.assertEqual(
                result["out_of_fold_sample_count"],
                summary["sample_count"],
            )
            self.assertGreaterEqual(
                result["pooled_out_of_fold_metrics"]["mae_percentage_points"],
                0.0,
            )

    def test_rejects_overwriting_existing_outputs(self) -> None:
        rows, summary = evaluate_leave_one_cycle_out(self.dataset_path)
        csv_path = self.root / "results.csv"
        metrics_path = self.root / "metrics.json"
        write_results(rows, summary, csv_path, metrics_path)

        with self.assertRaisesRegex(FileExistsError, "overwrite"):
            write_results(rows, summary, csv_path, metrics_path)

        saved = json.loads(metrics_path.read_text(encoding="utf-8"))
        self.assertEqual(saved["split_method"], "leave-one-complete-cycle-out")
        with csv_path.open(newline="", encoding="utf-8") as input_file:
            self.assertEqual(len(list(csv.DictReader(input_file))), 9)


if __name__ == "__main__":
    unittest.main()
