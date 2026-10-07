import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestRegressor

from src.data.prepare_panasonic_soc_proxy import MODEL_FEATURES
from src.evaluation.export_panasonic_forest_c import (
    export_models,
    render_header,
)


class PanasonicCExportTests(unittest.TestCase):
    def test_render_header_has_feature_order_and_prediction_api(self) -> None:
        model = RandomForestRegressor(
            n_estimators=2,
            max_leaf_nodes=4,
            random_state=42,
        )
        model.fit(
            [
                [4.1, -0.1, 25.0, 0.0],
                [3.9, -0.2, 25.1, 1.0],
                [3.7, -0.3, 25.2, 2.0],
                [3.5, -0.4, 25.3, 3.0],
            ],
            [100.0, 75.0, 50.0, 25.0],
        )

        header, structure = render_header(
            model,
            function_name="test_predict",
            macro_prefix="TEST_FOREST",
            artifact_role="test_model",
        )

        self.assertEqual(structure["tree_count"], 2)
        self.assertEqual(structure["feature_count"], 4)
        self.assertIn(", ".join(MODEL_FEATURES), header)
        self.assertIn("double test_predict(const float features[TEST_FOREST_FEATURE_COUNT])", header)
        self.assertIn("return NAN;", header)

    @unittest.skipUnless(shutil.which("cc"), "C99 compiler is not available")
    def test_compiled_headers_match_python_on_held_out_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_name:
            root = Path(temp_dir_name)
            dataset_path = root / "proxy.csv"
            baseline_path = root / "baseline.joblib"
            optimized_path = root / "optimized.joblib"
            metadata_path = root / "comparison.json"
            output_dir = root / "exported"
            validation_path = root / "validation.json"

            rows = []
            for cycle_id, offset in (
                ("train-cycle-a", 0.0),
                ("train-cycle-b", 0.05),
                ("test-cycle", 0.15),
            ):
                for index in range(8):
                    rows.append(
                        {
                            "cycle_id": cycle_id,
                            "voltage_v": 4.2 - 0.08 * index + offset,
                            "current_a": -0.1 * index,
                            "battery_temp_c": 25.0 + 0.2 * index,
                            "elapsed_time_s": float(index),
                            "soc_proxy_percent": 100.0 - 10.0 * index + offset,
                        }
                    )
            with dataset_path.open("w", newline="", encoding="utf-8") as output_file:
                writer = csv.DictWriter(
                    output_file,
                    fieldnames=[
                        "cycle_id",
                        *MODEL_FEATURES,
                        "soc_proxy_percent",
                    ],
                )
                writer.writeheader()
                writer.writerows(rows)

            metadata_path.write_text(
                json.dumps(
                    {
                        "target": "soc_proxy_percent",
                        "input_features": list(MODEL_FEATURES),
                        "split_method": "fixed complete-cycle holdout",
                        "train_cycle_ids": ["train-cycle-a", "train-cycle-b"],
                        "test_cycle_ids": ["test-cycle"],
                    }
                ),
                encoding="utf-8",
            )
            train_rows = [
                row for row in rows if row["cycle_id"].startswith("train-cycle")
            ]
            train_features = [
                [row[name] for name in MODEL_FEATURES] for row in train_rows
            ]
            train_targets = [row["soc_proxy_percent"] for row in train_rows]
            for path, tree_count in (
                (baseline_path, 5),
                (optimized_path, 3),
            ):
                model = RandomForestRegressor(
                    n_estimators=tree_count,
                    max_leaf_nodes=8,
                    random_state=42,
                    n_jobs=1,
                )
                model.fit(train_features, train_targets)
                joblib.dump(model, path)

            report = export_models(
                baseline_path,
                optimized_path,
                dataset_path,
                metadata_path,
                output_dir,
                validation_path,
            )

            self.assertTrue(report["validation"]["passed"])
            self.assertEqual(
                report["validation"]["models"][
                    "selected_baseline_100_trees"
                ]["sample_count"],
                8,
            )
            self.assertLessEqual(
                report["validation"]["models"][
                    "optimized_candidate_50_trees"
                ]["maximum_absolute_error"],
                1e-9,
            )
            self.assertTrue((output_dir / "panasonic_soc_proxy_selected_rf.h").is_file())
            self.assertTrue((output_dir / "panasonic_soc_proxy_rf_50_trees.h").is_file())
            self.assertTrue(validation_path.is_file())

            with self.assertRaisesRegex(FileExistsError, "overwrite"):
                export_models(
                    baseline_path,
                    optimized_path,
                    dataset_path,
                    metadata_path,
                    output_dir,
                    validation_path,
                )


if __name__ == "__main__":
    unittest.main()
