from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.ui.data import DATASET_COLUMNS, FEATURE_COLUMNS, load_proxy_dataset
from src.ui.service import predict_soc_proxy


class FixedPredictor:
    n_features_in_ = 4

    def __init__(self) -> None:
        self.received: np.ndarray | None = None

    def predict(self, features: np.ndarray) -> np.ndarray:
        self.received = features
        return np.asarray([63.5])


class UiDataAndPredictionTests(unittest.TestCase):
    def test_prediction_uses_model_feature_order(self) -> None:
        model = FixedPredictor()
        features = {
            "elapsed_time_s": 12.0,
            "battery_temp_c": 25.0,
            "current_a": -1.0,
            "voltage_v": 3.8,
        }

        prediction = predict_soc_proxy(model, features)

        self.assertEqual(prediction, 63.5)
        self.assertIsNotNone(model.received)
        np.testing.assert_array_equal(
            model.received,
            np.asarray([[3.8, -1.0, 25.0, 12.0]], dtype=np.float32),
        )

    def test_prediction_rejects_non_finite_inputs(self) -> None:
        model = FixedPredictor()
        features = {
            "voltage_v": float("nan"),
            "current_a": -1.0,
            "battery_temp_c": 25.0,
            "elapsed_time_s": 12.0,
        }

        with self.assertRaisesRegex(ValueError, "finite"):
            predict_soc_proxy(model, features)

    def test_prediction_rejects_missing_input_feature(self) -> None:
        with self.assertRaisesRegex(ValueError, "Missing model input"):
            predict_soc_proxy(
                FixedPredictor(),
                {"voltage_v": 3.8, "current_a": -1.0, "battery_temp_c": 25.0},
            )

    def test_dataset_loader_checks_schema_and_numeric_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "proxy.csv"
            pd.DataFrame(
                [
                    {
                        "cycle_id": "cycle-a",
                        "voltage_v": 3.8,
                        "current_a": -1.0,
                        "battery_temp_c": 25.0,
                        "elapsed_time_s": 1.0,
                        "soc_proxy_percent": 80.0,
                    }
                ],
                columns=DATASET_COLUMNS,
            ).to_csv(path, index=False)

            frame = load_proxy_dataset(path)

        self.assertEqual(list(frame.columns), list(DATASET_COLUMNS))
        self.assertEqual(len(FEATURE_COLUMNS), 4)
        self.assertEqual(frame.loc[0, "soc_proxy_percent"], 80.0)

    def test_dataset_loader_rejects_non_finite_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "proxy.csv"
            pd.DataFrame(
                [
                    {
                        "cycle_id": "cycle-a",
                        "voltage_v": np.inf,
                        "current_a": -1.0,
                        "battery_temp_c": 25.0,
                        "elapsed_time_s": 1.0,
                        "soc_proxy_percent": 80.0,
                    }
                ],
                columns=DATASET_COLUMNS,
            ).to_csv(path, index=False)

            with self.assertRaisesRegex(ValueError, "non-finite"):
                load_proxy_dataset(path)


if __name__ == "__main__":
    unittest.main()
