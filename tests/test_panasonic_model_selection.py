import unittest
from pathlib import Path

from src.models.compare_panasonic_models import MODEL_PARAMETERS
from src.models.select_panasonic_model import (
    build_decision,
    select_candidate,
)


class PanasonicModelSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [
            {
                "model": "RidgeRegression",
                "proxy_test_mae_percentage_points": "3.1",
                "proxy_test_rmse_percentage_points": "5.0",
                "proxy_test_r2": "0.97",
                "per_cycle_metrics_json": '{"cycle-4":{"mae_percentage_points":3.5}}',
            },
            {
                "model": "RandomForestRegressor",
                "proxy_test_mae_percentage_points": "2.2",
                "proxy_test_rmse_percentage_points": "3.0",
                "proxy_test_r2": "0.99",
                "per_cycle_metrics_json": '{"cycle-4":{"mae_percentage_points":2.1}}',
            },
            {
                "model": "ExtraTreesRegressor",
                "proxy_test_mae_percentage_points": "3.2",
                "proxy_test_rmse_percentage_points": "4.3",
                "proxy_test_r2": "0.98",
                "per_cycle_metrics_json": '{"cycle-4":{"mae_percentage_points":2.9}}',
            },
        ]

    def test_selects_lowest_mae_and_returns_per_cycle_evidence(self) -> None:
        selected = select_candidate(self.rows)

        self.assertEqual(selected["model"], "RandomForestRegressor")
        self.assertAlmostEqual(selected["mae"], 2.2)
        self.assertIn("cycle-4", selected["per_cycle"])

    def test_requires_all_comparison_candidates(self) -> None:
        with self.assertRaisesRegex(ValueError, "Expected exactly one result"):
            select_candidate(self.rows[:2])

    def test_decision_is_provisional_and_marks_selection_limitations(self) -> None:
        selected = select_candidate(self.rows)
        metadata = {
            "train_cycle_ids": ["train-1"],
            "test_cycle_ids": ["cycle-2", "cycle-4"],
            "train_sample_count": 10,
            "test_sample_count": 8,
        }
        candidates = [
            {
                "model": row["model"],
                "mae": float(row["proxy_test_mae_percentage_points"]),
                "rmse": float(row["proxy_test_rmse_percentage_points"]),
                "r2": float(row["proxy_test_r2"]),
            }
            for row in self.rows
        ]

        decision = build_decision(
            selected,
            candidates,
            metadata,
            Path("data.csv"),
            Path("selected.joblib"),
        )

        self.assertEqual(
            decision["decision_status"],
            "provisional_for_optimization_phase",
        )
        self.assertEqual(len(decision["other_candidates"]), 2)
        self.assertTrue(
            any("holdout" in limitation for limitation in decision["limitations"])
        )
        self.assertEqual(
            decision["model_parameters"],
            MODEL_PARAMETERS["RandomForestRegressor"],
        )


if __name__ == "__main__":
    unittest.main()
