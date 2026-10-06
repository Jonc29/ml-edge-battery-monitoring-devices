import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.data.eda_panasonic_proxy import (
    build_eda_summary,
    plot_correlations,
    plot_distributions,
)
from src.data.prepare_panasonic_soc_proxy import MODEL_FEATURES
from src.preprocessing.prepare_panasonic_model_data import TARGET


class PanasonicEdaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [
            {"cycle_id": "cycle-a"},
            {"cycle_id": "cycle-a"},
            {"cycle_id": "cycle-b"},
            {"cycle_id": "cycle-b"},
            {"cycle_id": "cycle-b"},
        ]
        self.features = np.array(
            [
                [3.0, -1.0, 21.0, 0.0],
                [3.2, -0.5, 22.0, 1.0],
                [3.4, 0.0, 23.0, 0.0],
                [3.6, 0.5, 24.0, 1.0],
                [3.8, 1.0, 25.0, 2.0],
            ]
        )
        self.targets = np.array([100.0, 75.0, 50.0, 25.0, -1.0])

    def test_summary_labels_proxy_and_reports_pooled_statistics(self) -> None:
        summary = build_eda_summary(self.rows, self.features, self.targets)

        self.assertEqual(summary["row_count"], 5)
        self.assertEqual(summary["cycle_count"], 2)
        self.assertEqual(summary["rows_per_cycle"]["cycle-b"], 3)
        self.assertEqual(summary["statistics"][TARGET]["minimum"], -1.0)
        self.assertIn("not measured SoC", summary["interpretation"])
        self.assertEqual(
            summary["columns"],
            [*MODEL_FEATURES, TARGET],
        )

    def test_writes_both_figures(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            distributions_path = Path(temp_dir) / "distributions.png"
            correlation_path = Path(temp_dir) / "correlation.png"

            plot_distributions(
                self.features,
                self.targets,
                distributions_path,
            )
            plot_correlations(
                self.features,
                self.targets,
                correlation_path,
            )

            self.assertGreater(distributions_path.stat().st_size, 0)
            self.assertGreater(correlation_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
