import unittest

from src.audit_soc_proxy import build_sensitivity_report


class SocProxySensitivityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.metadata = {
            "dataset": "test dataset",
            "reference_capacity_ah": 3.0,
            "capacity_reference_files": [
                {"member": "reference-low.mat", "discharge_throughput_ah": 2.0},
                {"member": "reference-high.mat", "discharge_throughput_ah": 4.0},
            ],
        }
        self.rows = [
            {
                "cycle_id": "cycle-1",
                "elapsed_time_s": "0",
                "soc_proxy_percent": "100",
            },
            {
                "cycle_id": "cycle-1",
                "elapsed_time_s": "1",
                "soc_proxy_percent": "50",
            },
        ]

    def test_compares_each_capacity_to_mean_reference_proxy(self) -> None:
        report = build_sensitivity_report(self.rows, self.metadata)
        scenarios = {
            scenario["name"]: scenario
            for scenario in report["capacity_scenarios"]
        }

        self.assertEqual(report["baseline_sample_count"], 2)
        self.assertEqual(scenarios["measured_reference_1"]["capacity_ah"], 2.0)
        self.assertEqual(scenarios["measured_reference_2"]["capacity_ah"], 4.0)
        self.assertEqual(scenarios["mean_reference"]["capacity_ah"], 3.0)
        self.assertAlmostEqual(
            scenarios["measured_reference_1"]["cycles"][0]["proxy_end_percent"],
            25.0,
        )
        self.assertAlmostEqual(
            scenarios["measured_reference_2"]["cycles"][0]["proxy_end_percent"],
            62.5,
        )
        self.assertEqual(
            scenarios["mean_reference"][
                "maximum_absolute_change_from_mean_reference_percentage_points"
            ],
            0.0,
        )

    def test_rejects_capacity_metadata_that_does_not_match_mean(self) -> None:
        self.metadata["reference_capacity_ah"] = 2.5

        with self.assertRaisesRegex(ValueError, "does not match the mean"):
            build_sensitivity_report(self.rows, self.metadata)

    def test_rejects_non_finite_proxy_sample(self) -> None:
        self.rows[1]["soc_proxy_percent"] = "nan"

        with self.assertRaisesRegex(ValueError, "non-finite"):
            build_sensitivity_report(self.rows, self.metadata)


if __name__ == "__main__":
    unittest.main()
