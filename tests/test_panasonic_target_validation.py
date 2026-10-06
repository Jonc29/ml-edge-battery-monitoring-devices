import unittest

import numpy as np

from src.data.validate_panasonic_target import (
    audit_ah_current_consistency,
    audit_discharge_reference,
    select_25c_capacity_comparisons,
    select_25c_drive_cycles,
)
from src.data.prepare_panasonic_soc_proxy import (
    derive_soc_proxy,
    select_pre_drive_capacity_references,
)


class AhCurrentConsistencyTests(unittest.TestCase):
    def test_matches_negative_discharge_current_and_ah(self) -> None:
        result = audit_ah_current_consistency(
            "cycle.mat",
            {
                "Time": np.array([0.0, 3600.0]),
                "Current": np.array([-1.0, -1.0]),
                "Ah": np.array([0.0, -1.0]),
                "Voltage": np.array([4.0, 3.0]),
            },
        )

        self.assertEqual(result["net_direction_from_ah"], "net_discharge")
        self.assertAlmostEqual(result["integrated_current_ah"], -1.0)
        self.assertAlmostEqual(result["absolute_difference_ah"], 0.0)

    def test_accounts_for_counter_offset(self) -> None:
        result = audit_ah_current_consistency(
            "offset.mat",
            {
                "Time": np.array([0.0, 3600.0]),
                "Current": np.array([1.0, 1.0]),
                "Ah": np.array([12.0, 13.0]),
                "Voltage": np.array([3.0, 4.0]),
            },
        )

        self.assertEqual(result["net_direction_from_ah"], "net_charge")
        self.assertAlmostEqual(result["ah_change"], 1.0)
        self.assertAlmostEqual(result["absolute_difference_ah"], 0.0)

    def test_allows_repeated_timestamps(self) -> None:
        result = audit_ah_current_consistency(
            "repeated-time.mat",
            {
                "Time": np.array([0.0, 0.0, 3600.0]),
                "Current": np.array([-1.0, -1.0, -1.0]),
                "Ah": np.array([0.0, 0.0, -1.0]),
                "Voltage": np.array([4.0, 4.0, 3.0]),
            },
        )

        self.assertEqual(result["observations"], 3)
        self.assertAlmostEqual(result["integrated_current_ah"], -1.0)

    def test_rejects_time_reversal(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not move backwards"):
            audit_ah_current_consistency(
                "bad-time.mat",
                {
                    "Time": np.array([0.0, -0.1]),
                    "Current": np.array([-1.0, -1.0]),
                    "Ah": np.array([0.0, 0.0]),
                    "Voltage": np.array([4.0, 3.0]),
                },
            )

    def test_selects_split_25c_drive_cycles_not_combined_log(self) -> None:
        members = [
            "Data/25degC/Drive cycles/25degC_Cycle_1.mat",
            "Data/25degC/Drive cycles/25degC_US06_HWFET_UDDS_LA92.mat",
            "Data/10degC/Drive cycles/10degC_Cycle_1.mat",
            "Data/25degC/Charges and Pauses/25degC_Charge.mat",
        ]

        self.assertEqual(
            select_25c_drive_cycles(members),
            ["Data/25degC/Drive cycles/25degC_Cycle_1.mat"],
        )

    def test_selects_only_25c_full_discharge_comparison_files(self) -> None:
        members = [
            "Data/25degC/1C discharge tests_start_of_tests/Dis1C_1.mat",
            "Data/25degC/1C discharge tests_end_of_tests/Dis1C_2.mat",
            "Data/25degC/1C discharge tests_start_of_tests/Dis1C_Rp.mat",
            "Data/10degC/1C discharge tests_start_of_tests/Dis1C_1.mat",
        ]

        self.assertEqual(
            select_25c_capacity_comparisons(members),
            [
                "Data/25degC/1C discharge tests_end_of_tests/Dis1C_2.mat",
                "Data/25degC/1C discharge tests_start_of_tests/Dis1C_1.mat",
            ],
        )

    def test_summarizes_discharge_throughput_not_per_cycle_capacity(self) -> None:
        result = audit_discharge_reference(
            "Data/25degC/1C discharge tests_end_of_tests/Dis1C_1.mat",
            {
                "Ah": np.array([2.8, 0.0]),
                "Voltage": np.array([4.2, 2.5]),
            },
        )

        self.assertEqual(result["test_period"], "end_of_tests")
        self.assertAlmostEqual(result["discharge_throughput_ah"], 2.8)
        self.assertFalse(result["per_drive_cycle_capacity_reference"])

    def test_derives_soc_proxy_from_measured_ah_and_capacity(self) -> None:
        actual = derive_soc_proxy(
            np.array([0.0, -1.0, -2.0]),
            reference_capacity_ah=2.5,
        )

        np.testing.assert_allclose(actual, [100.0, 60.0, 20.0])

    def test_does_not_clip_out_of_range_proxy(self) -> None:
        actual = derive_soc_proxy(
            np.array([0.0, -3.0]),
            reference_capacity_ah=2.5,
        )

        self.assertAlmostEqual(actual[-1], -20.0)

    def test_requires_positive_capacity(self) -> None:
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            derive_soc_proxy(np.array([0.0]), reference_capacity_ah=0.0)

    def test_selects_pre_drive_capacity_references_only(self) -> None:
        members = [
            "Data/25degC/C20 OCV and 1C discharge tests_start_of_tests/Dis1C_1.mat",
            "Data/25degC/C20 OCV and 1C discharge tests_start_of_tests/Dis1C_2.mat",
            "Data/25degC/1C discharge tests_end_of_tests/Dis1C_1.mat",
            "Data/10degC/C20 OCV and 1C discharge tests_start_of_tests/Dis1C_1.mat",
        ]

        self.assertEqual(
            select_pre_drive_capacity_references(members),
            [
                "Data/25degC/C20 OCV and 1C discharge tests_start_of_tests/Dis1C_1.mat",
                "Data/25degC/C20 OCV and 1C discharge tests_start_of_tests/Dis1C_2.mat",
            ],
        )


if __name__ == "__main__":
    unittest.main()
