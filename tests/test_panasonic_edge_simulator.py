import csv
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIMULATOR_SOURCE = PROJECT_ROOT / "edge/simulation/panasonic_edge_simulator.c"
BASELINE_MODEL = PROJECT_ROOT / "models/baseline/panasonic_soc_proxy_selected_rf.joblib"
OPTIMIZED_MODEL = PROJECT_ROOT / "models/optimized/panasonic_rf_50_trees.joblib"
EXAMPLE_FEATURES = np.asarray(
    [
        [4.1, -0.1, 25.0, 0.0],
        [3.9, -0.5, 25.3, 120.0],
        [3.7, -1.1, 26.1, 900.0],
        [3.4, -0.7, 27.5, 2400.0],
    ],
    dtype=np.float32,
)


@unittest.skipUnless(shutil.which("cc"), "C99 compiler is not available")
class PanasonicEdgeSimulatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.TemporaryDirectory(prefix="panasonic-edge-sim-")
        cls.executable = Path(cls.temp_dir.name) / "panasonic_edge_simulator"
        compile_result = subprocess.run(
            [
                shutil.which("cc") or "cc",
                "-std=c99",
                "-O2",
                "-Wall",
                "-Wextra",
                "-Werror",
                str(SIMULATOR_SOURCE),
                "-lm",
                "-o",
                str(cls.executable),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if compile_result.returncode != 0:
            raise RuntimeError(
                f"Simulator failed to compile: {compile_result.stderr}"
            )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir.cleanup()

    def test_both_modes_match_saved_python_estimators(self) -> None:
        input_text = "".join(
            ",".join(format(float(value), ".9g") for value in row) + "\n"
            for row in EXAMPLE_FEATURES
        )
        for mode, artifact_path, expected_label in (
            ("selected", BASELINE_MODEL, "selected_baseline"),
            ("candidate", OPTIMIZED_MODEL, "optimized_candidate_50_trees"),
        ):
            with self.subTest(mode=mode):
                result = subprocess.run(
                    [str(self.executable), "--model", mode],
                    input=input_text,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                output_rows = list(csv.DictReader(result.stdout.splitlines()))
                self.assertEqual(len(output_rows), len(EXAMPLE_FEATURES))
                self.assertTrue(
                    all(row["model"] == expected_label for row in output_rows)
                )
                self.assertEqual(
                    [int(row["sample_index"]) for row in output_rows],
                    list(range(len(EXAMPLE_FEATURES))),
                )

                model = joblib.load(artifact_path)
                expected = model.predict(EXAMPLE_FEATURES)
                actual = np.asarray(
                    [float(row["soc_proxy_percent"]) for row in output_rows]
                )
                np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-9)

    def test_rejects_malformed_or_non_finite_rows(self) -> None:
        for invalid_row in ("4.0,-0.2,25.0\n", "4.0,-0.2,nan,3.0\n"):
            with self.subTest(invalid_row=invalid_row):
                result = subprocess.run(
                    [str(self.executable)],
                    input=invalid_row,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Invalid input row 0", result.stderr)

    def test_rejects_unknown_model(self) -> None:
        result = subprocess.run(
            [str(self.executable), "--model", "unknown"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Unknown model", result.stderr)


if __name__ == "__main__":
    unittest.main()
