"""Export the Panasonic Random Forest models as portable C99 headers."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from src.data.prepare_panasonic_soc_proxy import DEFAULT_DATASET, MODEL_FEATURES
from src.models.compare_panasonic_models import (
    DEFAULT_METADATA as DEFAULT_COMPARISON_METADATA,
)
from src.models.train_panasonic_soc_proxy_baseline import load_proxy_csv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASELINE = PROJECT_ROOT / "models/baseline/panasonic_soc_proxy_selected_rf.joblib"
DEFAULT_OPTIMIZED = PROJECT_ROOT / "models/optimized/panasonic_rf_50_trees.joblib"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "edge/exported"
DEFAULT_VALIDATION = PROJECT_ROOT / "results/metrics/panasonic_c_export_validation.json"
PARITY_ABSOLUTE_TOLERANCE = 1e-9

EXPORTS = (
    {
        "name": "selected_baseline_100_trees",
        "path_name": "panasonic_soc_proxy_selected_rf.h",
        "function_name": "panasonic_soc_proxy_selected_rf_predict",
        "macro_prefix": "PANASONIC_SOC_PROXY_SELECTED_RF",
        "artifact_role": "provisional_selected_baseline",
    },
    {
        "name": "optimized_candidate_50_trees",
        "path_name": "panasonic_soc_proxy_rf_50_trees.h",
        "function_name": "panasonic_soc_proxy_rf_50_trees_predict",
        "macro_prefix": "PANASONIC_SOC_PROXY_RF_50_TREES",
        "artifact_role": "size_optimized_candidate_not_selected",
    },
)


def _format_c_array(values: list[str], *, values_per_line: int = 6) -> str:
    lines = []
    for start in range(0, len(values), values_per_line):
        lines.append("    " + ", ".join(values[start : start + values_per_line]))
    return ",\n".join(lines)


def render_header(
    model: RandomForestRegressor,
    *,
    function_name: str,
    macro_prefix: str,
    artifact_role: str,
) -> tuple[str, dict[str, int]]:
    """Render a forest using flat, read-only C arrays and iterative traversal."""
    if not isinstance(model, RandomForestRegressor):
        raise TypeError("C export supports only RandomForestRegressor artifacts")
    if model.n_features_in_ != len(MODEL_FEATURES):
        raise ValueError(
            f"Expected {len(MODEL_FEATURES)} model features; "
            f"found {model.n_features_in_}"
        )
    if not model.estimators_:
        raise ValueError("Cannot export a Random Forest with no trees")

    tree_offsets = [0]
    node_features: list[str] = []
    left_children: list[str] = []
    right_children: list[str] = []
    thresholds: list[str] = []
    leaf_values: list[str] = []
    for estimator in model.estimators_:
        tree = cast(Any, estimator.tree_)
        if tree.n_outputs != 1:
            raise ValueError("C export supports only single-output regression trees")
        if tree.node_count < 1:
            raise ValueError("A decision tree must contain at least one node")
        if tree.n_features > 127:
            raise ValueError("C export supports at most 127 input features")

        for node_index in range(tree.node_count):
            feature = int(tree.feature[node_index])
            left = int(tree.children_left[node_index])
            right = int(tree.children_right[node_index])
            if left == -1 or right == -1:
                if left != -1 or right != -1:
                    raise ValueError("Decision tree contains a partially defined leaf")
                node_features.append("-1")
                left_children.append("0")
                right_children.append("0")
                thresholds.append("0.0")
            else:
                if not (0 <= feature < len(MODEL_FEATURES)):
                    raise ValueError(f"Tree node uses invalid feature index {feature}")
                if not (0 <= left < tree.node_count and 0 <= right < tree.node_count):
                    raise ValueError("Decision tree contains an invalid child index")
                node_features.append(str(feature))
                left_children.append(str(left))
                right_children.append(str(right))
                thresholds.append(format(float(tree.threshold[node_index]), ".17g"))
            leaf_values.append(format(float(tree.value[node_index, 0, 0]), ".17g"))
        tree_offsets.append(tree_offsets[-1] + tree.node_count)

    if tree_offsets[-1] > np.iinfo(np.uint32).max:
        raise ValueError("Forest has too many nodes for the C export format")

    guard = f"{macro_prefix}_H"
    tree_count = len(model.estimators_)
    node_count = tree_offsets[-1]
    header = f"""/*
 * Generated from the {artifact_role} RandomForestRegressor.
 * Output is an approximate capacity-referenced SoC proxy, not validated physical SoC.
 * Feature order: {", ".join(MODEL_FEATURES)}.
 * Inputs are float32 to match scikit-learn tree prediction.
 */
#ifndef {guard}
#define {guard}

#include <math.h>
#include <stdint.h>

#define {macro_prefix}_FEATURE_COUNT {len(MODEL_FEATURES)}
#define {macro_prefix}_TREE_COUNT {tree_count}
#define {macro_prefix}_NODE_COUNT {node_count}

static const uint32_t {macro_prefix.lower()}_tree_offsets[{tree_count + 1}] = {{
{_format_c_array([str(value) for value in tree_offsets])}
}};

static const int8_t {macro_prefix.lower()}_node_features[{node_count}] = {{
{_format_c_array(node_features)}
}};

static const int32_t {macro_prefix.lower()}_left_children[{node_count}] = {{
{_format_c_array(left_children)}
}};

static const int32_t {macro_prefix.lower()}_right_children[{node_count}] = {{
{_format_c_array(right_children)}
}};

static const double {macro_prefix.lower()}_thresholds[{node_count}] = {{
{_format_c_array(thresholds, values_per_line=4)}
}};

static const double {macro_prefix.lower()}_leaf_values[{node_count}] = {{
{_format_c_array(leaf_values, values_per_line=4)}
}};

static inline double {function_name}(const float features[{macro_prefix}_FEATURE_COUNT])
{{
    if (features == 0) {{
        return NAN;
    }}
    for (int32_t feature_index = 0;
         feature_index < {macro_prefix}_FEATURE_COUNT;
         ++feature_index) {{
        if (!isfinite((double)features[feature_index])) {{
            return NAN;
        }}
    }}

    double prediction_sum = 0.0;
    for (uint32_t tree_index = 0;
         tree_index < {macro_prefix}_TREE_COUNT;
         ++tree_index) {{
        const uint32_t tree_start = {macro_prefix.lower()}_tree_offsets[tree_index];
        const uint32_t tree_end = {macro_prefix.lower()}_tree_offsets[tree_index + 1];
        uint32_t node_index = 0;
        uint32_t steps = 0;

        while (tree_start + node_index < tree_end && steps++ < tree_end - tree_start) {{
            const uint32_t flat_index = tree_start + node_index;
            const int8_t feature_index =
                {macro_prefix.lower()}_node_features[flat_index];
            if (feature_index < 0) {{
                prediction_sum += {macro_prefix.lower()}_leaf_values[flat_index];
                break;
            }}
            if (features[feature_index] <=
                {macro_prefix.lower()}_thresholds[flat_index]) {{
                node_index = (uint32_t)
                    {macro_prefix.lower()}_left_children[flat_index];
            }} else {{
                node_index = (uint32_t)
                    {macro_prefix.lower()}_right_children[flat_index];
            }}
        }}
        if (tree_start + node_index >= tree_end || steps > tree_end - tree_start) {{
            return NAN;
        }}
    }}
    return prediction_sum / {macro_prefix}_TREE_COUNT;
}}

#endif
"""
    return header, {
        "tree_count": tree_count,
        "node_count": node_count,
        "feature_count": len(MODEL_FEATURES),
    }


def _load_comparison_test_features(
    dataset_path: Path,
    comparison_metadata_path: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    comparison_metadata = json.loads(
        comparison_metadata_path.read_text(encoding="utf-8")
    )
    if comparison_metadata.get("input_features") != list(MODEL_FEATURES):
        raise ValueError("Comparison feature order does not match the C API")
    if comparison_metadata.get("target") != "soc_proxy_percent":
        raise ValueError("Comparison metadata has an unexpected target")
    train_ids = comparison_metadata.get("train_cycle_ids")
    test_ids = comparison_metadata.get("test_cycle_ids")
    if not isinstance(train_ids, list) or not train_ids:
        raise ValueError("Comparison metadata has no training cycle IDs")
    if not isinstance(test_ids, list) or not test_ids:
        raise ValueError("Comparison metadata has no held-out cycle IDs")
    if set(train_ids).intersection(test_ids):
        raise ValueError("Comparison metadata leaks cycles across the split")

    features, _, groups = load_proxy_csv(dataset_path)
    present_ids = set(np.unique(groups).tolist())
    declared_ids = set(train_ids) | set(test_ids)
    missing = declared_ids.difference(present_ids)
    if missing:
        raise ValueError(f"Dataset is missing split cycles: {sorted(missing)}")
    unexpected = present_ids.difference(declared_ids)
    if unexpected:
        raise ValueError(f"Comparison split omits cycles: {sorted(unexpected)}")
    test_features = features[np.isin(groups, test_ids)]
    if test_features.size == 0:
        raise ValueError("No samples found for the held-out cycles")
    return test_features, comparison_metadata


def _validate_compiled_headers(
    headers: dict[str, str],
    test_features: np.ndarray,
    python_predictions: dict[str, np.ndarray],
    *,
    compiler: str,
) -> dict[str, dict[str, float | int]]:
    """Compile a C99 harness and compare all held-out C predictions to Python."""
    compiler_path = shutil.which(compiler)
    if compiler_path is None:
        raise FileNotFoundError(f"C compiler not found on PATH: {compiler}")
    with tempfile.TemporaryDirectory(prefix="panasonic-c-export-") as temp_name:
        temp_dir = Path(temp_name)
        header_dir = temp_dir / "headers"
        header_dir.mkdir()
        for file_name, contents in headers.items():
            (header_dir / file_name).write_text(contents, encoding="utf-8")

        harness = temp_dir / "validate_export.c"
        harness.write_text(
            """#include <stdio.h>
#include "panasonic_soc_proxy_selected_rf.h"
#include "panasonic_soc_proxy_rf_50_trees.h"

int main(void)
{
    float features[4];
    while (scanf("%f %f %f %f",
                 &features[0], &features[1], &features[2], &features[3]) == 4) {
        printf("%.17g %.17g\\n",
               panasonic_soc_proxy_selected_rf_predict(features),
               panasonic_soc_proxy_rf_50_trees_predict(features));
    }
    return ferror(stdin) ? 1 : 0;
}
""",
            encoding="utf-8",
        )
        executable = temp_dir / "validate_export"
        compile_process = subprocess.run(
            [
                compiler_path,
                "-std=c99",
                "-O2",
                "-Wall",
                "-Wextra",
                "-Werror",
                f"-I{header_dir}",
                str(harness),
                "-lm",
                "-o",
                str(executable),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if compile_process.returncode != 0:
            raise RuntimeError(
                "Generated C headers failed to compile: "
                + compile_process.stderr.strip()
            )

        input_rows = "".join(
            " ".join(format(float(value), ".9g") for value in row) + "\n"
            for row in test_features
        )
        process = subprocess.run(
            [str(executable)],
            input=input_rows,
            check=False,
            capture_output=True,
            text=True,
        )
        if process.returncode != 0:
            raise RuntimeError(
                "Compiled C validation harness failed: "
                + process.stderr.strip()
            )
        c_predictions = np.loadtxt(process.stdout.splitlines(), dtype=float)
        if c_predictions.ndim == 1:
            c_predictions = c_predictions.reshape(1, -1)
        if c_predictions.shape != (test_features.shape[0], len(EXPORTS)):
            raise RuntimeError(
                "C validation returned shape "
                f"{c_predictions.shape}, expected "
                f"({test_features.shape[0]}, {len(EXPORTS)})"
            )

    report: dict[str, dict[str, float | int]] = {}
    for index, export in enumerate(EXPORTS):
        c_values = c_predictions[:, index]
        python_values = python_predictions[export["name"]]
        differences = np.abs(c_values - python_values)
        if not np.isfinite(c_values).all():
            raise ValueError(f"C export returned non-finite outputs for {export['name']}")
        max_difference = float(np.max(differences))
        if max_difference > PARITY_ABSOLUTE_TOLERANCE:
            raise AssertionError(
                f"C/Python prediction mismatch for {export['name']}: "
                f"maximum absolute error {max_difference:.12g} exceeds "
                f"{PARITY_ABSOLUTE_TOLERANCE}"
            )
        report[export["name"]] = {
            "sample_count": int(c_values.size),
            "maximum_absolute_error": max_difference,
            "mean_absolute_error": float(np.mean(differences)),
            "absolute_tolerance": PARITY_ABSOLUTE_TOLERANCE,
            "passed": True,
        }
    return report


def export_models(
    baseline_path: Path,
    optimized_path: Path,
    dataset_path: Path,
    comparison_metadata_path: Path,
    output_dir: Path,
    validation_path: Path,
    *,
    compiler: str = "cc",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Render, compile-check, and persist both approved forest artifacts."""
    input_paths = (
        baseline_path,
        optimized_path,
        dataset_path,
        comparison_metadata_path,
    )
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required export input not found: {path}")

    output_paths = [output_dir / item["path_name"] for item in EXPORTS]
    output_paths.append(validation_path)
    existing = [path for path in output_paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing C export output(s): "
            + ", ".join(str(path) for path in existing)
            + "; pass --overwrite to replace them."
        )

    artifacts = {
        EXPORTS[0]["name"]: baseline_path,
        EXPORTS[1]["name"]: optimized_path,
    }
    headers: dict[str, str] = {}
    model_records: dict[str, dict[str, Any]] = {}
    models: dict[str, RandomForestRegressor] = {}
    for export in EXPORTS:
        artifact_path = artifacts[export["name"]]
        model = joblib.load(artifact_path)
        header, structure = render_header(
            model,
            function_name=export["function_name"],
            macro_prefix=export["macro_prefix"],
            artifact_role=export["artifact_role"],
        )
        headers[export["path_name"]] = header
        models[export["name"]] = model
        model_records[export["name"]] = {
            "artifact_role": export["artifact_role"],
            "source_artifact": str(artifact_path),
            "source_artifact_sha256": hashlib.sha256(
                artifact_path.read_bytes()
            ).hexdigest(),
            "source_artifact_size_bytes": artifact_path.stat().st_size,
            "header": str(output_dir / export["path_name"]),
            "function": export["function_name"],
            "header_size_bytes": len(header.encode("utf-8")),
            **structure,
        }

    test_features, comparison_metadata = _load_comparison_test_features(
        dataset_path,
        comparison_metadata_path,
    )
    python_predictions = {
        export["name"]: np.asarray(
            models[export["name"]].predict(test_features),
            dtype=np.float64,
        )
        for export in EXPORTS
    }
    parity = _validate_compiled_headers(
        headers,
        test_features,
        python_predictions,
        compiler=compiler,
    )

    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    for export in EXPORTS:
        output_path = output_dir / export["path_name"]
        output_path.write_text(headers[export["path_name"]], encoding="utf-8")
        model_records[export["name"]]["header_size_bytes"] = output_path.stat().st_size

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "export_format": "standalone C99 headers",
        "compiler": shutil.which(compiler),
        "compiler_flags": ["-std=c99", "-O2", "-Wall", "-Wextra", "-Werror"],
        "feature_order": list(MODEL_FEATURES),
        "input_type": "float32",
        "output_type": "double; approximate proxy percentage units",
        "target_status": "approximate_capacity_referenced_coulomb_counting_proxy",
        "split_method": comparison_metadata["split_method"],
        "train_cycle_ids": comparison_metadata["train_cycle_ids"],
        "test_cycle_ids": comparison_metadata["test_cycle_ids"],
        "validation": {
            "method": (
                "Compiled C99 harness compared each generated prediction with "
                "the matching scikit-learn artifact over every held-out row."
            ),
            "passed": all(record["passed"] for record in parity.values()),
            "models": parity,
        },
        "models": model_records,
        "limitations": [
            "The target is a capacity-referenced coulomb-counting proxy, not validated physical SoC.",
            "The held-out cycles were previously used for model selection and optimization; parity validation is not an independent model evaluation.",
            "C headers contain model parameters only. They do not implement sensor drivers, filtering, SoC initialization, charge integration, or hardware-specific memory/power handling.",
            "The 50-tree model remains an optimization candidate and is not promoted over the provisionally selected baseline by this export.",
        ],
        "physical_hardware_tested": False,
        "physical_power_measured": False,
    }
    validation_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export the selected and optimized Panasonic Random Forest models "
            "to self-contained C99 headers and verify C/Python parity."
        )
    )
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--optimized", type=Path, default=DEFAULT_OPTIMIZED)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--comparison-metadata",
        type=Path,
        default=DEFAULT_COMPARISON_METADATA,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--compiler", default="cc")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    report = export_models(
        args.baseline,
        args.optimized,
        args.dataset,
        args.comparison_metadata,
        args.output_dir,
        args.validation,
        compiler=args.compiler,
        overwrite=args.overwrite,
    )
    print("C99 export and compiled C/Python parity validation complete.")
    for name, result in report["models"].items():
        parity = report["validation"]["models"][name]
        print(
            f"{name}: {result['tree_count']} trees, {result['node_count']:,} "
            f"nodes, {result['header_size_bytes']:,} header bytes, "
            f"maximum parity error {parity['maximum_absolute_error']:.3g}."
        )
    print(f"Validation record: {args.validation}")
    print("Proxy target only; no physical edge hardware or power was tested.")


if __name__ == "__main__":
    main()
