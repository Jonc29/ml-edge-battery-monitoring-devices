"""Streamlit dashboard for exploring the existing offline battery experiments."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.ui import data
from src.ui.service import load_saved_model, predict_soc_proxy

PAGE_NAMES = (
    "Dashboard",
    "Battery Monitoring",
    "ML Prediction",
    "Model Comparison",
    "Model Optimization",
    "Performance",
    "Dataset",
    "Edge Deployment",
    "System Information",
)
REQUIRED_FILES = (
    data.PROXY_DATASET,
    data.MODEL_COMPARISON,
    data.OPTIMIZATION_COMPARISON,
    data.RESOURCE_BENCHMARKS,
    data.MODEL_SELECTION,
    data.PROXY_METADATA,
    data.PREPROCESSING_LOG,
    data.DATASET_SUMMARY,
    data.C_EXPORT_VALIDATION,
    data.LEAVE_ONE_CYCLE_OUT,
    data.BASELINE_MODEL,
    data.OPTIMIZED_MODEL,
)

st.set_page_config(
    page_title="Battery Intelligence | Edge ML",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .stApp { background: #f4f7fb; color: #17263c; }
      [data-testid="stSidebar"] { background: #122238; }
      [data-testid="stSidebar"] * { color: #eaf1f8; }
      [data-testid="stSidebar"] [data-testid="stRadio"] label {
        border-radius: 8px; padding: 0.35rem 0.55rem;
      }
      [data-testid="stMetric"] {
        background: #fff; border: 1px solid #e0e7ef;
        border-radius: 12px; padding: 1rem;
      }
      .eyebrow {
        color: #417064; font-size: 0.75rem; font-weight: 700;
        letter-spacing: .12em; text-transform: uppercase;
      }
      .subtle { color: #627389; }
      div[data-testid="stAlert"] { border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Loading prepared Panasonic cycle data…")
def get_proxy_data() -> pd.DataFrame:
    return data.load_proxy_dataset()


@st.cache_data
def get_json(path: Path) -> dict[str, Any]:
    return data.load_json(path)


@st.cache_data
def get_model_comparison() -> pd.DataFrame:
    return data.load_model_comparison()


@st.cache_data
def get_optimization_comparison() -> pd.DataFrame:
    return data.load_optimization_comparison()


@st.cache_resource(show_spinner="Loading saved Random Forest…")
def get_model(variant: str):
    return load_saved_model(variant)


def format_bytes(value: int) -> str:
    return f"{value / 1_000_000:.2f} MB ({value:,} bytes)"


def show_metric_row(metrics: list[tuple[str, str, str | None]]) -> None:
    columns = st.columns(len(metrics))
    for column, (label, value, help_text) in zip(columns, metrics, strict=True):
        with column:
            st.metric(label, value, help=help_text)


def show_mode_notice() -> None:
    st.info(
        "**Offline Simulation · Panasonic 18650PF dataset**  \n"
        "Values shown are recorded dataset measurements or model estimates. "
        "No live sensor or physical edge device is connected."
    )


def render_header(title: str, description: str) -> None:
    st.markdown('<div class="eyebrow">Power-aware battery research</div>', unsafe_allow_html=True)
    st.title(title)
    st.caption(description)


def get_selected_sample(frame: pd.DataFrame, cycle_id: str, row_index: int) -> pd.Series:
    cycle = frame.loc[frame["cycle_id"] == cycle_id]
    if cycle.empty:
        raise ValueError(f"No prepared samples found for cycle {cycle_id!r}")
    if not 0 <= row_index < len(cycle):
        raise IndexError(f"Sample index {row_index} is outside cycle {cycle_id!r}")
    return cycle.iloc[row_index]


def dataset_cycle_choices(frame: pd.DataFrame, metadata: Mapping[str, Any]) -> list[str]:
    recorded = frame["cycle_id"].astype(str).drop_duplicates().tolist()
    held_out = metadata.get("test_cycle_ids", [])
    if not held_out:
        training_data = metadata.get("training_data", {})
        if isinstance(training_data, Mapping):
            held_out = training_data.get("test_cycle_ids", [])
    preferred = [cycle for cycle in held_out if cycle in recorded]
    return preferred + [cycle for cycle in recorded if cycle not in preferred]


def run_dashboard(frame: pd.DataFrame, selection: Mapping[str, Any]) -> None:
    render_header(
        "Battery intelligence dashboard",
        "A dataset-based view of battery signals, SoC-proxy estimation, and edge-oriented model work.",
    )
    show_mode_notice()
    cycles = dataset_cycle_choices(frame, selection)
    cycle_id = cycles[0]
    sample = get_selected_sample(frame, cycle_id, 0)
    prediction = predict_soc_proxy(
        get_model("selected"),
        {column: float(sample[column]) for column in data.FEATURE_COLUMNS},
    )
    model_size = data.BASELINE_MODEL.stat().st_size

    st.caption(
        f"Demonstration sample: {cycle_id}, first prepared row. "
        "It is a recorded dataset row, not a current sensor reading."
    )
    show_metric_row(
        [
            ("Battery voltage", f"{sample['voltage_v']:.3f} V", None),
            ("Battery current", f"{sample['current_a']:.3f} A", None),
            ("Battery temperature", f"{sample['battery_temp_c']:.2f} °C", None),
            ("Estimated SoC proxy", f"{prediction:.2f}%", "Model output; not validated physical SoC."),
        ]
    )
    show_metric_row(
        [
            ("Selected model", "Random Forest · 100 trees", None),
            ("Model artifact size", format_bytes(model_size), "Serialized joblib artifact on this computer."),
            ("Reference proxy", f"{sample[data.TARGET_COLUMN]:.2f}%", "Derived label for this dataset row."),
            ("Elapsed time", f"{sample['elapsed_time_s']:.1f} s", "Time from the selected drive-cycle file."),
        ]
    )
    st.subheader("Recorded cycle trend")
    cycle_frame = frame.loc[frame["cycle_id"] == cycle_id]
    chart_rows = cycle_frame.iloc[:: max(1, math.ceil(len(cycle_frame) / 2500))]
    st.line_chart(
        chart_rows,
        x="elapsed_time_s",
        y="soc_proxy_percent",
        y_label="SoC proxy (%)",
        x_label="Elapsed time (s)",
        height=260,
    )
    st.caption(
        "The reference is a capacity-referenced coulomb-counting proxy. "
        "Performance metrics describe agreement with that proxy only."
    )
    render_comparison_summary()


def run_battery_monitoring(frame: pd.DataFrame) -> None:
    render_header(
        "Battery monitoring",
        "Explore measured signals recorded in the selected 25 °C drive-cycle file.",
    )
    show_mode_notice()
    cycle_ids = frame["cycle_id"].astype(str).drop_duplicates().tolist()
    cycle_id = st.selectbox("Recorded drive cycle", cycle_ids, key="monitor_cycle")
    cycle = frame.loc[frame["cycle_id"] == cycle_id]
    chart_rows = cycle.iloc[:: max(1, math.ceil(len(cycle) / 4000))]
    st.caption(
        f"{len(cycle):,} prepared samples · {cycle['elapsed_time_s'].max():,.1f} s "
        "recorded duration. Charts are downsampled for display only."
    )
    left, right = st.columns(2)
    with left:
        st.subheader("Voltage")
        st.line_chart(
            chart_rows,
            x="elapsed_time_s",
            y="voltage_v",
            y_label="Voltage (V)",
            x_label="Elapsed time (s)",
        )
        st.subheader("Battery temperature")
        st.line_chart(
            chart_rows,
            x="elapsed_time_s",
            y="battery_temp_c",
            y_label="Temperature (°C)",
            x_label="Elapsed time (s)",
        )
    with right:
        st.subheader("Current")
        st.line_chart(
            chart_rows,
            x="elapsed_time_s",
            y="current_a",
            y_label="Current (A)",
            x_label="Elapsed time (s)",
        )
        st.subheader("Derived SoC proxy")
        st.line_chart(
            chart_rows,
            x="elapsed_time_s",
            y="soc_proxy_percent",
            y_label="SoC proxy (%)",
            x_label="Elapsed time (s)",
        )


def run_prediction(frame: pd.DataFrame, selection: Mapping[str, Any]) -> None:
    render_header(
        "ML prediction",
        "Run the saved estimator on explicit inputs or replay a recorded dataset sample.",
    )
    show_mode_notice()
    mode = st.radio(
        "Input mode",
        ("Replay a recorded sample", "Enter measurements"),
        horizontal=True,
    )
    variant_label = st.selectbox(
        "Estimator",
        ("Selected baseline · 100 trees", "Optimization candidate · 50 trees"),
    )
    variant = "selected" if variant_label.startswith("Selected") else "candidate"
    cycle_id: str | None = None
    sample: pd.Series | None = None
    if mode == "Replay a recorded sample":
        choices = dataset_cycle_choices(frame, selection)
        cycle_id = st.selectbox("Recorded cycle", choices, key="prediction_cycle")
        cycle = frame.loc[frame["cycle_id"] == cycle_id].reset_index(drop=True)
        row_index = st.slider(
            "Sample index within cycle",
            min_value=0,
            max_value=len(cycle) - 1,
            value=0,
            key="prediction_sample_index",
        )
        sample = cycle.iloc[row_index]
        st.caption("Replay values come from the prepared dataset; proxy reference is available for comparison.")
        inputs: dict[str, float] = {
            column: float(sample[column]) for column in data.FEATURE_COLUMNS
        }
        with st.expander("Input features"):
            st.dataframe(
                pd.DataFrame(
                    [{"feature": name, "value": value} for name, value in inputs.items()]
                ),
                hide_index=True,
                width="stretch",
            )
    else:
        st.caption("Initial values are taken from a real dataset sample; edit them before running if desired.")
        default_cycle = dataset_cycle_choices(frame, selection)[0]
        defaults = get_selected_sample(frame, default_cycle, 0)
        with st.form("manual_prediction"):
            voltage = st.number_input(
                "Voltage (V)", value=float(defaults["voltage_v"]), step=0.01, format="%.5f"
            )
            current = st.number_input(
                "Current (A)", value=float(defaults["current_a"]), step=0.05, format="%.5f"
            )
            temperature = st.number_input(
                "Battery temperature (°C)",
                value=float(defaults["battery_temp_c"]),
                step=0.1,
                format="%.5f",
            )
            elapsed = st.number_input(
                "Elapsed time (s)",
                min_value=0.0,
                value=float(defaults["elapsed_time_s"]),
                step=1.0,
                format="%.5f",
            )
            submitted = st.form_submit_button("Estimate SoC proxy", type="primary")
        if submitted:
            inputs = {
                "voltage_v": voltage,
                "current_a": current,
                "battery_temp_c": temperature,
                "elapsed_time_s": elapsed,
            }
            _display_prediction(inputs, variant, None, "Manual input")
        render_prediction_history()
        return

    if st.button("Run dataset replay", type="primary"):
        assert sample is not None
        _display_prediction(inputs, variant, float(sample[data.TARGET_COLUMN]), cycle_id or "")
    render_prediction_history()


def _display_prediction(
    inputs: Mapping[str, float],
    variant: str,
    reference: float | None,
    source: str,
) -> None:
    prediction = predict_soc_proxy(get_model(variant), inputs)
    difference = abs(prediction - reference) if reference is not None else None
    if variant == "selected":
        model_name = "Selected baseline · 100 trees"
    else:
        model_name = "Optimization candidate · 50 trees"
    st.success(f"Estimated SoC proxy: **{prediction:.3f}%**")
    if reference is not None:
        st.metric("Dataset proxy reference", f"{reference:.3f}%", f"{difference:.3f} percentage points absolute error")
        st.caption(
            "This per-row difference is a demonstration, not a replacement for "
            "held-out-cycle evaluation or physical SoC validation."
        )
    else:
        st.caption("No reference target exists for manually entered measurements; no error score is reported.")
    history = st.session_state.setdefault("prediction_history", [])
    history.insert(
        0,
        {
            "source": source,
            "model": model_name,
            "prediction_proxy_percent": prediction,
            "reference_proxy_percent": reference,
            "absolute_difference_percentage_points": difference,
        },
    )
    del history[20:]


def render_prediction_history() -> None:
    history = st.session_state.get("prediction_history", [])
    if history:
        st.subheader("Prediction history · this session")
        st.dataframe(pd.DataFrame(history), hide_index=True, width="stretch")
        if st.button("Clear prediction history"):
            st.session_state.pop("prediction_history", None)
            st.rerun()


def render_comparison_summary() -> None:
    comparison = get_model_comparison()
    st.subheader("Model comparison · fixed complete-cycle holdout")
    st.bar_chart(
        comparison.set_index("model")[
            [
                "proxy_test_mae_percentage_points",
                "proxy_test_rmse_percentage_points",
            ]
        ],
        y_label="Error (percentage points)",
    )
    st.dataframe(
        comparison[
            [
                "model",
                "proxy_test_mae_percentage_points",
                "proxy_test_rmse_percentage_points",
                "proxy_test_r2",
            ]
        ].rename(
            columns={
                "model": "Model",
                "proxy_test_mae_percentage_points": "Proxy MAE (pp)",
                "proxy_test_rmse_percentage_points": "Proxy RMSE (pp)",
                "proxy_test_r2": "R²",
            }
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Cycle holdout results are agreement with the derived proxy. "
        "The same cycles informed model selection; this is not an independent final test."
    )


def run_model_comparison() -> None:
    render_header(
        "Model comparison",
        "Compare the existing Ridge, Random Forest, and Extra Trees results.",
    )
    show_mode_notice()
    render_comparison_summary()
    selection = get_json(data.MODEL_SELECTION)
    st.info(f"Provisional selection: **{selection['selected_model']}**. {selection['reason']}")
    cycle_cv = get_json(data.LEAVE_ONE_CYCLE_OUT)
    cv_rows = [
        {
            "Model": name,
            "Pooled proxy MAE (pp)": values["pooled_out_of_fold_metrics"][
                "mae_percentage_points"
            ],
            "Pooled proxy RMSE (pp)": values["pooled_out_of_fold_metrics"][
                "rmse_percentage_points"
            ],
            "Pooled R²": values["pooled_out_of_fold_metrics"]["r2"],
            "Worst cycle by MAE": values["worst_cycle_by_mae"],
        }
        for name, values in cycle_cv["models"].items()
    ]
    with st.expander("Supplementary leave-one-cycle-out robustness results"):
        st.dataframe(pd.DataFrame(cv_rows), hide_index=True, width="stretch")
        st.caption(cycle_cv["interpretation"])


def run_optimization() -> None:
    render_header(
        "Model optimization",
        "Compare the selected baseline with the 50-tree size-optimized candidate.",
    )
    show_mode_notice()
    results = get_optimization_comparison()
    baseline = results.loc[results["variant"] == "selected_baseline"]
    candidate = results.loc[results["variant"] == "rf_50_trees"]
    if len(baseline) != 1 or len(candidate) != 1:
        raise ValueError("Optimization results must contain one baseline and one 50-tree candidate.")
    base = baseline.iloc[0]
    optimized = candidate.iloc[0]
    reduction = float(optimized["artifact_size_reduction_percent_vs_baseline"])
    mae_change = float(optimized["mae_change_percentage_points_vs_baseline"])
    show_metric_row(
        [
            ("Baseline · 100 trees", format_bytes(int(base["artifact_size_bytes"])), None),
            ("Candidate · 50 trees", format_bytes(int(optimized["artifact_size_bytes"])), None),
            ("Artifact size reduction", f"{reduction:.2f}%", "Serialized file size only; not measured power."),
            ("Proxy MAE change", f"+{mae_change:.3f} pp", "Positive means the candidate error is higher."),
        ]
    )
    st.subheader("Before and after")
    chart = pd.DataFrame(
        {
            "Artifact size (MB)": [
                float(base["artifact_size_bytes"]) / 1_000_000,
                float(optimized["artifact_size_bytes"]) / 1_000_000,
            ],
            "Proxy MAE (pp)": [
                float(base["proxy_test_mae_percentage_points"]),
                float(optimized["proxy_test_mae_percentage_points"]),
            ],
        },
        index=["Baseline · 100 trees", "Candidate · 50 trees"],
    )
    left, right = st.columns(2)
    with left:
        st.bar_chart(chart[["Artifact size (MB)"]], y_label="Serialized size (MB)")
    with right:
        st.bar_chart(chart[["Proxy MAE (pp)"]], y_label="MAE (percentage points)")
    st.dataframe(
        results[
            [
                "variant",
                "n_estimators",
                "artifact_size_bytes",
                "proxy_test_mae_percentage_points",
                "proxy_test_rmse_percentage_points",
                "proxy_test_r2",
            ]
        ],
        hide_index=True,
        width="stretch",
    )
    st.warning(
        "The optimization candidate is smaller, but the measurements are software "
        "artifact size and proxy agreement—not physical power savings. The shared "
        "holdout informed model selection and optimization."
    )


def run_performance() -> None:
    render_header(
        "Software performance",
        "Review measured inference latency, serialized size, and host-process memory.",
    )
    show_mode_notice()
    benchmark = get_json(data.RESOURCE_BENCHMARKS)
    st.warning(
        "Host-side software benchmark only. RSS includes the Python/scientific "
        "runtime and test data; it is not model-only RAM. Physical electrical "
        "power measurement is unavailable."
    )
    st.caption(
        f"Host: {benchmark['host']['processor']} · "
        f"Python {benchmark['host']['python_version']} · "
        f"Benchmark date: {benchmark['benchmark_timestamp_utc']}"
    )
    rows: list[dict[str, Any]] = []
    for model in benchmark["results"]:
        memory = model["memory"]
        for latency in model["latency"]:
            rows.append(
                {
                    "Model": model["model"],
                    "Artifact size (bytes)": model["artifact_size_bytes"],
                    "Batch size": latency["batch_size"],
                    "Median latency (ms/batch)": latency["latency_median_ms_per_batch"],
                    "P95 latency (ms/batch)": latency["latency_p95_ms_per_batch"],
                    "Median latency (µs/sample)": latency["median_latency_us_per_sample"],
                    "Process RSS high-water (MB)": memory["rss_high_water_bytes"] / 1_000_000,
                }
            )
    performance = pd.DataFrame(rows)
    st.dataframe(performance, hide_index=True, width="stretch")
    st.subheader("Median latency by batch size")
    st.line_chart(
        performance,
        x="Batch size",
        y="Median latency (ms/batch)",
        color="Model",
        y_label="Milliseconds per batch",
    )
    st.caption(benchmark["latency_method"])


def run_dataset(frame: pd.DataFrame) -> None:
    render_header(
        "Dataset and preprocessing",
        "What the current prototype learned from, and how its proxy target was prepared.",
    )
    show_mode_notice()
    summary = get_json(data.DATASET_SUMMARY)
    metadata = get_json(data.PROXY_METADATA)
    preprocessing = get_json(data.PREPROCESSING_LOG)
    show_metric_row(
        [
            ("Prepared records", f"{len(frame):,}", "1 Hz prepared samples, not unique raw archive observations."),
            ("Drive cycles", str(frame["cycle_id"].nunique()), None),
            ("Raw MATLAB files", str(summary["matlab_file_count"]), None),
            ("Sampling target", "1 Hz selected samples", "Logged observations selected near regular one-second targets; not interpolation."),
        ]
    )
    st.subheader("Model inputs and target")
    st.dataframe(
        pd.DataFrame(
            [
                {"Role": "Input", "Channel": "Voltage", "Column": "voltage_v", "Unit": "V"},
                {"Role": "Input", "Channel": "Current", "Column": "current_a", "Unit": "A"},
                {"Role": "Input", "Channel": "Battery temperature", "Column": "battery_temp_c", "Unit": "°C"},
                {"Role": "Input", "Channel": "Elapsed time", "Column": "elapsed_time_s", "Unit": "s"},
                {"Role": "Target", "Channel": "SoC proxy", "Column": "soc_proxy_percent", "Unit": "% (derived proxy)"},
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    st.subheader("Drive-cycle coverage")
    st.dataframe(
        pd.DataFrame(metadata["drive_cycle_summaries"])[
            [
                "cycle_id",
                "source_observations",
                "prepared_observations",
                "proxy_min_percent",
                "proxy_max_percent",
                "out_of_range_proxy_samples",
            ]
        ],
        hide_index=True,
        width="stretch",
    )
    st.write(
        f"Reference capacity: **{metadata['reference_capacity_ah']:.5f} Ah**, "
        "the mean of two measured pre-test 1C throughputs."
    )
    checks = preprocessing["checks"]
    st.write(
        f"Preprocessing checks: {preprocessing['rows_removed']} rows removed; "
        f"{checks['duplicate_cycle_time_keys']} duplicate cycle-time keys; "
        f"{checks['out_of_range_proxy_rows_preserved']} preserved out-of-range proxy rows. "
        "Values were not clipped."
    )
    raw_rows = summary["file_row_observation_count"]
    st.caption(
        f"The raw archive sums to {raw_rows:,} file-row observations; its dataset "
        "documentation notes duplicated contiguous/split drive-cycle records, "
        "so this is not a unique observation count."
    )
    if data.FEATURE_CORRELATION_FIGURE.is_file():
        st.image(str(data.FEATURE_CORRELATION_FIGURE), caption="Descriptive feature/target correlations")
    if data.FEATURE_DISTRIBUTIONS_FIGURE.is_file():
        st.image(str(data.FEATURE_DISTRIBUTIONS_FIGURE), caption="Pooled-row feature and proxy distributions")
    st.caption(metadata["limitations"][0])


def run_edge_deployment() -> None:
    render_header(
        "Edge deployment pathway",
        "A software export and host simulation that prepare for—not claim—physical deployment.",
    )
    show_mode_notice()
    validation = get_json(data.C_EXPORT_VALIDATION)
    stages = [
        ("Dataset measurements", "Available", "Panasonic 18650PF archive and prepared 25 °C cycles"),
        ("Preprocessing", "Available", "Validated local feature table and training-only scaler"),
        ("Python model", "Available", "Selected 100-tree Random Forest artifact"),
        ("Optimized candidate", "Available", "50-tree candidate; not promoted over selected baseline"),
        ("C99 export", "Available", "Standalone inference headers"),
        ("Host C99 simulator", "Available", "Serial-style CSV input on the development computer"),
        (
            "C99 parity validation",
            "Passed" if validation["validation"]["passed"] else "Failed",
            "Checked against Python on held-out rows",
        ),
        ("Physical microcontroller", "Not implemented", "Not implemented or tested"),
    ]
    for title, state, detail in stages:
        message = f"**{title} — {state}.** {detail}"
        if state in {"Available", "Passed"}:
            st.success(message)
        elif state == "Failed":
            st.error(message)
        else:
            st.warning(message)
    st.metric(
        "C/Python parity samples per model",
        f"{validation['validation']['models']['selected_baseline_100_trees']['sample_count']:,}",
        help="Prediction parity test only; does not establish physical SoC accuracy.",
    )
    st.caption(
        "The C simulator runs on the development computer. Headers do not include "
        "sensor drivers, SoC initialization, charge integration, or hardware-specific "
        "memory and power handling."
    )


def run_system_information(selection: Mapping[str, Any]) -> None:
    render_header(
        "System information",
        "Application state, model provenance, and evidence boundaries.",
    )
    show_mode_notice()
    model_size = data.BASELINE_MODEL.stat().st_size
    info = [
        ("Project", "Power-Aware Machine Learning for Edge Battery Monitoring Devices"),
        ("Application mode", "Offline dataset simulation"),
        ("Dataset", "Panasonic 18650PF · 25 °C drive cycles"),
        ("Target", "Approximate capacity-referenced coulomb-counting SoC proxy"),
        ("Selected model", f"{selection['selected_model']} · 100 trees · provisional"),
        ("Selected artifact size", format_bytes(model_size)),
        ("Software version", "Not recorded in the experiment metadata"),
        ("Physical device", "Not connected / not tested"),
        ("Physical power", "Not available — no physical edge hardware used"),
    ]
    st.dataframe(
        pd.DataFrame(info, columns=["Property", "Current value"]),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "No live readings, real battery SoC, microcontroller deployment, or "
        "electrical power savings are claimed."
    )


def main() -> None:
    missing = [path for path in REQUIRED_FILES if not path.is_file()]
    if missing:
        st.error("Some local experiment artifacts required by the dashboard are missing:")
        st.code("\n".join(str(path.relative_to(data.PROJECT_ROOT)) for path in missing))
        st.info(
            "Restore the prepared dataset and model/result artifacts, or run the "
            "corresponding preparation and experiment commands from README.md."
        )
        st.stop()

    with st.sidebar:
        st.markdown("## 🔋 Battery intelligence")
        st.caption("POWER-AWARE EDGE ML · RESEARCH PROTOTYPE")
        page = st.radio(
            "Navigation",
            PAGE_NAMES,
            key="page_navigation",
            label_visibility="collapsed",
        )
        st.divider()
        st.markdown("**Operating mode**")
        st.success("Offline Simulation")
        st.caption("No physical device connected")

    frame = get_proxy_data()
    selection = get_json(data.MODEL_SELECTION)
    if page == "Dashboard":
        run_dashboard(frame, selection)
    elif page == "Battery Monitoring":
        run_battery_monitoring(frame)
    elif page == "ML Prediction":
        run_prediction(frame, selection)
    elif page == "Model Comparison":
        run_model_comparison()
    elif page == "Model Optimization":
        run_optimization()
    elif page == "Performance":
        run_performance()
    elif page == "Dataset":
        run_dataset(frame)
    elif page == "Edge Deployment":
        run_edge_deployment()
    else:
        run_system_information(selection)


main()
