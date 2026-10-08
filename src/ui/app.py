"""Streamlit dashboard for exploring the existing offline battery experiments."""

from __future__ import annotations

import math
from html import escape
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.ui import data
from src.ui.service import load_saved_model, predict_soc_proxy

PAGE_NAMES = (
    "Overview",
    "Recorded signals",
    "Try an estimate",
    "Project details",
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
    page_title="Battery Monitor | Project Demo",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .stApp { background: #101013; color: #f5f7f8; }
      [data-testid="stHeader"] { background: rgba(16, 16, 19, .92); }
      [data-testid="stSidebar"] { background: #17171b; border-right: 1px solid #303037; }
      [data-testid="stSidebar"] * { color: #e9e9ec; }
      [data-testid="stSidebar"] [data-testid="stRadio"] label {
        border-radius: 8px; padding: 0.45rem 0.6rem;
      }
      [data-testid="stMetric"] {
        background: #1d1d22; border: 1px solid #35353c;
        border-radius: 12px; padding: 1rem;
      }
      .eyebrow {
        color: #38d6d0; font-size: 0.75rem; font-weight: 700;
        letter-spacing: .12em; text-transform: uppercase;
      }
      .subtle { color: #b4b4bc; }
      [data-testid="stCaptionContainer"] { color: #b4b4bc; }
      .offline-banner {
        margin: .5rem 0 1rem; padding: .85rem 1rem;
        border-left: 4px solid #27c7e5; border-radius: 8px;
        background: #17282d; color: #f0fcff; line-height: 1.5;
      }
      .offline-banner strong { color: #55e1dc; letter-spacing: .035em; }
      .instrument-panel {
        position: relative; overflow: hidden; isolation: isolate;
        display: grid; grid-template-columns: minmax(230px, .9fr) minmax(300px, 1.1fr);
        gap: 2rem; align-items: center; padding: 2rem 2.25rem;
        border: 1px solid #57575c; border-radius: 22px;
        background: linear-gradient(112deg, #101013 0%, #17171b 70%, #36363b 100%);
        box-shadow: inset 0 0 0 4px #080809, 0 16px 40px rgba(0,0,0,.24);
      }
      .instrument-panel:after {
        content: ""; position: absolute; inset: 0; z-index: -1; pointer-events: none;
        background: linear-gradient(112deg, transparent 67%, rgba(255,255,255,.07) 67.2%, rgba(255,255,255,.015) 100%);
      }
      .instrument-brand { color: #aaaab0; font-size: .76rem; letter-spacing: .36em; text-align: center; }
      .instrument-label { color: #38d6d0; font-size: .8rem; font-weight: 800; letter-spacing: .04em; text-transform: uppercase; }
      .instrument-subtitle { color: #f4f4f6; font-size: .92rem; font-weight: 650; margin-top: .15rem; }
      .charge-gauge { width: 190px; height: 190px; position: relative; margin: 1.1rem auto .7rem; }
      .charge-gauge svg { display:block; width:100%; height:100%; transform:rotate(-90deg); }
      .charge-track { fill:none; stroke:#29292e; stroke-width:10; }
      .charge-value { fill:none; stroke:#27c7e5; stroke-width:7; stroke-linecap:round; }
      .charge-center { position:absolute; inset:0; display:flex; flex-direction:column; align-items:center; justify-content:center; }
      .charge-number { color:#fff; font-size:2.8rem; font-weight:750; line-height:1; }
      .charge-unit { color:#d3d3d8; font-size:.9rem; margin-top:.3rem; }
      .gauge-caption { color:#b7b7bf; font-size:.78rem; text-align:center; }
      .instrument-readings { padding: .4rem 0; }
      .reading-row { padding: .8rem 0 .65rem; border-bottom:1px solid #45454b; }
      .reading-row:last-child { border-bottom:0; }
      .reading-head { display:flex; align-items:baseline; justify-content:space-between; gap:1rem; }
      .reading-name { color:#efeff1; font-size:.85rem; font-weight:650; }
      .reading-value { color:#fff; font-size:1.25rem; font-weight:700; white-space:nowrap; }
      .reading-unit { color:#b7b7bf; font-size:.74rem; margin-left:.2rem; }
      .reading-accent { height:3px; margin-top:.55rem; background:#f5a623; border-radius:4px; }
      .reading-accent.cyan { background:#27c7e5; }
      .sample-tag { display:inline-block; margin-top:1rem; padding:.32rem .58rem; border:1px solid #4b4b52; border-radius:20px; color:#d6d6da; font-size:.72rem; }
      @media (max-width: 1050px) {
        .instrument-panel { grid-template-columns:1fr; gap:.4rem; padding:1.4rem 1rem; }
        .charge-gauge { width:165px; height:165px; }
      }
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
    st.markdown(
        '<div class="offline-banner"><strong>OFFLINE DEMONSTRATION · RECORDED BATTERY DATA</strong><br>'
        'This screen replays laboratory measurements. No live battery or sensor is connected.</div>',
        unsafe_allow_html=True,
    )


def render_header(title: str, description: str) -> None:
    st.markdown('<div class="eyebrow">Battery monitor · research prototype</div>', unsafe_allow_html=True)
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
        "Battery overview",
        "A simple view of one recorded battery sample and its model estimate.",
    )
    show_mode_notice()
    cycles = dataset_cycle_choices(frame, selection)
    cycle_id = cycles[0]
    sample = get_selected_sample(frame, cycle_id, 0)
    prediction = predict_soc_proxy(
        get_model("selected"),
        {column: float(sample[column]) for column in data.FEATURE_COLUMNS},
    )
    gauge_value = min(100.0, max(0.0, prediction))
    circumference = 2 * math.pi * 51
    dash_offset = circumference * (1.0 - gauge_value / 100.0)
    st.markdown(
        f"""
        <section class="instrument-panel" aria-label="Recorded battery sample display">
          <div>
            <div class="instrument-brand">BATTERY MONITOR</div>
            <div class="instrument-label">MAIN BATTERY</div>
            <div class="instrument-subtitle">Estimated charge level</div>
            <div class="charge-gauge" role="img" aria-label="Estimated charge level {prediction:.1f} percent">
              <svg viewBox="0 0 120 120" aria-hidden="true">
                <circle class="charge-track" cx="60" cy="60" r="51"></circle>
                <circle class="charge-value" cx="60" cy="60" r="51"
                  stroke-dasharray="{circumference:.2f}" stroke-dashoffset="{dash_offset:.2f}"></circle>
              </svg>
              <div class="charge-center">
                <div class="charge-number">{prediction:.1f}</div>
                <div class="charge-unit">percent estimate</div>
              </div>
            </div>
            <div class="gauge-caption">Model estimate from a recorded test sample</div>
            <div class="sample-tag">RECORDED SAMPLE · PANASONIC 25 °C DATA</div>
          </div>
          <div class="instrument-readings">
            <div class="reading-row">
              <div class="reading-head"><span class="reading-name">VOLTAGE</span>
                <span class="reading-value">{float(sample['voltage_v']):.3f}<span class="reading-unit">V</span></span></div>
              <div class="reading-accent cyan"></div>
            </div>
            <div class="reading-row">
              <div class="reading-head"><span class="reading-name">CURRENT</span>
                <span class="reading-value">{float(sample['current_a']):.3f}<span class="reading-unit">A</span></span></div>
              <div class="reading-accent"></div>
            </div>
            <div class="reading-row">
              <div class="reading-head"><span class="reading-name">BATTERY TEMPERATURE</span>
                <span class="reading-value">{float(sample['battery_temp_c']):.2f}<span class="reading-unit">°C</span></span></div>
              <div class="reading-accent"></div>
            </div>
            <div class="reading-row">
              <div class="reading-head"><span class="reading-name">TIME INTO RECORDED CYCLE</span>
                <span class="reading-value">{float(sample['elapsed_time_s']):.0f}<span class="reading-unit">s</span></span></div>
              <div class="reading-accent cyan"></div>
            </div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("What does the charge estimate mean?"):
        st.write(
            "This is a model estimate based on recorded battery measurements, "
            "compared with an approximate reference calculated from the dataset. "
            "It is not a live reading or a verified measurement of the battery's true charge."
        )
        st.caption(
            f"Recorded reference for this sample: {float(sample[data.TARGET_COLUMN]):.2f}%. "
            "For the full model tests and limitations, open Project details."
        )
        st.caption(f"Source test file: {cycle_id}")
    st.subheader("Recorded charge reference over the test")
    cycle_frame = frame.loc[frame["cycle_id"] == cycle_id]
    chart_rows = cycle_frame.iloc[:: max(1, math.ceil(len(cycle_frame) / 2500))]
    st.line_chart(
        chart_rows,
        x="elapsed_time_s",
        y="soc_proxy_percent",
        y_label="Approximate reference (%)",
        x_label="Time in test (seconds)",
        height=260,
    )
    st.caption(
        "The reference is calculated from the recorded test data; it is approximate, not independently measured charge."
    )


def run_battery_monitoring(frame: pd.DataFrame) -> None:
    render_header(
        "Recorded battery signals",
        "Choose a laboratory test to view its recorded voltage, current, and temperature.",
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
        st.subheader("Approximate charge reference")
        st.line_chart(
            chart_rows,
            x="elapsed_time_s",
            y="soc_proxy_percent",
            y_label="Reference (%)",
            x_label="Time in test (s)",
        )


def run_prediction(frame: pd.DataFrame, selection: Mapping[str, Any]) -> None:
    render_header(
        "Try a charge estimate",
        "Replay a recorded sample or enter battery measurements to see an example estimate.",
    )
    show_mode_notice()
    st.caption(
        "The percentage is an approximate model estimate from the project's "
        "research data, not a verified measurement of a battery's true charge."
    )
    mode = st.radio(
        "Choose what to try",
        ("Use a recorded sample", "Enter measurements"),
        horizontal=True,
    )
    variant_label = "Selected baseline · 100 trees"
    with st.expander("Advanced: choose model version"):
        variant_label = st.selectbox(
            "Model version",
            ("Selected baseline · 100 trees", "Smaller candidate · 50 trees"),
        )
    variant = "selected" if variant_label.startswith("Selected") else "candidate"
    cycle_id: str | None = None
    sample: pd.Series | None = None
    if mode == "Use a recorded sample":
        choices = dataset_cycle_choices(frame, selection)
        cycle_id = st.selectbox("Recorded test", choices, key="prediction_cycle")
        cycle = frame.loc[frame["cycle_id"] == cycle_id].reset_index(drop=True)
        row_index = st.slider(
            "Position in recorded test",
            min_value=0,
            max_value=len(cycle) - 1,
            value=0,
            key="prediction_sample_index",
        )
        sample = cycle.iloc[row_index]
        st.caption("These are recorded measurements, not readings from a connected battery.")
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
        st.caption("Start with sample values, then change any measurement before requesting an estimate.")
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
            submitted = st.form_submit_button("Show estimate", type="primary")
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

    if st.button("Show estimate for this sample", type="primary"):
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
    st.success(f"Estimated charge level: **{prediction:.3f}%**")
    if reference is not None:
        st.metric("Approximate recorded reference", f"{reference:.3f}%", f"{difference:.3f} percentage points difference")
        st.caption(
            "This per-row difference is a demonstration, not a replacement for "
            "held-out-cycle evaluation or physical SoC validation."
        )
    else:
        st.caption("For entered measurements there is no recorded reference to compare against.")
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


def run_project_details(
    frame: pd.DataFrame, selection: Mapping[str, Any]
) -> None:
    st.markdown('<div class="eyebrow">Additional information</div>', unsafe_allow_html=True)
    st.title("Project details")
    st.caption("Technical evidence for assessment and further investigation.")
    section = st.selectbox(
        "Choose a topic",
        (
            "Model results",
            "Smaller model",
            "Computer performance",
            "Dataset and preparation",
            "C software export",
            "System notes",
        ),
    )
    if section == "Model results":
        run_model_comparison()
    elif section == "Smaller model":
        run_optimization()
    elif section == "Computer performance":
        run_performance()
    elif section == "Dataset and preparation":
        run_dataset(frame)
    elif section == "C software export":
        run_edge_deployment()
    else:
        run_system_information(selection)


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
        st.markdown("## 🔋 Battery monitor")
        st.caption("PROJECT DEMONSTRATION")
        page = st.radio(
            "Navigation",
            PAGE_NAMES,
            key="page_navigation",
            label_visibility="collapsed",
        )
        st.divider()
        st.markdown("**Data source**")
        st.success("Recorded laboratory data")
        st.caption("No live sensor is connected.")

    frame = get_proxy_data()
    selection = get_json(data.MODEL_SELECTION)
    if page == "Overview":
        run_dashboard(frame, selection)
    elif page == "Recorded signals":
        run_battery_monitoring(frame)
    elif page == "Try an estimate":
        run_prediction(frame, selection)
    else:
        run_project_details(frame, selection)


main()
