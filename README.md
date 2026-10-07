# Power-Aware Machine Learning for Edge Battery Monitoring Devices

This project implements the software experiments for a postgraduate research
project on estimating battery operating conditions with machine learning while
considering suitability for resource-constrained edge devices.

The research document (Chapters 1–3) is the primary specification. The
Panasonic 18650PF archive is the current dataset candidate. Its SoC labels are
derived proxies, not independent ground truth; see the documented assumptions
and limitations before interpreting model scores.

## Current status

- Phase 0 (research and implementation analysis): complete.
- Phase 1 (development environment): verified in the active Conda environment.
- Phase 2 (project structure): scaffolded.
- Phase 3 (dataset candidate investigation): archive inspected; Panasonic is
  the selected candidate for a derived SoC-proxy prototype.
- Phase 4 (initial raw-data exploration): MATLAB inventory and a representative
  measured-signal plot generated; SoC label validity remains unresolved.
- Phase 5 (target validation): measured Ah/current consistency audit added for
  separate 25 °C drive cycles; absolute SoC validity remains unresolved.
- Phase 6 (model setup): created an approximate capacity-referenced SoC proxy
  from measured pre-test 1C capacities and trained a grouped-holdout baseline.
  Scores quantify proxy agreement only.
- Roadmap Phase 3 (preprocessing): added input validation, a cleaning log, and
  a `StandardScaler` fitted only on complete-cycle training rows. Generated
  clean and scaled tables are local processed artifacts, not ground-truth data.
- Roadmap Phase 4 (EDA): representative measured-signal, distribution, and
  correlation figures have been generated for the prepared Panasonic proxy
  table.
- Roadmap Phase 5 (model comparison): Ridge Regression, Random Forest, and
  Extra Trees were compared on the same fixed three-cycle holdout.
- Roadmap Phase 6 (model selection): Random Forest is selected provisionally
  for optimization based on the fixed-holdout proxy-agreement results. This is
  not validated physical SoC performance.
- Roadmap Phase 7 (optimization): smaller Random Forest variants were measured
  for proxy agreement and serialized file size. The 50-tree model is half the
  baseline artifact size and has been benchmarked in Phase 8.
- Roadmap Phase 8 (resource benchmarking): the selected 100-tree model and
  50-tree candidate have host-side latency, serialized-size, and process-RSS
  measurements. These are software measurements, not physical edge-device
  results; the single-row timing is noisy on this host.
- Roadmap Phase 9 (C export): standalone C99 headers were generated for the
  selected baseline and the 50-tree optimization candidate. Compiled C
  predictions matched Python on all 37,326 held-out samples within 4.26e-14
  proxy percentage points.
- Roadmap Phase 10 (edge simulation): a host-compiled C program accepts
  serial-style CSV feature rows and returns proxy predictions for either
  exported forest. The simulator is not firmware and has not run on a physical
  edge device.
- Roadmap Phase 11 (cycle robustness evaluation): leave-one-complete-cycle-out
  comparisons evaluate each of the ten cycles once with all rows from that
  cycle held out. This is a supplementary proxy-robustness analysis, not an
  independent confirmatory test, because these cycles informed prior model
  selection and optimization. Pooled proxy MAE was 3.111 percentage points for
  Random Forest, 4.008 for Ridge Regression, and 4.523 for Extra Trees.
- Physical hardware testing and power measurement: not performed.

See the [dataset candidate investigation](./reports/dataset-investigation.md)
for sources, verified metadata, open questions, and the current recommendation.

## Scope and evidence boundary

The initial implementation uses a suitable, verified public battery dataset for
offline software experiments. It will not represent synthetic readings as
measurements or claim that sensors or physical hardware were used when they
were not.

The software experiments may measure prediction metrics, serialized model
size, local inference latency, and software memory/resource indicators using
documented methods. Local inference on a computer is an **edge inference
simulation**, not a physical edge-device test.

Physical sensor validation, microcontroller deployment, and electrical power
measurement require actual hardware and recorded measurements. Device power
will only be reported if it is physically measured; otherwise it will be
reported as unavailable. Model size or computer-side latency will not be
presented as a direct measurement of electrical power.

## Intended workflow

1. Investigate candidate public datasets and document their sources,
   measurements, labels, operating conditions, and limitations.
2. Select and justify the target and input features from verified dataset
   contents.
3. Explore and preprocess the approved dataset, keeping transformations fitted
   on training data only.
4. Choose a train/test strategy appropriate to the battery, cycle, and time
   structure to reduce leakage.
5. Train and evaluate appropriate baseline models.
6. Compare prediction quality with measured resource requirements.
7. Apply only technically appropriate optimization and compare it with the
   original model using controlled experiments.
8. Demonstrate local software inference and clearly label it as simulation
   unless physical edge hardware is actually used.

Metrics, configurations, plots, and other generated evidence will be saved
under `results/`. Dataset files are excluded from Git; their source, access
instructions, and any required checksums should be documented without
redistributing data contrary to its license.

## Project layout

```text
data/
  raw/                 Public dataset files (not committed)
  processed/           Derived data files (not committed)
notebooks/              Exploration and experiment notebooks
src/
  data/                 Dataset loading and validation
  preprocessing/        Reusable preprocessing logic
  models/               Model definitions and training helpers
  optimization/         Resource-reduction experiments
  evaluation/            Metrics and resource evaluation
models/
  baseline/             Baseline model artifacts
  optimized/            Optimized model artifacts
edge/
  inference/            Local inference interface
  simulation/           Software-only edge inference experiments
  firmware/             Reserved for a later hardware phase
  exported/             Deployment exports, if needed
results/
  metrics/
  plots/
  tables/
  comparisons/
reports/
tests/
```

## Development environment

The environment verified for the initial scaffold is Conda `ml_env` with
Python 3.10.20. Direct dependencies and relevant installed versions are listed
in `requirements.txt`. SciPy is included to read the candidate dataset's MATLAB
files.

To use the already-created Conda environment:

```bash
conda activate ml_env
cd "$HOME/ML EDGE BATTERY MONITORING DEVICES"
```

To open JupyterLab after activating the environment:

```bash
jupyter lab
```

To recreate the Python packages in a separate environment, install the
requirements there with:

```bash
python -m pip install -r requirements.txt
```

The commands above do not download a dataset. Dataset investigation and target
limitations are documented before any model result is interpreted.

After the approved Panasonic archive is present at `data/raw/`, regenerate its
file inventory, summary, and representative measured-signal plot with:

```bash
python -m src.data.explore_panasonic
```

This command inspects the raw MATLAB files but does not train a model. It
refuses to overwrite existing results; use `--overwrite` only when intentionally
regenerating those named outputs. The current generated inventory and summary
are in `results/tables/` and `results/metrics/`, and the plot is in
`results/plots/`.

Audit the measured Ah counter against current integrated over time across the
separate 25 °C drive-cycle files with:

```bash
python -m src.data.validate_panasonic_target
```

The audit writes `results/metrics/panasonic_soc_target_audit.json` and refuses
to overwrite it unless `--overwrite` is supplied. Along with internal charge
accounting, it records measured throughput from selected 25 °C 1C discharge
files as capacity comparisons. These are not per-drive-cycle capacity
measurements and do not establish absolute SoC.

To prepare the current clearly labelled proxy dataset:

```bash
python -m src.data.prepare_panasonic_soc_proxy
```

This creates a 1 Hz table for the ten separate 25 °C drive-cycle runs under
`data/processed/` and provenance metadata under `results/metrics/`. The target
assumes each run starts at 100% and uses the mean of the two measured 25 °C
start-of-tests 1C discharge throughputs (2.79826 Ah and 2.75160 Ah), not the
nominal 2.9 Ah. The measured `Ah` channel is used to calculate the proxy and is
not included in the model feature table. Values outside 0–100% are preserved,
not silently clipped.

An experimental RandomForest baseline with complete-cycle holdout can be run
with:

```bash
python -m src.models.train_panasonic_soc_proxy_baseline
```

Its metrics and model are saved to `results/metrics/` and `models/baseline/`.
Those scores are agreement with the derived proxy only, not validated physical
SoC accuracy or multi-cell generalization. Both commands refuse to replace
existing outputs unless `--overwrite` is supplied.

To check how much the proxy depends on the capacity-reference choice:

```bash
python src/audit_soc_proxy.py
```

This compares proxies recalculated with each measured pre-test capacity and
their mean, writing per-cycle and overall differences to
`results/metrics/panasonic_soc_proxy_sensitivity.json`. It uses the existing
proxy samples to rescale relative Ah change; the comparison is a sensitivity
check, not a confidence interval or validation against physical SoC.

Validate and prepare model inputs with a cycle-grouped training split:

```bash
python -m src.preprocessing.prepare_panasonic_model_data
```

The command writes a validated clean table and a scaled table under
`data/processed/`, plus the training-only scaler parameters and cleaning log
under `results/metrics/`. It preserves all proxy values, including values
outside 0–100%, and fails on malformed or duplicate time samples instead of
silently dropping them. These generated tables are local data artifacts and are
excluded from Git.

Generate pooled-row distribution and correlation figures for the inputs and
the explicitly approximate target with:

```bash
python -m src.data.eda_panasonic_proxy
```

Figures are saved under `results/figures/` and descriptive statistics and
correlations under `results/metrics/panasonic_eda_summary.json`. The pooled
time-series rows are not independent samples, so these figures are descriptive
only and do not establish causality or physical SoC accuracy.

Compare baseline regressors on a fixed complete-cycle holdout with:

```bash
python -m src.models.compare_panasonic_models
```

This writes `results/comparisons/model_comparison.csv` and split/provenance
metadata under `results/metrics/`. Cycle 2, Cycle 4, and LA92 are held out;
including Cycle 4 tests behavior on a run with proxy values outside 0–100%.
All scores measure agreement with the approximate proxy, and this comparison
does not automatically select a final model.

Select the candidate for the next optimization phase with:

```bash
python -m src.models.select_panasonic_model
```

The script chooses by lowest aggregate MAE (RMSE as a tie-breaker), records
the evidence and limitations in
`results/metrics/panasonic_model_selection.json`, and saves a selected-model
artifact in `models/baseline/`. That artifact is refit on the comparison
training cycles only; it does not train on the held-out cycles. Because the
holdout informed selection, it is not an independent final performance
estimate.

Test smaller Random Forest variants against the selected baseline with:

```bash
python -m src.models.optimize_panasonic_rf
```

The experiment keeps the train/test cycles and random seed fixed, compares
tree-count and leaf-count reductions, writes artifacts under
`models/optimized/`, and records proxy metrics and serialized sizes in
`results/comparisons/panasonic_rf_optimization.csv`. It measures artifact size
only; latency and memory are part of roadmap Phase 8. The held-out cycles were
used for model selection, so optimization results are exploratory and not an
independent final evaluation.

Benchmark the selected model and the 50-tree candidate for inference latency,
artifact size, and process memory with:

```bash
python -m src.evaluation.benchmark_panasonic_models
```

The script uses identical held-out-cycle inputs, times only `model.predict`,
and reports median, p95, and p99 latency for batch sizes 1, 32, and 256 after
warm-up calls. It runs each model in a fresh process, records the host and
software versions, and measures Linux process RSS before/after dataset loading,
model loading, and inference. Outputs are
`results/metrics/panasonic_resource_benchmarks.json` and
`results/comparisons/panasonic_resource_benchmarks.csv`. Existing outputs are
not replaced unless `--overwrite` is passed. RSS includes the Python and
scientific-computing runtime and test data; it is not model-only RAM. Host
latency and RSS do not establish microcontroller performance or power use.

Export both Random Forest artifacts as standalone C99 headers and validate
them against the Python estimators with:

```bash
python -m src.evaluation.export_panasonic_forest_c
```

The command requires the prepared proxy dataset, saved model artifacts,
comparison metadata, and a C99 compiler available as `cc` (override with
`--compiler`). It compiles a temporary validation harness, compares C and
scikit-learn predictions for all held-out rows, then writes
`edge/exported/panasonic_soc_proxy_selected_rf.h`,
`edge/exported/panasonic_soc_proxy_rf_50_trees.h`, and
`results/metrics/panasonic_c_export_validation.json`. Existing outputs are
preserved unless `--overwrite` is passed. Both headers expect four `float`
inputs in the order voltage (V), current (A), battery temperature (°C), and
elapsed time (s), and return an approximate proxy percentage as `double`.
They contain inference parameters only—not sensor drivers, target validation,
firmware, or hardware-specific memory/power handling.

Run the host-side edge inference simulation with:

```bash
mkdir -p build
cc -std=c99 -O2 -Wall -Wextra -Werror \
  edge/simulation/panasonic_edge_simulator.c -lm \
  -o build/panasonic_edge_simulator
printf '3.9,-0.5,25.3,120\n3.7,-1.1,26.1,900\n' \
  | build/panasonic_edge_simulator --model selected
```

The simulator reads one four-number CSV record per input line in the header's
feature order: voltage (V), current (A), battery temperature (°C), and elapsed
time (s). It writes `sample_index,model,soc_proxy_percent` rows to standard
output. Use `--model candidate` for the 50-tree model; `selected` is the
default. Malformed or non-finite rows fail explicitly. This demonstrates the
exported inference function and a serial-like text interface on the
development computer only. It does not implement a serial driver, sensors,
state initialization, charge integration, microcontroller firmware, or power
measurement.

Run the supplementary leave-one-cycle-out evaluation for all three model
families with:

```bash
python -m src.evaluation.evaluate_panasonic_cycle_cv
```

The script fits each model on all but one complete cycle, repeats this for each
cycle, and reports pooled out-of-fold metrics and per-cycle metrics. Results
are written to `results/comparisons/panasonic_leave_one_cycle_out.csv` and
`results/metrics/panasonic_leave_one_cycle_out.json`; existing outputs are
preserved unless `--overwrite` is passed. This remains proxy-agreement evidence
for one cell at 25 °C. Since these same cycles have already informed prior
model selection and optimization, this repeated grouped evaluation is not an
independent confirmatory test.

## Local research dashboard

Activate the project Conda environment, then run the Streamlit dashboard from
the repository root:

```bash
conda activate ml_env
python -m streamlit run src/ui/app.py
```

The dashboard reads the existing prepared Panasonic dataset, model artifacts,
and experiment result files; it does not retrain models or change the research
pipeline. Its prediction page supports both replaying a recorded dataset sample
and entering four explicit model inputs in the order voltage (V), current (A),
battery temperature (°C), and elapsed time (s).

The application is **Offline Simulation** only. Dataset signals are not live
sensor readings, the prediction target is a derived SoC proxy rather than
validated physical SoC, and software resource measurements are not physical
power measurements. The dashboard reports missing required local artifacts
instead of substituting example or synthetic values.

## Reproducibility principles

- Record dataset provenance, license, target definition, selected features,
  split strategy, and preprocessing decisions.
- Keep raw data separate from processed data.
- Use explicit experiment configurations and reproducible random seeds where
  appropriate.
- Fit preprocessing only on training data.
- Save model parameters and measured results; do not invent pending values.
- Record how latency, model size, and memory are measured.
- Clearly distinguish software experiments from physical hardware evidence.

## GitHub

This directory is initialized as a local Git repository on the `main` branch.
No GitHub remote has been configured yet. Before publishing, review dataset
licenses and ensure that local data, credentials, and other private files are
not committed. Raw and processed dataset files are excluded by `.gitignore`.