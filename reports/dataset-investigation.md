# Public Battery Dataset Investigation

**Review date:** 2026-10-06  
**Status:** Panasonic archive inspection and a first proxy-only baseline are
complete. The user selected the Panasonic-derived proxy route; its target is
explicitly approximate and physical SoC validation remains unresolved.

## Research fit

Chapters 1–3 call for battery voltage, current, temperature, and time; a
justified battery-state target; ML evaluation; and resource-aware edge
suitability. The implementation must not assume that a target or sensor
channel exists without verifying it in the dataset itself.

The candidate review prioritizes:

- traceable, public data and a clearly identified battery/test setup;
- measured electrical and thermal signals;
- a defensible target such as SoC or capacity-derived SoH;
- operating profiles that support meaningful held-out evaluation; and
- practical access and acceptable reuse terms.

## Candidate comparison

| Candidate | Source and battery | Measurements / target evidence | Coverage and access | Limitations and open checks |
|---|---|---|---|---|
| **Panasonic 18650PF Li-ion Battery Data** | [Mendeley Data, DOI 10.17632/wykht8y7tg.1](https://doi.org/10.17632/wykht8y7tg.1). One 2.9 Ah Panasonic 18650PF cell tested at the University of Wisconsin–Madison. The record is CC BY 4.0. | The readme lists timestamp, terminal voltage, current, amp-hours, watt-hours, power, battery-case temperature, chamber temperature, and elapsed time. Inspection confirms these nine channels in all 198 MATLAB files. SoC is **not a validated per-row target column**; a possible Ah-based reference must be justified and validated. | HPPC, EIS, C/20 tests, and drive cycles; ambient test conditions include 25, 10, 0, -10, and -20 °C, with additional temperature-rise/pause tests. The record reports a 182 MB archive. | One physical cell limits cell-to-cell generalization. Sampling interval varies; the readme states drive-cycle measurements occur in both contiguous and split files. Tests span aging, with end-of-test reference capacity reported as lower than earlier capacity. Unique usable observations and a reliable SoC reference remain unresolved. |
| **CALCE A123/LFP SOC test data** | [CALCE battery data](https://calce.umd.edu/battery-data). The CALCE page describes A123 LiFePO4-cell SOC tests. | The page describes low-current and incremental-current OCV tests, plus DST/FUDS dynamic tests and SoC estimation. It establishes test-level SOC conditions and temperature variation, but a per-sample target column and complete per-record schema have not been verified. Ambient test temperature is documented; availability of measured battery-case temperature is not confirmed for each relevant file. | The page describes public test-data downloads and experiments over multiple temperatures. | The page covers multiple CALCE battery studies and protocols; the exact files must be isolated and inspected before treating them as one dataset. Exact sample count, target-label availability, measurement columns, and reuse terms need verification. |
| **NASA PCoE Battery Data Set** | [NASA Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/), entry “5. Batteries,” credited to B. Saha and K. Goebel. | NASA describes Li-ion battery charge/discharge tests at different temperatures and says impedance is recorded as the damage criterion. The repository landing page does not establish the complete channel schema or a per-reading SoC label. | The repository lists a direct public archive link and citation. | Suitable as a possible aging/health candidate, but voltage/current/temperature/time fields, capacity/SoH ground truth, cell/test count, and archive structure need verification before selection. The repository description alone is not enough to assert those fields. |

### Observation counts

The downloaded Panasonic archive is 182 MB and contains 292 ZIP entries,
including directories. It contains 198 MATLAB `.mat` files, 59 `.csv` files,
one MATLAB `.m` file, and one readme `.txt` file. Counting the `Time` samples
across all 198 MATLAB files gives **8,532,692 file-row observations**. This is
the sum across files, not a count of unique observations: the readme explicitly
states that some drive-cycle observations occur in both contiguous and split
files. Therefore, 8,532,692 must not be reported as a deduplicated sample count.

All 198 MATLAB files loaded with the same `meas` structure and the same nine
fields: `TimeStamp`, `Voltage`, `Current`, `Ah`, `Wh`, `Power`,
`Battery_Temp_degC`, `Time`, and `Chamber_Temp_degC`. The channels had matching
sample counts in every file. Seven files contain a single timestamp/sample;
the remaining files contain multiple observations. One inspected 25 °C drive
cycle has 109,641 observations.

For that inspected drive cycle, `Time` runs from 0 to about 10,983.9 seconds
with increments varying around 0.1 seconds. This agrees with the readme's
warning that logging intervals vary. The file also confirms measured battery
case temperature and chamber temperature as distinct channels.

The archive passed ZIP integrity testing. Its local SHA-256 is
`0098dc90cfb645beb7a1daff1d9721f7eebfa85dca272e5f6adb6e6ca46c14d8`.

CALCE and NASA candidate subsets have not been downloaded or inspected at file
level, so their exact file and observation counts remain unknown.

## Preliminary recommendation

For a first supervised **SoC-estimation** software study, the **Panasonic
18650PF dataset is the leading candidate** because the dataset record explicitly
positions it for SoC algorithm evaluation, its readme documents voltage,
current, battery temperature, time, and amp-hour measurements, and it includes
dynamic profiles and multiple temperature conditions. It is the best-supported
match among the candidates reviewed, not a claim that it represents many cells
or every battery chemistry.

The proposed prediction task, subject to label verification, is:

```text
terminal voltage + current + battery-case temperature
                    ↓
            estimated SoC
```

The readme says amp-hour (`Ah`) is measured and reset before each charge, test,
or drive cycle, and explains that SoC levels for HPPC pulse sets can be
determined from Ah readings. It does **not** establish a validated per-row SoC
ground-truth column for every drive-cycle record. A derived proxy is therefore
used for initial model setup, with assumptions and capacity basis recorded
below; it is not an independent sensor measurement or validated SoC target.
The Ah column is excluded from model inputs.

Because this dataset follows one cell over multiple tests, it cannot support a
claim of generalization across independent cells. A defensible test should hold
out complete test runs or drive-cycle segments (and account for temperature
and duplicated contiguous/split records), rather than randomly splitting
individual time rows. The final strategy depends on the archive structure.

NASA PCoE may be a better alternative if the research decision prioritizes a
capacity/aging-derived SoH task, but its precise inputs and labels need further
verification. CALCE is a useful SoC alternative, especially for its LFP and
temperature-focused test design, but its per-sample labels and file subset must
be checked.

## Inspection status and remaining decision

**User approval received:** use a clearly labelled Panasonic-derived SoC
reference, do not switch datasets, and do not use nominal 2.9 Ah as ground
truth. Archive download, schema/sample inspection, proxy construction, and a
first proxy-only grouped-holdout baseline are complete.

The archive is stored locally under `data/raw/` and is excluded from Git by
`.gitignore`. Archive integrity, file inventory, MATLAB schemas, channel
lengths, and per-file observation counts were checked. The reported sum includes
rows from known duplicated contiguous/split drive-cycle records, so it is not
a unique-observation count. The proxy reference has been constructed; exact
physical SoC and per-run capacity remain unresolved.

Absolute SoC is still not validated. The exploratory baseline below must be
interpreted only as agreement with the derived proxy.

## Target-validation phase

The provider readme confirms that the Ah counter resets before a charge, test,
or drive cycle; discharge current and Ah are negative; the drive cycles are
preceded by charging; and drive-cycle tests at 25 °C end when voltage first
reaches 2.5 V. These facts make Ah/current agreement a useful integrity check,
but do not specify an independently measured initial SoC or a capacity value
for each run. The dataset's stated 2.9 Ah is nominal.

The reproducible audit in
[`src/data/validate_panasonic_target.py`](../src/data/validate_panasonic_target.py)
compares each separate 25 °C drive-cycle's measured Ah change with numerical
integration of current over elapsed time. It intentionally excludes the
combined continuous log because the provider documents overlap with the
separate drive-cycle files. The generated per-run differences and audited
sample counts are recorded in
[`results/metrics/panasonic_soc_target_audit.json`](../results/metrics/panasonic_soc_target_audit.json).
This check validates charge-accounting consistency only; it does not validate
absolute SoC or a target-label reconstruction.

The same audit records measured throughput for named 25 °C 1C discharge
records. The two start-of-tests discharge records from March 9–10 measure
2.798 Ah and 2.752 Ah to a minimum voltage of about 2.5 V; the four drive-cycle
files begin March 18–21. Two end-of-tests records from July 22 and 24 measure
2.434 Ah and 2.354 Ah. These are useful indicators of capacity variation over
the dataset, but are not a capacity measurement for each drive-cycle run.
Consequently, using one reference across all runs—or nominal 2.9 Ah—would add
unquantified label error.

**Decision:** absolute SoC remains unvalidated. Per the user-selected route, an
approximate capacity-referenced coulomb-counting target is used for exploratory
model setup:

```text
SoC proxy (%) = 100 × (1 + (Ah_sample − Ah_first_sample) / Q_reference)
Q_reference = mean(2.79826 Ah, 2.75160 Ah) = 2.77493 Ah
```

The two `Q_reference` values are measured throughput in the two available
25 °C start-of-tests 1C discharge files. The data builder assumes every
drive-cycle starts at approximately 100% after the documented charge procedure
and applies this one early-life mean capacity to all ten separate 25 °C cycle
files. It does **not** use the nominal 2.9 Ah rating. These assumptions do not
account for cell aging between runs, current-rate effects, or per-run starting
SoC deviations.

The generated table contains 112,701 samples at 1 Hz from 1,126,083 raw
observations across ten drive cycles. `Ah` is not among the feature columns;
the model uses terminal voltage, current, battery-case temperature, and
elapsed time. No clipping is applied: Cycle 4 reaches -0.8375% proxy and has
5,058 raw samples outside [0, 100] because its measured discharge throughput
slightly exceeds the shared capacity reference. This is retained as an explicit
proxy limitation, not presented as physical negative SoC.

An initial RandomForest baseline uses a seeded 80/20 `GroupShuffleSplit` by
complete drive-cycle file (eight train runs, two held-out runs). The forest
limits each tree to 128 leaf nodes to bound model size. On held-out Cycle 2 and
LA92, proxy-only MAE is 2.318 percentage points, RMSE is 3.136 percentage
points, and R² is 0.9875. The 1.8 MB serialized estimator's scores measure
agreement with the derived proxy; they are not validated SoC error estimates,
independent-cell generalization, or physical edge performance. The one-cell,
one-temperature scope remains a major limitation. The held-out cycles in this
split do not include Cycle 4, which contains the out-of-range proxy labels;
this baseline does not validate behavior on that condition. The model artifact is
[`models/baseline/panasonic_soc_proxy_rf.joblib`](../models/baseline/panasonic_soc_proxy_rf.joblib),
and the evaluation is
[`results/metrics/panasonic_soc_proxy_baseline.json`](../results/metrics/panasonic_soc_proxy_baseline.json).

### Capacity-reference sensitivity

The sensitivity audit in [`src/audit_soc_proxy.py`](../src/audit_soc_proxy.py)
recalculates the proxy using each of the two measured start-of-tests capacity
references separately, then compares both with the existing mean-reference
proxy. It reports sample- and cycle-level changes and out-of-range counts in
[`results/metrics/panasonic_soc_proxy_sensitivity.json`](../results/metrics/panasonic_soc_proxy_sensitivity.json).
This quantifies how much the chosen denominator moves the proxy; it is not a
confidence interval, a per-cycle capacity estimate, or independent evidence of
physical SoC. The relative Ah changes are reconstructed from the existing
mean-reference proxy samples for this comparison. On the prepared 112,701-row
1 Hz table, the 2.79826 Ah reference changes proxy values by a mean absolute
0.399 percentage points (maximum 0.841), with no samples outside [0, 100]. The
2.75160 Ah reference changes them by a mean absolute 0.406 points (maximum
0.855); 672 samples (about 0.60%) fall outside that range, and the minimum
proxy becomes -1.692%. The existing 2.77493 Ah mean-reference proxy has 506
out-of-range rows in this downsampled table. These shifts are measurable
denominator sensitivity, not estimates of actual SoC error.

No Ah-derived channel should be added to the estimator inputs. Before making
claims about physical SoC accuracy or expanding to other temperatures, the
initial-state and per-run capacity assumptions need independent validation.

## Preprocessing readiness

The prepared proxy table was checked for finite feature and target values,
empty cycle IDs, repeated exact rows, duplicate `(cycle_id, elapsed_time_s)`
keys, and time reversals within cycles. The current table has no malformed rows
or duplicate sample keys. It contains 506 proxy samples outside 0–100%; these
are retained and must not be silently clipped or removed. The nominal sampling
period is one second, but selected source observations have intervals ranging
from about 0.095 to 3.108 seconds, so the table is approximately sampled rather
than precisely interpolated at fixed one-second intervals.

The preprocessing command
[`src/preprocessing/prepare_panasonic_model_data.py`](../src/preprocessing/prepare_panasonic_model_data.py)
validates these conditions, writes a clean copy and a scaled feature table, and
records its decisions in `results/metrics/panasonic_preprocessing_log.json`.
`StandardScaler` parameters in `results/metrics/panasonic_feature_scaler.json`
are fit only on the seeded 80/20 complete-cycle training groups; the scaler is
then applied to both train and held-out feature rows. The proxy target and cycle
IDs are preserved, and `Ah` is not a feature. This split matches the initial
baseline split only. The later model comparison uses an expanded fixed
three-cycle holdout that also includes Cycle 4; its Ridge scaler is fit inside
that model's training-only pipeline. Do not reuse the scaler artifact from the
initial split for that comparison, because it includes Cycle 4 among its
training cycles. Scaling is not itself a model-performance result and does not
improve the validity of the proxy target.

## Exploratory data analysis

The descriptive EDA command
[`src/data/eda_panasonic_proxy.py`](../src/data/eda_panasonic_proxy.py)
generates pooled-row histograms for voltage, current, battery temperature,
elapsed time, and the approximate SoC proxy, plus a Pearson correlation
heatmap. The numerical summary is recorded in
[`results/metrics/panasonic_eda_summary.json`](../results/metrics/panasonic_eda_summary.json);
the figures are
[`results/figures/panasonic_feature_distributions.png`](../results/figures/panasonic_feature_distributions.png)
and
[`results/figures/panasonic_feature_correlation.png`](../results/figures/panasonic_feature_correlation.png).
These figures pool 112,701 time-series observations from ten cycles of one cell
at 25 °C. Rows within a cycle are correlated, cycle sizes differ, and the target
is derived from the measured Ah channel. Therefore, correlations and
distributions are descriptive only; they do not establish causation, validate
physical SoC, or provide independent-sample statistical inference.

In this pooled table, Pearson correlation is about 0.948 between voltage and
the proxy and -0.781 between elapsed time and the proxy; current has near-zero
linear correlation with the proxy (about 0.002). These are descriptive
associations only. In particular, elapsed time and voltage may track
discharge progress, and the proxy itself is calculated from Ah change, so the
strong correlations do not demonstrate independent estimation accuracy.

## Baseline model comparison

The initial comparison script
[`src/models/compare_panasonic_models.py`](../src/models/compare_panasonic_models.py)
evaluates Ridge Regression, Random Forest, and Extra Trees with the same fixed
whole-cycle split: seven training cycles (75,375 rows) and Cycle 2, Cycle 4,
and LA92 held out (37,326 rows). Cycle 4 is included because its proxy has
out-of-range values. Ridge fits its scaler on training data only; the tree
models use raw features. `Ah` is excluded from model inputs.

The results are in
[`results/comparisons/model_comparison.csv`](../results/comparisons/model_comparison.csv),
with split details in
[`results/metrics/panasonic_model_comparison_metadata.json`](../results/metrics/panasonic_model_comparison_metadata.json).
Overall held-out proxy-agreement metrics are:

| Model | MAE (percentage points) | RMSE (percentage points) | R² |
|---|---:|---:|---:|
| Ridge Regression | 3.126 | 5.075 | 0.9700 |
| Random Forest | 2.258 | 3.070 | 0.9890 |
| Extra Trees | 3.222 | 4.309 | 0.9784 |

Random Forest has the lowest error on this one split, but no final model has
been selected. These results measure agreement with the approximate proxy
only; they are not validated physical SoC accuracy, cell-to-cell
generalization, or edge performance. The Cycle 4 proxy is not a true physical
SoC label merely because it is included in the test split.

## Model selection decision

For Roadmap Phase 6, Random Forest is selected **provisionally for the
optimization phase**. On the same fixed three-cycle holdout, it has the lowest
aggregate MAE (2.258 percentage points) and RMSE (3.070 points) and highest
R² (0.9890) of the three candidates. It also has the lowest MAE on each of
Cycle 2, Cycle 4, and LA92. Its existing bounded configuration
(`max_leaf_nodes=128`, 100 trees, `min_samples_leaf=2`) offers a concrete
baseline to optimize.

This is an engineering choice to focus the next experiment, not a final claim
that Random Forest is universally best. The comparison holdout was used to
choose among candidates, which makes its reported metrics selection evidence,
not an independent final performance estimate. The selected artifact is
refitted only on the seven comparison training cycles; the three holdout cycles
remain excluded from fitting. The artifact and decision record are
[`models/baseline/panasonic_soc_proxy_selected_rf.joblib`](../models/baseline/panasonic_soc_proxy_selected_rf.joblib)
and
[`results/metrics/panasonic_model_selection.json`](../results/metrics/panasonic_model_selection.json).
All metrics remain proxy-agreement scores, not validated physical SoC
accuracy.

## Random Forest optimization trade-off

For Roadmap Phase 7, four smaller Random Forest variants were trained using the
same seven training cycles, three held-out cycles, features, random seed, and
minimum leaf size as the selected baseline. Tree count and maximum leaf nodes
were reduced; model artifact sizes are actual uncompressed `joblib` file sizes.
The detailed per-cycle results and model paths are in
[`results/comparisons/panasonic_rf_optimization.csv`](../results/comparisons/panasonic_rf_optimization.csv);
run configuration and limitations are in
[`results/metrics/panasonic_rf_optimization.json`](../results/metrics/panasonic_rf_optimization.json).

| Variant | Trees | Max leaf nodes | MAE (proxy points) | RMSE (proxy points) | Artifact size | Size reduction | MAE increase vs baseline |
|---|---:|---:|---:|---:|---:|---:|---:|
| Selected baseline | 100 | 128 | 2.258 | 3.070 | 1,868,529 bytes | — | — |
| `rf_50_trees` | 50 | 128 | 2.295 | 3.112 | 934,929 bytes | 50.0% | +0.037 |
| `rf_100_64_leaves` | 100 | 64 | 2.681 | 3.626 | 946,929 bytes | 49.3% | +0.423 |
| `rf_50_64` | 50 | 64 | 2.719 | 3.673 | 474,129 bytes | 74.6% | +0.461 |
| `rf_25_32` | 25 | 32 | 3.470 | 4.636 | 122,529 bytes | 93.4% | +1.212 |

The 50-tree/128-leaf candidate halves serialized size while increasing pooled
held-out proxy MAE by only 0.037 percentage points in this run. It also has
lower MAE than the other reduced-size variants. This makes it a useful
**candidate for Phase 8 benchmarks**, not a replacement claimed to be better:
latency and process RSS have since been measured as described in the Phase 8
section below. The 25-tree/32-leaf version
provides a more aggressive size reduction but a larger proxy-error increase.

These comparisons reuse the holdout that informed Phase 6 model selection.
Therefore, results quantify an exploratory accuracy/size trade-off against the
derived proxy, not an independent final test, physical SoC accuracy, hardware
latency, memory consumption, or device power.

## Host-side resource benchmark

Roadmap Phase 8 benchmarks the selected 100-tree Random Forest and the
50-tree/128-leaf candidate using the same fixed held-out cycles and feature
order. Each model ran in a fresh Python subprocess. The benchmark warms up the
estimator, times `model.predict` only, and samples Linux `/proc/self/status`
RSS around dataset loading, model loading, and inference. The saved estimators
retain their configured `n_jobs=-1`.

The measured host was an Intel Core i5-4210U at 1.70 GHz with 4 logical CPUs,
Python 3.10.20, NumPy 2.2.5, scikit-learn 1.7.2, and joblib 1.5.3. Results from
this run are:

| Model | Artifact bytes | Batch 1 median / p95 (ms) | Batch 32 median / p95 (ms) | Batch 256 median / p95 (ms) | RSS change during model load (bytes) |
|---|---:|---:|---:|---:|---:|
| Selected baseline (100 trees) | 1,868,529 | 39.010 / 65.802 | 38.688 / 63.085 | 47.722 / 63.086 | 102,400 |
| Candidate (50 trees) | 934,929 | 43.050 / 82.758 | 24.912 / 36.711 | 25.338 / 40.028 | 163,840 |

The 50-tree model is half the serialized size. Its measured median latency is
lower at batch sizes 32 and 256 in this run, but higher for single-row
inference; the broad tail timings and single-run variation mean this does not
establish a stable latency benefit. Repeat paired measurements and investigate
the saved `n_jobs=-1` inference setting before drawing a deployment conclusion.

The process high-water RSS was about 300.9 MB for the baseline worker and
301.1 MB for the candidate worker. These numbers include Python, NumPy,
scikit-learn, loaded test data, and runtime allocations. The observed model-load
RSS changes are process snapshots, not a direct measurement of model-only RAM;
the high-water mark may also be dominated by loading the CSV and can miss brief
peaks. They must not be presented as embedded-device memory requirements.

Machine-readable per-batch latency percentiles, current/high-water RSS
snapshots, host metadata, and measurement qualifications are saved in
[`results/metrics/panasonic_resource_benchmarks.json`](../results/metrics/panasonic_resource_benchmarks.json)
and
[`results/comparisons/panasonic_resource_benchmarks.csv`](../results/comparisons/panasonic_resource_benchmarks.csv).
This is a host-software benchmark only: no physical edge hardware or power
measurement was used. The same model-selection holdout was reused, so all
conclusions remain exploratory and proxy-specific.

## C99 edge export

Roadmap Phase 9 exports both the provisionally selected 100-tree baseline and
the 50-tree size-reduction candidate. Each standalone header stores tree
features, thresholds, children, and leaf values as read-only C arrays and
exposes a prediction function. Inputs are `float` values in the exact model
feature order—voltage (V), current (A), battery temperature (°C), and elapsed
time (s)—and output is a `double` in approximate proxy percentage units. The
export does not clip out-of-range estimates.

The headers were compiled using `cc -std=c99 -O2 -Wall -Wextra -Werror`.
Predictions from the compiled C functions were compared with their matching
scikit-learn artifacts over every one of the **37,326** held-out samples. For
each forest the maximum absolute difference was **4.26e-14** proxy percentage
points, below the 1e-9 tolerance. The baseline header contains 100 trees and
25,500 nodes (1,210,081 bytes); the candidate contains 50 trees and 12,750
nodes (606,478 bytes). The generated header sizes differ from the serialized
joblib artifact sizes; C array source text is not a measure of embedded RAM or
flash after a particular compiler/linker build.

The exports are available as
[`edge/exported/panasonic_soc_proxy_selected_rf.h`](../edge/exported/panasonic_soc_proxy_selected_rf.h)
and
[`edge/exported/panasonic_soc_proxy_rf_50_trees.h`](../edge/exported/panasonic_soc_proxy_rf_50_trees.h).
The C/Python parity results and split provenance are in
[`results/metrics/panasonic_c_export_validation.json`](../results/metrics/panasonic_c_export_validation.json).
The 50-tree model remains a candidate, not a replacement selected by this
export. Matching software predictions confirms the translation of the saved
estimators; it does not validate the approximate target as physical SoC, test
real sensors or firmware, or demonstrate hardware memory, latency, or power.

## Host-side edge inference simulation

Roadmap Phase 10 adds
[`edge/simulation/panasonic_edge_simulator.c`](../edge/simulation/panasonic_edge_simulator.c),
a C99 program that links either exported header at runtime selection. It accepts
one CSV record per input line with the exact four model features and emits
`sample_index,model,soc_proxy_percent` CSV output. The default uses the
provisionally selected 100-tree model; `--model candidate` uses the 50-tree
optimization candidate. Input records must contain exactly four finite numeric
values; malformed and non-finite rows produce an error and nonzero exit status.

The simulator compiled successfully with
`cc -std=c99 -O2 -Wall -Wextra -Werror`. Its selected and candidate outputs were
compared with their matching saved Python estimators for representative
float32 inputs, and both matched within 1e-9 proxy percentage points. The
focused Phase 10 tests also verify malformed-input and unknown-model handling.
See the
[`tests/test_panasonic_edge_simulator.py`](../tests/test_panasonic_edge_simulator.py)
test for the repeatable compiled parity check.

This is a **host-side inference simulation**, not firmware. It does not acquire
serial data from a physical device, read sensors, establish initial SoC, perform
charge integration, or measure device memory or power. The output is still
agreement with an approximate capacity-referenced proxy and must not be
interpreted as validated physical SoC.

## Leave-one-cycle-out robustness evaluation

Roadmap Phase 11 fits each of the three compared model families ten times,
holding out one complete 25 °C drive-cycle run in each fold. This produces one
out-of-fold prediction for every prepared row (112,701 total) without dividing
rows from the same cycle between training and testing. Ridge scaling is
contained in its pipeline and is fitted anew using only the training cycles in
each fold. Pooled metrics and unweighted per-cycle means are reported separately
because cycle lengths differ.

| Model | Pooled proxy MAE (percentage points) | Pooled proxy RMSE (percentage points) | Pooled R² |
|---|---:|---:|---:|
| Ridge Regression | 4.008 | 5.870 | 0.9570 |
| Random Forest (100 trees) | 3.111 | 4.680 | 0.9727 |
| Extra Trees (100 trees) | 4.523 | 6.873 | 0.9410 |

The Random Forest had the lowest pooled proxy MAE and RMSE among these fixed
model configurations. Its worst per-cycle MAE was on UDDS. This evaluation
supplements the prior fixed holdout; it does not replace it with an independent
confirmatory result. All ten cycles have already informed earlier model
selection and optimization. The experiment also remains limited to one cell,
one ambient temperature, and the approximate capacity-referenced target.
Results measure proxy agreement, not validated physical SoC accuracy.

Per-cycle metrics and fold membership are in
[`results/comparisons/panasonic_leave_one_cycle_out.csv`](../results/comparisons/panasonic_leave_one_cycle_out.csv);
pooled/unweighted summary metrics and the evaluation caveat are in
[`results/metrics/panasonic_leave_one_cycle_out.json`](../results/metrics/panasonic_leave_one_cycle_out.json).
The implementation and regression tests are in
[`src/evaluation/evaluate_panasonic_cycle_cv.py`](../src/evaluation/evaluate_panasonic_cycle_cv.py)
and [`tests/test_panasonic_cycle_cv.py`](../tests/test_panasonic_cycle_cv.py).

## Source notes

- Phillip Kollmeyer, *Panasonic 18650PF Li-ion Battery Data*, Mendeley Data,
  version 1, DOI [10.17632/wykht8y7tg.1](https://doi.org/10.17632/wykht8y7tg.1).
  The dataset page, accompanying readme, and 182 MB archive were reviewed.
- CALCE, [Battery Data](https://calce.umd.edu/battery-data). Candidate
  descriptions and test protocols were reviewed on the institutional page.
- NASA Ames PCoE, [Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/).
  The battery entry and cited archive availability were reviewed; the archive
  itself was not downloaded.
