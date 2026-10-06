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

## Source notes

- Phillip Kollmeyer, *Panasonic 18650PF Li-ion Battery Data*, Mendeley Data,
  version 1, DOI [10.17632/wykht8y7tg.1](https://doi.org/10.17632/wykht8y7tg.1).
  The dataset page, accompanying readme, and 182 MB archive were reviewed.
- CALCE, [Battery Data](https://calce.umd.edu/battery-data). Candidate
  descriptions and test protocols were reviewed on the institutional page.
- NASA Ames PCoE, [Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/).
  The battery entry and cited archive availability were reviewed; the archive
  itself was not downloaded.
