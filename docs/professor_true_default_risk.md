# Professor Model — Step 3 True Default Risk

Stage record: results, test counts, and later-work deferrals below describe
this step. See [current status](PROJECT_STATUS.md) and the
[documentation index](README.md) for the latest state across all three families.

Step 3 calculates evaluator-only default probabilities for the existing
**100,000** applicants. No applicant was regenerated and all registered Step 2
files, including its metadata and bank views, retain their original checksums.

## Exact equation and inputs

The supplied working transcription is implemented without coefficient changes:

```text
P_income = sigmoid(0.52 - 0.65 * log_Y)
P_other  = sigmoid(4 + 0.06 * D + epsilon)
true_PD  = P_income * P_other

sigmoid(z) = 1 / (1 + exp(-z))
```

`log_Y` is the saved natural logarithm of annual income in synthetic dollars.
`D` is the saved, untruncated DTI ratio; `epsilon` is the hidden borrower
factor. The numerical implementation uses `scipy.special.expit` for a stable
sigmoid. There is no clipping, artificial floor, intercept calibration, or
extra risk term.

Only `log_Y`, `D`, and `epsilon` enter the calculation. Y in levels, B, C, and
omega are excluded. B remains an audit group and C an observed signal of the
hidden factor. Neither is a direct true-risk input. The two sigmoid factors
are components of the defined mathematical product; treating them as
independent default events would require additional assumptions.

The professor's default-event horizon remains unresolved. This step evaluates
the probability equation as supplied; it does not convert it into per-period
survival or generate binary default events or repayment histories.

## Implementation and configuration

- `configs/professor_model/true_risk.yaml` records the frozen equation, expected
  row count, preview size, input directory, and separate output locations.
  The loader rejects changes to the equation or additional group terms.
- `src/fair_lending/professor_model/risk.py` owns the stable deterministic
  calculation and four-column truth contract.
- `risk_validation.py` validates equations/IDs, summarizes distributions and
  bins, calculates correlations, and audits groups and fixed-feature signs.
- `risk_io.py` reads existing inputs, verifies preserved checksums, and saves
  separate truth artifacts with their own provenance.
- `risk_figures.py` produces exactly the three diagnostic figures below.
- `experiments/generate_professor_true_risk.py` orchestrates the calculation.

Step 2's `baseline.yaml`, population-generation code, random seed, and metadata
are unchanged. The runner joins existing bank/private Parquet tables using
`applicant_id` and validates those inputs before calculating risk.

From the repository root:

```bash
python3 experiments/generate_professor_true_risk.py
```

To deliberately replace only Step 3 outputs after a restart:

```bash
python3 experiments/generate_professor_true_risk.py --overwrite
```

The calculation consumes no RNG draws. Same input data and configuration give
bitwise-identical truth. Separate `true_risk_metadata.json` records schema
`professor-true-risk-v1`, resolved configuration and fingerprint, configuration
file hash, source-file hashes and aggregate fingerprint, dependency/Python
versions, all registered Step 2 input checksums, new output checksums,
evaluator scope, and the no-floor/no-random-draw policy. Original metadata is
never updated to describe this later stage.

## Datasets and information boundary

All data paths below use this base directory:

```text
/Users/macbook/Hack/econ4994/data/synthetic/professor_model/baseline/
```

| File | Rows | Columns / purpose |
|---|---:|---|
| `simulation_truth.csv` | 100,000 | Exactly `applicant_id, P_income, P_other, true_PD`; evaluator only. |
| `simulation_truth.parquet` | 100,000 | Same schema and values; efficient canonical truth. |
| `simulation_truth_audit_preview_EVALUATOR_ONLY.csv` | 150 | `applicant_id, Y, log_Y, D, B, epsilon, C, P_income, P_other, true_PD`. |
| `true_risk_metadata.json` | One manifest | Independent Step 3 provenance and checksums. |

The preview contains the first 150 applicants for manual review. Its filename
and metadata label it evaluator-only, not a bank training table. It is not a
representative/group-balanced analysis sample. Neither the bank CSV nor bank
Parquet receives epsilon, risk components, or true_PD.

Numerical outputs are:

```text
results/metrics/professor_model/true_risk_validation.json
results/tables/professor_model/true_risk_distribution.csv
results/tables/professor_model/true_risk_group_comparison.csv
results/tables/professor_model/true_risk_correlations.csv
```

The distribution CSV has one row per risk quantity; the group CSV one row per
group; the correlations CSV one row per comparison variable. Values are raw
probabilities unless a display explicitly says percent.

## Distribution results

All SD figures use the sample convention (`ddof=1`). This table uses raw
probability units, not percentages:

| Quantity | Mean | Median | SD | Minimum | Maximum |
|---|---:|---:|---:|---:|---:|
| P_income | 0.001462240 | 0.001315447 | 0.000702570 | 0.000164001 | 0.011170255 |
| P_other | 0.972544432 | 0.982447607 | 0.031452574 | 0.405786327 | 0.999645352 |
| true_PD | 0.001422076 | 0.001278240 | 0.000685101 | 0.000161631 | 0.009440280 |

Mean true_PD is **0.142208%**; median is **0.127824%**. The range is
approximately **0.016163%–0.944028%**. Low risk is an implication of the
supplied equation, not a calibration target. The income component is generally
small and strongly shapes the product. The other component is usually near
one, limiting the size of DTI/hidden-factor effects in this population.

| Percentile | true_PD (probability) | true_PD (%) |
|---|---:|---:|
| p01 | 0.000441530 | 0.044153% |
| p05 | 0.000604532 | 0.060453% |
| p25 | 0.000942435 | 0.094244% |
| p50 | 0.001278240 | 0.127824% |
| p75 | 0.001743100 | 0.174310% |
| p95 | 0.002716712 | 0.271671% |
| p99 | 0.003740779 | 0.374078% |

| Descriptive risk bin | Applicants | Share |
|---|---:|---:|
| true_PD < 0.001 (below 0.1%) | 29,284 | 29.284% |
| 0.001 <= true_PD < 0.01 (0.1%–below 1%) | 70,716 | 70.716% |
| 0.01 <= true_PD < 0.05 (1%–below 5%) | 0 | 0% |
| true_PD >= 0.05 (5% or higher) | 0 | 0% |

## Economic interpretation and observed correlations

Holding other inputs fixed, the equation implies:

```text
d(true_PD)/d(log_Y) = -0.65 * P_income * (1-P_income) * P_other < 0
d(true_PD)/d(D)     =  0.06 * P_income * P_other * (1-P_other) > 0
d(true_PD)/d(epsilon) =      P_income * P_other * (1-P_other) > 0
```

Higher income therefore reduces constructed risk, while higher DTI and epsilon
increase it at fixed features. These signs passed validation for all applicants
and controlled perturbation tests on hand-constructed examples.

Marginal correlations across the actual sample are a different quantity:

| Variable versus true_PD | Pearson correlation | Spearman correlation |
|---|---:|---:|
| Y | -0.710838 | -0.996863 |
| log_Y | -0.946036 | -0.996863 |
| D | -0.000444 | -0.002063 |
| epsilon | +0.053405 | +0.050863 |
| C | +0.048422 | +0.047124 |

Income is the dominant marginal association. Epsilon and C are weakly
positively associated with risk. C has an association because it contains
information about epsilon; C does not directly enter the equation.

DTI has an approximately zero, slightly negative marginal association in this
realized sample despite its strictly positive fixed-feature effect. The small
coefficient and sigmoid saturation make its effect weak enough for finite-sample
relationships with the other varying inputs to dominate. This is reported as
a diagnostic warning, not corrected by changing coefficients. Correlations
are descriptive; they are not estimates of real-world causal effects.

## Group comparison

| Group | Applicants | Mean true_PD | Median true_PD | p95 true_PD |
|---|---:|---:|---:|---:|
| B=0 | 84,930 | 0.001421938 | 0.001277944 | 0.002713399 |
| B=1 | 15,070 | 0.001422851 | 0.001279180 | 0.002734086 |

The mean B=1 minus B=0 difference is approximately 0.000000912 in probability
units, or **0.000091 percentage points**. B is absent from risk and was
generated independently in Step 2. These small realized differences are
sampling differences; no group-specific risk treatment is encoded and no
exact equality is imposed.

## Validation, figures, and limits

Validation found **zero errors**: 100,000 rows/unique IDs, matching IDs,
finite/nonmissing values, strict probability bounds, both exact sigmoid
formulas, product identity, deterministic recalculation, and fixed-feature
direction signs all passed. Saved CSV and Parquet match exactly under
round-trip float parsing, and recalculation matches persisted truth.
All original input checksums were verified before and after export.

There are two interpretation warnings: the low probability scale and the
near-zero/slightly-negative marginal DTI correlation. Neither is a reason to
retune the specification. The original DTI tails are retained unchanged.
No empirical claim about actual mortgage default or discrimination follows.
Review the handwriting, income units, coefficients, and default-event horizon
with the professor before interpreting the values economically.

Three figures use all 100,000 applicants and display probability in percent:

1. `results/figures/professor_model/true_pd_histogram.png` — logarithmic
   probability axis and logarithmic-width bins; mean/median reference lines.
2. `results/figures/professor_model/true_pd_vs_log_income.png` — density scatter
   showing the income association, with logarithmic count color scale.
3. `results/figures/professor_model/true_pd_vs_dti.png` — density scatter showing
   the weak marginal DTI association; untruncated DTI is displayed in percent.

The three PNGs were rendered and inspected for readable labels and clipping.
The scatter plots are hexagonal count aggregations, not regression fits.

`tests/test_professor_true_risk.py` adds **32 tests** covering independent scalar
hand calculations, the exact frozen coefficients, product/bounds, reproducible
calculation, B/C exclusion, untouched inputs/bank view, invalid data, schema,
controlled directions, numerically stable extreme logits without floors,
ID alignment, sample SD/bin/group summaries, exports, and checksum enforcement.
The combined professor-population/risk focused suite passed **77 tests** with
no warnings.

The complete repository suite finished with **357 passed, 4 failed, and 32
existing deprecation warnings** in 200.02 seconds. The same four failures as
Step 2 occur in `tests/test_systemic_corrected_checkpoint.py`: its loader
constructs filenames using current Git revision `7a7e590`, whereas the saved
V2 corrected success records reference `8a9f71f`. This lookup problem remains
explicitly reported; unrelated V2 recovery code and artifacts were not changed.

## Recommended Step 4

After reviewing the low true-risk scale, Step 4 should add only the specified
noisy measurement `PN = 0.8 * true_PD + eta` in a separate artifact and noise
stream, with exact-identity and distribution checks. Retain the raw signal and
the 0.8 scaling. Bank regression, lending thresholds, profit, and allocation
remain separately scoped future steps. None is implemented by Step 3.
