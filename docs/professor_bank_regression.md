# Professor Model — Step 5 Bank Linear Regression

Stage record: results, test counts, and later-work deferrals below describe
this step. See [current status](PROJECT_STATUS.md) and the
[documentation index](README.md) for the latest state across all three families.

Step 5 fits only the professor's working OLS risk regression. It uses the
existing 100,000 applicants and raw PN measurements from Steps 2–4. No
population, true-risk, or PN values were regenerated or retuned. Lending,
profit, repayment histories, and portfolio optimization remain unimplemented
for this separate professor-model family.

## Inputs and information boundary

The bank reads only `applicants_bank_view.parquet` (100,000 × 5) and
`training_signals.parquet` (100,000 × 2), joined one-to-one by `applicant_id`.
The loader rejects missing, duplicate, or mismatched IDs, wrong schemas,
nonfinite values, and incomplete joins. It prints shapes, column names,
the first 10 joined rows, and the raw PN summary before fitting.

Only Y, D, C, B, and an intercept enter the design matrix. Applicant IDs and
split labels are administrative. PN is the raw target, not a predictor.
Neither hidden factors/noise (`epsilon`, `omega`, `eta`) nor truth/components
are supplied to the fit. Old metadata and files can be byte-hashed before
fitting, but hidden data tables are not numerically read by the bank loader.

The model JSON, full predictions, split file, and coefficient CSV are saved
and checksummed **before** the evaluator loads `simulation_truth.parquet`.
The truth loader verifies this freeze checkpoint. Evaluation never refits,
selects, rescales, or overwrites the frozen bank model or predictions.

## Fixed split and exact specification

The 80,000-training / 20,000-evaluation split is a researcher-chosen design,
not an assertion about the whiteboard. Sort IDs lexicographically, permute
their positions with PCG64 initialized from
`SeedSequence(entropy=4994, spawn_key=(6,))`, and assign the first 80,000
permuted positions to training. Save every ID's assignment. Population
children [0]–[4] and eta child [5] remain untouched. There is no validation
set, tuning, outcome-selected split, or model comparison.

Fit unregularized `statsmodels.OLS(...).fit(method="pinv")` on training rows
only, using the stable library solver, not an explicit inverse of X'X.
Predictors remain in exactly this order:

| Design term | Scale / transformation | Fitted coefficient |
|---|---|---:|
| `const` | Intercept equal to 1 | 0.0030197849371854635 |
| `Y` | Annual synthetic USD in levels, not log income | -1.0913735436754443e-08 |
| `D` | DTI ratio, not percent | -0.0023144630522321961 |
| `C` | Original continuous signal scale | -0.00040562204042523186 |
| `B` | Binary 0/1 group, retained as requested | 0.00071052614427191425 |

```text
bank_PD_raw = 0.0030197849371854635
            - 1.0913735436754443e-08 * Y
            - 0.0023144630522321961 * D
            - 0.00040562204042523186 * C
            + 0.00071052614427191425 * B
```

The design has full rank **5**. Its condition number is **651,112.0**, which
depends strongly on income's dollar scale relative to ratio/signal units.
Coefficients and all 100,000 predictions are finite. Units are intentionally
not changed to improve conditioning, and no sign constraints are imposed.
A separate deterministic refit reproduced the split, coefficients, and every
prediction exactly.

`bank_PD_bounded = clip(bank_PD_raw, 0, 1)` is a separate safeguard, not a
fitted model or probability calibration. Raw PN and raw predictions remain
unchanged. Nothing is divided by 0.8. No binary default labels are created.

## Held-out PN prediction

All following metrics use only the 20,000 held-out evaluation applicants.
Numbers in tables are raw probability/measurement units unless stated.

| Predictor of held-out PN | MAE | RMSE | R² |
|---|---:|---:|---:|
| OLS raw | 0.0793871070 | 0.0996671572 | 0.0000006933 |
| Constant training mean PN | 0.0793847677 | 0.0996672394 | -0.0000009568 |

The reference is fixed at **0.0014797010616993192**, the training PN mean,
not the evaluation mean. OLS has marginally lower RMSE but slightly worse
MAE. Its near-zero held-out R² shows essentially no improvement in predicting
individual noisy PN observations. Pearson correlation for a constant
prediction is explicitly undefined, not reported as zero.

## Hidden-risk recovery — evaluator only

| Reference | Estimate | MAE | RMSE | Mean error (estimate − truth/reference) | Pearson r |
|---|---|---:|---:|---:|---:|
| true_PD | Raw | 0.0005264084 | 0.0007221061 | +0.0000495850 | 0.560342 |
| true_PD | Bounded | 0.0004903232 | 0.0006452314 | +0.0000856701 | 0.581123 |
| 0.8 × true_PD | Raw | 0.0005943184 | 0.0007663578 | +0.0003355116 | 0.560342 |

The last row is the **scaled true-risk reference**, never a bank-observed
target or a calibration instruction. These evaluator values did not select
settings or alter the model. Raw true-risk MAE of 0.0005264084 is **0.052641
percentage points**, not 0.052641% relative error. Signal prediction and
hidden-risk recovery answer different questions. Clipping reduces error in
this sample but shifts the mean upward and creates boundary ties; that does
not demonstrate calibration or validate a future lending policy.

## Prediction distribution

| Evaluation raw statistic | Value |
|---|---:|
| Mean | 0.0014792180 |
| Median / p50 | 0.0015637005 |
| Sample SD (`ddof=1`) | 0.0008246255 |
| Minimum | -0.0142225369 |
| Maximum | 0.0041049801 |
| p01 | -0.0011251436 |
| p05 | 0.0000512974 |
| p95 | 0.0026039591 |
| p99 | 0.0030284551 |

Evaluation has **923 raw values below zero (4.615%)**, none above one,
**923 bounded zeros**, and no bounded ones. These ties are preserved; no
threshold or tie-breaking rule is implemented. The bounded evaluation mean
is 0.0015153032 (0.15153032% on a probability display). Across all 100,000
scored applicants there are 4,549 negative raw predictions and none above one;
those all-row counts are descriptive, not additional out-of-sample evidence.

## Evaluation group audit

| B | Applicants | Mean true_PD | Mean raw | Mean bounded | Raw MAE | Mean raw error |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 16,970 | 0.0014322376 | 0.0013749528 | 0.0014137714 | 0.0004779647 | -0.0000572848 |
| 1 | 3,030 | 0.0014150459 | 0.0020631719 | 0.0020839480 | 0.0007977251 | +0.0006481260 |

The baseline has no inserted group-specific true-risk treatment. A nonzero
fitted B coefficient and different group estimation errors in one noisy
population are not proof of discrimination. Retaining B follows the specified
regression; it does not authorize group shifts in the DGP or justify deleting
the coefficient after observing its estimate.

## Exact output locations

Dataset base: `/Users/macbook/Hack/econ4994/data/synthetic/professor_model/baseline/`.

| File | One row / purpose |
|---|---|
| `applicant_splits.csv` | Applicant; exactly `applicant_id, split`. |
| `bank_risk_estimates.csv` | Applicant scored by `professor_bank_ols_v1`; exactly `applicant_id, split, model_id, bank_PD_raw, bank_PD_bounded, prediction_out_of_bounds`. No PN, hidden truth, or noise. |
| `bank_risk_estimates.parquet` | Same 100,000 predictions and exact values/types. |
| `bank_regression_audit_preview_EVALUATOR_ONLY.csv` | First 150 evaluation IDs in sorted order; `information_scope, applicant_id, Y, D, C, B, PN, true_PD, bank_PD_raw, bank_PD_bounded`. Not a representative or group-balanced subsample. |
| `bank_regression_metadata.json` | Resolved configuration, source/config fingerprints, dependencies, Git revision/dirty state, cohort ID fingerprints, freeze checkpoint, and original/output checksums. |

Result base: `/Users/macbook/Hack/econ4994/results/`.

| Relative path | Contents |
|---|---|
| `tables/professor_model/bank_regression_coefficients.csv` | Five terms, full-precision coefficients, units. |
| `tables/professor_model/bank_regression_evaluation_metrics.csv` | Five reference/estimate rows: PN OLS and constant; truth raw and bounded; scaled-truth raw. |
| `tables/professor_model/bank_regression_evaluation_groups.csv` | Two evaluation B-group audit rows. |
| `tables/professor_model/bank_regression_prediction_summary.csv` | One evaluation prediction distribution/range summary. |
| `metrics/professor_model/bank_regression_model.json` | Transparent replayable coefficients, feature order, fit settings, training mean PN, and split fingerprints. No pickle. |
| `metrics/professor_model/bank_regression_validation.json` | Structural checks, numerical results, and interpretation notices; not a full-suite test log. |
| `figures/professor_model/bank_pd_raw_vs_true_pd.png` | All 20,000 evaluation points, identity line, zero, linear axes, negative values visible. |
| `figures/professor_model/bank_pd_raw_histogram.png` | All evaluation raw predictions, full range including negatives, zero marked. |

Both figures were rendered and visually inspected. CSV values are not rounded
on export; round-trip parsing agrees exactly with Parquet. Generated data and
results follow the repository's existing Git-ignore policy.

## Reproduction, validation, and limits

```bash
python3 experiments/fit_professor_bank_regression.py
# Explicitly replace only Step 5 outputs, if they already exist:
python3 experiments/fit_professor_bank_regression.py --overwrite
python3 -m pytest -q tests/test_professor_population.py tests/test_professor_true_risk.py tests/test_professor_noisy_signal.py tests/test_professor_bank_regression.py
python3 -m pytest -q
```

Configuration lives in `configs/professor_model/bank_regression.yaml`; reusable
fitting/evaluation and IO code is in `bank_regression.py` and `bank_io.py`.
The runner only orchestrates inspection, splitting, fitting, freeze, and
evaluation. The 34 new deterministic tests cover allowlists/units/intercept,
raw target, split coverage and isolation, prediction replay, metric/group
calculations, CSV/Parquet agreement, checksums, and loading truth only after
saved artifacts exist. Changing evaluation PN cannot change coefficients;
changing truth or PN cannot change predictions made with frozen coefficients.
All **146 professor-model tests passed without warnings**.

The complete repository suite finished with **426 passed, 4 failed, and 32
existing deprecation warnings** in 149.60 seconds. These are the same four
pre-existing `tests/test_systemic_corrected_checkpoint.py` failures:

- `test_corrected_replication_has_exactly_72_unique_complete_cells`
- `test_replication13_provenance_is_internally_consistent`
- `test_legacy_comparison_is_diagnostic_and_never_authorizes_reuse`
- `test_interim_summary_uses_five_corrected_records_only`

The loader includes current HEAD `7a7e590` in its run ID, whereas the saved V2
records reference `8a9f71f`; it therefore searches different filenames. There
are no new Step 5 failures. The 32 warnings are existing scikit-learn/SciPy
L-BFGS-B `disp`/`iprint` deprecations (26 from the ML benchmark tests, six from
sensitivity tests). Neither unrelated recovery code nor saved V2 records were
changed to bypass these failures.

The original 45-file pre-Step-5 snapshot and all 47 original artifacts covered
by the prior manifests remained byte-identical. Old V1/V2 scientific code,
configurations, portfolios, and Monte Carlo recovery were not modified.

Noise SD 0.1 is large compared with the approximately 0.0014 true-risk level.
Moreover the bank's specified Y-level linear model does not reproduce the
nonlinear, log-income true-risk equation or observe epsilon. Fitted slopes
are not recoveries of the sigmoid-factor coefficients. Wrong-signed finite
sample slopes are reported, not corrected by refitting or changing inputs.
This single synthetic population provides no empirical evidence about actual
default, real mortgage discrimination, or general estimator rankings.

The true default-event horizon, period calendar, and payment accounting still
need clarification before lending/profit interpretation. Step 5 stops at risk
estimation: no thresholds, lending decisions, profit calculations, repayment
histories, portfolio optimization, flexible ML, or Monte Carlo were added.
