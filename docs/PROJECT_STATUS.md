# Project Status

## Current phase

Phase 7 — synthetic sensitivity and Monte Carlo experiments completed.

## Completed

- Research context established.
- Repository scaffold established.
- Proposed 24-variable synthetic data schema documented.
- Causal ordering and four core simulation scenarios documented.
- Design-only YAML configuration schema established.
- Version-1 demographic, financial, loan, context, and approval assumptions
  calibrated and documented.
- Direct and upstream mild/moderate/strong treatment levels defined.
- Configuration-integrity tests established.
- Version-1 vectorized generator implemented for all four scenarios.
- Frozen fair-baseline intercept calibration and reproducibility metadata
  implemented.
- Structured generator validation and automated behavioral tests implemented.
- Four 10,000-row seed-4994 validation datasets generated with one shared,
  one-million-row fair-baseline intercept calibration.
- DTI tail diagnosed on a deterministic 100,000-row population.
- Property-request dependencies minimally recalibrated without changing
  payment assumptions or experimental treatments.
- Approval intercept recalibrated once and frozen after the population change.
- Four deterministic 100,000-row, seed-4994 scenario datasets generated for
  the first statistical experiment.
- Overall and race-specific outcome summaries and unpooled 95% confidence
  intervals for the raw Black-White approval gap implemented.
- Four predeclared statsmodels logistic specifications implemented with exact
  DGP transformations and explicit categorical references.
- Standardized adjusted probability contrasts and generator-based direct-effect
  ground truth implemented.
- Statistical recovery tables, figures, diagnostics, and deterministic analysis
  tests completed.
- Two controlled ML feature regimes implemented: race-blind primary and
  race-aware sensitivity.
- Deterministic 60/20/20 application-level splitting and validation-only
  hyperparameter selection implemented.
- Logistic regression, random forest, and histogram gradient boosting evaluated
  across all four scenarios.
- Overall prediction, race-group audit, disparity-reproduction, calibration,
  true-probability recovery, and synthetic-oracle metrics completed.
- Five ML benchmark figures and deterministic leakage/splitting/behavior tests
  completed.
- Seven-page Streamlit research dashboard implemented around the existing
  simulation, statistical, and ML APIs.
- In-memory scenario generation, bounded custom treatments, active-sample
  statsmodels analysis, and a fixed-seed direct-effect sensitivity curve added.
- Saved benchmark exploration, race-blind/race-aware comparison, group audits,
  calibration, true-probability recovery, and a no-retraining global-threshold
  explorer added.
- Mechanism comparison, 24-field data dictionary, reproducibility metadata,
  CSV/Parquet/JSON downloads, controlled missing-artifact messages, caching, and
  cross-page state implemented.
- Dashboard support tests, Streamlit script execution checks, and local server
  route checks completed.
- Resumable sensitivity framework implemented with stable SHA-256 run identities,
  atomic per-run JSON, common-random-number seeds, and deterministic parallel
  execution.
- Direct-effect, upstream-strength, mixed-mechanism, sample-size, and detection
  designs completed across 3,140 unique replication/settings.
- All four statsmodels specifications and both validation-tuned logistic ML
  regimes evaluated for every run; raw, adjusted, predicted, and synthetic-truth
  quantities retained separately.
- Monte Carlo bias, RMSE, empirical intervals, coverage, detection, sign recovery,
  false positives, mechanism signatures, and result-derived thresholds completed.
- Thirteen sensitivity figures, tidy result families, a paper-ready summary, and
  a read-only seventh dashboard page completed.

## Resolved calibration finding

The original one-million-row population placed 10.7893% at the 0.65 DTI
ceiling. Revision 2 reduces this to 4.2781% while retaining the unchanged 5%
threshold. The original and revised diagnostics are documented in
`docs/dti_recalibration.md`.

## Statistical recovery finding

In the 100,000-row runs, Model 2 estimated Black log-odds coefficients of
-0.263 and -0.272 in the direct and mixed scenarios, respectively, against the
configured -0.25 direct effect. The upstream-only coefficient moved from
-0.373 unadjusted to approximately zero with the DGP controls. Full results and
caveats are documented in `docs/statistical_recovery.md`.

## Machine-learning benchmark finding

Logistic regression was selected as the main ML model using mean validation log
loss. In the direct scenario, its race-blind prediction gap was approximately
zero while its race-aware gap was -2.747 percentage points. In the upstream
scenario, its race-blind gap was already -6.205 points because the shifted
financial features were available. In the mixed scenario, adding race changed
the gap from -6.423 to -10.414 points. These are disparities reproduced from
synthetic mechanisms, not claims about real lenders. Full results are in
`docs/ml_benchmark.md`.

## Sensitivity and Monte Carlo finding

Across 50 replications at 100,000 rows, Model 2 recovered the configured
moderate direct effect of -0.25 at a mean of -0.249 log odds. Moderate upstream
inequality generated a mean raw gap of -6.93 percentage points while the Model
2 adjusted contrast remained approximately zero. The fair-baseline Model 2
false-positive rate was 2%, with 98% interval coverage. The complete design,
thresholds, signatures, numerical policy, and limits are documented in
`docs/sensitivity_analysis.md`.

## Next

- Consolidate the synthetic findings into capstone-ready narrative and tables.
- Preserve the distinction between label prediction, true-probability recovery,
  disparity reproduction, and normative fairness.
- Continue deferring HMDA work until the planned synthetic workflow is complete.
