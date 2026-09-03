# Project Status

## Current phase

Phase 5 — first machine-learning benchmark completed.

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

## Next

- Build an interactive dashboard on top of the validated simulation,
  statistical, and ML APIs.
- Preserve the distinction between label prediction, true-probability recovery,
  disparity reproduction, and normative fairness.
- Continue deferring HMDA work until the synthetic workflow is complete.
