# Project Status

## Current phase

Phase 4 — first statistical recovery experiment completed.

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

## Next

- Define a machine-learning benchmark against this statistical reference.
- Preserve the separation among raw disparity, adjusted disparity, and known
  synthetic direct effects.
- Defer Streamlit and HMDA work until the synthetic method is established.
