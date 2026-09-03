# Project Status

## Current phase

Phase 3 — synthetic generator and population calibration validated.

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

## Resolved calibration finding

The original one-million-row population placed 10.7893% at the 0.65 DTI
ceiling. Revision 2 reduces this to 4.2781% while retaining the unchanged 5%
threshold. The original and revised diagnostics are documented in
`docs/dti_recalibration.md`.

## Next

- Define and implement the first descriptive and statistical modeling phase.
- Continue separating raw disparity, adjusted disparity, and known synthetic
  direct effects.
