# Synthetic Generator Design

## Scope

The version-1 generator implements the calibrated synthetic mortgage-
application data-generating process. It creates one application per row,
preserves the known true approval probability, supports the four documented
simulation worlds, and produces validation and reproducibility artifacts. It
does not estimate regressions, train classifiers, calculate research fairness
metrics, process HMDA, or provide a dashboard.

## Implementation architecture

The implementation is split into small modules under
`src/fair_lending/simulation/`:

- `config.py` loads and validates YAML and resolves a complete run config.
- `population.py` generates demographics, latent factors, context, financial
  characteristics, and loan/property characteristics.
- `approval.py` implements the transparent logistic score and fixed-feature
  direct-effect counterfactual.
- `calibration.py` solves and stores the common fair-baseline intercept.
- `generator.py` exposes the in-memory API and separate Parquet export.
- `validation.py` returns structured checks and writes validation summaries.
- `generate.py` is a small `python -m` command-line entry point.

The public in-memory API is `generate_synthetic_data(...)`. File writing is a
separate `save_synthetic_dataset(...)` operation so tests and later experiments
do not need temporary research files.

## Configuration resolution

`resolve_simulation_config(...)` merges the canonical baseline, the selected
effect level from the treatment library, the named scenario switches, and
run-level sample-size and seed overrides. Disabled mechanisms are materialized
as exact zero/identity maps. This makes scenario behavior explicit in saved
metadata and avoids duplicating switch logic in financial and approval code.

The resolver validates the four-world switch matrix, probability shares,
reference-category zeros, the version-1 upstream variable list, race-independent
context design, and null denial-reason policy. It reads and deep-copies YAML; a
run never edits a source config.

## Generation and causal order

The vectorized generation order is:

1. Independently draw race, ethnicity, sex, and age group.
2. Draw correlated standard-normal financial-stability and neighborhood-
   advantage factors.
3. Generate the three context variables from neighborhood advantage, with no
   race input.
4. Generate income and age-feasible employment history.
5. Generate credit score, liquid assets, and existing monthly debt.
6. Draw loan purpose, loan type, and occupancy type.
7. Generate property value and LTV, then derive loan amount exactly.
8. Calculate projected housing payment internally and derive/winsorize DTI.
9. Calculate the baseline underwriting score, add an enabled direct term,
   retain the true probability, and draw the Bernoulli approval label.

Financial stability and neighborhood advantage create documented positive
dependencies but are never persisted. Internal age and projected payment are
also discarded.

## Random-stream strategy

Each run initializes one NumPy `SeedSequence` from the recorded non-negative
integer seed. Six stable child streams cover demographics, latent factors,
context, financial variables, loans, and approval draws. The child spawn keys
are saved in metadata. This separates logically distinct randomness and keeps a
fixed config, seed, and code revision reproducible without global NumPy state.

Application IDs are deterministic row identifiers (`APP000000001`, ...), so
they introduce no additional randomness or protected-group encoding.

## Population dependencies and treatment injection

Income depends on age, financial stability, neighborhood advantage, and a
lognormal residual. Credit depends on financial stability and standardized log
income. Assets depend on income, financial stability, age, and a lognormal
residual. Positive monthly debt depends on income and financial stability, with
a separate structural-zero draw.

When upstream treatment is enabled, the selected race maps are added only to
the log location of income, the point location of credit score, and the
conditional log location of assets. In version 1 only Black has nonzero shifts.
The asset shift is conditional on already generated income, so it is not a
claim about the final marginal Black/White asset ratio. Direct discrimination
never modifies a financial field; it enters only the final log-odds score.

## Context, property, loans, and DTI

All context fields depend on the latent neighborhood factor but not race.
Neighborhood minority share uses a logistic construction and is not an
underwriting term. Property value depends on income, neighborhood income,
purpose, and occupancy. Following the DTI recalibration, its researcher-chosen
income elasticity is 0.90 and residual log SD is 0.18. LTV uses the calibrated
shifted/scaled beta mechanism.
The generator then enforces:

```text
loan_amount = property_value * loan_to_value_ratio
income_to_loan_ratio = annual_income / loan_amount
```

LTV and DTI are stored as decimal ratios. Projected principal and interest is
loan amount times the calibrated monthly payment factor; tax and insurance is
property value times the calibrated monthly property-cost factor. DTI combines
that projected payment with existing monthly debt and divides by gross monthly
income. Raw DTI is winsorized to its configured bounds, and its upper-bound
mass is always reported.

`diagnostics.py` can reconstruct `pre_clip_dti`, payment components, and
loan/income ratios in memory. These diagnostic fields never enter the persisted
schema. The reproducible candidate study is
`experiments/calibrate_population_v1.py`.

## Approval equation and frozen intercept

The approval implementation translates the documented transforms explicitly:
log income, credit score, employment history, log assets, DTI, LTV, and the
three selected loan-category effects. It contains no baseline race, ethnicity,
sex, age, context, property-value, loan-amount, or income-to-loan term. The only
race term is the experimental direct-effect map when its scenario switch is on.

`calibrate_intercept(...)` creates the designated one-million-row fair-baseline
population with seed 499400, computes the score without an intercept, and uses
deterministic bisection over the configured bounds. It solves only the intercept
to target a mean probability of 0.80. The resulting JSON artifact is reused for
every scenario and effect level. It is never re-solved to restore a treated
scenario's approval rate.

## Output and reproducibility metadata

Persisted datasets have exactly the 24 fields in `synthetic_schema.md`; latent
or intermediate values are not silently added. Parquet filenames include the
scenario, effect label, row count, and seed. Adjacent JSON records the full
resolved configuration, stable SHA-256 config fingerprint, intercept and its
calibration fingerprint, seed-stream assignment, timestamp, package version,
Git revision/dirty state when available, output path, and shape.

`approval_probability_true` is simulation ground truth. `approved` is a
Bernoulli draw from it. `denial_reason` remains null in every version-1 row.

## Validation rules

Structured validation covers exact schema/order, row and ID counts,
missingness, numerical bounds, age/employment consistency, exact derived
identities, allowed categories, demographic-share diagnostics, distribution
summaries, key positive correlations, approval probabilities and labels,
probability quantiles, near-boundary probabilities, all clipping rates, DTI
upper-bound mass, and scenario switches. Direct scenarios also compare White
and Black direct terms for the same non-race feature matrix.

Finite-sample racial approval-rate differences in `fair_baseline` are
diagnostics, not pass/fail assertions. Its strict checks are structural: no
race-specific upstream map, no direct race term, no race-conditioned context,
and no race term in baseline underwriting.

## Known simplifications and safeguards

- Demographic attributes are independently drawn.
- Latent factors organize correlations but are not fitted economic constructs.
- Employment history is an accumulated stock based on an unobserved age draw.
- One interest rate, term, and tax/insurance factor applies to all applications.
- Clipping and DTI winsorization can create boundary masses and are reported.
- The underwriting score is a research DGP, not a production lender model.
- Direct and upstream group effects exist only because the researcher inserts
  them; synthetic results cannot establish real-world discrimination.
- Race remains an audit/treatment variable rather than an ordinary baseline
  predictive feature.
