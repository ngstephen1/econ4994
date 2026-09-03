# Synthetic Sensitivity and Monte Carlo Analysis

## Objective

This experiment tests whether the project's descriptive, statsmodels, and
logistic machine-learning estimands behave consistently as the known synthetic
mechanisms and sample size change. It is a controlled recovery study, not an
estimate of real-world mortgage discrimination.

Every probability gap uses the **Black minus White** sign convention. A value
of `-0.03` is a three-percentage-point lower probability for Black applicants.

## Declared design

The routine uses the existing 24-field generator, frozen fair-baseline
intercept `2.0362202310934663`, canonical population configuration, and four
mechanism labels. It does not recalibrate the intercept or modify YAML files.

| Experiment | Settings | Replications | Rows per setting |
|---|---|---:|---:|
| Direct sweep | `0, -0.05, -0.10, -0.15, -0.25, -0.35, -0.50` log odds | 50 | 100,000 |
| Upstream sweep | `0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50` | 50 | 100,000 |
| Mixed grid | Direct `0, -0.10, -0.25, -0.50` × upstream `0, 0.50, 1.00, 1.50` | 30 | 100,000 |
| Sample-size study | Fair, moderate direct, moderate upstream, and moderate mixed | 50 | 1,000–100,000 |
| Detection curves | Seven direct effects | 50 | 5,000–100,000 |

Overlapping scientific cells are evaluated once and reused. The combined
design contains 3,140 unique replication/settings.

Upstream strength scales the canonical moderate treatment on its native DGP
scale:

- income log-location shift: `strength × log(0.90)`;
- credit-score shift: `strength × -25` points;
- liquid-assets log-location shift: `strength × log(0.75)`.

The displayed conditional income and asset multipliers are therefore
`0.90^strength` and `0.75^strength`.

## Reproducibility and execution

A NumPy `SeedSequence(4994)` child is assigned to each replication. The same
child seed is shared across mechanism settings and sample sizes for that
replication, creating common random numbers; distinct replication indices have
distinct child seeds.

Each run has a SHA-256 identity containing its treatment settings, row count,
replication index, seed, configuration fingerprint, analysis version, feature
regimes, and model policy. Completed records are written atomically to
`results/metrics/sensitivity_runs/`. Resume accepts only a complete record with
the exact run identifier, configuration fingerprint, and analysis version.
Failed or stale records are recalculated.

Two worker processes are used, with one numerical-library thread per process to
avoid BLAS oversubscription. One- and two-worker scientific outputs were exactly
equal in deterministic comparison runs. The schema-final routine completed all
3,140 runs with zero failures in 1,814.5 seconds (30.2 minutes); 16 exact-match
quick-run records were resumed.

No replication-level synthetic dataset is saved. Each generated frame is
released after its compact metrics and metadata have been persisted.

## Estimands

Every run records:

- the full-sample observed Black–White approval gap and its unpooled Wald 95%
  confidence interval;
- Black coefficients and standardized probability contrasts for statistical
  Models 0–3;
- Model 2 inference, convergence, conditioning, and optimizer diagnostics;
- held-out race-blind and race-aware logistic-regression predictions selected
  on validation log loss;
- the configured direct log-odds effect;
- the fixed-feature true direct probability contrast;
- the held-out true-probability group gap;
- the observed Bernoulli outcome gap.

The ML disparity-recovery error is:

```text
predicted Black–White mean-probability gap
− test-set Black–White synthetic-true-probability gap
```

These truths are intentionally distinct. In particular, the test-set true
group gap can contain upstream composition effects, while the fixed-feature
direct contrast isolates the inserted direct term.

## Inference and numerical policy

Detection means a two-sided Model 2 `p < 0.05`. Sign recovery is reported
separately. Coverage is inclusive containment of the configured direct
log-odds effect within the Model 2 95% interval.

Models use unpenalized statsmodels logit. At `n = 1,000`, a rare loan-category
dummy can contain only approvals or only denials, producing complete separation
for that nuisance coefficient. The predeclared stability policy removes only
such separated categorical dummy columns for that replication, never the Black
indicator or continuous underwriting controls. The removed names are recorded.
Newton is used first; an unpenalized BFGS fallback is available if needed.

Forty-five of 3,140 records used this separated-nuisance-dummy rule in Models 2
and 3. All four models converged in every completed record, and no retained
model had a maximum standard error above 10.

## Direct-effect recovery

At 100,000 rows, Model 2 closely recovered every configured direct effect.

| True direct log odds | Mean estimate | Monte Carlo SD | Bias | Detection | Coverage |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.002 | 0.024 | 0.002 | 2% | 98% |
| -0.05 | -0.049 | 0.024 | 0.001 | 40% | 98% |
| -0.10 | -0.098 | 0.023 | 0.002 | 96% | 98% |
| -0.15 | -0.148 | 0.022 | 0.002 | 100% | 100% |
| -0.25 | -0.249 | 0.021 | 0.001 | 100% | 100% |
| -0.35 | -0.349 | 0.022 | 0.001 | 100% | 98% |
| -0.50 | -0.500 | 0.022 | -0.000 | 100% | 98% |

The race-blind logistic model's mean prediction gap stayed near zero across the
direct-only sweep because neither race nor a modeled proxy carried the direct
term. The race-aware sensitivity model reproduced an increasingly negative gap.
This is a synthetic detectability result, not an argument for using race in
real underwriting.

## Upstream-inequality behavior

The raw approval gap became monotonically more negative as upstream strength
increased, while the correctly specified Model 2 standardized contrast remained
near zero.

| Upstream strength | Mean raw gap | Mean Model 2 adjusted gap | Mean race-blind ML gap |
|---:|---:|---:|---:|
| 0.00 | +0.01 pp | +0.02 pp | -0.04 pp |
| 0.25 | -1.61 pp | +0.02 pp | -1.60 pp |
| 0.50 | -3.33 pp | -0.00 pp | -3.23 pp |
| 0.75 | -5.09 pp | +0.00 pp | -4.93 pp |
| 1.00 | -6.93 pp | +0.00 pp | -6.68 pp |
| 1.25 | -8.84 pp | +0.01 pp | -8.51 pp |
| 1.50 | -10.80 pp | +0.02 pp | -10.39 pp |

Adjustment removes the deliberately modeled upstream pathway. It does not show
that upstream inequality is unimportant, and the adjusted coefficient is not a
causal estimate of real-world discrimination.

## Mixed mechanisms

The mixed grid behaved as designed. Raw and race-blind ML gaps became more
negative with upstream strength. Model 2 continued to recover the configured
direct component: across the 16 mixed cells, absolute mean coefficient bias was
at most 0.0036 log odds. The moderate mixed setting (`-0.25`, strength `1.0`)
had:

- mean raw gap: -10.60 percentage points;
- mean adjusted probability contrast: -3.20 percentage points;
- mean race-blind ML gap: -6.89 percentage points;
- mean race-aware ML gap: -10.64 percentage points;
- mean fixed-feature true direct contrast: -3.21 percentage points;
- Model 2 detection: 100%; coverage: 98%.

## Sample-size behavior

For the moderate direct setting, Model 2 Monte Carlo SD fell from 0.347 at
`n = 1,000` to 0.021 at `n = 100,000`. Detection rose from 24% to 100%.
Coverage ranged from 92% to 100% across the declared sizes. This illustrates
sampling uncertainty; it does not change the fixed DGP treatment.

## Result-derived thresholds

Thresholds are inferred from the completed results rather than hard-coded:

| Criterion at 100,000 rows | First reached value |
|---|---:|
| Negative Model 2 detection in at least 80% of replications | direct magnitude 0.10 |
| Raw gap below -2 pp in at least 80% | upstream strength 0.50 |
| Race-aware correct-sign recovery within 25% in at least 80% | direct magnitude 0.25 |
| Race-blind relative recovery below 50% in at least 80% | direct magnitude 0.05 |

The race-blind threshold is low because the direct-only treatment is omitted
from that feature regime by design.

## Mechanism signatures

The signatures are descriptive synthetic diagnostics:

- **Pattern A:** raw, adjusted, and race-blind ML evidence is practically null;
- **Pattern B:** raw and adjusted statistical signals appear, the race-blind ML
  gap is small, and the race-aware gap is practically nonzero;
- **Pattern C:** a raw and race-blind ML gap appears while Model 2 is not
  statistically nonzero;
- **Pattern D:** raw, adjusted, and both ML gaps appear, with the race-aware
  magnitude at least one point larger than race-blind;
- unmatched cases are **Other / ambiguous**.

At 100,000 rows, the modal fair signature was Pattern A (96%), moderate direct
was Pattern B (98%), moderate upstream was Pattern C (98%), and moderate mixed
was Pattern D (100%). These labels are not causal classifiers for observational
data.

## Outputs

The routine writes tidy run and summary CSVs for direct, upstream, mixed, and
sample-size/detection families, plus:

- `monte_carlo_detection.csv`;
- `monte_carlo_coverage.csv`;
- `mechanism_signatures.csv`;
- `detection_thresholds.csv`;
- `sensitivity_paper_summary.csv`;
- 13 Matplotlib figures under `results/figures/`;
- experiment metadata and atomic per-run JSON under `results/metrics/`.

Generated artifacts are Git-ignored. Reproduce them with:

```bash
python3 experiments/run_sensitivity.py --experiment all --resume --workers 2
```

Use `--quick` for the 16-run smoke design. The seventh dashboard page reads
precomputed summaries only and never launches Monte Carlo work.

## Limits

- Synthetic recovery validates behavior only under the specified DGP.
- Observed gaps are not automatically discrimination.
- Adjusted gaps are model-dependent associations, not automatic causal effects.
- A model that reproduces the synthetic lender is not therefore normatively fair.
- Logistic coefficients are non-collapsible; changes across Models 0–3 are not
  formal mediation effects.
- No HMDA data, SHAP, neural network, fairness optimization, random forest, or
  boosted-tree Monte Carlo analysis is part of this experiment.
