# DTI Recalibration

## Original problem

The first generator calibration placed 10.7893% of the one-million-row fair
calibration population at the 0.65 DTI ceiling. Seed-4994 development samples
placed approximately 10.50%–10.84% at that boundary, exceeding the predeclared
5% maximum. The threshold was not raised and DTI was not altered after the
approval outcome.

## Diagnostic methodology

The diagnosis generated a deterministic 100,000-row `fair_baseline` population
with seed 4994. `pre_clip_dti` and its payment components were reconstructed in
memory from the same equation used by the generator. These fields were never
added to the persisted 24-column schema.

The diagnostic decomposed annual and monthly income, existing debt, property
value, loan amount, LTV, projected principal and interest, estimated tax and
insurance, total housing expense, total obligation, and unclipped DTI. It also
compared four DTI bands and tested a small sequence of interpretable population
dependency changes. Reproducible tables are written by
`experiments/calibrate_population_v1.py` under `results/tables/`.

The implemented equation matched the documented equation exactly:

```text
monthly gross income = annual_income / 12
principal and interest = loan_amount * 0.00648598
tax and insurance = property_value * 0.0012
pre_clip_dti = (existing debt + principal and interest + tax and insurance)
               / monthly gross income
```

There was no arithmetic bug.

## Root cause

The dominant problem was the loan request relative to income, rather than an
incorrect payment factor. `pre_clip_dti` correlated 0.8494 with loan amount
divided by annual income. The original property equation used an income
elasticity of 0.55 and an independent residual log SD of 0.30. Consequently,
property value and requested loan principal did not fall enough with income in
the lower-income tail.

Applicants below 0.50 DTI had median income of $106,914, median loan principal
of $289,558, and median loan/income of 2.78. Applicants at or above 0.65 had
median income of $56,546 but a similar median loan of $320,031, producing median
loan/income of 5.88. Their median existing debt was also higher ($903 versus
$495), but the principal-and-interest contribution and the loan/income
correlation identify the request-capacity mismatch as the larger mechanism.

Loan purpose and occupancy mixes changed little across the bands. The highest
DTI band had somewhat more FHA/VA lending and higher median LTV, but neither
category composition nor LTV alone explained the excess boundary mass.

## Candidate changes

Candidates were evaluated sequentially, not through automated optimization.
All rates below use the same 100,000 rows and seed 4994.

| Candidate | DTI ceiling |
|---|---:|
| Original: property elasticity 0.55, residual SD 0.30 | 10.592% |
| Residual SD 0.25 only | 9.216% |
| Property income elasticity 0.70 only | 8.899% |
| Property income elasticity 1.00 only | 6.757% |
| Existing-debt income elasticity 1.00 only | 8.361% |
| Property elasticity 0.90 plus debt elasticity 1.00 | 5.660% |
| Property elasticity 0.90 plus residual SD 0.25 | 5.682% |
| Property elasticity 0.85 plus residual SD 0.20 | 4.883% |
| Property elasticity 0.90 plus residual SD 0.20 | 4.572% |
| Property elasticity 0.90 plus residual SD 0.18 | 4.198% |

Single-parameter candidates were rejected because they remained above 5%.
Changing existing-debt dependence alone did not address the main loan/income
mechanism. The 0.85/0.20 property candidate was rejected because 4.883% left
almost no margin. The 0.90/0.20 candidate passed but retained a smaller margin
than preferred. Payment and tax/insurance factors were not candidates because
their documented mathematics was correct and their empirical anchors had not
changed.

## Selected calibration

Two logically connected, researcher-chosen synthetic dependency parameters in
the property-value equation changed:

| Parameter | Before | After |
|---|---:|---:|
| Income elasticity | 0.55 | 0.90 |
| Residual log standard deviation | 0.30 | 0.18 |

This makes requested property values more closely reflect borrower income and
reduces independent request variation. It does not impose an approval-based
loan cap, condition on future outcomes, or modify any group specifically.

The $400,000 property reference median, income distribution, debt distribution,
LTV mechanism, 6.75%/360-month payment assumption, tax/insurance factor,
demographic shares, latent-factor correlation, underwriting slopes, DTI ceiling,
and every direct/upstream experimental treatment remain unchanged.

The property dependency parameters are B1 researcher-chosen synthetic
parameters. They are not empirical estimates of borrower demand or lender
behavior. The rate/payment factor remains an empirically anchored
approximation. This recalibration seeks internal coherence, not HMDA replication.

## Before and after DTI

| Statistic | Before | After |
|---|---:|---:|
| Mean | 0.4246 | 0.4077 |
| Standard deviation | 0.1821 | 0.1279 |
| Minimum | 0.0514 | 0.0641 |
| p25 | 0.2972 | 0.3239 |
| Median | 0.3923 | 0.3915 |
| p75 | 0.5153 | 0.4711 |
| p90 | 0.6589 | 0.5636 |
| p95 | 0.7628 | 0.6321 |
| p99 | 1.0081 | 0.8122 |
| Maximum | 2.4490 | 2.2788 |
| Above 0.50 | 27.457% | 18.787% |
| Above 0.60 | 14.722% | 6.909% |
| At or above 0.65 | 10.592% | 4.198% |

The final one-million-row calibration population has 4.2781% at the DTI
ceiling, also below the 5% criterion.

## Before and after borrower/loan summaries

| Median or tail statistic | Before | After |
|---|---:|---:|
| Annual income | $94,078 | $94,078 |
| Credit score | 720 | 720 |
| Liquid assets | $37,426 | $37,426 |
| Existing monthly debt | $559 | $559 |
| Property value | $396,155 | $395,672 |
| Loan amount | $297,359 | $296,182 |
| LTV | 0.7654 | 0.7654 |
| Median loan/income | 3.1833 | 3.2068 |
| p95 loan/income | 6.1277 | 4.7558 |

The central borrower and loan profiles remain stable while the problematic
loan/income tail contracts. After recalibration, the remaining high-DTI band is
more strongly characterized by high existing debt: its median existing debt is
$1,632 and median loan/income is 4.07. This residual tail is retained rather
than eliminated because some high-obligation applications are useful and
plausible in a synthetic application population.

## Approval-intercept recalibration

The old population-specific intercept was `2.2422267822548747`. After the
property dependency change, the designated one-million-row fair population was
regenerated with seed 499400 and deterministic bisection was rerun.

- New intercept: `2.0362202310934663`
- Target mean probability: `0.80`
- Achieved mean probability: `0.8000000002705238`
- New config fingerprint:
  `359d34523d9e044da56d2fae3bb300a8fcce5028bbe2e030336070520033967c`

The new intercept is frozen across all four scenarios. The current artifact
records the previous calibration and the reason `population/DTI recalibration`;
the complete old artifact is also retained as
`results/metrics/calibrated_intercept_pre_dti_recalibration.json`.

## Limitations

- Strengthening property-income dependence raises their correlation from about
  0.65 to 0.90. This is a transparent synthetic design choice and may be varied
  in future sensitivity experiments.
- The calibration controls the canonical population, not every finite sample;
  clipping rates must still be checked on each run.
- The remaining extreme unclipped DTI values largely reflect existing-debt and
  residual combinations. They are winsorized transparently rather than deleted.
- No synthetic parameter should be presented as an estimate of actual mortgage
  applicants, underwriting behavior, or discrimination.
