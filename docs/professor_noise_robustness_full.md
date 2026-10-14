# Professor Model — Step 8B Full Noise-Only Monte Carlo Study

Stage record: results, test counts, and later-work deferrals below describe
this step. See [current status](PROJECT_STATUS.md) and the
[documentation index](README.md) for the latest state across all three families.

## Research question and scope

Across repeated noisy measurements of the same underlying applicants, how
stable are risk estimates, lending decisions, group approval gaps, and true
expected portfolio profits? This study is predeclared at **R=200** independent
eta draws. It conditions on one synthetic population, not new applicants or
new training/evaluation splits. **All 200 replications are complete and
validated:** five scientifically identical pilot reuses and 195 new noise
replications, with zero failed attempts. The repository-wide test suite still
has four unrelated, pre-existing V2 artifact-lookup failures, detailed below.

## Frozen scientific design

The existing 100,000 applicant IDs and Y, D, B, C, epsilon, omega, and true_PD
are unchanged. The original 80,000 training and 20,000 evaluation assignments,
contracts, principals, payment schedules, true expected profits, capital, and
oracle decisions are fixed. No population generation, true-risk recalibration,
new feature, realized default outcome, or new lending policy is introduced.

Only measurement noise is generated:

```text
eta_ir ~ Normal(0, standard deviation 0.1)
PN_ir = 0.8 * saved_true_PD_i + eta_ir
```

PN is unbounded and retained raw, including negatives. No rescaling, clipping,
sigmoid, recentering, or variance change occurs. `SeedSequence(499408).spawn(200)`
defines child keys `[0]` through `[199]`, each using PCG64 and ascending
lexicographic applicant-ID order. The first five children match the pilot.

Both operational fits reuse the unchanged Step 8A functions:

- Group-aware unregularized OLS: `PN_r ~ 1 + Y + D + C + B`.
- Group-blind unregularized OLS: `PN_r ~ 1 + Y + D + C`.

Income remains in annual dollar levels, D in ratios, and C in original signal
units. Fits use training labels only; evaluation labels do not select settings.
Hidden truth/noise factors do not enter either bank's feature matrix. Raw
predictions and `clip(raw,0,1)` predictions remain separate. Clipping is an
accounting safeguard, not probability calibration.

## Identical accounting and restricted cutoff policy

T=30, theta=0.10, r=0.05 **per period**, ell=theta·Y, L=30·ell, and
scheduled payment `ell*(1+r)^t` are inherited without modification:

```text
expected_receipts_bank_i = sum[t=1..30] (1-bank_PD_bounded_i)^t * payment_it
expected_profit_bank_i = expected_receipts_bank_i - L_i
approved_i(tau) = 1{bank_PD_bounded_i <= tau}
```

The exact saved budget is **$1,826,087,220.7466326**, never recomputed. Each
policy maximizes perceived profit over common-cutoff prefixes, with exact
binary-float summation, indivisible score ties, full requests, no skipping,
and a separate null-cutoff reject-all action. Independent rational-arithmetic
audits validate every cutoff candidate. Oversized zero-score blocks are
reported, not split to manufacture lending. These are not unrestricted
knapsack allocations. True portfolio profit is evaluated only after a bank
freeze, using saved evaluator values; the oracle is read, never reoptimized.

## Pilot reuse, provenance, persistence, and failures

The full study has its own wrapper schema and R=200 configuration. Pilot
reuse requires the original pilot schema, identical per-replication scientific
settings (excluding the wrapper's R/schema fields), recorded source contents,
dependencies, seeds, and valid immutable records. Predictions, accounting,
candidate cutoffs, allocations, and evaluator metrics are replayed exactly
from saved coefficients and compared with the pilot before adoption. Reused
records retain the pilot runtime separately; new adoption/validation runtime
is measured, not falsely reported as fresh model fitting.

Compatibility uses the **source paths recorded in the pilot manifest**.
The old pilot CLI fingerprints a directory glob, which is enlarged by the new
Step 8B modules and can reject a direct old-CLI resume in the expanded tree.
We do not rewrite the pilot code or manifest to bypass that guard. Instead,
the full-study adoption validates every originally recorded source/input hash
and exact scientific replay under the original pilot identity. The distinction
between compatible per-replication science and a changed outer directory
inventory is explicit, not a silent provenance relabeling.

The immutable full manifest records configuration, scientific/source/settings
fingerprints, dependencies, Python, Git revision/dirty state, and original
file checksums, including all pilot results. Git HEAD alone is not scientific
provenance. Every result has a stable SHA-256 run ID and matching bank-freeze
hash. Raw rows are released after each replication and are not persisted.

Attempt JSONs are immutable and retained on both success and failure. Failed
attempts record stage, exception, traceback, and runtime. Invalid saved results
are moved into an evidence-preserving `invalid_records/` directory and are
not counted as complete; retries produce separate attempts. Resume reuses
only semantically valid successes with complete policy/group coverage, valid
model contracts, arithmetic, budget checks, and decomposition. Summary-only
cannot draw noise, fit, or optimize. An incomplete study is explicitly
reported and cannot produce a completed 200-run summary.

## Evaluator-only OLS decomposition

Because OLS is linear in the target for fixed X:

```text
beta_hat(PN) = beta_hat(0.8 * true_PD) + beta_hat(eta)
```

One evaluator-only, noise-free OLS uses the same sorted training IDs and
`[1,Y,D,C,B]` matrix. It is saved once, never used by either operational bank
or to choose approvals. For each replication, the noise B coefficient is
the corresponding row of the same statsmodels pseudoinverse multiplied by
training eta. This is equivalent to the eta-target OLS coefficient without
creating an additional operational model. Resume loads the signal reference
rather than refitting it.

Decomposition uses predeclared absolute tolerance **1e-12** and relative
tolerance **1e-10**. A nonzero signal B coefficient is possible from finite
sample associations and nonlinear misspecification despite no structural
B term in true risk. It is not evidence of a discriminatory preference.

## Summary definitions and denominators

- Sample SD (ddof=1): variation across individual noise realizations.
- MCSE of the mean: SD / sqrt(number of defined successful replications).
- Quantiles p2.5/p5/p25/p75/p95/p97.5: empirical outcome spread using NumPy's
  linear interpolation, not real-world discrimination confidence intervals.
- Approval gaps: B=1 rate minus B=0 rate, displayed in percentage points.
- Absolute-gap change: `abs(blind gap) - abs(aware gap)`; negative means smaller
  blind absolute disparity.
- Profit change: blind minus aware **true expected** profit, not perceived or
  realized profit. Oracle differences remain signed restricted-cutoff comparisons.
- Original-magnitude event: `abs(aware gap in pp) >= 14.385705`, using the
  requested rounded threshold explicitly. This is not an empirical p-value.
- Coefficient/gap/profit sign and paired-comparison frequencies use 200
  replications. Zero/tied events are reported separately from positive/negative.
- Tie and reject-all frequencies use 200 per policy and 400 for pooled
  policy-replication reporting. Every frequency includes count and denominator.
- Reject-all remains a legitimate observation: its approval gap and profits
  are zero. Means of approved-borrower risk and null cutoffs are undefined;
  summaries report defined-value counts rather than inventing values.

## Reproduction and artifact layout

```bash
python3 experiments/run_professor_noise_full.py
python3 experiments/run_professor_noise_full.py --resume
python3 experiments/run_professor_noise_full.py --summarize-only
python3 -m pytest -q tests/test_professor_noise_full.py tests/test_professor_noise_pilot.py
python3 -m pytest -q tests/test_professor_*.py
python3 -m pytest -q
```

To reproduce the publication figures with the verified caption spacing, use
the saved presentation-only Matplotlib profile when regenerating summaries:

```bash
MATPLOTLIBRC="$PWD/results/metrics/professor_model/noise_robustness_full/matplotlibrc" \
  python3 experiments/run_professor_noise_full.py --summarize-only
```

This profile changes only `figure.constrained_layout.h_pad` to 0.22. Initial
figure renders are retained under `initial_figure_render/`; the final four
PNGs were visually inspected for caption/axis separation, alignment, and
clipping. Scientific source fingerprints and all five CSV hashes stayed
unchanged during the layout correction. `publication_rendering.json` records
the profile and figure hashes separately from scientific provenance.

No row-count, noise-SD, or replication-count override is offered by the CLI.
The frozen scientific configuration is `configs/professor_model/noise_robustness_full.yaml`.
New reusable code lives in `noise_full.py` and `noise_full_io.py`; orchestration
lives in `experiments/run_professor_noise_full.py`. Step 8A code is unchanged.

All paths are relative to `/Users/macbook/Hack/econ4994/`:

| Directory | Files |
|---|---|
| `results/metrics/professor_model/noise_robustness_full/` | One immutable `manifest.json`; `replications/replication_000.json` through `replication_199.json`; immutable `attempts/` and bank `freezes/`; `signal_reference_EVALUATOR_ONLY.json`; completion reports, session timing and `latest_report.json`. |
| `results/tables/professor_model/noise_robustness_full/` | `noise_full_results.csv` (one policy/replication row); `noise_full_coefficient_summary.csv`; `noise_full_group_outcome_summary.csv`; `noise_full_tie_diagnostics.csv`; `noise_full_summary.csv`. |
| `results/figures/professor_model/noise_robustness_full/` | Exactly four figures: `beta_B_distribution.png`, `approval_gap_distributions.png`, `absolute_gap_changes.png`, `true_profit_changes.png`. |

The tidy results table repeats paired-level contrasts on its two policy rows;
paired summaries deduplicate by replication and use 200, not 400, observations.
All fitted coefficients remain in the immutable replication JSONs; the
coefficient-summary CSV aggregates each of the nine policy/term combinations.
Generated-output Git-ignore rules remain unchanged. Nothing is committed or
pushed as part of this step.

## Completed run and preservation checks

Exactly **200 unique replication indices** were attempted and successfully
recorded, with **200 immutable success attempts**, zero failed attempts, and
zero invalid saved records. The first five pilot records passed exact replay
without refitting; the remaining 195 replications added 390 operational OLS
fits. The complete evidence represents 400 policy/replication cells and 400
operational fits including the pilot's ten fits.

The initial invocation took **641.545 seconds (10 minutes 41.5 seconds)**;
cumulative recorded attempt runtime was **639.509 seconds**. Tests ran
alongside part of this invocation, so this is an observed session runtime,
not an isolated performance benchmark. A subsequent resume reused all 200
records and computed/adopted zero replications (1.545 seconds). Summary-only
also completed without generation, fitting, optimization, or new attempts.

All **285 independently snapshotted original source, configuration, test,
dataset, and result files** remain byte-identical, including Step 8A evidence.
Record identities, complete cell coverage, arithmetic, bank-freeze ordering,
oracle identity, and original dollar-budget feasibility passed validation.
The full-study fingerprint is
`99b52586da3cee55addb0a72c47bac7fc3cb7b7cd66f94707d2e7164f88d7f36`.

A final inventory check detected newly created Finder metadata at
`results/figures/professor_model/.DS_Store`. It contained no scientific data
and was moved recoverably to
`results/metrics/professor_model/noise_robustness_full/os_metadata_evidence/figures_professor_model.DS_Store`.
The strict guard was not bypassed; the matching original inventory and
summary-only check were restored without changing scientific code.

## Coefficient and approval-gap distributions

Each statistic below uses 200 paired noise replications. SD describes
between-draw variation; MCSE describes precision of the mean. The p2.5/p97.5
columns describe the empirical central outcome spread, not a causal interval.
All requested intermediate quantiles, minima, and maxima are in
`noise_full_summary.csv` and the coefficient/group summary files.

| Metric | Mean | Median | Sample SD | MCSE | p2.5 | p97.5 |
|---|---:|---:|---:|---:|---:|---:|
| Aware B coefficient, probability per B unit | 0.000063278 | 0.000021057 | 0.001063127 | 0.000075174 | -0.001909077 | 0.002188145 |
| Aware approval gap, pp | +8.839521 | 0.000000 | 36.206101 | 2.560158 | -46.657711 | +82.128929 |
| Blind approval gap, pp | +0.365663 | +0.443872 | 0.599209 | 0.042370 | -1.200024 | +1.453878 |
| Blind minus aware gap, pp | -8.473858 | 0.000000 | 36.141512 | 2.555591 | -81.585226 | +45.852011 |
| Blind minus aware absolute gap, pp | -28.021777 | -22.850167 | 23.791382 | 1.682305 | -81.585226 | +0.002092 |

| Event | Positive | Negative | Zero |
|---|---:|---:|---:|
| Aware B coefficient | 105/200 (52.5%) | 95/200 (47.5%) | 0/200 (0%) |
| Aware approval gap | 92/200 (46.0%) | 95/200 (47.5%) | 13/200 (6.5%) |
| Blind approval gap | 151/200 (75.5%) | 38/200 (19.0%) | 11/200 (5.5%) |
| Blind minus aware true profit | 141/200 (70.5%) | 47/200 (23.5%) | 12/200 (6.0%) |

The blind policy has a strictly smaller absolute approval gap in **183/200
(91.5%)** draws, not every draw. The aware gap reaches or exceeds the original
**14.385705 pp absolute magnitude in 131/200 (65.5%)** draws. Its observed
range is -71.591043 to +89.133765 pp; the blind range is -1.663292 to
+1.832899 pp. The original gap is therefore not a stable directional outcome
of this fixed-population/noise experiment. These frequencies do not establish
intentional discrimination or normative superiority of a policy.

## Economic results and tail behavior

All values refer to **true expected** portfolio profit, not bank beliefs or
realized default paths. The fixed oracle has 3,406 approvals and
**$2,370,206,703.87** true expected profit. Mean true profits are
**$2,200,289,797.71** aware and **$2,224,906,027.46** blind. Mean signed
oracle-minus-policy differences are **$169,916,906.16** and **$145,300,676.40**,
respectively; medians are $12,290,283.52 and $9,838,486.39. These are comparisons
between restricted cutoff families, not unrestricted-optimal regret bounds.

| Blind minus aware true profit, USD | Value |
|---|---:|
| Mean | +$24,616,229.75 |
| Median | +$651,806.68 |
| Sample SD | $235,673,271.09 |
| MCSE | $16,664,616.81 |
| p2.5 / p97.5 | -$4,279,050.08 / +$7,122,265.83 |
| Minimum / maximum | -$6,816,666.05 / +$2,365,919,379.46 |

The mean is **not a typical gain**. In replication 8 the aware policy rejects
all while the blind policy funds 4,744 applicants (difference
+$2,361,202,289.11). Replication 92 has the same pattern with 3,861 blind
approvals (+$2,365,919,379.46). These two policy discontinuities dominate the
mean and SD. They remain in every primary summary and in the full-range
profit figure, which uses a labeled symmetric-log dollar axis. The empirical
central 95% interval does not encompass those rare extreme cases; it must
be read together with the extrema and median. Blind profit is higher in
70.5%, lower in 23.5%, and equal in 6% of draws, not uniformly improved.

## Clipped-zero ties and reject-all

| Diagnostic | Aware (n=200) | Blind (n=200) | Pooled (n=400 policy cells) |
|---|---:|---:|---:|
| Reject-all selected | 13 (6.5%) | 11 (5.5%) | 24 (6.0%) |
| Zero block exceeds capital | 13 (6.5%) | 11 (5.5%) | 24 (6.0%) |
| Zero block would require over 90% of capital | 17 (8.5%) | 17 (8.5%) | 34 (8.5%) |
| Feasible zero block requires over 90% of capital | 4 (2.0%) | 6 (3.0%) | 10 (2.5%) |

The over-90% event includes infeasible blocks; those blocks **do not actually
consume capital**. All reject-all cases here follow an infeasible indivisible
lowest-risk zero block. Since larger cutoff prefixes cannot remove that
block, no nonempty cutoff is feasible. Both policies reject all in eleven
replications; only the aware policy does so in two additional replications.
No ties were split and no funded portfolio exceeded the frozen budget.
Clipping is consequently not a harmless display operation: it interacts
with the already specified common-cutoff policy through large indivisible ties.

## Signal decomposition and probability recovery

The one evaluator-only noise-free projection gives
**beta_B_signal = 0.00000384199494513**. For all 200 draws,
`beta_B_hat = beta_B_signal + beta_B_noise` within the predeclared tolerance;
the maximum absolute residual is **2.25514e-17**. The B coefficient's
between-noise SD (0.001063127) is much larger than that fixed signal component.
A small finite-population linear-projection coefficient is not a structural
group coefficient or evidence of preferential treatment.

| Mean evaluation error across 200 draws | Aware | Blind |
|---|---:|---:|
| Raw prediction RMSE versus held-out PN | 0.0999959562 | 0.0999951012 |
| Raw prediction MAE versus true_PD | 0.0007440443 | 0.0006792367 |
| Bounded prediction MAE versus true_PD | 0.0006873190 | 0.0006329107 |

Raw and bounded metrics are separately identified in the summary CSV; neither
bounding nor the preserved 0.8 signal multiplier is probability calibration.
Noise sensitivity and cutoff discontinuities, rather than a new fitted
mechanism, explain the repeated outcomes investigated in this step.

## Interpretation boundary and next research

The study measures conditional sensitivity to risk-measurement noise on one
researcher-defined population. It does not identify intentional discrimination,
ensure equal approval under group-blind estimation, establish a general profit
advantage, or describe real mortgage lenders. The true-risk oracle is not
defined as normatively fair and is not an unrestricted-optimality bound.
The 5%-per-period growing-payment contract without costs/discounting remains
stylized. New-population variation, alternative noise scales or contracts,
and coefficient-level interventions require separately specified experiments;
none is introduced here.

Next, review the noise-to-signal scale, indivisible clipped-zero policy, and
stylized contract assumptions with the professor. A separately specified
coefficient-level matched intervention or new-population experiment could
address different questions, but neither is launched automatically. The
noise-only study is complete; unrelated V2 provenance-test failures should
be resolved separately before describing the entire repository as validated.

## Validation design

The 24 new deterministic tests cover exact first-five seed/target/model
equivalence, 200 unique streams, unchanged fixed inputs, frozen oracle,
feature allowlists, evaluator-only OLS decomposition against an independent
eta regression, loading the signal reference without refitting, immutable
freeze/result consistency, failed-attempt retention/retry, invalid-record
quarantine, complete cell coverage, group/profit arithmetic, MCSE and empirical
quantiles, explicit frequency denominators, deterministic aggregation,
unique run IDs, and summary-only execution without computation.
Together with the unchanged pilot and prior professor tests, the focused
suite passed **344 tests without warnings**.

The full suite reported **624 passed, 4 failed, and 32 existing deprecation
warnings** in 138.93 seconds (executed alongside the Monte Carlo run).
No new professor-model tests failed. The four unchanged failures are:

- `test_corrected_replication_has_exactly_72_unique_complete_cells`
- `test_replication13_provenance_is_internally_consistent`
- `test_legacy_comparison_is_diagnostic_and_never_authorizes_reuse`
- `test_interim_summary_uses_five_corrected_records_only`

They are existing V2 checkpoint lookups in
`tests/test_systemic_corrected_checkpoint.py`: current HEAD `7a7e590` produces
different run-ID filenames from saved revision `8a9f71f`. The unrelated
recovery code and records were not changed or bypassed. Warnings are the
existing scikit-learn/SciPy L-BFGS-B `disp`/`iprint` deprecations: 26 ML
benchmark and six sensitivity warnings. The repository-wide suite is not
all-green, even though the professor-model suite passes.
