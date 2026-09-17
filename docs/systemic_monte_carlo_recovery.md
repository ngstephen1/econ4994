# Prompt 17 recovery audit

## Prompt 17B predeclared mini-batch

Before executing Prompt 17B, corrected replication indices **0, 1, 2, and 3**
were selected by deterministic numeric order, without inspecting their legacy
outcome values. Together with already corrected replication 13, the checkpoint
set is `[0, 1, 2, 3, 13]`. The four new runs will use the corrected implementation
fingerprinted by replication 13. The legacy copies of those indices remain
`provisional_legacy`; corrected records are `validated_corrected`. No legacy
scientific output is eligible for the corrected summary.

## Evidence and boundaries

The last committed scientific milestone is Prompt 16 (`8a9f71f`). Version 1,
the calibrated economic population, opportunity effects, risk equations,
model grid, and neutral-world budget fractions are unchanged by this recovery.
The prior Prompt 17 artifacts remain **provisional legacy evidence**: 19 records
marked successful, replication 13 failed, and 1,368 saved portfolio summaries.
They are not automatically reusable under the corrected provenance contract.

The design remains master seed 499417, 10,000 applicants per full replication,
strengths 0, 0.10, 0.20, 0.40, additive and nonlinear risk worlds, and three
policies (oracle, traditional logit, HistGB). Within each replication, budgets
are 100%, 40%, and 20% of neutral-world evaluation requests and stay fixed
across treatment strengths. This gives 72 portfolios per replication. Named
SeedSequence children preserve independent populations and within-replication
common random numbers. The target of 50 successful replications would contain
3,600 portfolios; it has not been completed.

The long-running retry (PID 98309) was confirmed by its command and stopped
with SIGTERM. Its absence was checked before modifications. A pre-repair copy
of source, configuration, tests, records, summaries, and figures is retained
locally under `results/recovery/pre_repair_v30sCS/`. Original results remain in
place. Earlier console logs and overwritten attempt durations cannot be
reconstructed from the surviving records; we do not invent them.

## Solver correction

The former budget guard narrowed the feasible set. It is removed. For requests
represented exactly in cents, costs are converted to integer cents and divided
by their integer greatest common divisor. The budget bound is floored in those
same integer units. This preserves precisely the original set of feasible
binary portfolios; it is not a reduction by an arbitrary numerical allowance.
Non-cent-valued requests retain the original dollar formulation.

The original decimal-dollar constraint is checked again after solving, along
with binary integrality, nonpositive-profit exclusion, the reported objective,
and solver-reported optimality with MIP gap at most 32 machine epsilons
(`7.105427357601002e-15`). The raw gap is retained, not rounded to zero.
These are numerical solver certificates, not a
symbolic proof for arbitrary floating-point profit values. Each solve has a
60-second limit and native HiGHS integrality tolerance `1e-9`. Time limits and
infeasible candidates remain failures. SciPy's specific notice that it forwards
this native option is suppressed; the option itself is saved in diagnostics.

### Reproduced numerical failure

On replication 13, nonlinear risk, strength 0.40, oracle policy, 40% budget:

- Original budget: `$241,599,559.87200004`.
- Original dollar formulation reproduced a rounded budget residual of
  `+$0.21799996`, exceeding its original tolerance of approximately `$0.02416`.
- Integer cents alone was insufficient: one candidate binary coordinate was
  `8.941906141091138e-7` from an integer, with the same rounded violation.
- Integer cents plus tighter integrality tolerance solved the saved case in
  0.736 seconds: funded principal `$241,598,781.41`, unused budget
  `$778.46200004`, objective `$93,147,586.48686095`, reported gap zero.
- A separate 300-applicant test returned an optimal objective/bound pair differing
  by one floating-point step (relative gap `1.4573221841752937e-16`). This motivated
  the explicit roundoff allowance, not an economic relaxation.

The actual failing principal/profit vectors are retained as a portable regression
fixture in `tests/fixtures/systemic_r13_knapsack.json`; it has no demographic,
applicant-characteristic, or payment-history panel. Full replay diagnostics remain
under ignored `results/recovery/` paths.

Optional diagnostic checkpoints contain only request keys, principal, perceived
profit, budget, solver options, candidate vector, status, integrality error,
and budget residual—not applicant panels or repayment histories. They are
local ignored artifacts. Replication 13 uses a unique diagnostic directory.

Replay a captured case without training any models:

```bash
python3 experiments/replay_portfolio_diagnostic.py PATH_TO_CASE.json \
  --output-directory results/recovery/solver_replay
```

## Provenance and resumption

- Run identity incorporates actual economic-lending source/config file hashes,
  dependency versions, Python/platform, analysis version, seed, and Git HEAD.
- Source/config changes during a replication prevent a success result.
- Every completed attempt has an immutable JSON copy. An atomic latest-status
  pointer is separate; retries cannot erase recorded failures.
- Records have checksums and are checked for expected cells, unique keys,
  required finite metrics, budgets, feasibility, and unguarded optimal solves.
- New output paths contain `recovery`; legacy files are neither relabeled nor
  combined with these records.
- Summary-only checks all requested records before doing anything. Missing or
  invalid records raise an error without generation, fitting, or solving.
- Secondary audit errors are explicit missing audit results, not solver failures.
- Failure fractions above 5% stop sequential execution or further parallel
  batches. Any completed invocation containing failures exits nonzero.
- Runtime metadata distinguishes invocation wall time from the sum of recorded
  attempt compute times. Neither reconstructs interrupted legacy runtime.
- Each worker uses one native math thread, without changing estimator settings.
  This prevents process-level concurrency from multiplying unrestricted native
  thread pools. Its measured runtime is reported separately from the legacy runs.

## Interpretation and acceptance

Twenty successful replications would be an interim checkpoint. The configured
target remains 50. No additional batch is launched automatically during recovery.
Single-replication SD and MCSE are undefined, not zero. Distribution plots retain
separate budget categories rather than pooling multiple correlated portfolios
as independent replications.

Before publishing uncertainty conclusions, the remaining repeated-seed study
must be authorized and completed with compatible provenance and explicit
successful-replication denominators. A small-population, same-seed
one-versus-two-worker test checks numerical determinism, not laptop throughput
at the full 10,000-applicant size.

## Verification results

The recovery checkpoint previously passed 277 tests. After adding the Prompt
17B checkpoint tests, the focused recovery/Monte Carlo suite passed **54 tests
in 67.21 seconds** and the complete suite passed **284 tests with 32 warnings
in 85.82 seconds**.
The warnings are the existing sklearn/SciPy L-BFGS-B `disp`/`iprint`
deprecations (26 in ML benchmark tests, six in sensitivity tests).
The one-/two-worker integration test runs the real generation, fitting,
allocation, and audit pipeline at 300 applicants for seeds/indices 0 and 1,
with all four strengths, both risk worlds, and the frozen model grid.
Outputs match exactly after removing timing and process peak-memory fields.

The separate provisional legacy audit reports 20 attempts, 19 marked successes,
1,368 portfolio rows, zero *reported aggregate* budget violations, and a
denominator of 19 for each summary cell. All 20 records lack source-content
provenance. Aggregate feasibility cannot independently certify the missing
selection vectors or original source state; reuse remains unauthorized.
Corrected budget-separated figures and summary CSV are under
`results/recovery/legacy_provisional/`, explicitly labeled provisional.

Rebuild this read-only-input audit without launching fits or solves:

```bash
python3 experiments/summarize_legacy_systemic_provisional.py
```

## Corrected replication 13: accepted checkpoint

The final corrected run completed in **404.2239 seconds**: 364.9327 seconds
fitting models and 33.2634 seconds solving portfolios. Process high-water RSS
was approximately 996 MB on this Mac; this is not a per-stage memory measure.
The record includes 72 portfolios, 24 model-performance rows, eight ML
selections, 144 secondary audit rows, 54 nonzero-treatment switcher-transition
rows, and 16 population-pathway rows.

All portfolios satisfy the original budget. The largest exact decimal-dollar
budget residual was `-$24.53600002`; the largest raw reported MIP gap was
`1.1929640385926911e-15`. There were no secondary audit exceptions; 124 of 144
audit fits were identified. Unidentified audits remain explicitly excluded
from coefficient summaries, not interpreted as zero effects.

Accepted record:
`results/metrics/systemic_mc_recovery_runs/bacf2df0a7ae3fc7017ef1172f92a104e483374361c55107936073b4928e7aa9.json`.
The earlier failed cent-only attempt is retained separately in immutable
attempt history. Recorded cumulative attempt compute time is 786.8048 seconds
for those two recovery attempts; isolated probes, tests, and interrupted legacy
execution are not included in that measure.

Summary-only resumption was exercised successfully:

```bash
python3 experiments/run_systemic_monte_carlo.py \
  --replication-index 13 --summarize-only
```

It resumed the record with **zero newly executed replications**. The measured
load/preflight portion was 0.0883 seconds (not the duration of the earlier fit,
and not the entire process including import/output overhead).
Corrected tables use `results/tables/v2_systemic_mc_recovery_*.csv`; metadata
uses `results/metrics/v2_systemic_mc_recovery*.json`. With only one corrected
replication, SD/MCSE remain missing and uncertainty figures are not presented.

### What remains

- Corrected validated study count: **1**, not 20 or 50.
- Legacy successes: **19**, preserved but not authorized for automatic reuse.
- Scientific target: **50 successful replications**, unchanged.
- At the one observed corrected runtime, 19 further sequential runs to reach
  an interim 20 would take about **2.1 hours**; 49 to reach 50 about **5.5 hours**.
  These are simple projections from one run, not a throughput benchmark or
  guaranteed completion time. Full-size two-worker scaling has not been measured
  for the corrected implementation.
- No additional full-size population was run, and no research parameters were
  retuned. All processes launched for recovery and tests have completed.
- No files were staged, committed, or pushed.

## Prompt 17B: corrected five-run checkpoint

The predeclared indices `[0, 1, 2, 3, 13]` all passed the corrected record
validator. Indices 0–3 were computed in two bounded two-worker batches; no
legacy cache was accepted. The final status is **5/5 corrected successful** and
**0 final-status failures**. Replication 13 retains one earlier failed corrected
attempt in its immutable history.

### Provenance and resume

All five records share:

| Contract | Fingerprint/version |
|---|---|
| Scientific Monte Carlo config | `36a2c202cd0eba18b38058ce80164e8a2f807efb8b60895cadbf792146709fea` |
| Frozen systemic config | `59cf4f50249620c678e77ba8da55c1cc6ac9b8cad8950554bc6bb578558df64b` |
| Relevant source content | `90da2aba0e9315544114bd73ad07d365702c97b666eb07d5cce78b0c97461fed` |
| Dependencies | `1889b540050d6c7efd8603133e8bac35767f0b9ad8bf95a4e20f9cad67a136f1` |
| Solver settings | `1d96069b9d3e41a7e803a4d8305df5802979d4edd1a9e605d7aa00f3de55ea6e` |
| Analysis schema | `systemic-opportunity-mc-v2-recovery` |

Replication 0 was then invoked with `--summarize-only`. It reused the validated
record, reported zero newly executed replications, and took 0.0673 seconds for
the recorded load/preflight portion. No generation, fitting, or optimization
was launched.

### Runtime

`Cumulative` includes every recorded corrected attempt for an index, so the
failed pre-repair attempt for index 13 is counted. Times are seconds.

| Index | Latest successful | Model fit | Solver | Recorded attempts | Cumulative |
|---:|---:|---:|---:|---:|---:|
| 0 | 402.020 | 352.652 | 42.449 | 1 | 402.020 |
| 1 | 397.957 | 350.451 | 40.553 | 1 | 397.957 |
| 2 | 398.512 | 351.798 | 39.839 | 1 | 398.512 |
| 3 | 409.203 | 360.007 | 42.251 | 1 | 409.203 |
| 13 | 404.224 | 364.933 | 33.263 | 2 (one failed) | 786.805 |

Across per-index cumulative time: minimum 397.957, median 402.020, mean
478.899, maximum 786.805, and sample SD 172.183 seconds. Forty-five remaining
sequential successful replications project to **5.03 hours** using the median
and **5.99 hours** using the cumulative mean. These are planning estimates, not
guarantees; the remaining run was not launched.

### Solver checkpoint

- 360 portfolios attempted and 360 optimal within the recorded roundoff rule.
- Zero time limits, failed portfolios, and budget violations.
- Maximum reported MIP gap: `1.5809011396958814e-15`.
- Maximum signed budget residual: `-$0.64799998`; therefore maximum positive
  violation and normalized feasibility violation are both zero.
- Exactly 360 solver-input diagnostic files exist for the five successful
  attempts. No error replay was triggered in the four new runs.
- Every record contains exactly 72 unique cells: two risk worlds × four
  strengths × three policies × three budgets. Required metrics are finite.

### Interim scientific checkpoint

This is an **INTERIM CORRECTED 5-REPLICATION SUMMARY**, not the final Prompt 17
result. At strengths 0.10, 0.20, and 0.40, mean Group B opportunity-access
changes were -2.157, -4.439, and -8.956 percentage points. Mean opportunity
switcher counts were 107.8, 222.0, and 448.0 applicants, respectively.

For the strongest treatment and 40% budget, selected mean effects were:

| Risk world | Policy | Δ Group A funding | Δ Group B funding | Δ B−A gap | Δ true expected portfolio profit |
|---|---|---:|---:|---:|---:|
| Additive | Oracle | +0.263 pp | -0.120 pp | -0.382 pp | -$143,033 |
| Additive | Traditional | +0.222 pp | -0.120 pp | -0.341 pp | -$145,296 |
| Additive | ML | -0.273 pp | +0.215 pp | +0.487 pp | -$35,643 |
| Nonlinear | Oracle | +0.162 pp | -0.079 pp | -0.241 pp | -$125,565 |
| Nonlinear | Traditional | +0.082 pp | +0.020 pp | -0.061 pp | -$117,538 |
| Nonlinear | ML | +0.361 pp | -0.197 pp | -0.558 pp | -$135,722 |

At that cell, mean oracle regret was $62,673 for traditional versus $852,691
for ML in the additive world, and $527,578 versus $572,536 in the nonlinear
world. These five-run values are descriptive only. Across the complete interim
table, many funding and gap effects crossed zero across seeds (187 of 324
nonzero-strength delta cells). No final interval, sign-stability conclusion,
model ranking, or robustness claim is supported.

Full mean/SD/MCSE/median/min/max results are in
`results/tables/v2_systemic_mc_corrected_checkpoint_summary.csv`.

### Legacy diagnostic

For corrected indices 0–3, all 11,520 comparable downstream numeric values
matched the provisional legacy records exactly. Twenty-seven solver-gap values
differed by at most `1.5809011396958814e-15`, and solver status strings did not
differ. Legacy replication 13 failed, while its corrected record succeeded.
Applicant-level portfolio decisions cannot be compared because neither compact
record persisted decision vectors. Therefore aggregate agreement is reassuring
but cannot establish identical portfolios. The legacy records also lack the
corrected source provenance, so they remain **not scientifically reusable** and
are not blended with corrected results.

### Decision

**A. READY_FOR_FULL_50.** The mini-batch achieved 5/5 successes, complete unique
cells, finite required metrics, consistent provenance, clean solver behavior,
read-only resume reuse, unchanged scientific parameters, and a manageable
resumable runtime estimate. This classification authorizes planning the next
run; it does not mean Prompt 17 is complete and does not launch the remaining 45.
