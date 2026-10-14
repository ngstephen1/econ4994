# Project Status

## Current research: professor's lending model

**Steps 1–8B are complete.** This family is separate from the preserved Version 1
approval framework and Version 2 repayment/portfolio experiments.

| Step | Validated work | Details |
|---|---|---|
| 1 | Repository audit and working whiteboard specification | [Specification](professor_whiteboard_model.md) |
| 2 | 100,000 applicants, seed 4994, bank/private information separation | [Population](professor_population_generation.md) |
| 3 | Hidden two-factor true default risk, exact frozen equations | [True risk](professor_true_default_risk.md) |
| 4 | Raw PN = 0.8 true_PD + eta; noise and truth remain evaluator-only | [Noisy measurement](professor_noisy_risk_signal.md) |
| 5 | Group-aware OLS on 80,000 training IDs, 20,000 held out | [Bank regression](professor_bank_regression.md) |
| 6A | 30-period income-linked contracts and survival-weighted expected profit | [Loan economics](professor_loan_economics.md) |
| 6B | Profit-maximizing common bounded-PD cutoff under fixed capital | [Bank cutoff](professor_lending_threshold.md) |
| 7A | Evaluator true-risk cutoff; bank policies remain frozen | [Oracle comparison](professor_oracle_comparison.md) |
| 7B | Matched group-aware versus group-blind OLS/cutoff comparison | [Group-blind comparison](professor_group_blind_comparison.md) |
| 8A | Five-draw noise-only pilot; adopted after provenance validation | [Pilot record](professor_noise_robustness_pilot.md) |
| 8B | 200 independent noise draws: five pilot reuses and 195 new replications | [Full study](professor_noise_robustness_full.md) |

### Design and interpretation

Applicants, hidden true risk, the original 80,000/20,000 split, contracts,
capital, and the true-risk oracle are fixed across the noise study. Only eta
changes. The two banks fit raw PN using Y, D, C, with or without B.

PD is a constant **conditional per-period default probability**, not a
whole-loan default rate. The working contract uses T=30, theta=0.10, and
r=0.05 **per period**, with growing payments. The budget is the saved
$1,826,087,220.7466326. Cutoff policies include indivisible exact ties and
reject-all; they are not unrestricted portfolio optimization.

The original single-noise-draw aware/blind/oracle comparison is preserved.
It must not be substituted for averages across the full noise study.

### Latest evidence

All 200 replications succeeded, yielding 400 policy-level result rows, with
zero failed attempts. Group-blind lending has a smaller absolute B=1 minus B=0
approval gap in 183/200 replications (91.5%). Mean aware/blind gaps are
+8.840/+0.366 percentage points.

Blind-minus-aware true expected profit has mean +$24.616M but median +$651,807.
Two differences above +$2.36B dominate the mean; both remain in the primary
results. Blind profit is higher/lower/equal in 70.5%/23.5%/6.0% of draws.
Reject-all occurs in 13 aware and 11 blind draws.

These are noise-conditional synthetic findings, not independent-population
uncertainty or proof that group-blind lending is fair or always profitable.
There is no inserted discrimination intervention or flexible-model comparison
in this professor-model family.

See the [data guide](professor_model_data_guide.md), [full study](professor_noise_robustness_full.md),
and [eight-slide progress report](../reports/professor_model/step8b_progress/README.md).

## Validation checkpoint

Fresh documentation-cleanup validation:

- Professor-model suite: **344 passed, no warnings**, in 11.96 seconds.
- Full suite: **624 passed, 4 failed, 32 deprecation warnings**, in 92.19 seconds.
- All four failures are in `tests/test_systemic_corrected_checkpoint.py`.
  The V2 loader constructs saved-record identities using current Git HEAD;
  the corrected artifacts were generated under an older revision.
- No scientific source, saved experiment, or test was modified to suppress
  these failures. This is not an all-green repository.
- The warnings concern deprecated SciPy L-BFGS-B options in existing V1
  scikit-learn tests.

Artifact-dependent tests require matching local data, which is not bundled in
a fresh Git clone. Stage-specific test counts in older documents and slides
remain historical records; the counts above describe this validation run.

## Preserved research families

| Family | Completed work | Remaining work |
|---|---|---|
| V1 approval disparities | Four-world generator; statistical recovery; ML; seven-page Streamlit dashboard; 3,140 sensitivity runs | Capstone synthesis; preserved tag `v1-approval-benchmark` |
| V2 economic lending | Conditional repayment accounting; calibrated population; traditional/HistGB estimates; nonlinear world; MILP portfolios; direct belief and systemic opportunity mechanisms | Full systemic repeated-seed study remains incomplete |
| Professor model | Steps 1–8B including the 200-draw noise-only study and progress report | Next experimental design requires professor review |

### V2 systemic checkpoint is not the professor noise study

The corrected V2 checkpoint contains **five validated replications and 360
portfolios**, indices 0, 1, 2, 3, and 13. The target remains 50 successes;
the remaining 45 corrected runs were not launched.

The legacy folder has 19 provisional successes plus one solver failure,
representing 1,368 saved portfolios. Legacy records are excluded from corrected
aggregation. See [recovery audit](systemic_monte_carlo_recovery.md).
The existing checkpoint lookup failures require a separate provenance-safe
repair; adding more replications does not resolve record identity.

## Next decisions

1. Review the noise scale, growing-payment contract, per-period interest
   interpretation, and clipped-zero tie behavior with the professor.
2. Predeclare a matched B-coefficient intervention before attributing observed
   gaps to a specific discrimination mechanism.
3. If requested, design separate noise-scale or independent-population
   robustness experiments; do not silently retune the frozen study.
4. Repair V2 checkpoint identity/resume tests separately before any authorized
   extension of its corrected 50-replication study.
5. Continue deferring HMDA until the synthetic workflow and interpretation
   are ready. Keep causal claims distinct from descriptive gaps.

## Documentation policy

[Documentation index](README.md) identifies the three research families.
[Project History](PROJECT_HISTORY.md) preserves the previous detailed chronology
and numerical stage records. No historical research document was deleted.
[Preserved run commands](running_v1_v2.md) and the root README provide navigation
without implying that expensive experiments should be rerun automatically.
