# Project Status

Updated **2026-10-28**. Start with the
[current research guide](CURRENT_RESEARCH.md) for reading order, result files,
policy definitions and next decisions. This page reflects completed work;
the documentation refresh did not run experiments or tests. A subsequent
pre-merge check ran the professor-model tests, as recorded below.

## Current research: professor's lending model

**Complete through Step 10B, the post-10B review, and the six-case raw-ordering
sensitivity.** The original 200-draw noise study and the 200-draw
fixed-coefficient comparison are complete. The raw-ordering follow-up covers
only the canonical case and five selected draws; its full 200-draw extension
is **not run**. Professor review of the policy and accounting interpretation
is the next research decision. This family is separate from the preserved
Version 1 approval framework and Version 2 repayment/portfolio experiments.

**Step 10A is complete:** documentation-only evidence synthesis and scientific
interpretation audit, with a [traceable results matrix](professor_model_evidence_synthesis.md)
and [prioritized roadmap/go-no-go conditions](professor_model_next_steps.md).
At that stage, the proposed next study was to isolate the fitted B contribution
on the canonical saved aware model and compare it with the blind refit; Step
10B had not been run. No scientific baselines, experiments, figures, or
historical test results were changed or rerun for the Step 10A synthesis.

**Step 10B is complete, superseding the Step 10A proposal above:** the
[fixed-coefficient comparison](professor_fixed_coefficient_comparison.md)
validated the canonical case and all 200 saved noise draws (603 policy cells,
zero failures), without refitting, regenerating noise, or reoptimizing W/F.
Canonical Z−W/Z−F true-profit differences are +$853,048.84/+$327,762.07;
across the 200 draws Z beats W in 138 cases and F in 69, with rare tie/capital
discontinuities dominating means. There is no universal profit or fairness
improvement. The focused/dependency tests passed 38/190 tests without warnings;
the full suite was not rerun and the four known V2 failures remain disclosed.

**Post-10B results review is complete:** the
[saved-evidence review and follow-up design](professor_post10b_review.md)
confirm that all 603 policies treat every request as profitable and choose
the largest capital-feasible prefix. Three reject-all disagreements account
for $11.88M of the $12.82M mean Z−W profit difference. All 603 clipped-zero
blocks replay exactly; the frozen baseline and Step 10B hashes remain valid.
At that review, the relevant suite passed **206 tests with no warnings** and
raw ordering was a design proposal. The subsequent implementation is recorded
below. The review itself performed no new fit, noise draw or lending search.

**Raw-ordering follow-up is complete for the six declared diagnostic cases:**
the [raw-ordering sensitivity](professor_raw_ordering_sensitivity.md) compares
the canonical case and saved draws 008/064/089/092/154 with their frozen
bounded policies. Accounting probabilities and per-request profits are
unchanged. All nine reject-all policies become funded portfolios; the other
nine remain exactly unchanged. This supports the clipping/tie/capital
explanation without establishing a fairness or prediction improvement.
The final targeted suite recorded **249 tests with no warnings**, including
43 new tests; the full repository suite was not rerun. The final scientific
audit verified 1,654 protected pre-existing files unchanged. The 200-draw
extension was not run. All earlier baselines and reports remain preserved.

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
| 9A | Frozen group-aware no-direct branch, blind-minus-0.1B direct branch, and unadjusted blind control; risk/tie diagnostics only | [Revised scenarios](professor_revised_scenarios.md) |
| 9B | Three exact common-cutoff policies, frozen before evaluator-only comparison with the original aware/blind/oracle references | [Revised lending comparison](professor_revised_lending_comparison.md) |
| 9C | Five predeclared score variants; three valid cutoff policies; positive-floor sensitivity, clipping/tie/capital audits and preserved references | [Mechanism audit](professor_mechanism_audit.md) |
| 10A | Documentation-only evidence synthesis and historical next-study design | [Evidence synthesis](professor_model_evidence_synthesis.md) |
| 10B | Canonical plus 200 saved-draw coefficient-removal comparison, with frozen aware/blind references | [Fixed-coefficient comparison](professor_fixed_coefficient_comparison.md) |
| Post-10B review | Saved-output explanation of profitable prefixes, reject-all tails and blind-refit level shifts; follow-up design only | [Review and next study](professor_post10b_review.md) |
| Raw-ordering sensitivity | Six selected diagnostic cases, unchanged probability accounting, exact raw prefixes and frozen bounded references | [Results and reproduction](professor_raw_ordering_sensitivity.md) |

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

### Preserved Step 8B noise evidence

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
Steps 1–8B contain no inserted discrimination intervention. Step 9A separately
adds the professor's exact **−0.1B** perceived-default-risk adjustment, using
the canonical frozen fits, not a Monte Carlo draw. It favors B=1 in risk
assessment. There is still no flexible-model comparison in this family.

### Step 9A risk-only checkpoint

Three scenarios each reuse the same 20,000 evaluation applicants, yielding
60,000 risk rows. The no-direct branch exactly reproduces the frozen
group-aware fit, including its fitted B term. The direct branch uses the
frozen group-blind fit minus 0.1B; a third unadjusted blind control isolates
that manual treatment. “No direct” means no manual adjustment, not exclusion
of B. PN retains its original meaning and is not used to build the scores.

All 3,030 B=1 applicants receive exactly −10 percentage points of raw
adjustment and become bounded zero. Total zeros are 923 in no-direct,
3,764 in direct, and 883 in the blind control. The direct zero-score group's
requested principal is $1,276,685,167.37, 69.9137% of the unchanged budget;
it is feasible as a whole. This is **only a capital diagnostic**, not a
selected portfolio, approval finding, or profit result.

All 905 pre-step scientific files remain unchanged; 133 frozen source/reference
files, exact coefficient replay, original IDs, and CSV/Parquet values passed
checks. No data regeneration, model training, lending optimization, or
profit experiment was performed. See [Step 9A](professor_revised_scenarios.md).

### Step 9B lending checkpoint

The no-direct and blind-control policies exactly reproduce the original aware
and blind decisions (3,900 and 3,802 approvals). The direct policy approves
4,990 applicants: all 3,030 B=1 applicants and 1,960 B=0 applicants. Its
B=1 minus B=0 gap is **+88.450206 pp**, versus −14.385705 pp for aware and
+0.155468 pp for blind control. All three zero-score blocks are funded whole;
the direct block contains 3,764 requests and uses 69.9137% of capital.

Direct true expected profit is $2,359,267,451.62, **$7,332,311.99 below blind
control** and $6,807,025.23 below the frozen aware bank, despite higher
bank-perceived profit. Relative to blind control, 2,450 B=1 applicants gain
funding and 1,262 B=0 applicants lose it. These are matched synthetic results,
not fairness findings or empirical discrimination estimates. Oracle comparisons
remain signed restricted-cutoff comparisons, not unrestricted optimal bounds.

All 54,436 candidates passed independent Fraction enumeration. The 60,000
new decision rows were saved/frozen before numeric truth or frozen-reference
decisions were loaded. Independent audits reproduced all six portfolios,
12 group rows, and 15 allocation-overlap pairs. All 963 pre-step scientific
files, including Step 9A work, remain byte-identical. Existing fits, scores,
contracts, capital, oracle, and prior experiments were not changed. See
[Step 9B report](professor_revised_lending_comparison.md).

### Step 9C controlled mechanism audit

A exactly reproduces the frozen direct policy; E exactly reproduces the blind
control. B preserves negative direct scores for diagnosis only. C applies only
the zero floor and exactly equals A because no direct raw score exceeds one;
it remains diagnostic-only. Neither B nor C is used for lending.

D uses the predeclared first percentile (`method="linear"`) of the 19,117
strictly positive frozen blind-control evaluation scores:
**0.00013923734234194752**, or **0.013923734234194752%**. Only the 3,764 exact
zeros change (734 B=0, 3,030 B=1); 161 smaller positive scores remain unchanged.
The tie moves to the positive floor rather than splitting. Its principal is
69.913701% of the budget; including all lower scores requires 74.769253%,
which remains feasible.

A and D select the same cutoff and exactly the same 4,990 approvals. Their
gap remains +88.450206 pp, true expected profit remains $2,359,267,451.62,
and unused capital remains $61,706.89. D reduces perceived profit by
$7,854,813.35. Both policies remain $7,332,311.99 below blind control in
true expected profit. This one floor does not remove the allocation change,
but it preserves the shared tie and cannot identify an additive decomposition
of adjustment, clipping, ties, and capital. Synthetic findings are not empirical
discrimination or fairness estimates.

All 51,595 candidates passed independent rational enumeration. New decisions
were frozen before truth access. All 986 pre-step scientific artifacts are
byte-identical, 21 output hashes and 18 source hashes match, and all three
new figures were visually inspected. No refitting, new draws, new capital,
oracle reoptimization, Monte Carlo, ML, HMDA, or dashboard changes were added.
See the [Step 9C report and full artifact inventory](professor_mechanism_audit.md).

See the [data guide](professor_model_data_guide.md), [full study](professor_noise_robustness_full.md),
and [eight-slide progress report](../reports/professor_model/step8b_progress/README.md).

## Validation status

Latest recorded pre-merge professor-model check:

- `python3 -m pytest -q tests/test_professor_*.py`: **534 passed, no warnings**,
  in 20.82 seconds. This covers the professor-model test family, not the
  entire repository. No scientific experiment was rerun.

Recorded targeted suite and artifact audit from the raw-ordering study:

- **249 passed, no warnings**, in 8.28 seconds, including 43 new raw-ordering
  tests. This is a targeted dependency suite, not the full professor-model
  or repository suite.
- Six selected diagnostic cases and 18 new raw policies validated, with zero
  failed attempts. All 18 frozen reference cells were reproduced.
- Resume and summary-only checks started no computation or writes.
- [Final validation record](../results/metrics/professor_model/raw_ordering/d0421da2ea691f2e/final_validation.json)
  binds the results, source snapshots, checks and test command.

Latest recorded repository-wide run remains the **historical Step 9C** run:

- Full suite: **717 passed, 4 failed, 32 deprecation warnings**, in 105.31 seconds.
- Its professor-model suite recorded **437 passed, no warnings**, in 17.77
  seconds. Later targeted counts must not be substituted for that broader scope.
- All four failures are in `tests/test_systemic_corrected_checkpoint.py`.
  The V2 loader constructs saved-record identities using current Git HEAD;
  the corrected artifacts were generated under an older revision.
- No existing scientific source, saved experiment, or test was modified to suppress
  these failures. This is not an all-green repository.
- The warnings concern deprecated SciPy L-BFGS-B options in existing V1
  scikit-learn tests.

Artifact-dependent tests require matching local data, which is not bundled in
a fresh Git clone. Stage-specific test counts in older documents and slides
remain historical records. The documentation refresh itself did not run
tests; the later pre-merge check is reported separately above. The repository
is not represented as fully green.

## Preserved research families

| Family | Completed work | Remaining work |
|---|---|---|
| V1 approval disparities | Four-world generator; statistical recovery; ML; seven-page Streamlit dashboard; 3,140 sensitivity runs | Capstone synthesis; preserved tag `v1-approval-benchmark` |
| V2 economic lending | Conditional repayment accounting; calibrated population; traditional/HistGB estimates; nonlinear world; MILP portfolios; direct belief and systemic opportunity mechanisms | Full systemic repeated-seed study remains incomplete |
| Professor model | Steps 1–10B, evidence review and six selected raw-ordering diagnostics | Professor review of ranking versus accounting PD, stylized economics and noise; any 200-draw raw-ordering extension remains unrun |

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
2. Review the completed six-case raw-ordering diagnostic with the professor.
   Decide whether raw scores may rank requests while bounded PD determines
   cash flows, or whether this remains a policy-class sensitivity only. Do not
   infer Monte Carlo frequencies from selected diagnostic cases or launch the
   200-draw extension automatically. Keep coefficient removal, blind refitting
   and the separate manual −0.1B treatment distinct. More lending or a smaller
   group gap does not by itself establish fairness.
3. If requested, design separate noise-scale or independent-population
   robustness experiments; do not silently retune the frozen study.
4. Repair V2 checkpoint identity/resume tests separately before any authorized
   extension of its corrected 50-replication study.
5. Continue deferring HMDA until the synthetic workflow and interpretation
   are ready. Keep causal claims distinct from descriptive gaps.

## Documentation policy

[Current research guide](CURRENT_RESEARCH.md) provides up-to-date navigation
through the professor-model results and explicitly supersedes earlier planning
language. The older [documentation index](README.md) identifies the three
research families but reflects its Step 8B snapshot.
[Project History](PROJECT_HISTORY.md) preserves the previous detailed chronology
and numerical stage records. No historical research document was deleted.
[Preserved run commands](running_v1_v2.md) retain V1/V2 entry points. Earlier
READMEs, roadmaps, reports and slides remain in place when their bytes are
registered as experiment evidence; current navigation is supplied here and
in the current guide. Documentation updates do not authorize rerunning studies.
