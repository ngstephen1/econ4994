# Project Status

Updated **2026-11-03**. Start with the
[current research guide](CURRENT_RESEARCH.md) for reading order, result files,
policy definitions and next decisions. This page records the completed
minimum-5% successor and its validation. Earlier checkpoints and test results
below retain their original scope.

The current publication package contains both smaller-penalty successor
implementations, frozen configurations, tests, reports, editable slide decks,
PDFs and compact hash-verified CSV/figure snapshots. Full generated applicant
data, model binaries and build intermediates remain local and ignored. Use the
current research guide for the latest links; older checksum-bound documentation
remains preserved. Repository publication does not rerun scientific experiments.

## Current research: professor's lending model

**Minimum 5% Group B approval update completed, November 3:** the fifth common
candidate, **+0.0003125B**, gives **383/3,030 approvals (12.64%)** in Scenario 1
and **154/3,030 (5.08%)** in Scenario 2, below Group A's **20.33%/22.46%**.
Four earlier candidates fail the common criterion; the sixth was not tried.
True expected profit falls versus control by **$23,312.55/$354,515.28**, and
increases versus the preserved +0.005B policy by **$3.710M/$2.629M**.

The original models, split, four control/ML policies, contracts and capital are
unchanged. No new fits, noise or populations were generated. The inclusive 5%
criterion uses bank approval counts, with all final bank policies frozen before truth
evaluation. The prediction-cohort calibration remains illustrative. The new
method comparison separates approval parity from fairness and shows ML's
**28.1%/13.3%** profit reductions alongside unused capital.

Tests: **73 new**, **269 focused/dependency**, **720 professor-model**, all
passing without warnings. Independent rational enumeration reproduces both
selected cutoffs. All 2,012 pre-existing files are preserved except the two
permitted current documentation edits. See the
[minimum-5% report](professor_five_percent_penalty.md) and
[new 14-slide delivery](../reports/professor_model/five_percent_penalty/README.md).
The unrelated full suite and its four known V2 failures were not rerun or changed.

### Preserved +0.005B checkpoint

**Final smaller-penalty update completed, November 3:** +0.005B was the first
common candidate and satisfies positive minority approval below Group A in
both scenarios. Group B receives **3/3,030 loans (0.10%)** in Scenario 1 and
**1/3,030 (0.03%)** in Scenario 2. No further coefficient was tried.
True expected portfolio profit falls versus control by **$3.733M/$2.983M**.
All four control/Gradient Boosting policies remain exactly frozen. Training
uses **80,000 applicants**, and loans/profits use only the **20,000 prediction
applicants**. No new fitting, ML selection, noise or population was generated.

This is an explicitly approval-calibrated illustrative choice on the prediction
cohort, not a claim of untouched confirmatory policy evaluation. The updated
14-slide deck defines Group A and minority Group B and reports tiny positive
rates with exact counts. New tests: **63 passed**; focused/dependency tests:
**297 passed**; professor-model suite: **647 passed**; all without warnings.
See [the final study](professor_small_penalty.md) and
[new delivery](../reports/professor_model/small_penalty/README.md). No automatic
commit, push, email or additional scientific extension was performed.

### Preserved November 2 checkpoint

**November 2 follow-up completed:** the corrected **+0.01B** treatment and
two-scenario OLS/gradient-boosting comparison now have one matched validated
run, six policies and twelve group rows. Scenario 2 shifts B=1 epsilon by
+1 SD, updates the credit signal, and includes B in OLS/ML; all earlier
scientific work is preserved. See the [new study and results](professor_nov2_followup.md).

The penalty funds zero B=1 applicants in both scenarios and reduces true
portfolio profit versus each control by **$3.821M/$3.179M**. Gradient boosting
funds more applicants but has lower true profit and **72.4%/87.2%** capital
utilization because its next indivisible score block exceeds the fixed budget.
B=1 mean true PD increases modestly, from **0.142285% to 0.144759%**, under
the unchanged near-saturated true-risk function. No outcome sign or ML win was
imposed. The fresh professor-model suite passes **584 tests with no warnings**.
The scientific run added no 200-draw extension, independent population or HMDA
analysis. Git publication is a separate repository operation.

The [November delivery folder](../reports/professor_model/nov2_followup/README.md)
contains the editable 14-slide deck, PDF handout, and a compact Git-tracked
snapshot of six result tables and four plots (PNG and vector PDF). The full
generated datasets, fitted models, cutoff candidates and build files remain
local under existing ignore rules. Published copies are checked against the
completed run's artifact hashes; they do not replace or modify the originals.

### Preserved October checkpoint

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
| November 2 follow-up | Corrected +0.01B; two matched risk worlds; same-feature OLS/HistGB within each world; four six-bar figures and updated 14-slide presentation | [Protocol, results and validation](professor_nov2_followup.md) |
| Final smaller-penalty update | Common +0.005B gives positive but lower minority approval in both worlds; frozen models; prediction-only profit audit; four plots and new 14-slide presentation | [Final protocol, results and validation](professor_small_penalty.md) |
| Minimum 5% update | Fifth common candidate +0.0003125B meets the new 5% minimum in both worlds; frozen controls/ML; exact search audit, method comparison and new 14-slide delivery | [Latest protocol, results and validation](professor_five_percent_penalty.md) |

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

Latest minimum-5% professor-model check, November 3:

- `python3 -m pytest -q tests/test_professor_*.py`: **720 passed, no warnings**,
  in 25.51 seconds; includes 73 new tests.
- New/reused dependency check: **269 passed, no warnings**, in 6.30 seconds.
- Independent saved-file audit reproduces both selected cutoffs with rational
  arithmetic and confirms frozen control/ML policies and prediction-only profit.
- The unrelated full suite was not rerun; its four historical V2 lookup failures
  remain untouched.

Pre-publication verification for the combined successor package repeated the
professor-model suite: **720 passed, no warnings**, in 31.38 seconds. All **52
published evidence copies** and **four PPTX/PDF delivery files** match their
recorded hashes; current documentation links resolve. No scientific experiment
was rerun for publication.

Preserved earlier November 3 +0.005B check:

- `python3 -m pytest -q tests/test_professor_*.py`: **647 passed, no warnings**,
  in 21.57 seconds, including 63 new smaller-penalty tests.
- Focused new/dependency tests: **297 passed, no warnings**, in 7.14 seconds.
  New test file alone: **63 passed**, no warnings, in 3.59 seconds.
- One candidate tested in both scenarios; exact positive-but-lower group
  approval criterion passed in both without evaluating smaller candidates.
- Independent audits reproduce four frozen control/ML score/decision frames
  and both prediction-only true-profit files exactly. No new fits occurred.
- Parent artifacts and all existing scientific files are unchanged. The
  1,933-file preservation inventory allows only the current guide/status edits.
- Summary-only replay starts no computations and changes no saved bytes/mtimes.

Historical November 2 professor-model check:

- `python3 -m pytest -q tests/test_professor_*.py`: **584 passed, no warnings**,
  in 25.91 seconds. Fifty new tests cover the November study.
- Focused new/dependency checks: **234 passed, no warnings**, in 7.88 seconds.
  New study alone: **50 passed, no warnings**, in 3.55 seconds.
- Six policy cutoffs and all 76,743 candidates independently verified with
  rational arithmetic; original Scenario 1 control decisions replay exactly.
- Reloaded ML models reproduce predictions exactly; independent OLS replay
  error is below 3.47e-18. Summary-only changes no saved bytes or mtimes.
- Original 1,846-file inventory is preserved except the explicitly allowed
  current guide/status edits; private truth/noise and bank artifacts are sealed
  before evaluator release. Source and supplied PDF remain unchanged.

Historical pre-merge professor-model check:

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
| Professor model | Steps 1–10B, six raw-ordering diagnostics, November OLS/penalty/ML comparison, preserved +0.005B and new minimum-5% successor with prediction-only profits | Review minimum-5% slides with Sergio and settle final scope; further robustness work remains unrun |

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

1. Review the minimum-5% figures and explicit training/prediction separation
   with Sergio. The new common coefficient gives 383 and 154 Group B approvals;
   report method-specific parity and profit comparisons accurately. Settle
   whether the package completes the final slides, then proceed with capstone
   synthesis.
2. Review the earlier six-case raw-ordering diagnostic separately.
   Decide whether raw scores may rank requests while bounded PD determines
   cash flows, or whether this remains a policy-class sensitivity only. Do not
   infer Monte Carlo frequencies from selected diagnostic cases or launch the
   200-draw extension automatically. Keep coefficient removal, blind refitting
   and the separate manual −0.1B treatment distinct. More lending or a smaller
   group gap does not by itself establish fairness.
3. If requested, predeclare additional noise-scale or independent-population
   checks. The November comparison uses one matched realization, not 200 new
   replications. Begin final capstone synthesis once the scope is settled.
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
