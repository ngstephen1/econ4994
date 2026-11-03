# Current Research Guide

Updated **2026-11-03**. Use this page and
[Project Status](PROJECT_STATUS.md) for the present repository state.
Earlier reports preserve the conclusions and plans recorded at each stage;
some are checksum-bound inputs to later experiments and remain unchanged.
This guide supersedes their planning language without rewriting the evidence.

## Read these first

| Question | Current document |
|---|---|
| What meets Sergio's revised minimum 5% minority approval request? | [Minimum 5% approval study](professor_five_percent_penalty.md) |
| Where are the latest slides, plots and method comparisons? | [Minimum 5% delivery](../reports/professor_model/five_percent_penalty/README.md) |
| What changed after the professor requested a smaller penalty? | [Final smaller-penalty study](professor_small_penalty.md) |
| Where are the final slides and compact result snapshot? | [Smaller-penalty delivery](../reports/professor_model/small_penalty/README.md) |
| What are the professor's corrected two-scenario OLS/penalty/ML results? | [November 2 follow-up](professor_nov2_followup.md) |
| Where are the preserved November 2 slides and result tables/plots? | [November delivery and evidence snapshot](../reports/professor_model/nov2_followup/README.md) |
| What is complete, what remains, and which tests were run? | [Project Status](PROJECT_STATUS.md) |
| What happened when only the fitted B contribution was removed? | [Step 10B fixed-coefficient comparison](professor_fixed_coefficient_comparison.md) |
| Why do profit averages and clipped ties behave this way? | [Post-10B evidence review](professor_post10b_review.md) |
| What happened when raw scores ordered lending while bounded PD determined profits? | [Completed raw-ordering sensitivity](professor_raw_ordering_sensitivity.md) |
| What do the underlying risk and accounting equations mean? | [Whiteboard specification](professor_whiteboard_model.md), [loan economics](professor_loan_economics.md) |

The **current deliverable** is the minimum-5% successor. The fifth common
candidate, **+0.0003125B**, funds Group B **383/3,030 (12.64%)** in Scenario 1
and **154/3,030 (5.08%)** in Scenario 2, below Group A in both. The first four
candidates fail the common criterion; no finer search or sixth candidate was
run. True expected profit is **$23,312.55/$354,515.28 lower than control**.
The four control/ML policies and original training/prediction split are frozen.
Selection remains an illustrative approval-based calibration on prediction
applicants. See the new report for exact thresholds, search records and caveats.

The **preserved +0.005B deliverable** is the prior smaller-penalty successor. The first
candidate, **+0.005B**, gives minority Group B **3/3,030 approvals (0.10%)** in
Scenario 1 and **1/3,030 (0.03%)** in Scenario 2, below Group A's rate in both.
No smaller candidate was tried. The models remain frozen, with **80,000
training / 20,000 prediction applicants** and **profits only on prediction
loans**. This is a disclosed approval-based demonstration calibration, not an
untouched confirmatory evaluation. See the report for the full selection rule.

The **preserved November 2 deliverable** has two matched worlds,
three methods per world, four six-bar plots, and an updated 14-slide deck.
Scenario 1 uses Y/D/C. Scenario 2 adds a +1 SD B=1 epsilon shift, updates C
consistently, and uses Y/D/C/B. That study uses **+0.01B**, applied
before clipping. These are new studies; the earlier minus 0.1B results below
remain historical evidence rather than being overwritten.

The [Step 10A roadmap](professor_model_next_steps.md) and
[evidence synthesis](professor_model_evidence_synthesis.md) describe a proposal
that has since been executed as Step 10B. The post-10B review's raw-ordering
proposal has also since been implemented for six selected cases. Read those
sections as historical design records, not pending implementation requests.
The older root README, documentation index, data guide and eight-slide deck
primarily describe the earlier Step 8B checkpoint.
The checksum-bound historical files remain unchanged so the completed studies'
read-only verification commands continue to work. This guide is the current
entry point for the latest research and publication package.

## Completed studies and their different denominators

| Study | Completed scope | What the repetition means |
|---|---|---|
| Original professor-model noise study, Step 8B | 200 saved noise draws, aware/blind policies | Different eta on one fixed population and split; five pilot draws are included, not additional |
| Fixed-coefficient comparison, Step 10B | One canonical case plus the same 200 saved draws, W/Z/F | 201 evaluation cases, not independent borrower populations; no new fits or noise |
| Post-10B review | Saved-output audit and follow-up design | Explanation of existing evidence, not another experiment |
| Raw-ordering sensitivity | Canonical plus draws 008, 064, 089, 092, 154; 18 new raw policies and 18 frozen references | Deliberately selected diagnostics, not Monte Carlo frequency estimates |
| November 2 follow-up | Two matched worlds × OLS control / OLS +0.01B / HistGB regressor | One original noise realization; Scenario 2 changes epsilon/C/truth/PN and the feature set, not an independent population replication |
| Final smaller-penalty successor | Same frozen worlds/models; first common +0.005B candidate succeeds in both; only two penalty policies recalculated | No fitting, noise draws or population generation; parameter chosen on prediction-group approval counts for illustration |
| Minimum 5% successor | Fifth common candidate +0.0003125B gives 12.64%/5.08% Group B approval, strictly below A | Five paired candidates; original models and cohorts frozen; no independent population or new noise |

All use the same 100,000 applicant identities, original 80,000/20,000 split,
evaluation contracts and capital. The first four studies keep the original
population; the November systemic world changes its declared epsilon/C/risk
pathway. The raw-ordering study did **not** extend
to the other 195 saved noise draws. Its two source-version executions replayed
the same six cases and produced identical scientific tables; they are not
12 independent cases. Details are in its execution-history section.

## Policy dictionary

| Label | Definition |
|---|---|
| W: `frozen_aware` | Saved OLS with Y, D, C and B, including its fitted B contribution |
| Z: `aware_B_zero` | `raw_W - saved_beta_B*B`, preserving the aware intercept and other slopes; no refit |
| F: `frozen_blind` | Saved refitted OLS with Y, D and C; intercept and other slopes can differ from W |
| `bounded_frozen` | Original common cutoff on `clip(raw,0,1)`, with exact bounded ties indivisible |
| `raw_order` | New common cutoff on raw score, preserving exact raw ties; cash flows still use the unchanged bounded PD |

The separate Step 9A/9B **−0.1B** manual adjustment and Step 9C positive-floor
sensitivity are different interventions. Neither is Z or the raw-ordering
study. Negative raw scores and cutoffs are ordering values, never negative
default probabilities. Clipping bounds a score but does not calibrate it.

## Findings to communicate

- Latest 5% successor: Group B receives 383/3,030 and 154/3,030 loans;
  Group A receives 3,450/16,970 and 3,811/16,970. The common penalty is
  +0.0003125B (+0.03125 pp on the raw default score).
- Latest 5% successor: true expected portfolio profit falls by only $23,312.55
  and $354,515.28 versus control, and rises by $3.710M/$2.629M versus the
  previous +0.005B policy. Profit did not determine selection.
- ML comparison: Scenario 1 OLS is closer to approval parity (absolute gaps
  0.155468 versus 0.725295 pp); Scenario 2 ML is closer (0.072794 versus
  14.564860 pp). ML profit is 28.1%/13.3% below control with 72.4%/87.2%
  capital use. These are descriptive comparisons, not a general equity ranking
  or quantified decomposition of the profit difference.
- The +0.005B findings immediately below remain the preserved prior study.
- Final successor: +0.005B preserves positive but very low minority approval:
  3 loans (0.099010%) in Scenario 1 and 1 loan (0.033003%) in Scenario 2.
  Group A approval rates are 25.050088% and 25.132587%, respectively.
- Final successor: true expected portfolio profit decreases versus control by
  **$3.733M/$2.983M**. Selection uses approval counts only, not true profit.
- Final successor: all four control/Gradient Boosting policies remain exactly
  the parent policies. Training is 80,000; all loans and profits concern only
  the 20,000 prediction applicants. ML's tie/capital limitations persist.
- The November 2 +0.01B findings below remain the earlier, preserved study.
- November 2: the corrected +0.01B penalty funds no B=1 applicants and lowers
  true expected portfolio profit by $3.821M in Scenario 1 and $3.179M in
  Scenario 2 relative to each matched control. Both outcomes are observed,
  not forced by acceptance tests.
- November 2: B=1 population mean true PD rises from 0.142285% to 0.144759%
  under the +1 SD epsilon shift; B=0 remains exactly unchanged. Scenario 2
  includes B as a fitted covariate, so its control is not group-blind.
- November 2: gradient boosting approves more applicants but has lower true
  profit; indivisible prediction ties leave 27.6%/12.8% of capital unused.
  A smaller approval gap or more approvals is not a complete fairness result.
- The following fixed-coefficient/raw-ordering findings belong to the earlier
  studies and are not recomputed as November results.
- Canonical coefficient removal changes the approval gap from −14.385705 pp
  to +0.155468 pp and increases true expected profit by $853,048.84 relative
  to W. Z and F have identical group counts but differ by one exchanged loan.
- Across 200 saved draws, Z has higher true profit than W in 138 cases and F
  in 69. Neither comparison establishes universal improvement. Rare
  reject-all disagreements dominate the mean dollar differences.
- All 603 Step 10B case/policy combinations perceive every request as
  profitable. The selected cutoff is therefore the largest capital-feasible
  prefix under the declared policy, not unrestricted portfolio optimization.
- Raw ordering changes all nine selected reject-all policies into funded
  portfolios. The other nine portfolios, including all canonical ones, are
  exactly unchanged. Probabilities and per-request profits do not change.
- Avoiding universal rejection is not a fairness guarantee. In draw 154,
  aware raw ordering still has a +88.254442 pp group approval gap.

These are synthetic mechanism findings. They do not establish real-world
mortgage discrimination, calibrated default probabilities, general model
rankings or independent-population uncertainty. Dollar profits use the
stylized undiscounted 30-period growing schedule with 5% interest per period;
they are not annual realized profits or standard mortgage returns.

## Latest CSVs to open

All paths below are relative to `/Users/macbook/Hack/econ4994/`. The latest
penalty-study links use compact, published CSV snapshots that are available
in a fresh clone. Full generated CSV, Parquet and JSON datasets remain local
and ignored by Git. Join applicant-level files using `applicant_id`.

| File | One row / useful columns |
|---|---|
| [Latest 5% group results](../reports/professor_model/five_percent_penalty/evidence/tables/group_results.csv) | Twelve group cells with exact approvals, denominators, profit contributions and normalized values |
| [Latest 5% portfolios](../reports/professor_model/five_percent_penalty/evidence/tables/portfolio_results.csv) | Six policies with selected delta, true/perceived profit, capital use and approval gaps |
| [Latest selection log](../reports/professor_model/five_percent_penalty/evidence/tables/penalty_search.csv) | Ten candidate × scenario attempts, exact 5% checks and common stopping decision |
| [Latest method comparisons](../reports/professor_model/five_percent_penalty/evidence/tables/method_comparisons.csv) | Six policies' absolute approval gaps, true-profit differences versus control and capital use |
| [Versus previous +0.005B](../reports/professor_model/five_percent_penalty/evidence/tables/previous_penalty_comparisons.csv) | Six scenario × scope rows with signed profit and allocation changes |
| [Final group results](../reports/professor_model/small_penalty/evidence/tables/group_results.csv) | Twelve scenario × method × group rows; exact counts, positive minority rates and funded-loan true profits |
| [Final portfolio results](../reports/professor_model/small_penalty/evidence/tables/portfolio_results.csv) | Six policies, selected delta, true/perceived profit and capital use |
| [Penalty selection log](../reports/professor_model/small_penalty/evidence/tables/penalty_search.csv) | Two rows: +0.005B checked once in each scenario and selected in both |
| [Training/prediction audit](../reports/professor_model/small_penalty/evidence/tables/split_audit.csv) | Two worlds' split fingerprints, 80,000/20,000 counts and prediction-only profit confirmation |
| [November group results](../results/tables/professor_model/nov2_followup/group_results.csv) | Twelve scenario × method × B rows; approval rates, funded-loan true profit totals, profit per eligible applicant |
| [November portfolio results](../results/tables/professor_model/nov2_followup/portfolio_results.csv) | Six scenario × method rows; approval gap, total true/perceived profit, capital use and ties |
| [November penalty displacement](../results/tables/professor_model/nov2_followup/penalty_comparisons.csv) | Six scenario × scope rows; lost/replacement approvals and their signed true-profit difference |
| [November systemic risk audit](../results/tables/professor_model/nov2_followup/systemic_risk_audit.csv) | Four scenario × B rows; mean epsilon, C, sigmoid factor and true PD |
| [Raw-ordering policies](../results/tables/professor_model/raw_ordering/d0421da2ea691f2e/policies.csv) | 36 case × source × ordering rows; `number_approved`, `approval_gap_pp`, `true_profit_USD`, `perceived_profit_USD`, capital and cutoff units |
| [Raw-ordering comparisons](../results/tables/professor_model/raw_ordering/d0421da2ea691f2e/comparisons.csv) | 54 signed comparisons; filter `comparison_kind` to distinguish within-model ordering changes from between-model comparisons |
| [Raw-ordering groups](../results/tables/professor_model/raw_ordering/d0421da2ea691f2e/groups.csv) | 72 case × policy × B rows; group counts, rates, principal and true profit |
| [Canonical fixed-coefficient policies](../results/tables/professor_model/fixed_coefficient/cd7f3ee268cfc4d0/canonical_policies.csv) | Three W/Z/F policy rows for the original noise realization |
| [Noise fixed-coefficient summaries](../results/tables/professor_model/fixed_coefficient/cd7f3ee268cfc4d0/noise_summary.csv) | 24 metric/comparison rows, n=200 each; mean, median, SD, MCSE, empirical percentiles, extrema and sign counts |
| [Noise fixed-coefficient paired results](../results/tables/professor_model/fixed_coefficient/cd7f3ee268cfc4d0/noise_paired.csv) | 600 draw × contrast rows; Z−W, Z−F and F−W remain distinct |

The [raw-ordering report](professor_raw_ordering_sensitivity.md) lists its
exact canonical CSV/Parquet paths and metadata. Canonical bank input and
decision files each contain 60,000 applicant-by-source rows for the same
20,000 people, not 60,000 distinct applicants. Noise-case applicant rows
remain in memory only. The earlier
[data guide](professor_model_data_guide.md) remains useful for the original
population, hidden truth, contracts and Step 8B tables; its Drive bundle is
an earlier snapshot, not a bundle of all later results.

[Open the raw-ordering comparison figure](../results/figures/professor_model/raw_ordering/d0421da2ea691f2e/diagnostic_portfolio_changes.png).

## Validation and reproducibility

The latest minimum-5% run passed **73 new tests**, **269 focused/dependency
tests**, and **720 professor-model tests**, all with no warnings. The full
professor-model check took 25.51 seconds. Independent exact arithmetic
reproduced both selected cutoffs and verified frozen policies and prediction-
only profits. The unrelated full suite and its four V2 lookup failures were
not changed or rerun. The prior 647-test result below remains historical.

The fresh November 3 check ran `python3 -m pytest -q tests/test_professor_*.py`:
**647 passed, no warnings**, in 21.57 seconds. This includes 63 new smaller-
penalty tests. The focused/dependency suite passed 297 tests with no warnings.
Independent saved-file checks confirm unchanged control/ML scores and masks,
prediction-only true-profit files, and read-only summary replay.

The historical November 2 check ran `python3 -m pytest -q tests/test_professor_*.py`:
**584 passed, no warnings**, in 25.91 seconds. This includes 50 new November
tests; the focused/dependency check passed 234 tests. This is the
professor-model test family, not the full repository suite. The previous
pre-merge result of 534 passed remains historical. The raw-ordering study separately
recorded **249 targeted tests with no warnings**, including 43 new tests.
The last recorded full suite remains the Step 9C result:
**717 passed, 4 failed, 32 existing warnings**. The four V2 checkpoint lookup
failures are unresolved. The documentation refresh ran no tests or experiments;
the later pre-merge test check is historical. No scientific experiment was
rerun for that merge. The November run is a separate approved study with its
own source/input/dependency fingerprints and bank/evaluator freeze.

See the [raw-ordering validation record](../results/metrics/professor_model/raw_ordering/d0421da2ea691f2e/final_validation.json)
and [evidence index](../results/metrics/professor_model/professor_model_evidence_index.json).
They distinguish canonical results, conditional noise repetitions, selected
diagnostics, source versions and historical evidence. The raw-ordering audit
verified 1,654 protected files unchanged and archived exact code/configuration
contents; these counts describe that recorded validation, not new test runs.

For an existing local installation, this command only reads saved evidence:

```bash
python3 experiments/run_professor_raw_ordering.py --summarize-only
```

It requires matching local artifacts and source/dependency fingerprints.
The reproduction commands in individual reports can execute studies; use
them deliberately rather than rerunning research merely to browse results.
The existing Streamlit dashboard covers Version 1, not this professor model.

For the completed November evidence, this command validates saved hashes and
summaries without fitting, optimizing or writing:

```bash
python3 experiments/run_professor_nov2_followup.py --summarize-only
```

The [November report](professor_nov2_followup.md) documents the exact datasets,
four plots, model selection, six policies, 12 group cells and limitations.

For the final successor, this read-only command verifies saved results and
their parent bindings without refitting or optimizing:

```bash
python3 experiments/run_professor_small_penalty.py --summarize-only
```

The [final study report](professor_small_penalty.md) and
[delivery folder](../reports/professor_model/small_penalty/README.md) contain
the selected smaller coefficient, four new plots, training/prediction audit,
14-slide update and compact evidence snapshot. Earlier deliveries are preserved.

For the latest 5% successor, read-only inspection is:

```bash
python3 experiments/run_professor_five_percent_penalty.py --summarize-only
```

See the [latest report](professor_five_percent_penalty.md) and
[14-slide delivery and evidence snapshot](../reports/professor_model/five_percent_penalty/README.md).

## Next decisions

1. Review the new minimum-5% results with Sergio before the November 12
   meeting. Group B now receives 383 and 154 loans, with one common penalty;
   all profits still use prediction applicants only. Distinguish approval
   parity from a general fairness claim and report ML's material profit gap.
2. Decide whether these one-realization results suffice for the final slides
   and capstone. Additional noise draws or independent populations require a
   separate design; the full raw-ordering extension remains unrun.
3. Retain the frozen stylized contract, noise and bounded-score cutoff when
   interpreting these results. A realism-oriented revision or raw-order
   extension is a separate study, not an automatic correction.
4. Handle the V2 checkpoint identity issue separately. The corrected V2
   checkpoint has five replications/360 portfolios, and its 50-run target is
   incomplete; it is not the completed professor-model 200-noise-draw study.
5. Continue deferring HMDA, larger Monte Carlo work and dashboard extensions.
   The smaller-penalty successor added none of those, performed no new fits,
   and did not commit, push or send an email automatically.
