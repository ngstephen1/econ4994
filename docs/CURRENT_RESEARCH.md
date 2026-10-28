# Current Research Guide

Updated **2026-10-28**. Use this page and
[Project Status](PROJECT_STATUS.md) for the present repository state.
Earlier reports preserve the conclusions and plans recorded at each stage;
some are checksum-bound inputs to later experiments and remain unchanged.
This guide supersedes their planning language without rewriting the evidence.

## Read these first

| Question | Current document |
|---|---|
| What is complete, what remains, and which tests were run? | [Project Status](PROJECT_STATUS.md) |
| What happened when only the fitted B contribution was removed? | [Step 10B fixed-coefficient comparison](professor_fixed_coefficient_comparison.md) |
| Why do profit averages and clipped ties behave this way? | [Post-10B evidence review](professor_post10b_review.md) |
| What happened when raw scores ordered lending while bounded PD determined profits? | [Completed raw-ordering sensitivity](professor_raw_ordering_sensitivity.md) |
| What do the underlying risk and accounting equations mean? | [Whiteboard specification](professor_whiteboard_model.md), [loan economics](professor_loan_economics.md) |

The [Step 10A roadmap](professor_model_next_steps.md) and
[evidence synthesis](professor_model_evidence_synthesis.md) describe a proposal
that has since been executed as Step 10B. The post-10B review's raw-ordering
proposal has also since been implemented for six selected cases. Read those
sections as historical design records, not pending implementation requests.
The older root README, documentation index, data guide and eight-slide deck
primarily describe the earlier Step 8B checkpoint.

## Completed studies and their different denominators

| Study | Completed scope | What the repetition means |
|---|---|---|
| Original professor-model noise study, Step 8B | 200 saved noise draws, aware/blind policies | Different eta on one fixed population and split; five pilot draws are included, not additional |
| Fixed-coefficient comparison, Step 10B | One canonical case plus the same 200 saved draws, W/Z/F | 201 evaluation cases, not independent borrower populations; no new fits or noise |
| Post-10B review | Saved-output audit and follow-up design | Explanation of existing evidence, not another experiment |
| Raw-ordering sensitivity | Canonical plus draws 008, 064, 089, 092, 154; 18 new raw policies and 18 frozen references | Deliberately selected diagnostics, not Monte Carlo frequency estimates |

All use the same 100,000 synthetic applicants, original 80,000/20,000 split,
evaluation contracts and capital. The raw-ordering study did **not** extend
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

All paths below are relative to `/Users/macbook/Hack/econ4994/`. Generated
CSV, Parquet and JSON files are local evidence and may be absent in a fresh
clone. Join applicant-level files using `applicant_id`.

| File | One row / useful columns |
|---|---|
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

The latest pre-merge check ran `python3 -m pytest -q tests/test_professor_*.py`:
**534 passed, no warnings**, in 20.82 seconds. This is the professor-model
test family, not the full repository suite. The raw-ordering study separately
recorded **249 targeted tests with no warnings**, including 43 new tests.
The last recorded full suite remains the Step 9C result:
**717 passed, 4 failed, 32 existing warnings**. The four V2 checkpoint lookup
failures are unresolved. The documentation refresh ran no tests or experiments;
the later pre-merge test check is recorded separately here. No scientific
experiment was rerun for the merge.

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

## Next decisions

1. Review whether raw scores may order loans while bounded PD determines
   cash flows. Keep the current raw-ordering result as a sensitivity unless
   the professor adopts that policy interpretation.
2. Confirm the time horizon, per-period rate, growing-payment contract,
   low risk scale and PN measurement noise before specifying a separate
   realism-oriented revision. Preserve the frozen studies.
3. Decide whether a full saved-draw raw-ordering comparison would answer a
   new question. It remains unrun; selected diagnostics do not estimate its
   frequencies. Independent-population uncertainty is another design.
4. Handle the V2 checkpoint identity issue separately. The corrected V2
   checkpoint has five replications/360 portfolios, and its 50-run target is
   incomplete; it is not the completed professor-model 200-noise-draw study.
5. Continue deferring HMDA and a professor-model flexible-ML comparison.
   Neither is implemented or authorized by a documentation refresh.
