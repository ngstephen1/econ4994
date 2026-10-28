# Professor Model — Step 9B Revised Lending Comparison

## Research question and scope

How does the professor-specified **−0.1 × B** adjustment change lending,
group approval rates, and evaluator-only true expected profit when applicants,
contracts, capital, and the policy family remain fixed?

This is one deterministic comparison on the original **20,000 evaluation
applicants** (16,970 B=0, 3,030 B=1). It reuses the validated
[Step 9A risk estimates](professor_revised_scenarios.md) without changing a
single score. No regression fitting, population/noise generation, oracle
reoptimization, Monte Carlo, HMDA, ML, or Step 9C work was performed.

## Policies and unchanged assumptions

| New policy / scenario ID | Frozen bounded decision score |
|---|---|
| `professor_no_direct_group_aware` | Clip the original group-aware OLS prediction to [0,1] |
| `professor_direct_minus_0p1_B` | Clip original group-blind raw OLS prediction **minus 0.1B** to [0,1] |
| `group_blind_no_adjustment_reference` | Clip original group-blind OLS prediction to [0,1] |

The original fitted coefficients, PN target, training/evaluation split, and
Step 9A CSV/Parquet scores are frozen. The direct adjustment is additive raw
probability, **−10 percentage points for B=1**, not a log-odds penalty or
0.1% change. It favors B=1 in perceived default risk. “No-direct” means no
manually imposed adjustment, not exclusion of B from the fitted model.

Three additional, separately labeled reference policies are read unchanged:
`frozen_group_aware_bank`, `frozen_group_blind_bank`, and
`frozen_true_risk_oracle`. The oracle knows probabilities, not future events.

Every new policy uses original contracts: T=30, theta=0.10, r=0.05 **per
period**, ell=theta·Y, principal L=30·ell, and payment_t=ell·(1+r)^t.
The existing `profit.assess_bank_profit` computes perceived receipts/profit:

```text
expected_receipts_bank = sum[t=1..30] payment_t * (1-estimated_PD_bounded)^t
expected_profit_bank = expected_receipts_bank - L
```

Default is absorbing with constant conditional per-period probability.
There are no added costs, discounts, recovery, pricing changes, or new
contract options. Evaluator true profit is read from the **existing**
`true_expected_profit.parquet`; it is not recalculated or fitted.

## Cutoff optimization and exact verification

The exact budget is read from the original bank selection:
**$1,826,087,220.7466326**. It is not rounded, rescaled, or recomputed.

```text
approved_i(tau) = 1{estimated_PD_bounded_i <= tau}
maximize_tau sum_i approved_i(tau) * expected_profit_bank_i
subject to sum_i approved_i(tau) * L_i <= saved capital
```

`revised_lending.optimize_scenario` delegates to the unchanged
`lending.choose_cutoff` through an observable-only adapter. The legacy model
ID is used only inside that adapter; exported records identify their actual
scenario/policy, not a newly fitted model. Group, truth, oracle decisions,
and fairness targets are excluded from the cutoff input schema.

The original algorithm uses exact binary-float integer prefix sums. It
evaluates every distinct bounded score endpoint plus a separate reject-all
action, keeps exact score ties indivisible, never skips applicants, and uses
the lowest cutoff/reject-all-first tie rule. This is not unrestricted knapsack.
Every candidate and selected result is independently checked with the existing
`noise_pilot.exact_cutoff_diagnostics` **Fraction enumeration only**; no pilot
replication, random draw, or fitting function is invoked.

| Scenario | Exact selected cutoff | Approved | Rejected | All candidates | Feasible candidates |
|---|---:|---:|---:|---:|---:|
| No-direct, aware | 0.0009333057709409261 | 3,900 | 16,100 | 19,079 | 2,979 |
| Direct, blind −0.1B | 0.0006623478981412278 | 4,990 | 15,010 | 16,238 | 1,228 |
| Blind control | 0.0009481206961177917 | 3,802 | 16,198 | 19,119 | 2,921 |

All **54,436** candidates are saved, including infeasible candidates and the
three reject-all candidates. All three selected policies fund complete
requests, keep ties whole, and satisfy the original exact budget.

## Decision freeze and information boundary

Before any numeric true-risk, true-profit, or frozen-reference decision read:

1. Verify Step 9A metadata/configuration/source hashes, all original inputs,
   CSV/Parquet equality, original cohort IDs, and exact saved bounded scores.
2. Compute perceived profits from only bounded scores and original contracts.
3. Optimize and independently verify all three policies.
4. Save all decisions, perceived assessments, candidates, and selections.
5. Seal their hashes in `bank_freeze_checkpoint.json`.

Only after rechecking the freeze does the evaluator load saved true-profit
rows and original aware/blind/oracle decisions. Altered or missing freezes
block evaluation. Private bytes may be hashed for preservation before the
freeze, but private tables are not numerically read for bank decisions.

New lender-facing decision files contain only information scope, scenario
and policy IDs, applicant ID, bounded estimated risk, cutoff, approval flag,
requested/funded principal, and decision status. They exclude B, true_PD,
true profit, PN, and private factors. Perceived assessments likewise exclude
hidden truth. Evaluation tables carry
`EVALUATOR_ONLY_NOT_FOR_LENDER_DECISIONS` on every row.

## Portfolio comparison

The new no-direct and blind-control policies exactly reproduce the original
aware and blind **individual decisions, cutoffs, capital, and bank objectives**.
The comparison CSV retains all six labels. The table below combines those
identical new/reference pairs for readability, not additional observations.

| Policy | Approved / rate | Funded principal | Capital use | Unused capital | Bank-perceived profit | True expected profit |
|---|---:|---:|---:|---:|---:|---:|
| No-direct / frozen aware | 3,900 / 19.50% | $1,825,953,398.73 | 99.992672% | $133,822.01 | $2,393,797,751.16 | $2,366,074,476.85 |
| Direct | 4,990 / 24.95% | $1,826,025,513.85 | 99.996621% | $61,706.89 | $2,411,377,270.28 | $2,359,267,451.62 |
| Blind control / frozen blind | 3,802 / 19.01% | $1,825,778,044.19 | 99.983069% | $309,176.56 | $2,392,136,130.18 | $2,366,599,763.61 |
| Frozen oracle | 3,406 / 17.03% | $1,825,849,955.62 | 99.987007% | $237,265.12 | Not a bank estimate | $2,370,206,703.87 |

The frozen oracle cutoff remains 0.0008343666409349597. None of its decisions
was optimized again. The CSV also reports average true profit per approved
applicant and mean approved true PD. Missing statistics are blank, not zero.

Direct true expected profit is **$6,807,025.23 below the frozen aware bank**,
**$7,332,311.99 below the matched blind control**, and **$10,939,252.25 below
the frozen oracle**. Relative to the blind control, perceived profit rises by
about $19.24M while true profit falls by $7.33M. Approving more applicants
does not guarantee higher true economic value under the fixed capital pool.

CSV signed-profit fields use **policy minus frozen reference**. They are not
clipped to zero or labeled unrestricted optimal regret. Different score
rankings yield different feasible cutoff portfolios; the oracle is not a
universal upper bound or fairness rule. All monetary values are nominal
expected values under the stylized contract, not realized mortgage returns.

## Group outcomes and comparison directions

Approval rates divide by every evaluation applicant in the group. Funded
principal per applicant includes rejected requests as zero. Risk means
condition on approval. The oracle has no estimated bounded risk, so that
specific field is undefined for its group rows; true risk is reported.

| Policy | B=0 approved / rate | B=1 approved / rate | B=1 minus B=0 gap (pp) | Gap minus frozen aware (pp) | Gap minus frozen blind (pp) |
|---|---:|---:|---:|---:|---:|
| No-direct / frozen aware | 3,679 / 21.679434% | 221 / 7.293729% | −14.385705 | 0 | −14.541172 |
| Direct | 1,960 / 11.549794% | **3,030 / 100%** | **+88.450206** | **+102.835911** | **+88.294739** |
| Blind control / frozen blind | 3,222 / 18.986447% | 580 / 19.141914% | +0.155468 | +14.541172 | 0 |
| Frozen oracle | 2,871 / 16.918091% | 535 / 17.656766% | +0.738675 | +15.124380 | +0.583207 |

Direct versus blind control isolates the imposed post-fit adjustment within
this synthetic system. It gains **2,450 B=1 approvals** and displaces
**1,262 B=0 approvals**. B=1 approval rises 80.858086 pp while B=0 approval
falls 7.436653 pp. The gap change of 88.294739 pp is a difference of group
rate differences, not an individual applicant's probability change.

Direct versus no-direct also changes the fitted base specification. Its
102.835911-pp gap change must not be described as the same isolated effect.
A difference of gaps can exceed 100 pp without an approval rate exceeding
100%. Neither a smaller gap nor higher profit by itself proves fairness or
nondiscrimination. This is a researcher-defined synthetic intervention.

## Pairwise allocation overlap

All **15 unordered pairs** among six labeled policies are reported with four
exhaustive categories each, for **60 rows**. Jaccard is the approved-set
intersection divided by union; it is undefined if both policies reject all.
Reference/new-policy duplicates are retained deliberately and have identical
funded sets. Pairwise Jaccard repeats on the four category rows and must not
be treated as four independent estimates.

| First / second policy | Approved by both | First only | Second only | Rejected by both | Jaccard |
|---|---:|---:|---:|---:|---:|
| No-direct / direct | 2,181 | 1,719 | 2,809 | 13,291 | 32.5086% |
| Direct / blind control | 2,540 | 2,450 | 1,262 | 13,748 | 40.6270% |
| Direct / frozen oracle | 2,207 | 2,783 | 1,199 | 13,811 | 35.6600% |

Each category reports total requested principal and potential true expected
profit, plus each policy's expected-profit contribution from requests it
actually funds. `rejected_by_both` profit is explicitly **hypothetical
unfunded-request value**, with zero contribution to either selected portfolio.
Potential profits for a request rejected by one policy are not earnings of
that rejecting policy. Expected contributions are not realized earnings.

## Zero-score ties and capital competition

| Scenario | B=0 / B=1 zeros | Total zeros | Zero-block principal | Budget share | Feasible? | Selected action |
|---|---:|---:|---:|---:|---|---|
| No-direct | 851 / 72 | 923 | $709,224,665.32 | 38.838488% | Yes | Approve all 923 |
| Direct | 734 / 3,030 | **3,764** | **$1,276,685,167.37** | **69.913701%** | Yes | **Approve all 3,764** |
| Blind control | 734 / 149 | 883 | $691,901,043.03 | 37.889814% | Yes | Approve all 883 |

Counts, group denominators, principal, and feasibility were recomputed from
saved scores/contracts and checked against frozen Step 9A diagnostics, not
hardcoded to the anticipated 3,764. No policy rejects all in this run, and
no tie is split. Oversized-zero-block behavior is retained and tested.

Clipping erases distinctions among all negative raw scores, creating one
shared zero-score block. The direct block fits capital but consumes roughly
70% of it, changing who can be added by the remaining common cutoff.
All B=1 applicants are funded because their scores join that feasible block,
not because the optimizer contains a group approval target. Their true risk
is unchanged and is not zero; mean true PD among funded B=1 applicants under
direct is approximately 0.141505% per period.

## Exact output locations and schema roles

All paths are relative to `/Users/macbook/Hack/econ4994/`.

| Directory | Files / row meaning |
|---|---|
| `data/processed/professor_model/revised_lending/` | `scenario_lending_decisions.csv` / `.parquet`: applicant × scenario, 60,000 bank-only rows; `scenario_bank_assessments.csv` / `.parquet`: same keys, perceived receipts/profit only |
| `results/tables/professor_model/revised_lending/` | `portfolio_comparison.csv`: six labeled policies; `group_outcome_audit.csv`: 12 policy/group rows; `pairwise_allocation_overlap.csv`: 60 pair/category rows; `zero_score_tie_diagnostics.csv`: three scenarios; `cutoff_candidates.csv`: 54,436 bank-only candidates |
| `results/metrics/professor_model/revised_lending/` | `preservation_manifest.json`: pre-edit hashes; `bank_selections.json`: all selections/checks; `bank_freeze_checkpoint.json`: sealed bank artifacts and input/source identity; `validation.json`: machine-readable checks; `run_manifest.json`: configuration, source/dependency versions, event order, all output hashes |
| `results/figures/professor_model/revised_lending/` | `true_expected_portfolio_profit.png`; `group_approval_rates.png` |

Both figures were rendered and visually inspected. Profit axes include zero;
approval axes show the full 0–100% range with annotation headroom. Hatched
reference bars distinguish frozen benchmarks from Step 9B policies. CSVs keep
raw probabilities, fraction-valued rates, explicitly named percentage-point
gaps, and dollar-valued monetary fields. No extra workbook was introduced.

New implementation: `configs/professor_model/revised_lending.yaml`,
`src/fair_lending/professor_model/revised_lending.py`, `revised_lending_io.py`,
`revised_lending_figures.py`, `experiments/run_professor_revised_lending.py`,
and `tests/test_professor_revised_lending.py`.

```bash
python3 experiments/run_professor_revised_lending.py
python3 -m pytest -q tests/test_professor_revised_lending.py
python3 -m pytest -q tests/test_professor_*.py
python3 -m pytest -q
```

The runner refuses existing outputs; there is no overwrite/refit/regenerate
option. Inspect the saved files instead of rerunning a completed comparison.
Generated data/results retain the repository's existing Git-ignore policy.

## Preservation, validation, and limits

The pre-edit manifest covers **963 files**, including all uncommitted Step 9A
implementation/configuration/tests/documentation and outputs. Every recorded
file remains byte-identical. Original population, PN, fits, scores, contracts,
capital, true-profit accounting, bank/blind/oracle decisions, Step 8A/8B
results, and V1/V2 science were preserved. No code was changed to weaken
unrelated V2 tests. No staging, commit, or push was performed.

An independent saved-file audit reproduced all six portfolio totals, all 12
group counts/rates/funding/profit cells, and every count, Jaccard, and potential
profit in all 15 pairs. Exact Fraction budget checks passed. New CSV/Parquet
decisions match and their scores exactly match the frozen Step 9A source.
All candidates passed independent exact enumeration. Frozen-aware and
frozen-blind allocations were reproduced exactly rather than merely in total.

Fresh test results:

- Step 9B focused tests: **25 passed, no warnings** (4.79 seconds).
- All professor-model tests: **404 passed, no warnings** (16.67 seconds).
- Full repository: **684 passed, 4 failed, 32 existing warnings** (99.75 seconds).

The four failures remain in `tests/test_systemic_corrected_checkpoint.py`:
72-cell completeness, replication-13 provenance, legacy comparison, and the
five-record interim summary. The unchanged loader constructs filenames using
current Git HEAD, whereas saved corrected V2 records refer to an older
revision. No V2 code, record, or test was changed or suppressed. The warnings
are existing scikit-learn/SciPy L-BFGS-B `disp`/`iprint` deprecations, 26 from
the ML benchmark and six from sensitivity tests. The full repository is not
all-green; there are no new professor-model failures or warnings.

The new tests cover exact score/ID/contract reuse; strict decision allowlists;
independence from truth; exact budget boundaries; whole and oversized zero
ties; deterministic candidate/tie choices; group rates and per-applicant
funding; signed profit comparisons; overlap/Jaccard and hypothetical-profit
labels; undefined approved-risk statistics for reject-all; source preservation;
immutable exports; truth blocked before freeze or after tampering; honest
reference reproduction; readable figures; and rejection of mutated in-memory
decisions or perceived profits before they can be frozen.

The adjustment strongly favors one group in perceived risk, but true profit
falls in this particular comparison. This is a finite, single-population
synthetic finding, not proof that a mechanism always lowers profit, that one
policy is fair, or that actual lenders discriminate. Large nominal profits
reflect the stylized 5%-per-period growing-payment contract. Clipping is not
calibration. No Step 9C, new-population study, ML, or Monte Carlo was launched.
