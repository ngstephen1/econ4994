# Professor Model — Step 9C Controlled Mechanism Audit

## Predeclared design

This protocol was written before calculating the new floor or optimizing any
Step 9C policy. Its scientific settings are in
`configs/professor_model/mechanism_audit.yaml`. All applicants, contracts,
fits, Step 9A/9B artifacts, capital, accounting, and frozen references remain
unchanged. This is a synthetic mechanism audit, not empirical causal evidence.

| Variant | Definition | Lending role |
|---|---|---|
| A: `A_frozen_direct` | Existing blind raw prediction −0.1B, clipped to [0,1] | Valid baseline; must reproduce frozen Step 9A/9B |
| B: `B_unclipped_direct_diagnostic` | Same direct raw score, including negatives | Diagnostic only; never used as a probability or policy |
| C: `C_zero_floor_diagnostic` | max(direct raw, 0), without an upper cap | Diagnostic only; test equality with A when no raw score exceeds one |
| D: `D_positive_floor_sensitivity` | Start with A and replace only exact zeros with a fixed positive floor | Valid, separately labeled sensitivity |
| E: `E_frozen_blind_control` | Existing frozen group-blind bounded predictions | Valid no-adjustment control |

### Positive-floor rule and rationale

Use **q=0.01**, the first percentile, with NumPy `method="linear"`, on the
strictly positive frozen blind-control evaluation probabilities. Exclude
zeros, retain all groups, and do not inspect truth or outcomes to choose the
quantile. This supplies a small nonzero risk on the existing prediction scale,
not an arbitrary externally calibrated probability. Replace **only A's exact
zeros**. Positive A scores below the floor remain unchanged. Record the exact
floor, source count, quantile, affected count, and any collision with existing
scores. If no positive control value exists, fail rather than invent a floor.

The rule is a researcher-chosen sensitivity, not a professor-derived parameter
or probability calibration. It relocates the common clipped-zero tie; it does
**not** split or remove that tie. Do not tune q after inspecting allocation.

### Upper bound and identification limits

A and E retain clipping to [0,1]. D remains in [0,1] because its floor derives
from positive bounded values. C deliberately has no upper cap: report scores
above one if they exist, flag them as invalid probabilities, and keep C out of
lending regardless of its observed range. B is likewise never a policy.

This design quantifies score-distribution changes and tests whether removing
the exact zero level changes allocation. It does **not** provide a full
additive decomposition of adjustment, clipping, ties, and capital: the raw
negative-score branch is not a valid counterfactual probability policy, and
D preserves the shared tie. A unchanged allocation under D would not prove
clipping or ties irrelevant under other floors or budgets. No coefficient-only
intervention, tie splitting, new budget, refitting, or Monte Carlo is added.

## Frozen inputs, accounting, and information boundary

The audit reuses the same **20,000 evaluation applicants** (16,970 B=0 and
3,030 B=1), existing contracts, and exact saved budget
**$1,826,087,220.7466326**. The five variants produce 100,000 *variant rows*,
not 100,000 new or independent applicants. Three valid policies produce
60,000 decision rows. Original PN, fits, applicant draws, true risk, contracts,
and oracle decisions were not changed.

The original contract remains T=30, theta=0.10, r=0.05 **per period**, with
`ell_i = theta * Y_i`, `L_i = T * ell_i`, and
`payment_it = ell_i * (1+r)^t`. PD means conditional per-period default risk.
The unchanged bank accounting is:

\[
R_i(p)=\sum_{t=1}^{30}\ell_i(1+r)^t(1-p)^t,\qquad
\Pi_i(p)=R_i(p)-L_i.
\]

The existing exact cutoff search maximizes summed perceived profit among
common-score prefixes, under the original capital limit. The rule is
`score <= tau`; exact ties are indivisible, reject-all is separate from a
zero cutoff, and exact objective ties select the lowest cutoff. There is no
applicant skipping, group-specific cutoff, new profit formula, or unrestricted
portfolio optimization. All valid policies remain profitable at their
selected prefixes; the next entire risk-group expansion is capital-infeasible.

The bank stage accepts only applicant IDs, variant labels, valid scores,
and existing contracts. Group is retained in the diagnostic/audit table but
excluded from the cutoff input. Neither PN nor numerical hidden truth is
loaded to construct variants or choose cutoffs. All A/D/E risk, assessment,
candidate, selection, and decision artifacts are saved and hash-frozen before
the evaluator loader releases existing true profit and original reference
decisions. True profit is read from the frozen original artifact, not rebuilt.

The evaluator's fixed-width six-policy API is reused through temporary label
adapters only. Exported IDs identify A/D/E accurately; D remains a positive-floor
sensitivity of the blind direct score, not a newly fitted group-aware model.

## Observed score and clipping diagnostics

There are **no direct raw scores above one**, so C exactly equals A numerically.
B and C remain diagnostic-only irrespective of their observed range.

| Raw direct scope | Applicants | Mean score ×100 | Minimum ×100 | Maximum ×100 | Negative scores |
|---|---:|---:|---:|---:|---:|
| Overall | 20,000 | −1.367149715 | −10.532622733 | 0.369957366 | 3,764 |
| B=0 | 16,970 | 0.148188146 | −1.411844377 | 0.369957366 | 734 |
| B=1 | 3,030 | −9.854041959 | −10.532622733 | −9.649603628 | 3,030 |

Values in this table are percentage-point score units; negative values are
not probabilities. The manual adjustment is exactly **−0.1**, or **−10 pp**,
for every B=1 applicant. It is large relative to the frozen prediction scale.
All B=1 raw scores consequently become negative. Lower clipping raises their
mean from −9.854041959 pp to zero. Across all applicants, clipping raises the
mean score by **1.495760428 pp**. This is a score-scale calculation, not an
allocation-effect decomposition or probability calibration.

The floor's strictly positive control source contains **19,117** of 20,000
values. The predeclared calculation yields:

```
quantile = 0.01
method = linear
floor = 0.00013923734234194752
floor × 100 = 0.013923734234194752 percent
replaced = 3,764 exact zeros (734 B=0; 3,030 B=1)
existing nonzero scores equal to the floor = 0
```

D changes no other score: **161 positive scores below the floor remain
unchanged**. It has zero exact-zero scores but still one **3,764-applicant
tie**, now at the positive floor. A and D each have 16,237 unique scores and
one tied block; E has 19,118 unique scores and one 883-applicant zero tie.
The selected cutoff boundary has one applicant in all three policies.

## Ties and capital

| Policy | Audited tie | Size | B=0 / B=1 | Principal | Share of fixed budget | Selected treatment |
|---|---|---:|---:|---:|---:|---|
| A | Zero | 3,764 | 734 / 3,030 | $1,276,685,167.37 | 69.913701% | Funded whole |
| D | Positive floor | 3,764 | 734 / 3,030 | $1,276,685,167.37 | 69.913701% | Funded whole |
| E | Zero | 883 | 734 / 149 | $691,901,043.03 | 37.889814% | Funded whole |

Block cost alone does not establish that a positive-floor block is reachable:
the rule must also fund every lower score. D's 161 lower-score requests plus
the relocated tie form a **3,925-applicant** prefix costing
**$1,365,351,773.15**, or **74.769253%** of capital. That prefix is feasible.
The selected cutoff is higher than the floor and funds the entire block.
No policy selects reject-all or splits a tie.

## Lending, group outcomes, and economic comparisons

A exactly reproduces the frozen Step 9A direct scores and Step 9B direct
decisions, selected cutoff, funded principal, and perceived objective.
E likewise exactly reproduces the frozen blind control. Original aware and
oracle decisions are read as references and never reoptimized.

| Quantity | A: frozen direct | D: positive floor | E: frozen blind control |
|---|---:|---:|---:|
| Selected cutoff (probability scale) | 0.0006623478981412278 | 0.0006623478981412278 | 0.0009481206961177917 |
| Approvals | 4,990 | 4,990 | 3,802 |
| Overall approval rate | 24.95% | 24.95% | 19.01% |
| B=0 approvals / rate | 1,960 / 11.549794% | 1,960 / 11.549794% | 3,222 / 18.986447% |
| B=1 approvals / rate | 3,030 / 100% | 3,030 / 100% | 580 / 19.141914% |
| B=1 minus B=0 approval gap | +88.450206 pp | +88.450206 pp | +0.155468 pp |
| Funded principal | $1,826,025,513.85 | $1,826,025,513.85 | $1,825,778,044.19 |
| Capital utilization | 99.996621% | 99.996621% | 99.983069% |
| Unused capital | $61,706.89 | $61,706.89 | $309,176.56 |
| Bank-perceived expected profit | $2,411,377,270.28 | $2,403,522,456.93 | $2,392,136,130.18 |
| Evaluator true expected profit | $2,359,267,451.62 | $2,359,267,451.62 | $2,366,599,763.61 |
| Mean true PD among approved | 0.112139% | 0.112139% | 0.077254% |
| Mean true profit per approved request | $472,799.09 | $472,799.09 | $622,461.80 |

**D changes zero approval decisions.** A/D overlap is 4,990 approvals, no
exclusive approvals, 15,010 joint rejections, and Jaccard 1.0. Relative to A,
D has zero changes in group rates, capital use, or true expected profit.
It reduces the bank-perceived portfolio objective by **$7,854,813.35**.
Estimated profitability changes without changing the maximizing feasible
cutoff in this sensitivity.

Relative to E, both A and D approve 1,188 more applicants: 2,450 B=1 applicants
gain funding and 1,262 B=0 applicants lose it. Their approval gap is
**88.294739 pp higher** and true expected profit is **$7,332,311.99 lower**.
They share 2,540 approvals with E, with Jaccard **0.406269993602047** and
13,748 joint rejections.

| Signed true-profit difference | A | D | E |
|---|---:|---:|---:|
| Minus frozen direct | $0.00 | $0.00 | +$7,332,311.99 |
| Minus frozen blind control | −$7,332,311.99 | −$7,332,311.99 | $0.00 |
| Minus frozen aware bank | −$6,807,025.23 | −$6,807,025.23 | +$525,286.77 |
| Minus frozen oracle | −$10,939,252.25 | −$10,939,252.25 | −$3,606,940.25 |

The group CSV separately reports counts, approval rates, funded principal per
applicant, approved estimated/true PD, and true expected profit. In A, B=1's
estimated PD is zero while mean true PD is **0.141505%**: zero estimated risk
does not mean zero true risk. Pairwise output covers all 15 pairs among A/D/E
and the three frozen references, with four disjoint allocation categories each.
Potential profits on unfunded requests are explicitly hypothetical. The oracle
comparison is signed and restricted to its existing common-cutoff family;
it is not unrestricted optimal regret.

## Interpretation and next research decision

The experiment establishes four distinct facts: the direct adjustment moves
B=1 raw scores down by 10 pp; lower clipping maps all their negative scores to
zero; clipping creates an indivisible shared tie; and the fixed budget admits
that tie in full. Removing the exact zero level under the declared floor
**does not remove the allocation change**. The relocated block and the
next-prefix capital restriction remain operative.

This does not assign a percentage of the allocation change to clipping or
ties. B cannot serve as a valid negative-probability policy, and D preserves
the tie. One floor and one fixed capital budget do not characterize all
alternative policy designs. Smaller gaps do not establish fairness; higher
profit does not establish nondiscrimination. These researcher-defined
synthetic mechanisms do not prove real-world discrimination. No fitted B
coefficient was isolated from refitting.

The recommended next step is professor review of the **−0.1 probability-scale
adjustment**, the treatment of negative predictions, and the indivisible-tie
rule. A separately authorized study could predeclare other valid risk/tie or
capital counterfactuals; it should not tune them to achieve a desired gap.
No such extension was run here.

## Reproducibility and exact artifact inventory

Repository root: `/Users/macbook/Hack/econ4994/`. All paths below are relative
to that root. The CLI refuses to overwrite an existing audit:

```bash
python3 experiments/run_professor_mechanism_audit.py
```

| Folder | Exact files | Contents / one row |
|---|---|---|
| `data/processed/professor_model/mechanism_audit/` | `variant_scores.csv`, `variant_scores.parquet` | 100,000 rows: one applicant × variant; raw origin, transformed score, group, validity and clipping/floor flags; no truth |
| Same data folder | `policy_decisions.csv`, `policy_decisions.parquet` | 60,000 rows: one applicant × A/D/E policy; frozen score, cutoff, approval, requested/funded principal |
| Same data folder | `bank_assessments.csv`, `bank_assessments.parquet` | 60,000 rows: one applicant × valid policy; perceived receipts/profit; no hidden truth |
| `results/tables/professor_model/mechanism_audit/` | `variant_definitions.csv` | Five definitions; policy/diagnostic eligibility and floor parameters |
| Same tables folder | `risk_clipping_diagnostics.csv` | 15 rows: variant × overall/B=0/B=1; score units in pp, negative/zero/clipping/tie counts |
| Same tables folder | `tie_capital_diagnostics.csv` | Three policy rows; unique/tied scores, block size/cost, capital shares, group composition, selected boundary, floor-prefix feasibility |
| Same tables folder | `cutoff_candidates.csv` | 51,595 rows: every unique-score prefix plus reject-all, including infeasible candidates, for A/D/E |
| Same tables folder | `portfolio_comparison.csv` | Six rows: A/D/E plus frozen aware/blind/oracle; true profit, capital, rates and signed differences |
| Same tables folder | `group_outcome_audit.csv` | 12 rows: policy × B; counts, rates, economic values and evaluator-only true PD |
| Same tables folder | `pairwise_allocation_overlap.csv` | 60 rows: 15 policy pairs × four disjoint allocation categories; hypothetical unfunded profit labeled |
| `results/metrics/professor_model/mechanism_audit/` | `positive_floor_derivation.json` | Exact deterministic floor, source size, affected counts and C/A equality |
| Same metrics folder | `bank_selections.json` | Frozen selections and exact-enumeration checks for A/D/E |
| Same metrics folder | `bank_freeze_checkpoint.json` | Pre-truth checkpoint; 18 source hashes, configuration/input identities, and 12 bank artifact hashes |
| Same metrics folder | `preservation_manifest.json` | Pre-edit SHA-256 inventory of 986 existing scientific artifacts; progress status is the explicitly allowed documentation edit |
| Same metrics folder | `validation.json` | Information-boundary, reproduction, aggregation, preservation and budget results |
| Same metrics folder | `run_manifest.json` | Configuration, dependencies, Git revision/dirty state, event order and 21 other output hashes |
| `results/figures/professor_model/mechanism_audit/` | `raw_and_transformed_scores.png` | Diagnostic B raw distributions versus valid A/D/E distributions; separate score scales |
| Same figures folder | `tie_blocks_and_capital.png` | Zero and relocated positive ties, sizes and principal shares of capital |
| Same figures folder | `approval_and_true_profit.png` | Group approval rates and evaluator true profit; frozen references hatched |

Code is isolated in `mechanism_audit.py`, `mechanism_audit_io.py`, and
`mechanism_audit_figures.py` under `src/fair_lending/professor_model/`, with
`experiments/run_professor_mechanism_audit.py`, the dedicated YAML, and
`tests/test_professor_mechanism_audit.py`. None of the old scientific
implementations or result files was modified.

### Validation evidence

The 33 new deterministic tests pass without warnings. All professor-model tests
pass: **437 passed, no warnings**. These cover exact scores and clipping,
strict-positive floor derivation, zero-only replacement, unchanged IDs and
contracts, diagnostic exclusion, truth/group feature allowlists, exact-budget
and oversized-tie behavior, floor-prefix feasibility, aggregation, overlap,
source invalidation, pre-truth freezing, tamper rejection, CSV/Parquet round
trips, deterministic reruns, and preservation.

A separate read-only saved-file audit reproduced all **51,595** cutoff
candidates with rational arithmetic, all six portfolio aggregates, all 12
group rows and all 60 overlap cells. A/E match original Step 9A/9B files
exactly; A/D approval masks match exactly. All 21 output hashes and 18 source
hashes match. All **986 pre-existing scientific files are byte-identical**.
All three production figures were rendered and visually inspected.

The fresh full suite reports **717 passed, 4 failed, 32 existing deprecation
warnings**, in 105.31 seconds. The same four pre-existing failures are in
`tests/test_systemic_corrected_checkpoint.py`: current-HEAD run identities
cannot locate corrected V2 artifacts saved under an earlier Git revision.
The failures are `test_corrected_replication_has_exactly_72_unique_complete_cells`,
`test_replication13_provenance_is_internally_consistent`,
`test_legacy_comparison_is_diagnostic_and_never_authorizes_reuse`, and
`test_interim_summary_uses_five_corrected_records_only`. No recovery code,
saved records, or tests were changed to suppress them. The warnings are the
existing SciPy L-BFGS-B option deprecations in V1 ML/sensitivity tests.
The repository is not fully green. No experiments outside Step 9C, commits,
staging, or pushes were performed.
