# Professor Model — Raw-Ordering Sensitivity

Completed **2026-10-28**. This implements the six-case follow-up described in
the [post-10B review](professor_post10b_review.md). It changes the score used
to order applications while preserving every applicant's economic probability
and potential profit. Original implementations, contracts, coefficients,
cutoffs, decisions and reports remain frozen.

**Main finding:** all nine previously reject-all policies in the selected
cases fund loans under raw ordering. The other nine portfolios are exactly
unchanged. This supports the clipped-tie/capital explanation. It is a change
in the allowable lending portfolios, not better risk prediction, probability
calibration, or a general fairness improvement.

## Fixed design and interpretation

The YAML specification was written before the first production optimization:
`configs/professor_model/raw_ordering.yaml`. The cases are the canonical
realization and saved draws **008, 064, 089, 092 and 154**. These five draws
were deliberately selected from existing results to diagnose the mechanism.
They are not a random sample or a new Monte Carlo study. Every case reuses
the same 20,000 evaluation applicants from the same 100,000-person population.

| Source policy | Frozen risk definition |
|---|---|
| W: `frozen_aware` | Existing aware OLS scores and fitted B contribution |
| Z: `aware_B_zero` | W raw score minus that case's saved B coefficient times B; other coefficients unchanged |
| F: `frozen_blind` | Existing fully refitted blind OLS scores |

For each source, compare its frozen bounded-score policy with a new raw-score
policy. Negative raw scores are ordering values, **never probabilities**:

\[
s_i=raw_i,\qquad p_i=\operatorname{clip}(raw_i,0,1),
\qquad approved_i(\tau)=\mathbf{1}\{s_i\leq\tau\}.
\]

The economic quantities remain

\[
\Pi_i(p_i)=\sum_{t=1}^{30}\ell_i(1.05)^t(1-p_i)^t-L_i,
\qquad \ell_i=0.1Y_i,\quad L_i=30\ell_i.
\]

The bank maximizes summed perceived profit over complete raw-score prefixes
under the unchanged **$1,826,087,220.7466326** capital limit. Each issued
request is funded in full. There is no skipping, group-specific cutoff,
arbitrary tie splitting, slack or currency rounding. Exact raw-score ties
are indivisible. Reject-all is separate from a cutoff of zero; exact objective
ties retain reject-all first, then the lowest raw cutoff.

As before, default is conditional per period with absorbing default, r=5%
**per period**, and calendar duration is unspecified. No costs, discounting,
recovery or prepayment are added. This is the existing stylized growing-payment
contract, not standard mortgage amortization or annual bank-profit accounting.

## What the comparison isolates

`ordering_score` and `accounting_PD` are distinct fields. Existing clipped
PD and per-request perceived profit agree exactly with their frozen source.
The original bounded-probability API was not relaxed or edited. The cutoff
input allowlist has IDs, source label, raw ordering score, accounting PD,
principal and perceived profit; it excludes B, hidden truth, PN and outcomes.
B can still influence the frozen aware risk score, as originally specified.

Clipping is monotone, so every bounded-score prefix also exists in the raw
family. Raw ordering adds endpoints inside formerly clipped score blocks
while retaining exact raw ties. It therefore cannot lower the optimized
**perceived** objective for the same source model. This property is checked
using exact arithmetic; it is not a guarantee about true profit or fairness.

The final execution independently checked **360,018 raw candidate endpoints**,
including reject-all. All 18 raw policies have 20,000 unique raw scores in
these data. The mapping covers **287,781 original bounded endpoints** and
adds **72,237 endpoints**. Baseline endpoints are mapped and their selected
portfolios replayed; the baseline cutoffs are never reoptimized.

## Verified lending outcomes

Each entry is **original bounded approvals → raw-order approvals**.

| Case | W: aware | Z: coefficient removed | F: blind refit |
|---|---:|---:|---:|
| Canonical | 3,900 → 3,900 | 3,802 → 3,802 | 3,802 → 3,802 |
| Draw 008 | 0 → 5,107 | 0 → 4,741 | 4,744 → 4,744 |
| Draw 064 | 0 → 3,898 | 3,701 → 3,701 | 0 → 3,701 |
| Draw 089 | 4,331 → 4,331 | 0 → 4,163 | 4,161 → 4,161 |
| Draw 092 | 0 → 4,222 | 0 → 3,865 | 3,861 → 3,861 |
| Draw 154 | 0 → 4,812 | 3,454 → 3,454 | 0 → 3,454 |

Reject-all changes from **9/18 bounded policies to 0/18 raw policies**.
These are descriptive counts in selected diagnostic cases, not estimated
population frequencies. Every previously lending portfolio has exactly the
same approved IDs, capital, group rates and profits after the ordering change.
The canonical case is an especially useful unchanged control.

All new raw policies use **99.981%–99.997%** of capital. For the nine revived
policies, the selected raw cutoff is negative, so all funded applications
have accounting PD zero while retaining their distinct raw ranks. A negative
cutoff does not imply negative default risk. Zero accounting PD also does
not imply zero evaluator true risk.

## Economic results and why the gains are large

The following are **raw-order minus original true expected portfolio profit**
in **billions of synthetic dollars**, rounded to six decimal places. Full
precision is in `comparisons.csv`.

| Case | W | Z | F |
|---|---:|---:|---:|
| Canonical | 0.000000 | 0.000000 | 0.000000 |
| Draw 008 | +2.358274 | +2.361265 | 0.000000 |
| Draw 064 | +2.366734 | 0.000000 | +2.368213 |
| Draw 089 | 0.000000 | +2.363881 | 0.000000 |
| Draw 092 | +2.363195 | +2.365862 | 0.000000 |
| Draw 154 | +2.360590 | 0.000000 | +2.369824 |

The large positive changes are switches from zero lending to a near-full-
capital portfolio under the stylized nominal cash flows. They are not
improvements in predicted PD: the risk estimates and per-request accounting
did not change. All requests are individually perceived and truly profitable
under the frozen economics, so lending rather than rejecting all produces
positive true value here. A toy test explicitly verifies that expanding the
perceived feasible family can instead reduce true profit when model beliefs
and true profitability disagree.

When an original bounded policy already funded loans, its selected positive
cutoff admitted the whole clipped-zero block. New endpoints *inside* that
block have lower summed perceived profit than the already selected larger
prefix because all incremental profits are positive. That explains the nine
unchanged outcomes. The comparison diagnoses this particular decision rule;
it does not establish raw-score ordering as an optimal real-world policy.

## Group audit: avoiding reject-all is not a fairness result

In draw 154, W originally rejected everyone: both group rates and their gap
were zero. Under raw ordering it funds **1,814/16,970 B=0 applicants** and
**2,998/3,030 B=1 applicants**, a **+88.254442 pp** B=1-minus-B=0 gap.
Other W raw gaps include −29.005195 pp in draw 008 and +29.617516 pp in draw
064. The frozen group coefficient still affects raw rankings.

Thus a zero gap under universal rejection should not be interpreted as a
fairness success, and removing the clipping blockage does not remove group
differences embedded in the fitted score. The underlying population and true
risk contain no group treatment in this family. These results are synthetic
policy diagnostics, not evidence of actual mortgage discrimination.

## Information boundary and implementation

The new modules are isolated under `src/fair_lending/professor_model/`:
`raw_ordering.py`, `raw_ordering_io.py` and `raw_ordering_figures.py`.
`experiments/run_professor_raw_ordering.py` provides orchestration.

W/F predictions and Z's exact coefficient intervention are replayed from
bank-only Step 10B freezes. All three saved bounded cutoffs and decision
fingerprints are reproduced. No old cutoff-search function is called.
Only the new raw policies are optimized, using the existing exact integer
sum helper and a separate Fraction check of every endpoint.

Each case saves and hash-freezes its three new selections, input/decision/
candidate fingerprints and prefix-mapping checks **before** parsing its
historical evaluator record or loading true profit. True values never select
the ordering rule or a cutoff. No PN or eta vector is reconstructed.

Experiment identity binds configuration, relevant source contents, dependency
versions and historical evidence. The exact source files and configuration
are also archived in the run directory. Successes, freezes and attempts are
immutable; failed attempts are retained. Resume validates identities and
complete cell coverage. Summary-only is read-only and cannot launch cases.

## Files to open and reproduction

Final experiment fingerprint:
`d0421da2ea691f2e0e39c3c53894d9b30a056bcecb16ae6b0274013fd12bbbf8`.
All directories below are relative to `/Users/macbook/Hack/econ4994/`.

| Folder | Files and row meaning |
|---|---|
| `results/tables/professor_model/raw_ordering/d0421da2ea691f2e/` | **`policies.csv`**: 36 case × source × ordering rows; approval counts/gaps, cutoffs, capital and perceived/true profit. **Open first.** |
| Same tables folder | **`comparisons.csv`**: 54 signed first-minus-second comparisons; 18 within-source ordering contrasts and 36 between-model contrasts, kept distinct. **Open second.** |
| Same tables folder | `groups.csv`: 72 case × policy × B rows; `overlap.csv`: 216 comparison × disjoint allocation-category rows. Unfunded potential profits are hypothetical. |
| `data/processed/professor_model/raw_ordering/d0421da2ea691f2e/a9707d67f8ba493fbfdc1b8c8f90b53a/` | Canonical new policies only: `bank_inputs.csv`/`.parquet`, `raw_order_decisions.csv`/`.parquet`, `raw_cutoff_candidates.csv`, `bounded_prefix_mapping.csv`. No hidden truth. |
| `results/metrics/professor_model/raw_ordering/` | `preservation_manifest.json`: 1,654 pre-existing scientific/documentation file hashes; status/index are the explicit additive-edit exceptions. |
| `results/metrics/professor_model/raw_ordering/d0421da2ea691f2e/` | `manifest.json`, `latest_report.json`, `final_validation.json`, six `records/`, six `freezes/`, six `attempts/`, and `source_snapshot/`. Noise-case applicant rows remain in memory only. |
| `results/figures/professor_model/raw_ordering/d0421da2ea691f2e/` | `diagnostic_portfolio_changes.png`: matched approval counts and true-profit differences; rendered and visually inspected. |

The exact canonical UUID and all output hashes are registered in
`final_validation.json`. The CSVs are companions for manual inspection,
not a second authoritative population.
Each canonical input/decision file has 60,000 applicant-by-source rows,
representing the same 20,000 applicants under three policies.

```bash
# Resume the six declared cases; no full 200-draw extension:
python3 experiments/run_professor_raw_ordering.py --resume
# Read-only report; creates no files and launches no scoring/optimization:
python3 experiments/run_professor_raw_ordering.py --summarize-only
```

## Validation and execution history

The final targeted suite passed **249 tests, no warnings, in 8.28 seconds**,
including **43 new deterministic tests**. These cover signed ordering scores,
unchanged bounded accounting, exact ties/budgets/objective ties/reject-all,
negative profits, feasible-family inclusion, independent rational checks,
row-order invariance, no baseline optimization, information-boundary gates,
tamper and source-change rejection, complete resume cells and read-only CLI.
The entire repository suite was not rerun. Four known V2 checkpoint lookup
failures remain unresolved and are not suppressed or called passing.

Independent saved-file checks reproduced all 36 policy masks and economic
values, 18 frozen reference cells, 72 group rows and 216 overlap rows.
Canonical CSV/Parquet pairs agree exactly; persisted canonical candidates
and prefix mappings passed the rational audit again. All six cases resumed
with computation disabled. The actual summary-only CLI ran with computation
and writing disabled, leaving all 57 then-existing final-run artifacts
unchanged in content and modification time. All 26 archived code/configuration
files match their registered hashes.

Final production comprises six cases, 18 new raw optimizations and 36 policy
cells including frozen references, with zero failed attempts. Successful-case
runtime totals **12.48 seconds**, excluding preflight, tests and rendering;
this is not a hardware benchmark. An initial execution used the same six
cases before a legend-layout fix and stronger record checks. Both source
versions and evidence remain saved; all four scientific CSV tables match
exactly. There were **36 raw searches across the two source versions**, not
12 independent cases or new populations/noise draws. The initial namespace
is explicitly marked superseded for delivery.

No original scientific implementation, YAML, dataset, decision or frozen
report was changed. No new fit, noise, ML model, HMDA, unrestricted portfolio allocation,
payment assumption, pricing rule, commit or push was added.

## Recommended next research decision

Review the policy interpretation with the professor: is the bounded PD itself
required to order lending, or can an unbounded score determine order while
bounded probabilities determine cash flows? If the former is essential,
retain this as a sensitivity rather than replacing the primary model.

The six cases already establish the specific tie/capital mechanism; do not
automatically run the other 195 saved draws or describe these selected cases
as uncertainty estimates. A later authorized 200-draw comparison could measure
conditional frequency, using the same frozen evidence. Separately resolve
the economic event/calendar/rate interpretation, low true-risk scale and
large measurement noise before a realism-oriented model revision. Preserve
these results rather than retuning their assumptions after observing them.
