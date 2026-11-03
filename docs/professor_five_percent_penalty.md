# Professor Follow-up: Minimum 5% Group B Approval

## Predeclared successor

This separate successor follows Sergio's request for at least 5% minority Group B
approval. Existing November 2 and +0.005B scientific code, results, models and
presentations remain unchanged. The protocol below preceded execution; validated
results and their scope follow it.

Reuse the frozen two risk scenarios, one noise realization, 100,000 applicant
identities, 80,000 training / 20,000 prediction split, and original contracts and
$1,826,087,220.7466326 capital budget. All lending and profit calculations use
prediction applicants only (16,970 Group A and 3,030 Group B). No fitting, new
noise or population generation is permitted. Scenario 1 uses Y/D/C; Scenario 2
retains its existing +1 SD epsilon shift, consistent C update and Y/D/C/B features.

## Finite bank-side selection

Try the common descending candidates exactly in this order:

```
0.005, 0.0025, 0.00125, 0.000625, 0.0003125, 0.00015625
```

Calculate `raw_penalty = control_raw + delta * B`, then clip to [0,1]. Reoptimize
only the penalty policy in each scenario under the existing bounded-score common
cutoff, full requests, indivisible exact ties and separate reject-all action.
Preserve the four control/Gradient Boosting policies exactly.

Stop at the first candidate meeting BOTH conditions in BOTH scenarios:

```
20 * approved_B >= n_B
approved_B * n_A < approved_A * n_B
```

The 5% boundary is inclusive. With 3,030 Group B applicants, at least 152 must be
funded; 151 fails. Group B must remain strictly below Group A. Do not assume
monotone approval responses or refine between candidates. Retain all attempts.
An exhausted grid reports `criterion_unmet` without inventing a coefficient or
publishing a successful updated result.

Selection uses bank-side approval counts only, not true risk or true profit.
Freeze selected scores, assessments, cutoffs and decisions before evaluator truth
is numerically loaded. Hash verification alone may read existing evidence bytes
before that boundary. This is an illustrative policy calibration on prediction
cohort approval rates; it is not an untouched confirmatory policy evaluation.

## Reporting and interpretation

Report signed profit/approval changes versus OLS control and the preserved +0.005B
policy. Profit is an outcome, never a selection condition. Group plots use within-
group approval denominators and total true expected profit from funded prediction
loans; companion tables include profit per eligible applicant.

Describe approval-rate parity separately from normative fairness. The unchanged
control/ML comparison has small gaps in Scenario 1, with OLS closer to parity,
and a substantially smaller ML gap in Scenario 2. The existing ML profit
reductions versus OLS are approximately 28.1% and 13.3%; they are material, not
"slightly lower." Show capital use alongside profit. Do not assign a quantified
share of that loss to unused capital or infer a general model ranking.

Deliver four six-bar plots, a new 14-slide editable PowerPoint with matching PDF,
source references/notes and a compact hash-verified evidence snapshot. Slides
9–14 remain explicitly historical. Update current research/status only after
validation. Keep the prior delivery and report unchanged.

## Validation protocol

Test exact 151/152 boundary behavior, strict rate inequality, common coefficient,
first-success stopping, no-success behavior, truth exclusion, unchanged Group A
scores, frozen control/ML identities, prediction-only cohorts, cutoff and budget
arithmetic, group profit aggregation and plotted values. Use independent rational
checks, run new/reused dependency tests and the professor-model suite, then inspect
all 14 slide/PDF pages. The unrelated full suite is not automatically rerun; its
four historical V2 lookup failures remain disclosed and untouched.

## Validated execution and selected coefficient

The study completed on **November 3, 2026**. The first common passing candidate
is **+0.0003125B**, an increase of **0.03125 percentage points** in the raw
per-period default score for Group B, before clipping. Five candidates were
examined; the sixth was not tried. The search stopped without refinement.

| Common delta | Scenario 1 B approvals / rate | Scenario 2 B approvals / rate | Both pass? |
|---:|---:|---:|---|
| 0.005 | 3 / 0.099010% | 1 / 0.033003% | No |
| 0.0025 | 25 / 0.825083% | 11 / 0.363036% | No |
| 0.00125 | 118 / 3.894389% | 52 / 1.716172% | No |
| 0.000625 | 245 / 8.085809% | 107 / 3.531353% | No |
| **0.0003125** | **383 / 12.640264%** | **154 / 5.082508%** | **Yes** |

At 0.000625, Scenario 1 alone passes. Using one common coefficient requires
continuing until Scenario 2 also passes. The result is therefore not a 5% target
in each world: 5% is the inclusive minimum, and Scenario 1 exceeds it. Approval
counts are not required to match across scenarios. No outcome monotonicity was
assumed by the selector.

### Selected penalty results: prediction applicants only

| Quantity | Scenario 1: no systemic differences | Scenario 2: systemic differences |
|---|---:|---:|
| Group A approvals / eligible | 3,450 / 16,970 | 3,811 / 16,970 |
| Group A approval rate | 20.329994% | 22.457278% |
| Group B approvals / eligible | **383 / 3,030** | **154 / 3,030** |
| Group B approval rate | **12.640264%** | **5.082508%** |
| B minus A approval gap | -7.689730 pp | -17.374769 pp |
| Total approvals | 3,833 / 20,000 | 3,965 / 20,000 |
| True expected portfolio profit | $2,366,576,451.06 | $2,365,498,960.02 |
| Selected penalty minus OLS control true profit | **-$23,312.55** | **-$354,515.28** |
| Relative true-profit difference versus control | -0.000985% | -0.014985% |
| Selected penalty minus previous +0.005B true profit | **+$3,710,054.65** | **+$2,628,740.43** |
| Capital utilization | 99.991444% | 99.991721% |

The observed profit loss is much smaller than under +0.005B. In Scenario 1,
large signed changes in group contributions nearly offset: Group A contributes
about $83.110M more and Group B about $83.134M less than control. Scenario 2
has approximately +$42.189M and -$42.543M contributions. These are matched,
post-freeze economic results; a desired profit loss did not determine selection.
A smaller loss does not make the direct penalty nondiscriminatory.

All four control/ML portfolios and their risk diagnostics remain exactly the
November 2 results. Their comparison is separate from the newly calibrated
penalty:

| Risk scenario | OLS control absolute approval gap | ML absolute approval gap | ML true profit minus OLS | ML relative profit difference | ML capital use |
|---|---:|---:|---:|---:|---:|
| No systemic differences | 0.155468 pp | 0.725295 pp | -$665,377,195.32 | -28.115324% | 72.406537% |
| Systemic differences | 14.564860 pp | 0.072794 pp | -$315,708,132.10 | -13.344365% | 87.245197% |

Scenario 1 has small approval gaps for both methods, with OLS closer to parity.
Scenario 2 has substantially more equal ML approval rates. Approval parity is
one descriptive measure, not a general equity verdict. The ML profit differences
are material under the current policy. Indivisible score blocks leave capital
unused, and risk rankings also differ; this study does not isolate how much of
the loss comes from each mechanism. It cannot establish a universal fairness-
profit tradeoff or general superiority of either model.

## Artifacts and reproduction

Repository-relative output namespace: `professor_model/five_percent_penalty/`.
The new runner refuses existing outputs; it never silently overwrites evidence.

| Folder | Files and row meanings |
|---|---|
| `data/processed/professor_model/five_percent_penalty/` | `predictions`, `assessments`, `decisions` as CSV/Parquet: 120,000 rows each, one prediction applicant × scenario × method. `loan_contracts` is the unchanged 20,000-row prediction contract table. Original applicant/inner splits and group audit are retained. Two separately labeled `*_true_profit_EVALUATOR_ONLY` CSV/Parquet tables contain 20,000 prediction rows each. |
| `results/tables/professor_model/five_percent_penalty/` | `penalty_search.csv`: ten candidate × scenario rows; `group_results.csv`: twelve scenario × method × group rows; `portfolio_results.csv` and `method_comparisons.csv`: six policy rows each; `penalty_comparisons.csv` and `previous_penalty_comparisons.csv`: six scenario × scope rows each. Also risk diagnostics, split audit, unchanged parent ML/systemic tables and all selected-policy cutoff candidates. |
| `results/metrics/professor_model/five_percent_penalty/` | Configuration, parent/model identities, preservation inventory, selection/attempt records, pre-truth bank freeze, validation and completed manifest. The manifest binds 43 output files and 243 parent content identities. |
| `results/figures/professor_model/five_percent_penalty/` | Four six-bar plots, each in PNG and vector PDF. Approval plots show exact counts and a 5% reference line; profit plots report total evaluator true expected funded-loan profit by group. |
| `reports/professor_model/five_percent_penalty/` | New 14-slide PowerPoint, matching PDF handout, README and compact hash-verified evidence snapshot. The new slide content and charts are editable; six historical appendix pages preserve source images. |

```bash
# First execution only; refuses existing output files.
python3 experiments/run_professor_five_percent_penalty.py

# Verified read-only inspection: no fits, optimization or writes.
python3 experiments/run_professor_five_percent_penalty.py --summarize-only
```

The recorded scientific run took **27.36 seconds**, excluding tests, independent
audit and presentation work. It performed zero fits, new noise draws or new
populations. All final bank policies were frozen before numerical true-profit
loading. The 80,000/20,000 split and ML's original 64,000/16,000 internal training
split are unchanged. The prediction cohort is held out from model fitting, but
its group approval counts inform this explicitly illustrative policy choice.

## Validation evidence and limits

- New deterministic tests: **73 passed, no warnings**, in 3.74 seconds.
- New/reused dependency tests: **269 passed, no warnings**, in 6.30 seconds.
- Professor-model suite: **720 passed, no warnings**, in 25.51 seconds.
- Independent saved-file audit reproduces selected cutoffs by rational arithmetic
  over **19,161 and 19,100** prefixes, including reject-all, and matches exact
  approval masks and counts.
- Control/ML prediction, assessment and decision frames match the frozen parent
  exactly after removing the successor context fields. Contracts and split files
  are byte-identical. Copied true-profit tables are also byte-identical and contain
  prediction IDs only.
- The first +0.005B attempt exactly reproduces the historical bank selections.
  All ten attempts preserve the budget and independent exact cutoff checks.
- Read-only summary commands for the new, +0.005B and November 2 studies succeed.
  They change no new output bytes or modification times.
- The preservation inventory covers **2,012 pre-existing files**. Scientific
  artifacts and prior deliveries remain unchanged; only the two explicitly
  permitted current guide/status documents are updated.
- Four research figures and all fourteen slides are visually inspected. Native
  package checks verify four editable charts with embedded workbook snapshots,
  two method tables, fourteen speaker-note sections and exact plotted values.
  The PDF is an appearance-matched image-backed handout. No native Microsoft
  PowerPoint application-opening check is claimed.

The unrelated full repository suite was not rerun. Its four previously recorded
V2 checkpoint lookup failures remain unresolved and untouched. No additional
robustness study, tie-policy change, economic-parameter retuning, HMDA work,
commit, push or email was performed. These are single-realization synthetic
results under stylized undiscounted contracts, not empirical lending evidence.
