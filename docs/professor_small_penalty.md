# Final Professor Update: Smaller Direct Penalty

## Predeclared successor

This November 3 protocol implements the professor's final smaller-penalty
request. The successor completed and validated on **November 3, 2026**. The completed
[November 2 study](professor_nov2_followup.md), its +0.01B results, and all
earlier scientific artifacts remain unchanged.

The final demonstration uses Group A (B=0) and minority Group B (B=1).
The bank faces uncertainty: it does not observe true default risk or every
determinant of risk. It observes income, DTI, a noisy credit proxy, and
historical-style noisy risk measurements PN for estimating its models.
These are synthetic measurements, not an empirical mortgage dataset.

Three methods are compared in each of two risk scenarios:

| Method | Role |
|---|---|
| OLS control | Naive linear estimation of the noisy risk measurement |
| OLS + direct penalty | Same fitted control score plus a positive delta times B, explicitly raising minority-group perceived default risk |
| Gradient Boosting | Flexible predictive comparator; superiority is evaluated rather than presumed |

Scenario 1 has equal generating distributions across groups, apart from
sampling variation. In Scenario 2, the existing upstream epsilon shift makes
Group B riskier and updates C consistently. This is a modeled difference from
outside the current bank decision, potentially representing past inequality;
it is not an empirical claim about any demographic group. Keep the existing
Y/D/C features in Scenario 1 and Y/D/C/B in Scenario 2.

## Existing training and prediction separation

The previous study already fits models using **80,000 training applicants**
and makes loans only among **20,000 prediction applicants**. OLS uses the
training rows; Gradient Boosting uses a 64,000/16,000 fitting/validation split
within training and is then refit on all 80,000 training applicants. PN is
the continuous fitting target and is never a prediction feature.

This successor reuses the frozen model artifacts, scores and four control/ML
policies exactly. It does not refit, retune, create another split, sample noise,
or generate applicants. All profit, approval and capital calculations use the
prediction cohort only, with 16,970 Group A and 3,030 Group B applicants.

## Smaller-penalty selection

Predeclare this exact common descending sequence:

```
0.005, 0.0025, 0.00125, 0.000625, 0.0003125, 0.00015625
```

For each candidate, calculate `raw_penalty = raw_control + delta * B` and
then clip to [0,1]. Delta=0.005 means +0.5 percentage points on the raw
default-score scale. Reoptimize only the penalty policy's cutoff in each
world. Preserve the original budget of $1,826,087,220.7466326, contracts,
conditional per-period default accounting, full requests, bounded-PD ordering,
indivisible exact ties and separate reject-all action.

Stop at the first candidate satisfying both worlds' exact integer conditions:

```
approved_B > 0
approved_B * n_A < approved_A * n_B
```

Use one coefficient in both worlds. No minimum percentage beyond positive
approval is imposed. If every candidate fails, preserve all attempts and report
the condition unmet; do not broaden the sequence, use zero, change the risk
model or tune other assumptions. A profit loss is not a selection criterion.

Selection uses bank-side group approval counts, not evaluator true risk or
true profit. Freeze the chosen scores, perceived assessments, cutoffs and
decisions before evaluator-only true-profit files are loaded. Selecting delta
on the prediction cohort's group approval rates is an **illustrative policy
calibration** requested by the professor. Model fitting remains held out, but
the prediction cohort is not untouched for this policy-design choice. Do not
describe the selected pattern as an unbiased confirmatory discovery.

## Planned deliverables and checks

Use the new `small_penalty` namespace for the selection log, bank/evaluator
CSV/Parquet files, six-policy/twelve-group tables, four plots, provenance and
the new 14-slide PowerPoint/PDF. Reference the parent fitted models by exact
hash instead of refitting them. Preserve prior reports and presentation files.

Approval plots use each group's full prediction cohort; profit plots sum true
expected bank profit contributed by funded loans. Include normalized profit
values in companion tables. Show sufficiently precise positive rates and
actual funded counts so a tiny nonzero bar is not labeled zero. The new deck
explicitly states **training 80,000 / prediction 20,000 / profits only on the
prediction subset**. Its six historical appendix pages remain clearly labeled.

Validate exact treatment/clipping, invariant Group A raw scores, frozen
control/ML identity, prediction-only IDs, deterministic first-success stopping,
common delta, explicit no-success behavior, no truth-driven selection, exact
budget/tie/cutoff arithmetic, group aggregation and artifact preservation.
Run new tests, reused dependencies and the professor-model suite. The four
previously recorded V2 checkpoint lookup failures remain a separate known
issue; no full unrelated scientific rerun is planned.

## Execution and results

The first candidate, **delta=0.005**, meets the professor's positive-but-lower
approval condition in **both** scenarios. No smaller candidate was evaluated.
The chosen increment is +0.5 percentage points before clipping. Only the two
new penalty policies were optimized; the four control/ML policies were reused.

| Prediction result | Scenario 1: no systemic differences | Scenario 2: systemic differences |
|---|---:|---:|
| Common delta | 0.005 | 0.005 |
| Group A funded / eligible | 4,251 / 16,970 | 4,265 / 16,970 |
| Group A approval rate | 25.050088% | 25.132587% |
| Minority Group B funded / eligible | **3 / 3,030** | **1 / 3,030** |
| Minority Group B approval rate | **0.099010%** | **0.033003%** |
| B minus A approval gap | -24.951078 pp | -25.099584 pp |
| True expected bank profit from funded Group B loans | $7.410M | $2.897M |
| True expected portfolio profit | $2,362.866M | $2,362.870M |
| Penalty minus control true expected profit | **-$3,733,367.21** | **-$2,983,255.71** |

The result satisfies the exact requested condition, but minority approval is
still **very low**. The presentation therefore shows **0.10% (3 loans)** and
**0.03% (1 loan)**, rather than describing the penalty as mild or implying a
substantial minority approval rate. No extra minimum rate was imposed after
looking at results, and no additional penalty was tried for visual effect.

All prediction requests remain truly profitable under the stylized contract.
In Scenario 1, 577 Group B applicants funded by control lose funding, while
1,029 Group A applicants gain funding. In Scenario 2, the corresponding
counts are 216 and 578. The true profit of lost requests is offset partly by
the profit of replacement requests; the table reports the resulting signed
portfolio difference. These are expected synthetic values, not realized
bank losses. Group A raw scores are unchanged, but its allocations can change
through competition for the shared budget.

OLS control and Gradient Boosting results are exactly the parent results.
In particular, ML still uses 72.4065%/87.2452% of available capital because
the next indivisible prediction block is infeasible. It is a flexible model
comparison, not an assumed performance improvement. The scenario-specific
information sets and modest systemic true-risk gap also remain unchanged.

### Training and prediction audit

The saved split and fitted model metadata independently confirm 80,000
training and 20,000 prediction IDs, with zero overlap. Gradient Boosting's
64,000/16,000 internal split and final 80,000-row refit were verified, not
performed again. The source dataset calls the prediction split `evaluation`;
this successor retains those original labels and IDs.

Every contract, decision, true-profit row and group denominator belongs to
the same 20,000 prediction applicants. No training-cohort profits are counted.
Fitting remains held out; selecting delta from prediction-cohort approval
rates is the disclosed illustrative policy calibration above.

### Files and reproduction

Paths are relative to the repository root. Main dataset tables have CSV and
Parquet companions; summaries are CSV. All new outputs are immutable.

| Location | Contents |
|---|---|
| `configs/professor_model/small_penalty.yaml` | Frozen candidate sequence, selection rule, cohort sizes and economic-policy assumptions |
| `data/processed/professor_model/small_penalty/` | Context-labeled predictions, perceived assessments, decisions, original contracts/splits/group audit, and separately labeled evaluator-only true-profit tables |
| `results/tables/professor_model/small_penalty/` | `penalty_search.csv` (two rows: first candidate in each world), `split_audit.csv`, `portfolio_results.csv` (six), `group_results.csv` (twelve), risk/displacement tables, selected cutoff candidates and unchanged parent ML/systemic audit tables |
| `results/metrics/professor_model/small_penalty/` | Resolved config, parent/model reuse bindings, attempted-candidate records, `selection.json`, pre-truth bank freeze, validation and completed run manifest |
| `results/figures/professor_model/small_penalty/` | Four six-bar plots, each as PNG and vector PDF; counts and positive-rate precision are explicit |
| `reports/professor_model/small_penalty/` | New 14-slide editable PowerPoint, matching PDF handout, and compact hash-verified evidence snapshot; earlier deliveries remain unchanged |

```bash
# First execution only: refuses existing output files.
python3 experiments/run_professor_small_penalty.py

# Read and verify existing evidence; starts no fitting, optimization or writes.
python3 experiments/run_professor_small_penalty.py --summarize-only
```

The completed manifest binds 41 output files, 196 parent content identities,
source/configuration contents, dependencies, Git revision and dirty-worktree
state. It records zero new model fits, noise draws and populations. Numeric
evaluator true profit is loaded only after the chosen bank artifacts are
frozen. The recorded run took about nine seconds, excluding tests, independent
audits and presentation production.

### Validation evidence

- New toy-fixture tests: **63 passed, no warnings**, in 3.59 seconds.
- New and relevant dependency checks: **297 passed, no warnings**, in 7.14 seconds.
- Fresh professor-model suite: **647 passed, no warnings**, in 21.57 seconds.
- Independent saved-file comparison reproduces all four frozen control/ML
  score and decision frames exactly after removing successor context fields.
- The two copied evaluator true-profit files exactly match the parent and
  contain only prediction IDs. The selected raw-score adjustment and clipping
  replay exactly in both scenarios, with unchanged Group A raw scores.
- All exact cutoff/budget checks pass, including independent rational
  enumeration. Approval success uses integer cross-products, not rounded rates.
- Summary-only replay changes no saved bytes or modification times and records
  zero writes, computations or fits.
- The preservation inventory contains **1,933 pre-existing files**; all prior
  scientific files remain unchanged. Only the explicitly allowed current
  research guide and project status are updated after validation.

The unrelated full repository suite was not rerun. Its previously recorded
four V2 checkpoint lookup failures remain unresolved and were not modified or
suppressed. These 647 passing tests describe the professor-model family only.
No additional robustness study, alternative tie policy, HMDA analysis,
economic-parameter retuning, commit, push or email was performed.
