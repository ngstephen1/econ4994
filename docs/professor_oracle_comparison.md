# Professor Model — Step 7A True-Risk Oracle Benchmark

Stage record: results, test counts, and later-work deferrals below describe
this step. See [current status](PROJECT_STATUS.md) and the
[documentation index](README.md) for the latest state across all three families.

## Purpose and information boundary

This evaluator-only comparison asks what happens when the common-cutoff lender
knows true default probabilities rather than the frozen bank's estimated
probabilities. The oracle knows **probabilities, not future realized defaults**.
No default outcomes are generated. It is not a fitted model or automatically
a fair policy.

The existing 20,000 evaluation applicants, contract terms, true-risk values,
bank estimates, profit calculations, split assignments, and Step 6B decisions
are reused unchanged. Original manifests, IDs, schemas, input/output hashes,
and source/configuration fingerprints are verified before use.

The bank's saved cutoff remains **0.0009333057709409261**. Its allocation is
read from `lending_decisions.parquet`; no bank fit or search is invoked in
the Step 7A workflow. The bank-facing tables still exclude hidden truth.

## Same accounting and exact saved capital

The oracle uses exactly Step 6A's working contract: T=30, theta=0.10,
r=0.05 **per period**, ell=theta*Y, L=T*ell, and scheduled receipts
`ell*(1+r)^t`. Expected receipts are weighted by `(1-true_PD)^t`, interpreting
PD as constant conditional per-period default. There are no costs,
discounting, prepayment, or recovery. Saved expected profits are not changed.

Capital is read from the frozen Step 6B selection JSON, not recalculated
from rounded tables or hard-coded into the implementation:

```text
A = 1,826,087,220.7466326
```

The oracle optimization is:

```text
d_i(tau_oracle) = 1{true_PD_i <= tau_oracle}

maximize_tau_oracle sum_i d_i * expected_profit_true_i
subject to sum_i d_i * L_i <= A
```

All requests are fully funded or rejected. Equal stored probabilities are
indivisible. Reject-all is included, represented by a null cutoff if chosen.
Equal maximum objectives choose the lowest cutoff, with reject-all first.

The unchanged Step 6B `choose_cutoff` implementation supplies the exact
binary-float integer-prefix algorithm. An adapter inside `oracle.py` maps
truth risk/objective into the legacy search schema **only inside the evaluator**.
Outputs are immediately relabeled as true-risk oracle results; the internal
bank-shaped view is neither persisted nor passed to a bank learning workflow.
No old source code or bank validation policy was changed to permit leakage.

## Important limit of the regret interpretation

Both policies use a common-cutoff *form*, identical applicants/contracts, and
the same capital budget. However, cutoffs on different scores generally
produce **different feasible portfolio families**. A bank-ranked prefix need
not be attainable by any true-PD cutoff. Consequently, nonnegative oracle
minus bank profit is **not a universal mathematical guarantee** for this
restricted benchmark. It would require an appropriate common feasible set or
additional assumptions; unrestricted allocation is deliberately not added.

We retain the requested name `economic_regret` for the signed difference:

```text
economic_regret = oracle true expected profit - bank true expected profit
regret_percent = 100 * economic_regret / oracle true expected profit
```

The result is never clipped to zero or used to select a different bank policy.
If oracle profit is zero, the percentage is explicitly undefined. A deterministic
test demonstrates a negative comparison when an indivisible low-risk tie
blocks every oracle loan but the bank can issue a profitable subset under its
own ordering. The actual canonical comparison below is positive, as hoped,
but should not be presented as a general value-of-information theorem or
unrestricted-optimality bound.

## Actual policy comparison

| Quantity | Bank, frozen | True-risk oracle |
|---|---:|---:|
| Cutoff score | bank_PD_bounded | true_PD |
| Exact cutoff | 0.0009333057709409261 | **0.0008343666409349597** |
| Cutoff in percent | 0.0933305771% | 0.0834366641% |
| Approved | 3,900 | **3,406** |
| Rejected | 16,100 | 16,594 |
| Approval rate | 19.50% | **17.03%** |
| Funded principal | $1,825,953,398.73 | **$1,825,849,955.62** |
| Capital utilization | 99.99267165% | 99.98700691% |
| Unused capital | $133,822.01 | $237,265.12 |
| True expected portfolio profit | $2,366,074,476.85 | **$2,370,206,703.87** |
| Mean true PD among approved | 0.07939238% | **0.06661497%** |
| Mean true profit per approved applicant | $606,685.76 | $695,891.57 |

The oracle-minus-bank true-profit difference is **+$4,132,227.01877594**,
or **0.1743403650% of oracle profit**. The bank's Step 6B perceived profit of
$2,393,797,751.16 is not substituted for its true profit in this subtraction.

The oracle evaluates **20,001 candidates including reject-all**, with **3,407
feasible candidates**. There are no exact true-PD ties in this saved cohort.
All true profits are positive and the selected oracle cutoff is the largest
feasible endpoint. The next endpoint would require $1,826,211,105.53, above
the original capital; no partial funding or tie splitting fills the remainder.
Independent `fractions.Fraction` enumeration reproduced the exact cutoff and
objective, and verified both budgets against exact sums of saved amounts.

## Allocation overlap and disagreement

| Allocation group | Applicants | Total true expected profit of requests | Mean true PD |
|---|---:|---:|---:|
| Funded by both | **2,516** | $1,906,017,990.90 | 0.06367584% |
| Bank only | **1,384** | $460,056,485.95 | **0.10796378%** |
| Oracle only | **890** | $464,188,712.97 | **0.07492379%** |
| Rejected by both | **15,210** | $3,013,500,215.97 | 0.16324481% |

Jaccard overlap of funded sets is **0.525260960334**, or **52.5261%**:
2,516 / (2,516 + 1,384 + 890). The last row reports hypothetical per-request
true profits, not profits earned by either portfolio. The difference between
oracle-only and bank-only profit reconciles the aggregate regret up to
floating-point reporting precision.

The bank selects higher-risk borrowers on average than the oracle in this
sample, both across approved sets (0.07939238% versus 0.06661497%) and across
the disagreement groups. The oracle approves fewer applicants but achieves
slightly higher true profit with nearly identical funded capital. These are
observed results, not assumptions used to tune the policies.

## Group audit

Approval-rate denominators are all evaluation applicants in each group:
16,970 for B=0 and 3,030 for B=1. Funded principal per applicant includes
rejections as zero, not just issued loans. Risk means use approved borrowers.

| Policy | B | Approved | Approval rate | Funded principal per applicant | Mean true PD among approved |
|---|---:|---:|---:|---:|---:|
| Bank, frozen | 0 | 3,679 | **21.679434%** | $98,344.57 | 0.08075723% |
| Bank, frozen | 1 | 221 | **7.293729%** | $51,830.39 | 0.05667165% |
| Oracle | 0 | 2,871 | **16.918091%** | $90,368.16 | 0.06667713% |
| Oracle | 1 | 535 | **17.656766%** | $96,469.38 | 0.06628141% |

- Bank B=1 minus B=0 approval gap: **-14.38570492 percentage points**.
- Oracle B=1 minus B=0 approval gap: **+0.73867493 percentage points**.
- Bank gap minus oracle gap: **-15.12437985 percentage points**.

This is a descriptive matched-policy comparison. The oracle is not defined
to equalize group outcomes, and a smaller gap does not establish fairness.
The gap difference is not an identified causal effect of discrimination.
B is in the bank's fitted regression but absent from the true-risk DGP;
finite-sample estimation, score ordering, and budget composition all matter.

## Files and information roles

Every new artifact is evaluator-only. One oracle-decision row represents one
of the same 20,000 evaluation applicants, never a new population.

| Repository-relative output | Contents |
|---|---|
| `data/processed/professor_model/oracle_lending_decisions.csv` | Exact fields: `information_scope, applicant_id, policy_id, true_PD, optimal_tau, approved, requested_principal, funded_principal, decision_status`. |
| `data/processed/professor_model/oracle_lending_decisions.parquet` | Identical decisions and full-precision values/types. |
| `results/tables/professor_model/bank_vs_oracle_summary.csv` | Two policy rows: cutoff, funding, true value, approved risk, regret and overlap. Repeated comparison-level fields describe this one matched comparison. |
| `results/tables/professor_model/bank_vs_oracle_group_audit.csv` | Four policy/group rows with explicit applicant denominators. |
| `results/tables/professor_model/bank_vs_oracle_disagreements.csv` | Four exhaustive allocation groups: counts, shares, principal, true profit, mean true PD. |
| `results/tables/professor_model/oracle_threshold_candidates.csv` | All oracle cutoff endpoints, including reject-all, exact-search feasibility and selected flag. |
| `results/metrics/professor_model/oracle_comparison_validation.json` | Exact oracle selection, comparison/gap metrics, structural checks, interpretation cautions. |
| `data/processed/professor_model/oracle_comparison_metadata.json` | Original/output checksums, resolved YAML, actual source/config fingerprints, dependencies, saved budget/bank cutoff, and no-refit/no-bank-reoptimization flags. |
| `results/figures/professor_model/bank_vs_oracle_profit.png` | True expected portfolio profits, zero-based USD-billions axis. |
| `results/figures/professor_model/bank_vs_oracle_group_approval.png` | Group approval percentages under both policies. |

Both figures were rendered and visually inspected for alignment, readable
units, and clipping. CSV/Parquet decisions agree exactly using round-trip
float parsing. No hidden truth is written into any old bank artifact.

New implementation files are `configs/professor_model/oracle_comparison.yaml`,
`src/fair_lending/professor_model/oracle.py`, `oracle_io.py`,
`experiments/run_professor_oracle_comparison.py`, and
`tests/test_professor_oracle.py`. Existing scientific code is unchanged.

```bash
python3 experiments/run_professor_oracle_comparison.py
# Explicitly replace only Step 7A outputs after a previous run:
python3 experiments/run_professor_oracle_comparison.py --overwrite
python3 -m pytest -q tests/test_professor_oracle.py
python3 -m pytest -q tests/test_professor_population.py tests/test_professor_true_risk.py tests/test_professor_noisy_signal.py tests/test_professor_bank_regression.py tests/test_professor_loan_economics.py tests/test_professor_lending_threshold.py tests/test_professor_oracle.py
python3 -m pytest -q
```

The **21 new tests** cover exact cutoff-search reuse, independent toy
enumeration, true-PD ordering, indivisible ties, reject-all, original bank
allocation and absolute budget, no refitting/bank reoptimization, signed regret
and percentage arithmetic, overlap/gap calculations, group denominators,
tampered-bank rejection, evaluator labeling, and export/checksum preservation.
All **250 professor-model tests passed without warnings**.

The complete repository suite finished with **530 passed, 4 failed, and 32
existing deprecation warnings** in 104.41 seconds. The same four pre-existing
failures remain in `tests/test_systemic_corrected_checkpoint.py`: exact-cell
coverage, replication-13 provenance, legacy-comparison isolation, and the
five-record interim summary. The loader's current-HEAD (`7a7e590`) run IDs
do not locate the older saved-record filenames (`8a9f71f`). No unrelated
recovery code or saved records were altered. The warnings are existing
scikit-learn/SciPy L-BFGS-B `disp`/`iprint` deprecations: 26 ML benchmark and
six sensitivity warnings. There are no new Step 7A failures.

All **89 pre-step data/source/configuration/result snapshot files** and **94
registered original artifacts** remain byte-identical, including the original
bank allocation and exact cutoff. All nine new output checksums and the new
source fingerprints were independently verified. No old scientific module or
configuration was edited; only project-status documentation was updated.

## Limits and recommended Step 7B

This remains one synthetic, researcher-defined population under stylized
growing payments and no costs/discounting. Large nominal profit levels are not
empirical mortgage profitability. The benchmark knows risks, not future events,
and optimizes only true-risk thresholds, not all possible feasible subsets.
No new model, group-blind fit, intervention, pricing change, Monte Carlo,
unrestricted knapsack, or realized outcomes were added.

Recommended next: review the score-dependent feasible-set caveat and results
with the professor, then separately specify a matched **group-blind OLS
sensitivity** if that is the Step 7B research question. Keep the current bank,
oracle, split, contracts, and absolute budget frozen as references. No such
comparison is implemented or run automatically here.
