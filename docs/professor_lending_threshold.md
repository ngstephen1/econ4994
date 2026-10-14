# Professor Model — Step 6B Profit-Maximizing Common Cutoff

Stage record: results, test counts, and later-work deferrals below describe
this step. See [current status](PROJECT_STATUS.md) and the
[documentation index](README.md) for the latest state across all three families.

## Research question and preserved inputs

Given the bank's frozen estimated risks and nominal expected profits, which
**one common risk cutoff** maximizes perceived portfolio profit under a 40%
capital budget? How do the resulting decisions differ across synthetic groups,
and how does their perceived profit compare with evaluator-only true profit?

This step uses the saved 20,000 evaluation contracts and bank-profit rows from
Step 6A. Their ID sets exactly match the evaluation IDs in the authoritative
80,000/20,000 split. No applicants, predictions, contracts, rates, loan amounts,
or expected-profit values were regenerated, refitted, or changed.

## Decision rule, budget, and objective

```text
d_i(tau) = 1{bank_PD_bounded_i <= tau}
funded_principal_i = d_i(tau) * requested_principal_i

total_requested_principal = 4,565,218,051.866581
A = 0.40 * total_requested_principal
  = 1,826,087,220.7466326

maximize_tau sum_i d_i(tau) * expected_profit_bank_i
subject to sum_i d_i(tau) * requested_principal_i <= A
```

The budget is computed from the unrounded saved amounts using
`0.40 * math.fsum(requested_principal)`. The numbers above are the full Python
float representations used, not rounded-dollar inputs. No numerical slack,
budget guard, or cent-rounding is introduced. Requests are fully funded or
rejected; the policy cannot skip a lower-risk applicant to fund somebody else.

Step 6B explicitly uses **bounded** PD operationally. This supersedes Step 1's
initial raw-score proposal; the raw predictions themselves remain unchanged
and available for diagnostics. The bank does not receive true PD or true profit
when selecting its cutoff.

## Complete candidate search and tie handling

1. Sort by bounded PD and then applicant ID for deterministic processing.
2. Treat exactly equal stored risks as one indivisible group. No approximate
   grouping, rounding, tie splitting, or budget-filling rationing is allowed.
3. Accumulate principal and perceived profit at each group endpoint.
4. Evaluate every distinct cutoff, including infeasible candidates for audit.
5. Include a separate reject-all candidate with profit/principal zero.
6. Select the feasible candidate with the highest exact perceived-profit sum.
   Equal objectives select the lowest cutoff; reject-all precedes finite cutoffs
   and wins a tie at zero profit.

All finite input floats are exact integers divided by powers of two.
`float.as_integer_ratio()` supplies those representations; aligned integer
prefix sums make budget, objective, and tie comparisons exact **for the saved
binary floating-point inputs**. Exported totals are their nearest floating-point
representations. This numerical technique changes neither the feasible set nor
the objective. Complexity is O(n log n) sorting plus O(n) accumulation, not
unrestricted knapsack optimization.

Reject-all is represented by null `optimal_tau` in JSON/Parquet and an empty
CSV value, equivalent to an extended cutoff of minus infinity. It is not tau=0:
tau=0 funds every applicant whose bounded score is zero. Each finite observed
cutoff represents all real cutoffs up to, but excluding, the next risk value
that produce the same portfolio.

## Observed optimal bank decision

| Quantity | Result |
|---|---:|
| Optimal tau, raw probability units | **0.0009333057709409261** |
| Optimal tau, percent | **0.0933305771%** |
| Approved / rejected | **3,900 / 16,100** |
| Approval rate | **19.5%** |
| Funded principal | **$1,825,953,398.73** |
| Available capital | **$1,826,087,220.75** |
| Capital utilization | **99.99267165%** |
| Unused capital | **$133,822.01** |
| Bank-expected portfolio profit | **$2,393,797,751.16** |
| Mean bank-expected profit per approved request | **$613,794.30** |
| Mean principal per approved request | **$468,193.18** |

There are **19,079 candidates including reject-all**, of which **2,979 are
feasible**. All requests have positive bank-expected profit, and the selected
candidate is indeed the largest feasible cutoff; neither result is hard-coded.
An independent rational-arithmetic (`fractions.Fraction`) enumeration of all
candidate endpoints reproduced the exact same cutoff and objective.

### Is capital binding?

The budget is **not exhausted exactly**: $133,822.01 remains. But it constrains
the next admissible policy expansion. The next cutoff,
`0.0009337470481588183`, must add one full request of **$299,569.90**. Its total
principal would be **$1,826,252,968.63**, exceeding capital by **$165,747.89**.
Thus capital blocks further expansion in the discrete cutoff family. Do not
describe this as zero slack or silently spend the remainder on a higher-risk
request outside the common-cutoff rule.

The bank funds 40% of requested capital but only 19.5% of applicants because
request sizes differ: approved requests average $468,193.18, versus the full
evaluation mean of $228,260.90. This is not partial financing.

### Exact ties and the clipped-zero scores

There is one tied-risk group: **923 applicants at bounded PD=0**. All 923 are
approved together, consuming **$709,224,665.32** of capital. The selected
cutoff itself has only one applicant, so no tie straddles the chosen boundary.
These ties constrain the allowable portfolios and are preserved; this step
does not run a counterfactual raw-score policy to quantify their causal effect.

## Risk distributions among approved/rejected

The table displays **percent**, not raw probability units. CSV/JSON fields
retain raw probability units; SD is a sample SD.

| Operational bank PD | Approved | Rejected |
|---|---:|---:|
| Applicants | 3,900 | 16,100 |
| Mean | 0.044471% | 0.177464% |
| Median | 0.050666% | 0.172690% |
| Sample SD, percentage points | 0.033185 | 0.049748 |
| Minimum | 0% | 0.093375% |
| p05 | 0% | 0.105203% |
| p95 | 0.089519% | 0.267385% |
| Maximum | 0.093331% | 0.410498% |

## Post-decision hidden-truth evaluation

The bank cutoff, all 20,000 decisions, all candidate rows, and the bank-only
portfolio summary are saved and checksummed **before**
`true_expected_profit.parquet` is loaded. The truth loader requires this
freeze checkpoint and rejects altered bank artifacts. Protected group and
raw-score audit values are joined after the freeze as well. Evaluator results
never overwrite or reselect the bank policy.

| Evaluator quantity | Result |
|---|---:|
| Bank-expected portfolio profit | $2,393,797,751.16 |
| True expected portfolio profit | $2,366,074,476.85 |
| Estimation error, bank minus true | **+$27,723,274.31** |

The bank overestimates the value of this chosen portfolio. This difference is
**not regret**: no oracle policy or alternative allocation benchmark is fitted
or optimized in this step. Both totals are expected nominal profits under the
Step 6A working contract, not realized profits.

## Group audit — evaluator only

| B | Applicants | Approved | Approval rate | Funded principal | Funded principal per applicant |
|---:|---:|---:|---:|---:|---:|
| 0 | 16,970 | 3,679 | 21.679434% | $1,668,907,329.24 | $98,344.57 |
| 1 | 3,030 | 221 | 7.293729% | $157,046,069.49 | $51,830.39 |

Per-applicant funded principal uses **all applicants in that group**, counting
rejections as zero, not only borrowers receiving a loan.

| B | Mean bank PD among approved | Mean true PD among approved | Total true expected profit |
|---:|---:|---:|---:|
| 0 | 0.044922% | 0.080757% | $2,161,524,721.40 |
| 1 | 0.036964% | 0.056672% | $204,549,755.45 |

- B=1 minus B=0 approval gap: **-14.38570492 percentage points**.
- Primary normalized funded-principal gap: **-$46,514.18 per applicant**.
- Difference in total funded principal: **-$1,511,861,259.75**, which also
  reflects the substantially different group sizes.

B is intentionally included in the professor's fitted OLS model; it is absent
from the simulator's true-risk equation. These single-population group gaps
do not establish intentional discrimination or empirical mortgage-lending
effects. No discrimination intervention was introduced.

## Files and reproduction

| Exact repository-relative output | Contents / role |
|---|---|
| `data/processed/professor_model/lending_decisions.csv` | 20,000 rows, exactly `applicant_id, model_id, bank_PD_bounded, optimal_tau, approved, requested_principal, funded_principal, decision_status`; **no hidden truth**. |
| `data/processed/professor_model/lending_decisions.parquet` | Same decisions and full-precision values; boolean approved and complete requests. |
| `results/tables/professor_model/lending_portfolio_summary.csv` | One frozen bank-only summary, including risk distributions and capital diagnostics. |
| `results/tables/professor_model/lending_threshold_candidates.csv` | All candidates: tau, approved count, funded principal, feasibility, bank profit, group size, candidate type/ID, selected flag. |
| `results/tables/professor_model/lending_group_audit.csv` | Two evaluator-only group rows. |
| `results/tables/professor_model/lending_evaluator_summary.csv` | Separate evaluator-only true value, estimation error, approval/funding gaps. |
| `results/metrics/professor_model/lending_selection.json` | Frozen bank selection, input allowlist, tie/numerical policy, configuration fingerprint and checks. |
| `results/metrics/professor_model/lending_validation.json` | Decision, optimality, lifecycle, export, preservation, and evaluator results. |
| `data/processed/professor_model/lending_metadata.json` | Evaluator provenance, resolved YAML, source/configuration hashes, dependencies, cohort fingerprint, input/output checksums, freeze checkpoint. |
| `results/figures/professor_model/lending_profit_vs_cutoff.png` | Bank profit versus cutoff; feasible candidates shaded/green, infeasible grey, chosen cutoff and separate reject-all marked. |

The figure was rendered and visually inspected for legibility and clipping.
Generated files follow the existing Git-ignore policy.

Reusable code lives in `src/fair_lending/professor_model/lending.py` and
`lending_io.py`; the YAML is `configs/professor_model/lending_threshold.yaml`.
Orchestration is separate:

```bash
python3 experiments/run_professor_lending_threshold.py
# Explicitly replace only Step 6B outputs after a previous run:
python3 experiments/run_professor_lending_threshold.py --overwrite
python3 -m pytest -q tests/test_professor_lending_threshold.py
python3 -m pytest -q tests/test_professor_population.py tests/test_professor_true_risk.py tests/test_professor_noisy_signal.py tests/test_professor_bank_regression.py tests/test_professor_loan_economics.py tests/test_professor_lending_threshold.py
python3 -m pytest -q
```

The 37 new deterministic tests cover exhaustive cutoff enumeration, exact
budget/one-ULP boundaries, tiny objective differences, ties, no skipping,
negative-profit prefixes, reject-all, deterministic tie-breaking and row-order
invariance, strict observable-only inputs, saved-before-truth gating, tampered
records, unchanged originals, group/error arithmetic, and CSV/Parquet equality.
All **229 professor-model tests passed without warnings**.

The complete repository suite finished with **509 passed, 4 failed, and 32
existing deprecation warnings** in 122.64 seconds. There are no new Step 6B
failures. The four pre-existing failures remain in
`tests/test_systemic_corrected_checkpoint.py`: exact-cell coverage,
replication-13 provenance, legacy-comparison isolation, and the five-record
interim summary. The loader's current-HEAD run IDs (`7a7e590`) do not locate
the older saved-record filenames (`8a9f71f`). No recovery code or saved records
were altered to bypass this. The warnings are the existing scikit-learn/SciPy
L-BFGS-B `disp`/`iprint` deprecations (26 ML benchmark, six sensitivity).

All **76 pre-step snapshot data/source/configuration/result files** and
**80 registered original artifacts** remain byte-identical. No new random
draws, refitting, new risk model, loan pricing, partial financing, default
outcomes, unrestricted allocation, ML comparison, Monte Carlo, or Streamlit
changes were added. Original Step 6A and earlier scientific code is unchanged.

## Limitations and stopping point

The large positive profit levels inherit the stylized growing-payment,
per-period-rate, no-cost/no-discount assumptions, not realistic mortgage
calibration. The bank still uses noisy, uncalibrated OLS predictions clipped
at zero. Choosing one common score threshold deliberately restricts the policy
family; it is not a claim of unrestricted portfolio optimality. Only a single
synthetic population is analyzed. Step 6B stops after this validated common
cutoff and audit; no oracle comparison or next-stage experiment runs here.
