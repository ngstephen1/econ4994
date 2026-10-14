# Professor Model — Step 6A Loan Contracts and Expected Profit

Stage record: results, test counts, and later-work deferrals below describe
this step. See [current status](PROJECT_STATUS.md) and the
[documentation index](README.md) for the latest state across all three families.

Step 6A values one potential loan request for each of the **20,000 existing
held-out evaluation applicants**. It does not choose which loans to issue.
No applicants, true-risk values, PN measurements, split assignments, or fitted
predictions were regenerated or changed. No bank regression was refitted.

## Working assumptions and event definition

These are the explicitly requested Step 6A working interpretations of the
professor-inspired model, not independently verified transcriptions of the
whiteboard or calibrated mortgage-market facts:

- T = 30 periods, theta = 0.10, and **r = 0.05 per period**.
- Period calendar duration is not specified. Step 1's annual-rate suggestion
  is not the convention used here. No annual-to-period conversion is inferred.
- Saved PD is used as a constant **conditional per-period default probability**:
  the probability of default before the current payment, conditional on having
  reached that period without default. Default is absorbing, so receipt of
  payment t requires surviving all periods 1,...,t.
- For bank accounting, p = the existing `bank_PD_bounded`; for evaluator
  accounting, p = the saved `true_PD`. Neither is re-estimated or rescaled.
- There are no transaction/funding costs, discounting, prepayment, or recovery.

The event interpretation is new accounting context, not a change to the saved
true-risk equation or files. PD is not treated as whole-loan default risk.
The probability of receiving every scheduled payment under this working
interpretation would be (1-p)^30, not 1-p. No realized repayment histories or
binary default events are simulated in this step.

## Exact contract and accounting equations

```text
ell_i = theta * Y_i = 0.10 * Y_i
L_i = T * ell_i = 30 * ell_i = 3 * Y_i

scheduled_payment_it = ell_i * (1 + r)^t,   t = 1,...,30
P(receive payment t | information) = (1 - p_i)^t

expected_receipts_i(p_i)
  = sum[t=1..30] (1-p_i)^t * ell_i * (1+r)^t
  = ell_i * sum[t=1..30] [(1-p_i)(1+r)]^t

expected_profit_i(p_i) = expected_receipts_i(p_i) - L_i

bank:      p_i = bank_PD_bounded_i
evaluator: p_i = true_PD_i
```

Principal is subtracted exactly once. Payments begin at t=1, not t=0.
The bank entry point does not accept `true_PD`, true profit, or raw risk scores.
The existing raw prediction remains in its original artifact and in the
evaluator preview; it is never inserted into the survival equation.

This is **not conventional mortgage amortization**: payments grow
geometrically rather than being derived from an amortizing outstanding
balance. Annual income is used numerically to set the base amount ell; no
monthly affordability test or income evolution is imposed. L is a requested
principal, not a funded amount. The calculations describe potential nominal
receipts/profit if a request were issued, not realized cash flows or a portfolio.

## Break-even and hand-checkable validation

Let q = (1-p)(1+r). For positive ell and T, break-even requires
sum(q^t) = T. Since q is nonnegative, this holds at q=1:

```text
p_break_even = r / (1+r)
             = 0.05 / 1.05
             = 0.047619047619047616
```

This is **4.76190476% conditional default per period**. It is an accounting
benchmark, not a lending threshold selected or applied by this implementation.
For fixed income/terms, profit decreases with p; p below this benchmark implies
positive profit, and p above it implies negative profit.

With Y=$1,000, ell=$100, T=30, L=$3,000, and r=0.05:

| Default probability | Expected receipts | Expected profit |
|---|---:|---:|
| 0 | $6,976.0789878164 | $3,976.0789878164 |
| 1 | $0 | -$3,000 |
| 0.05/1.05 | $3,000 | Approximately $0 |

At r=0 and p=0, receipts equal principal and profit is zero.
A separate two-period fixture has scheduled payments $105 and $110.25,
principal $200, and p=0.1. Its receipts are
`105*0.9 + 110.25*0.81 = 183.8025`, with profit **-$16.1975**.

The implementation uses the direct finite-period sum, avoiding division by
q-1 in a geometric-series shortcut. Across all 20,000 actual contracts, the
largest absolute break-even residual was **$1.862645149e-9**, within the
declared tolerance of `1e-12 * principal`. Zero-rate/zero-default, certain
default, all-payments, arithmetic, finite values, and nonnegative receipts
also passed. An independent scalar `math.fsum` calculation agreed for every
bank/true receipt, with maximum absolute differences below $1.5e-8.

## Observed evaluation-population results

All monetary values are synthetic USD; no rounding is applied to stored data.
SDs below are sample SDs (ddof=1). Summaries refer only to the evaluation set.

| Quantity | Mean | Median |
|---|---:|---:|
| Annual income Y | $76,086.97 | $59,698.90 |
| Base amount ell | $7,608.70 | $5,969.89 |
| Requested principal L | $228,260.90 | $179,096.71 |
| Bank expected receipts | $519,677.01 | $403,347.48 |
| True expected receipts | $520,449.07 | $406,396.49 |
| Bank expected profit | $291,416.11 | $224,126.34 |
| True expected profit | $292,188.17 | $227,350.95 |

Total requested principal is **$4,565,218,051.87**. No budget or funded
portfolio is constructed from that total.

| Distribution | Sample SD | Minimum | p05 | p95 | Maximum |
|---|---:|---:|---:|---:|---:|
| Principal L | $183,724.37 | $9,803.61 | $55,971.98 | $567,632.80 | $4,443,878.16 |
| Bank expected profit | $243,511.38 | $12,131.30 | $69,072.98 | $747,139.91 | $5,889,736.86 |
| True expected profit | $241,005.38 | $9,579.24 | $67,549.41 | $736,839.28 | $5,858,006.32 |

**100% of requests are profitable under both bank and true risk**, with **0%
profit-sign disagreement**. This was observed, not imposed as a validation
target. The evaluation bank PD range is 0–0.0041049801 and true PD range is
0.0001616307–0.0086220872; both lie below the break-even probability.
No parameters were retuned to create unprofitable requests.

Define error as `expected_profit_bank - expected_profit_true`:

- Mean error: **-$772.06** per request.
- MAE: **$4,400.06**.
- RMSE: **$6,371.84**.

## Group audit — evaluator only

| Scope | Applicants | Mean Y | Median Y | Mean ell | Median ell | Mean principal | Median principal |
|---|---:|---:|---:|---:|---:|---:|---:|
| B=0 | 16,970 | $75,844.05 | $59,571.16 | $7,584.40 | $5,957.12 | $227,532.14 | $178,713.48 |
| B=1 | 3,030 | $77,447.48 | $60,076.11 | $7,744.75 | $6,007.61 | $232,342.43 | $180,228.32 |

| Scope | Total requested principal | Mean bank receipts | Mean true receipts | Mean bank profit | Mean true profit |
|---|---:|---:|---:|---:|---:|
| B=0 | $3,861,220,476.74 | $518,902.75 | $518,764.22 | $291,370.60 | $291,232.08 |
| B=1 | $703,997,575.13 | $524,013.41 | $529,885.35 | $291,670.98 | $297,542.91 |

| Scope | Mean profit error | Profit MAE | Profit RMSE | Bank profitable | True profitable | Sign disagreement |
|---|---:|---:|---:|---:|---:|---:|
| Overall | -$772.06 | $4,400.06 | $6,371.84 | 100% | 100% | 0% |
| B=0 | +$138.52 | $3,888.88 | $5,703.66 | 100% | 100% | 0% |
| B=1 | -$5,871.93 | $7,263.02 | $9,262.29 | 100% | 100% | 0% |

These are descriptive consequences of the saved estimated probabilities and
the specified contract. They do not prove discrimination; no new group-specific
treatment was added, and no funding decisions exist for this family yet.

## Files, schema, and information separation

Processed-data base:
`/Users/macbook/Hack/econ4994/data/processed/professor_model/`.
Each main table contains exactly one row for each of the 20,000 evaluation IDs.

| Files | Exact schema / information role |
|---|---|
| `loan_contracts.csv`, `.parquet` | `applicant_id, Y, ell_i, L_i, T, theta, r, payment_rule_id`; bank-observable contract, no hidden truth. |
| `bank_expected_profit.csv`, `.parquet` | `applicant_id, model_id, bank_PD_bounded, expected_receipts_bank, expected_profit_bank`; bank assessment only, no true PD/profit. |
| `true_expected_profit.csv`, `.parquet` | `applicant_id, true_PD, expected_receipts_true, expected_profit_true`; **EVALUATOR ONLY**. |
| `loan_economics_preview_EVALUATOR_ONLY.csv` | First 150 sorted evaluation IDs, with `information_scope=EVALUATOR_ONLY_NOT_FOR_LENDER_DECISIONS`; contracts, B, raw/bounded scores, truth, both receipts/profits, and profit error. |
| `loan_economics_metadata.json` | Evaluator provenance, configuration/source fingerprints, dependencies, original/new checksums, schemas, and information roles. |

Additional evaluator outputs:

- `results/tables/professor_model/loan_economics_summary.csv`: three rows,
  overall/B=0/B=1; means, medians, totals, profitable fractions, sign disagreement,
  and error metrics. Saved fractions are 0–1, not formatted percent strings.
- `results/tables/professor_model/loan_economics_distributions.csv`: 24 rows,
  eight variables for each scope; mean, median, sample SD, minimum/maximum,
  and p01/p05/p25/p50/p75/p95/p99, with explicit units.
- `results/metrics/professor_model/loan_economics_validation.json`: accounting,
  break-even, export, information-separation, and preservation checks.

No complete 600,000-row payment schedule is persisted: it is exactly
regenerable from the contracts and `payment_rule_id`, and the small vectorized
schedule is computed in memory. No charts were requested or added.

## Implementation, reproduction, and tests

- `configs/professor_model/loan_economics.yaml`: explicit working terms,
  per-period risk interpretation, exclusions, and separate output locations.
- `src/fair_lending/professor_model/contracts.py`: risk-free contract/schedule.
- `src/fair_lending/professor_model/profit.py`: probability validation,
  survival-weighted receipts/profit, separate bank/evaluator entry points,
  and evaluator diagnostics.
- `src/fair_lending/professor_model/economics_io.py`: saved Step 5 manifest
  validation, exact evaluation selection, export, and provenance.
- `experiments/calculate_professor_loan_economics.py`: orchestration only.

```bash
python3 experiments/calculate_professor_loan_economics.py
# Explicitly replace only Step 6A outputs if they already exist:
python3 experiments/calculate_professor_loan_economics.py --overwrite
python3 -m pytest -q tests/test_professor_loan_economics.py
python3 -m pytest -q tests/test_professor_population.py tests/test_professor_true_risk.py tests/test_professor_noisy_signal.py tests/test_professor_bank_regression.py tests/test_professor_loan_economics.py
python3 -m pytest -q
```

The 46 new deterministic tests cover the exact schema/principal/schedule,
special cases, risk bounds, bank/true survival, arithmetic, break-even and
neighboring signs, monotonicity, ID alignment, held-out selection, separation,
no regeneration/refitting, original-file checksums, deterministic outputs,
CSV/Parquet equality, group diagnostics, and overwrite protection.
All **192 professor-model tests passed without warnings**.

The complete repository suite finished with **472 passed, 4 failed, and 32
existing warnings** in 176.79 seconds. There are no new Step 6A failures.
The same four pre-existing failures remain in
`tests/test_systemic_corrected_checkpoint.py`: exact-cell coverage,
replication-13 provenance, legacy-comparison isolation, and the five-record
interim summary. The checkpoint loader incorporates current HEAD `7a7e590`
into its run ID, while saved records reference `8a9f71f`, producing different
lookup filenames. No recovery code or records were modified to bypass this.
The warnings are existing scikit-learn/SciPy L-BFGS-B `disp`/`iprint`
deprecations: 26 from ML benchmark tests and six from sensitivity tests.

All **61 files in the pre-step data/configuration/source/result snapshot** and
**64 registered prior artifacts** retained their checksums. Exported CSVs
match Parquet exactly with round-trip float parsing. Metadata records actual
source-content hashes (not just the Git revision), the full resolved YAML,
dependency versions, evaluation-ID fingerprint, and ten output checksums.
Previous manifests and scientific implementations remain unchanged.

## Interpretation limits and next boundary

Large positive nominal profits follow from growing payments, low PD, and no
discounting/costs; they are not evidence of realistic mortgage profitability.
Clipping 923 negative evaluation scores to zero remains Step 5's safeguard,
not calibrated default prediction. The original PN scaling/noise and OLS
functional form remain unchanged. This single synthetic population does not
establish empirical claims about actual lending or discrimination.

Step 6A stops here: no thresholds, approvals, budget selection, portfolio
optimization, new risk model, ML comparison, discrimination treatment, or
Monte Carlo was added. A later separately requested step may define lending
and its treatment of raw/bounded scores and boundary ties; this step makes
no such policy choice.
