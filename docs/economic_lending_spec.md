# Version 2 Economic Lending Specification

## Purpose

Version 2 starts a separate economic lending model while preserving the
Version 1 approval-disparity benchmark. The new model follows this conceptual
chain:

```text
applicant characteristics
    -> true repayment risk
    -> lender-estimated repayment risk
    -> expected loan receipts
    -> expected profit
    -> constrained lending allocation
```

Prompts 10A and 10A.1 implement the baseline economic accounting. Prompt 10B
adds the calibrated applicant population, true repayment DGP, and realized
repayment histories. Prompt 11 adds a traditional lender logit estimated from
historical at-risk payment rows while keeping truth evaluator-only. Prompt 12
adds one controlled flexible-model benchmark using the same observable
information and target. These phases do not add discrimination, HMDA analysis,
pricing optimization, or continuous loan-amount optimization.

## Source Labels

### Professor / whiteboard-derived concepts

- Applicants have characteristics that affect repayment risk.
- A true repayment process exists but is hidden from the lender.
- A lender eventually estimates risk rather than knowing the true risk model.
- Lending decisions can be framed as expected-receipt and expected-profit
  calculations.
- Bank lending is constrained by total available funds.
- The allocation decision should be separated from group labels so later
  discrimination mechanisms can be introduced deliberately.

### Working assumptions adopted through Prompt 10B

- There are two abstract applicant groups: `A` and `B`.
- Group is retained only for later auditing.
- Groups `A` and `B` use the same applicant-generating process.
- Group does not enter repayment truth, transaction costs, profit, or allocation.
- Each applicant requests one fixed loan option.
- The hand-check fixture uses fixed principal; the population has one
  applicant-specific requested amount with common rate, term, and transaction
  cost.
- There are `T` repayment periods.
- `rho_true` is the conditional probability of making the next scheduled
  payment given that every earlier scheduled payment was made.
- The first baseline holds `rho_true` constant across a borrower's contract
  periods and treats default as absorbing.
- Zero recovery is assumed after default in the first baseline.
- Expected monetary profit equals expected receipts minus principal lent minus
  transaction cost.
- The bank has total lendable principal `K`.
- The fixed-request allocation problem funds either the full requested loan or
  zero for each applicant.
- There is no forced target approval or funding rate.

### Unresolved assumptions left configurable

- Whether the payment schedule should be simple interest, amortizing, or another
  board-derived formula.
- Whether the conditional repayment probability should vary across periods.
- Whether recovery after default should remain zero.
- Whether transaction costs vary by applicant, loan, lender, or context.
- Whether future lender-estimated risk should be correctly specified,
  misspecified, or learned by ML.
- Whether future discrimination mechanisms act through true risk, estimated
  risk, prices, transaction costs, information, or allocation.
- Whether future allocation should allow continuously variable loan amounts.

## Baseline Schema

The logical tables are:

| Table | Purpose |
|---|---|
| `applicants` | Applicant ID, group, and characteristics used to form hidden repayment risk. |
| `loan_options` | One requested contract per applicant in the fixed-request baseline. |
| `payment_schedule` | Scheduled payments by applicant and period. |
| `simulation_truth` | Hidden true conditional per-period and full-contract repayment probabilities. |
| `policy_assessments` | Lender-perceived repayment probability, expected receipts, and expected profit; hidden truth is excluded. |
| `lending_decisions` | Whether each applicant is funded and funded principal. |
| `loan_outcomes` | Absorbing realized repayment histories and realized accounting. |
| `run_manifest` | Run metadata, seed, budget, and schema version. |

Prompt 10B persists `applicants`, `loan_options`, `simulation_truth`, and
`loan_outcomes`. Prompt 11 adds evaluation-cohort `policy_assessments`. Payment
schedules remain lazily generated, and lending decisions remain deferred.
Schema contracts are implemented in `src/fair_lending/economic_lending/schema.py`.

## Traditional Lender Information Set

The primary lender model estimates `paid_this_period` from historical at-risk
payment rows using the six observable transformations in the true DGP. It is an
unregularized logistic regression fitted only on `historical_train`. Group,
identifiers, cohort, period index, hidden truth, and future outcomes are excluded
from the design matrix. Exact estimates, validation, survival-selection
diagnostics, and perceived-profit results are documented in
`docs/traditional_lender_risk_model.md`.

## Flexible Model Benchmark

The Prompt 12 comparison freezes `traditional_logit_v1` and fits one
`HistGradientBoostingClassifier`, `ml_histgb_v1`, to the same unweighted at-risk
payment rows. Both estimators use the same six substantive observable variables;
group, IDs, cohort labels, period index, truth, and future outcomes remain
excluded. ML settings are chosen on `historical_validation` log loss, and the
`evaluation` cohort is used only for final comparison.

Under the current additive-logistic truth, the traditional model recovers the
hidden probability and expected profit more accurately than HistGB. This is an
expected benchmark result rather than a reason to modify the DGP. Full model,
calibration, group-audit, and economic results are documented in
`docs/ml_repayment_risk_benchmark.md`.

## Nonlinear Risk Sensitivity

Prompt 13 preserves that control world and introduces a separate
`nonlinear_v1` world on the identical applicants and contracts. It adds a
convex high-DTI penalty, weak-credit/high-LTV interaction, and bounded asset
buffer using only the same six observable variables. Only the nonlinear
intercept is calibrated, targeting the baseline mean full-repayment probability.
Traditional logit and HistGB are then re-estimated on new, paired-uniform
repayment histories. The design and mixed empirical result are documented in
`docs/nonlinear_risk_sensitivity.md`.

## Population-Scale Portfolio Allocation

Prompt 14 applies the frozen oracle, traditional, and HistGB assessments to the
same 2,000 evaluation applicants under common 100%, 40%, and 20% capital
budgets. A SciPy/HiGHS binary MILP chooses whole requested loans or zero and
never forces capital exhaustion. Portfolios are evaluated using hidden true
expected profit and a shared realized outcome within each world. Economic
regret is oracle true expected portfolio profit minus policy true expected
portfolio profit. Full results are in `docs/portfolio_allocation_benchmark.md`.

## Direct Belief-Distortion Experiment

Prompt 15 preserves truth and all neutral artifacts, then applies a researcher-
defined Group B penalty to the log odds of the repayment probability used by
each lender. Base estimates remain unchanged; perceived receipts and profit are
recomputed from the distorted probability before the complete portfolio is
re-optimized. A separate evaluator-only true-risk reference isolates the pure
mechanism from ordinary estimation error. Design, matched gap changes,
spillovers, and economic costs are in `docs/direct_belief_discrimination.md`.

## Applicant Characteristics

The development generator creates:

- `applicant_id`
- `group`, either `A` or `B`
- `cohort`
- `age_years`
- `annual_income`
- `credit_score`
- `employment_years`
- `liquid_assets`
- `existing_monthly_debt`
- `property_value`
- `requested_loan_amount`

The baseline generator draws financial characteristics independently of group.
This is a structural requirement for the no-discrimination baseline, not an
empirical claim.

## Repayment Event and Time Horizon

For applicant `i`, `rho_true_i` means the conditional probability that the
applicant makes the next scheduled payment, given that every previous
scheduled payment was made. It is not a default probability and it is not a
whole-loan repayment probability.

The baseline holds this conditional probability constant across periods and
treats default as absorbing. Therefore:

```text
survival_probability_i(t) = rho_true_i ** t
full_repayment_probability_true_i = rho_true_i ** T
```

`survival_probability_i(t)` is the probability that payment `t` is received.
The full-repayment probability is the probability that all `T` payments are
received. The approval or funding decision is a separate allocation outcome;
it is not a repayment event.

The working hidden repayment model is logistic-linear in six declared
transformed financial and contract variables:

```text
rho_true_i = sigmoid(z_i)
```

where:

```text
sigmoid(x) = 1 / (1 + exp(-x))
```

The sigmoid is a probability-valid working implementation, not a definitive
transcription of the professor's unresolved "linear risk model" notation.
Group does not enter the formula. The population score uses transformed income,
credit score, employment history, liquid assets, first-period DTI, and requested
LTV. Exact transformations and calibration evidence are documented in
`docs/economic_population_calibration.md`. The tiny hand-checkable fixture
separately supplies scores corresponding to conditional probabilities of
`0.90`, `0.75`, `0.50`, and `0.95`.

## Payment Formula

The working baseline contract uses equal scheduled bank payments based on
simple interest:

```text
total_promised_receipts = principal * (1 + periodic_interest_rate * T)
scheduled_payment = total_promised_receipts / T
```

For the tiny fixture:

```text
principal = 100
periodic_interest_rate = 0.20
T = 2
total_promised_receipts = 100 * (1 + 0.20 * 2) = 140
scheduled_payment = 70 per period
```

This is not a standard mortgage amortization formula. It is identified by
`payment_rule_id = equal_payments_simple_interest_v1`, and the period-level
schedule is kept separate from repayment survival so a later prompt can replace
the contract rule without changing the risk event.

## Expected Receipts

Expected receipts weight each scheduled payment by the probability that the
borrower survives through that payment period:

```text
expected_receipts_true_i =
    sum over t=1..T of (rho_true_i ** t) * scheduled_payment_it
```

This equals `rho_true * payment` when `T = 1`, but it does not generally equal
`rho_true * total_promised_receipts` when `T > 1`.

For realized accounting, `default_period` is the first scheduled payment not
made. Period 1 default produces zero receipts; period 2 default preserves only
payment 1; and `null` or positive infinity represents completion of all
payments.

## Expected Profit

Expected monetary profit is:

```text
expected_profit_true_i =
    expected_receipts_true_i
    - requested_principal_i
    - transaction_cost_i
```

Transaction cost is charged exactly once when a positive loan is issued. A
zero-principal action has zero receipts, transaction cost, and profit.

## Budget And Allocation

Let `x_i` equal 1 if applicant `i` is funded and 0 otherwise. Prompt 10A solves:

```text
maximize    sum_i x_i * expected_profit_true_i
subject to  sum_i x_i * requested_principal_i <= K
            x_i in {0, 1}
```

For now, the implementation enumerates all subsets. This is transparent for
small development fixtures and is verified against exhaustive enumeration in
tests. A later large-scale prompt can replace it with a scalable optimizer.

## Tiny Hand-Checkable Fixture

The fixture uses four applicants. Every applicant requests a `$100` loan, the
bank budget is `$200`, the transaction cost is `$5`, and the contract schedules
two `$70` payments. Expected receipts equal `70 * (rho_true + rho_true ** 2)`.

| applicant_id | group | rho_true | rho_true^2 | expected_receipts_true | expected_profit_true |
|---:|:---:|---:|---:|---:|---:|
| 1 | A | 0.90 | 0.8100 | 119.700 | 14.700 |
| 2 | B | 0.75 | 0.5625 | 91.875 | -13.125 |
| 3 | A | 0.50 | 0.2500 | 52.500 | -52.500 |
| 4 | B | 0.95 | 0.9025 | 129.675 | 24.675 |

With `K = 200`, the selected applicants are `1` and `4`:

```text
funded principal = 200
total expected profit = 39.375
unused funds = 0
```

This example is intentionally small enough to verify by hand.

## Research Cautions

- This baseline is not a model of real bank behavior.
- The true repayment probability is synthetic truth, not observed HMDA data.
- No group effect exists unless a later scenario deliberately inserts one.
- Funding differences in later versions must be separated into mechanisms:
  true risk, estimated risk, pricing, transaction cost, and allocation.
- Version 2 should not be interpreted as a replacement for Version 1. It is a
  new economic model family preserved separately from the completed approval
  benchmark.
