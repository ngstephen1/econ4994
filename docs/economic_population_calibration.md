# Version 2 Economic Population Calibration

## Purpose and Evidence Status

Prompt 10B defines the baseline applicant population and hidden true repayment
data-generating process for Version 2. All numerical values are a
**researcher-chosen plausible synthetic calibration**, not empirical estimates
of U.S. mortgage applicants, lenders, or defaults.

The baseline has no discrimination mechanism. Abstract Groups `A` and `B` are
drawn independently of all financial variables and do not enter loan requests,
true repayment risk, contracts, outcomes, or true profit.

## Logical Outputs

| Output | Role | Hidden from a future lender? |
|---|---|:---:|
| `applicants` | Applicant, cohort, financial characteristics, property, and requested amount | No |
| `loan_options` | One requested contract and payment-burden diagnostics per applicant | No |
| `simulation_truth` | Conditional and full-contract true repayment probabilities | Yes |
| `loan_outcomes` | Realized default timing, receipts, and profit | Outcomes become historical labels only after realization |

The development artifacts contain 10,000 applicants. Cohorts are assigned at
the applicant level: 60% `historical_train`, 20% `historical_validation`, and
20% `evaluation`. Applicant IDs are globally unique, so payment histories
cannot be split across cohorts.

## Population Variables

The persisted applicant schema is:

```text
applicant_id
group
cohort
age_years
annual_income
credit_score
employment_years
liquid_assets
existing_monthly_debt
property_value
requested_loan_amount
```

The applicant table contains no true probability, expected-profit field,
future outcome, lender decision, or latent factor.

## Dependency Design

Generation follows this order:

```text
age -> feasible employment history

financial stability -----> employment, income, assets, debt, property, LTV
          |                                      |
          +---- correlated ----------------------+ 
                         creditworthiness -> credit, debt, LTV

income + assets + property + latent stability
    -> requested LTV and requested loan amount

applicant + requested contract
    -> first-period payment, DTI, LTV
    -> true conditional repayment probability
    -> absorbing realized default history
```

Financial stability and creditworthiness are correlated standard-normal latent
factors with correlation `0.35`. They create interpretable dependence but are
not persisted. Group is produced by its own random stream and is not an input
to this graph.

## Distribution Choices

| Variable | Synthetic construction | Bounds |
|---|---|---|
| Age | Scaled Beta(2.4, 2.2) | 21–75 years |
| Employment | Feasible history since age 18 times a bounded stability-dependent fraction | 0 to `age - 18` |
| Income | Conditional lognormal using stability, employment, and creditworthiness | $25,000–$300,000 |
| Credit score | Conditional normal using creditworthiness and stability, rounded | 500–850 |
| Liquid assets | Conditional lognormal using income, stability, and age | $500–$750,000 |
| Existing monthly debt | Bounded share of monthly income using a logistic index | 2%–42% of monthly income |
| Property value | Conditional lognormal using income and stability | $100,000–$1,500,000 |
| Requested LTV | Linear index using assets/property, stability, and creditworthiness plus noise | 0.45–0.95 |
| Requested amount | `property_value × requested_ltv` | Positive and no more than 95% of property value |

Bounds are transparent clipping rules. In the 100,000-row calibration, the
largest clipping share was `3.092%` for requested LTV, mainly at its lower
bound. All other clipping shares were approximately `1.04%` or lower.

Conditional lognormal income/assets/property mechanisms, bounded credit, and
latent dependence were adapted from useful Version 1 design patterns. Version
1's approval equation, 80% approval target, DTI cap, and discrimination
parameters were not reused.

## Requested Contract and Payment Burden

Every applicant requests one loan. The development contract uses:

```text
payment_rule_id = equal_payments_simple_interest_v1
periodic_interest_rate = 0.004
T = 120 periods
transaction_cost = 1500
```

This is a stylized 120-period development horizon aligned with monthly income
and debt measures. It is not a literal 360-month mortgage or a standard
amortization schedule.

```text
scheduled_payment =
    requested_loan_amount * (1 + 0.004 * 120) / 120

first_period_dti =
    (existing_monthly_debt + scheduled_payment) / (annual_income / 12)

requested_ltv = requested_loan_amount / property_value
```

The schedule remains lazily generated for the population; the four persisted
outputs do not duplicate 120 schedule rows per applicant.

## True Repayment DGP

The main specification is probability-valid logistic risk:

```text
rho_true = sigmoid(z)
```

with:

```text
z = 5.45
  + 0.10 * log(annual_income / 90000)
  + 0.45 * ((credit_score - 700) / 50)
  + 0.08 * ((employment_years - 10) / 10)
  + 0.15 * log((liquid_assets + 1000) / 46000)
  - 0.30 * ((first_period_dti - 0.45) / 0.10)
  - 0.20 * ((requested_ltv - 0.80) / 0.10)
```

Group, cohort, applicant ID, future outcomes, lender decisions, and
discrimination parameters are excluded. Coefficient signs encode the intended
relationships: credit, reserves, income, and employment stability raise
conditional repayment probability; DTI and LTV lower it.

`rho_true` is the probability of making the next scheduled payment conditional
on all earlier payments having been made. Default is absorbing:

```text
P(receive payment t) = rho_true ** t
P(complete all payments) = rho_true ** 120
```

### Linear-probability sensitivity design

The professor's handwritten "linear risk model" is not treated as identical to
the logistic implementation. A future sensitivity specification may use
`rho_true = alpha + beta'X` only on a declared admissible covariate support. It
must be rejected or recalibrated if it requires widespread clipping at zero or
one. It is documented in the config but is not the baseline DGP.

## Realized Outcomes

For each active loan and period, a uniform draw is compared with `rho_true`.
Simulation stops at the first missed payment; no further repayment outcomes are
drawn for that loan. Outputs record:

- `default_period`, the first scheduled payment not made;
- `completed_all_payments`;
- `realized_total_receipts`; and
- `realized_profit = receipts - principal - transaction_cost`.

The repayment-outcome stream is independent of the group, demographic, latent,
financial, request, and cohort streams.

## Calibration Targets

Broad targets were declared before accepting the calibration:

| Quantity | Accepted range |
|---|---:|
| Mean conditional repayment probability | 0.990–0.999 |
| Mean full-repayment probability | 0.40–0.90 |
| Realized default rate | 0.10–0.60 |
| Positive expected-profit share | 0.10–0.90 |

These ranges prevent degenerate outcomes and preserve both profitable and
unprofitable requests. They are research-design targets, not empirical claims.

## 100,000-Applicant Calibration Results

Seed `4994` produced 50.107% Group A and 49.893% Group B.

### Financial distributions

| Quantity | Mean | Median |
|---|---:|---:|
| Annual income | $108,912 | $98,783 |
| Credit score | 705.02 | 705 |
| Liquid assets | $74,044 | $52,881 |
| Existing monthly debt | $1,384.73 | $1,251.80 |
| Property value | $416,823 | $375,352 |
| Requested loan | $293,316 | $265,775 |
| Requested LTV | 0.7161 | 0.7249 |
| First-period DTI | 0.5805 | 0.5690 |

### Risk and outcomes

| Quantity | Result |
|---|---:|
| Mean `rho_true` | 0.992572 |
| Median `rho_true` | 0.995369 |
| `rho_true` p5 / p25 / p75 / p95 | 0.976907 / 0.991038 / 0.997618 / 0.999083 |
| Mean full-repayment probability | 0.535548 |
| Median full-repayment probability | 0.572896 |
| Full-repayment p5 / p25 / p75 / p95 | 0.060586 / 0.339502 / 0.751155 / 0.895787 |
| Realized full-repayment rate | 0.5354 |
| Realized default rate | 0.4646 |

Among loans that defaulted, the median first missed payment was period 43; the
25th, 75th, and 95th percentiles were periods 18, 75, and 110. This confirms
that the 120-period horizon is consequential even though one-period repayment
probabilities are high.

### Economics

| Quantity | Result |
|---|---:|
| Mean true expected receipts | $317,769 |
| Mean true expected profit | $22,954 |
| Positive expected-profit share | 65.451% |
| Nonpositive expected-profit share | 34.549% |

Both profit signs are well represented without forcing a target fraction.

### Group-invariance diagnostic

Observed Group B minus Group A differences in the 100,000-row sample were:

| Quantity | Difference |
|---|---:|
| Annual income | -$86.88 |
| Credit score | -0.045 points |
| Liquid assets | +$195.23 |
| Property value | -$36.83 |
| Requested loan | -$329.22 |
| First-period DTI | -0.00050 |
| Requested LTV | -0.00086 |
| `rho_true` | -0.0000048 |
| Full-repayment probability | +0.00079 |
| Realized completion rate | +0.00217 |

These are sampling differences, not configured group effects. Structural tests
also verify that changing every group label leaves financial generation and
true repayment probabilities unchanged.

## Reproducibility and Artifacts

Seed `4994` is split with `numpy.random.SeedSequence` into named streams for
group, demographics, latent factors, financial variables, loan requests,
cohorts, and repayment outcomes. The accepted configuration fingerprint is:

```text
68869774a4c19e6b81fbd660c71e387b75d0d9a504d3aba9ac905c2a00ad55f2
```

Run the saved 10,000-row development generation with:

```bash
python3 experiments/run_economic_population_calibration.py
```

The command writes the four Parquet outputs under
`data/synthetic/economic_lending/v2_baseline/`, a JSON validation report under
`results/metrics/`, and a group diagnostic CSV under `results/tables/`.

## Limitations

- Numerical choices are synthetic research assumptions, not externally
  validated mortgage-population estimates.
- The equal-payment simple-interest contract is a working accounting device.
- Conditional repayment probability is constant over a loan's periods.
- There is no recovery after default and no prepayment.
- There is one requested option per applicant and no pricing choice.
- The simulator knows hidden truth; future lenders must not receive truth
  fields or future outcomes as predictors.
- No discrimination, lender estimation, ML, HMDA, or allocation experiment is
  introduced in this phase.
