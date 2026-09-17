# Traditional Lender Repayment-Risk Model

## Research Purpose

Prompt 11 asks how accurately a transparent traditional lender can estimate
the conditional probability of the next scheduled payment from observed
synthetic repayment histories. This phase isolates parameter-estimation error:
the lender knows the relevant observable variables and transformations but not
the true coefficients or any applicant's hidden probability.

There is no group-specific treatment, discrimination mechanism, flexible ML
model, pricing decision, or portfolio optimization in this experiment.

## Information Boundary

The lender receives applicant and requested-contract characteristics plus
realized payment histories from `historical_train`. It never receives:

- `repayment_probability_per_period_true`;
- `full_repayment_probability_true`;
- true expected receipts or profit;
- latent financial factors;
- future default paths; or
- evaluation outcomes during fitting or model selection.

Group remains in the history table only for evaluator-side auditing. It is
excluded from the primary design matrix. The evaluation cohort is used only
after the coefficient artifact has been fitted.

## At-Risk Payment Table

One row represents an applicant still at risk at the beginning of a payment
period. `paid_this_period` is one if that scheduled payment was made and zero
for the first missed payment. Rows stop immediately after default.

For example, default in period 4 contributes outcomes `1, 1, 1, 0`; a completed
120-period contract contributes 120 ones. Applicant-level cohort assignments
are inherited without resplitting payment rows.

The development table contains:

| Cohort | Applicants | At-risk payment rows |
|---|---:|---:|
| `historical_train` | 6,000 | 524,756 |
| `historical_validation` | 2,000 | 170,690 |
| `evaluation` | 2,000 | 172,184 |
| Total | 10,000 | 867,630 |

Historical training applicants contribute an average of `87.459` rows and a
median of `120` rows.

## Primary Model

The primary policy is `traditional_logit_v1`, an unregularized logistic
regression fitted only on `historical_train`:

```text
P(paid_this_period = 1 | reached period, X) = sigmoid(alpha + beta'X)
```

It does not use group, applicant ID, cohort, period index, hidden truth, or any
future outcome. Because true conditional risk is constant over time, no period
effect is included.

The fitted design matrix uses the same six transformations as the true DGP:

```text
log_annual_income      = log(annual_income / 90000)
credit_score_50        = (credit_score - 700) / 50
employment_years_10    = (employment_years - 10) / 10
log_liquid_assets      = log((liquid_assets + 1000) / 46000)
first_period_dti_10pp  = (first_period_dti - 0.45) / 0.10
requested_ltv_10pp     = (requested_ltv - 0.80) / 0.10
```

There is no hyperparameter search. Validation metrics are reported, but the
validation or evaluation cohorts do not alter the specification.

## Coefficient Recovery

| Term | True | Fitted | Fitted − true | Relative error |
|---|---:|---:|---:|---:|
| Intercept | 5.4500 | 5.4855 | +0.0355 | +0.65% |
| Log income | 0.1000 | 0.2380 | +0.1380 | +138.03% |
| Credit score / 50 | 0.4500 | 0.4184 | -0.0316 | -7.02% |
| Employment years / 10 | 0.0800 | 0.0848 | +0.0048 | +6.03% |
| Log liquid assets | 0.1500 | 0.1338 | -0.0162 | -10.80% |
| First-period DTI / 0.10 | -0.3000 | -0.3082 | -0.0082 | +2.72% |
| Requested LTV / 0.10 | -0.2000 | -0.1638 | +0.0362 | -18.08% |

The income coefficient has the largest relative deviation. Income is correlated
with assets, employment, DTI, property/request scale, and the internal latent
factors, so individual coefficient recovery is noisier than probability
recovery. Coefficients are not interpreted using independent-row standard
errors because payment rows are clustered within applicants.

## Survival Selection

Higher-`rho_true` applicants survive longer and contribute more training rows.
The correlation between payment-row count and hidden `rho_true` is `0.477`.

| `rho_true` quintile | Mean `rho_true` | Mean rows contributed |
|---:|---:|---:|
| 1, lowest | 0.98013 | 50.48 |
| 2 | 0.99227 | 78.92 |
| 3 | 0.99537 | 94.22 |
| 4 | 0.99728 | 103.10 |
| 5, highest | 0.99876 | 110.58 |

Row count correlations were `+0.374` with credit score, `-0.438` with DTI, and
`-0.368` with LTV. Thus, the at-risk table overrepresents safer borrowers in
later periods. Under the constant correctly specified conditional hazard, the
unweighted row likelihood nevertheless targets next-payment probability among
at-risk observations and recovered probabilities accurately.

## Realized Next-Payment Prediction

| Cohort/model | Brier | Log loss | ROC-AUC |
|---|---:|---:|---:|
| Validation traditional logit | 0.005602 | 0.033053 | 0.7186 |
| Validation true-risk oracle | 0.005602 | 0.033024 | 0.7200 |
| Evaluation traditional logit | 0.005409 | 0.031759 | 0.7423 |
| Evaluation true-risk oracle | 0.005409 | 0.031762 | 0.7424 |

The oracle is the synthetic true conditional probability evaluated against a
Bernoulli payment realization. It does not know the future outcome. Its metrics
show the irreducible outcome uncertainty and finite-sample variation; a fitted
model can be marginally better on a realized sample without being truer.

## Hidden-Truth Probability Recovery

On 2,000 evaluation applicants:

| Metric | Result |
|---|---:|
| MAE | 0.000364 |
| RMSE | 0.000662 |
| Mean prediction error | +0.0000657 |
| Correlation with `rho_true` | 0.997693 |

Predicted and true quantiles were also close:

| Quantile | Predicted | True |
|---|---:|---:|
| P1 | 0.952896 | 0.953220 |
| P5 | 0.975451 | 0.975117 |
| P25 | 0.990835 | 0.990673 |
| P50 | 0.995269 | 0.995225 |
| P75 | 0.997558 | 0.997580 |
| P95 | 0.999107 | 0.999105 |
| P99 | 0.999572 | 0.999575 |

The saved calibration table reports mean predicted probability, hidden true
probability, and realized at-risk payment frequency in ten prediction bins.

## Group Audit

Group is excluded from the lender model. Evaluation results are:

| Group | N | Mean predicted | Mean true | Mean error | MAE | RMSE |
|:---:|---:|---:|---:|---:|---:|---:|
| A | 986 | 0.991985 | 0.991933 | +0.000052 | 0.000384 | 0.000710 |
| B | 1,014 | 0.992496 | 0.992417 | +0.000079 | 0.000345 | 0.000612 |

Small differences reflect finite-sample composition and estimation, not a
configured group effect.

## Perceived Expected Profit

The lender's estimated probability is inserted into the validated multi-period
cash-flow equation. No probability distortion is applied, so
`repayment_probability_base` equals `repayment_probability_used`.

Evaluation profit-estimation results:

| Metric | Result |
|---|---:|
| Mean error | +$1,696.10 |
| MAE | $4,029.93 |
| RMSE | $6,458.39 |
| Profit-sign agreement | 97.8% |
| Wrong-profit-sign rate | 2.2% |
| False profitable rate among truly nonpositive requests | 4.03% |
| False unprofitable rate among truly positive requests | 1.17% |

These are economic classification diagnostics, not approval errors or fairness
metrics. No funding-budget optimization occurs in Prompt 11.

## Sensitivity Estimators

### Applicant-weighted logit

Each applicant receives total weight one, divided across their observed at-risk
rows. On evaluation applicants it produced MAE `0.02401`, RMSE `0.03243`, and
mean error `-0.02401`; realized Brier score was `0.006014` and log loss was
`0.042627`.

This weighting changes the estimand substantially: an early defaulter's single
failure receives far more row weight than a late defaulter's failure. It is
retained as a sensitivity diagnostic, not selected as the primary model merely
because every applicant receives equal total weight.

### Linear probability model

The unbounded evaluation predictions ranged from `0.97510` to `1.00888`.
`9.2%` exceeded one and none fell below zero. Therefore the LPM is not treated
as a naturally probability-valid lender model. A separately labeled clipped
version had evaluation MAE `0.00257`, RMSE `0.00630`, Brier `0.005415`, log loss
`0.033990`, and ROC-AUC `0.7415`.

Clipping is disclosed and the clipped LPM is not used for the baseline policy
assessment.

## Reproducible Artifacts

- `data/synthetic/economic_lending/v2_baseline/policy_assessments.parquet`
  contains 2,000 evaluation assessments with no hidden truth columns.
- `results/models/traditional_logit_v1.json` stores coefficients, transforms,
  cohort restrictions, sample sizes, configuration fingerprint, and code
  revision without pickle.
- `results/metrics/traditional_lender_risk_model.json` stores primary, oracle,
  survival-selection, LPM, weighted, and economic metrics.
- `results/tables/` contains coefficient recovery, calibration, group audit,
  profit audit, and evaluator-only perceived-versus-true profit tables.

## Limitations

- Repeated payment rows are dependent within applicants; coefficient inference
  is secondary and ordinary row-level standard errors are not reported.
- The lender is intentionally given the correct covariates and transforms, so
  this isolates estimation error rather than functional-form discovery.
- Constant conditional repayment risk omits duration dependence and changing
  borrower circumstances.
- The synthetic population and contract are researcher-chosen calibrations.
- Hidden truth can evaluate the model in simulation but would not exist in real
  lending data.
