# Version 2 Flexible Repayment-Risk Benchmark

## Objective

This experiment asks whether a flexible machine-learning estimator improves
conditional next-payment risk estimates when the traditional lender already
observes the relevant covariates and uses an approximately correct logistic
functional form. It also tests whether probability differences translate into
more accurate request-level expected-profit estimates.

The result is deliberately allowed to favor either model. Under the current
additive-logistic synthetic truth, the traditional model performed better than
the flexible estimator. The data-generating process was not changed in response.

## Fair Comparison Design

Both models estimate the same event:

```text
P(next scheduled payment succeeds | borrower reached the period, X)
```

They use annual income, credit score, employment years, liquid assets,
first-period DTI, and requested LTV. The frozen `traditional_logit_v1` uses the
existing DGP-aligned transforms. The new `ml_histgb_v1` uses the corresponding
raw numeric fields so it can learn nonlinear splits and interactions. Neither
model receives group, applicant ID, cohort, period index, hidden truth, latent
factors, default period, future outcomes, or evaluation labels during fitting
or selection.

Applicant-level cohorts remain authoritative. The 524,756
`historical_train` at-risk rows fit the models, 170,690
`historical_validation` rows select the ML settings, and 172,184 `evaluation`
rows provide the final comparison. Each validation and evaluation cohort has
2,000 applicants. No post-default payment rows are included.

## Models and Selection

The traditional benchmark is the already-frozen, unregularized logistic model;
it was loaded from its transparent coefficient artifact and was neither
retuned nor refitted for this comparison.

The flexible estimator is scikit-learn's
`HistGradientBoostingClassifier`, with log loss, random state 4994, minimum
leaf size 50, L2 regularization 1.0, and early stopping disabled. The
predeclared validation grid crossed learning rates 0.03/0.05, iterations
100/200, and maximum leaf nodes 15/31. The eight trials were ranked first by
validation log loss, then Brier score, then ROC-AUC. The selected settings were
learning rate 0.03, 100 iterations, and 15 maximum leaf nodes. Evaluation
outcomes were not used in this choice.

## Probability Recovery

Errors below are in probability units and use evaluator-only synthetic truth.

| Cohort | Model | MAE | RMSE | Mean error | Correlation with truth |
|---|---|---:|---:|---:|---:|
| Validation | Traditional logit | 0.000352 | 0.000615 | 0.000108 | 0.997016 |
| Validation | HistGB | 0.001166 | 0.002266 | -0.000049 | 0.956109 |
| Evaluation | Traditional logit | 0.000364 | 0.000662 | 0.000066 | 0.997693 |
| Evaluation | HistGB | 0.001341 | 0.002982 | 0.000072 | 0.952270 |

On evaluation, ML's MAE was 0.000977 higher and its RMSE was 0.002320 higher.
Its MAE was about 3.68 times and its RMSE about 4.50 times the traditional
model's. Both absolute errors remain small in this high-repayment-probability
population.

| Model | Error p1 | p5 | p25 | p50 | p75 | p95 | p99 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Traditional | -0.001783 | -0.000715 | -0.000132 | 0.000014 | 0.000228 | 0.001048 | 0.002382 |
| HistGB | -0.006433 | -0.002737 | -0.000664 | -0.000171 | 0.000495 | 0.003428 | 0.008377 |

The paired ML-minus-traditional estimate had mean 0.0000066, median -0.000190,
standard deviation 0.003021, 5th/95th percentiles -0.002912 and 0.003216,
maximum absolute difference 0.045956, and correlation 0.949786.

## Realized Next-Payment Prediction

Realized labels contain irreducible Bernoulli noise, so these metrics answer a
different question from recovery of hidden `rho_true`.

| Evaluation model | Brier | Log loss | ROC-AUC | Average precision |
|---|---:|---:|---:|---:|
| Traditional logit | 0.005409 | 0.031759 | 0.742327 | 0.997861 |
| HistGB | 0.005411 | 0.031814 | 0.740461 | 0.997830 |
| True-risk oracle | 0.005409 | 0.031762 | 0.742363 | 0.997863 |

The oracle is the true conditional probability, not a model that knows future
payment draws. Its realized metrics need not beat every fitted estimate in one
finite sample. Accuracy is omitted because successful payments are highly
prevalent.

## Calibration

Ten equal-applicant-count bins compare mean prediction with both mean true
probability and realized at-risk payment frequency. On evaluation, the
applicant-weighted mean absolute bin gap against truth was 0.000070 for the
traditional model, 0.000305 for HistGB, and zero by construction for the
oracle. Payment-row-weighted absolute gaps against observed rates were
0.000504, 0.000648, and 0.000577, respectively. Sampling noise explains why
the oracle's realized-frequency gap is not zero.

## Group Audit

Group is excluded from both estimators and the baseline DGP is group-neutral.
Finite-sample differences do not prove fairness.

| Group | n | Mean true rho | Traditional mean rho | Traditional MAE | Traditional RMSE | ML mean rho | ML MAE | ML RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 986 | 0.991933 | 0.991985 | 0.000384 | 0.000710 | 0.992062 | 0.001310 | 0.002949 |
| B | 1,014 | 0.992417 | 0.992496 | 0.000345 | 0.000612 | 0.992434 | 0.001371 | 0.003014 |

The B-minus-A mean prediction-error difference was +0.000026 for the
traditional model and -0.000112 for HistGB.

## Expected-Profit Estimation

Both probabilities pass through identical survival-weighted contract and
profit functions. Dollar errors compare perceived and hidden true expected
profit for 2,000 evaluation requests.

| Model | Mean error | MAE | RMSE | Wrong sign | False profitable | False unprofitable |
|---|---:|---:|---:|---:|---:|---:|
| Traditional logit | $1,696 | $4,030 | $6,458 | 2.20% | 4.03% | 1.17% |
| HistGB | -$1,336 | $14,038 | $21,438 | 6.15% | 8.90% | 4.61% |

These are request-level estimation diagnostics, not portfolio regret and not
approval decisions. Portfolio allocation remains outside this prompt.

## Break-Even Disagreement

| Traditional assessment | ML assessment | Applicants | Mean true expected profit |
|---|---|---:|---:|
| Positive | Positive | 1,236 | $73,286 |
| Positive | Nonpositive | 59 | $11,430 |
| Nonpositive | Positive | 50 | -$16,745 |
| Nonpositive | Nonpositive | 655 | -$76,576 |

The models disagreed for 109 applicants (5.45%). In both discordant cells, mean
true profit aligns with the traditional classification. This is an
evaluator-side diagnostic, not a funding decision or final economic regret.

## Predictive Importance

Validation permutation importance measured increase in log loss. First-period
DTI ranked highest (0.001154), followed by credit score (0.000685), requested
LTV (0.000208), liquid assets (0.000131), annual income (0.000075), and
employment years (0.000026). Predictive importance is not causal importance.

## Interpretation and Limitations

The traditional model performed better. This is expected: hidden truth is
additive logistic, and the traditional lender has the correct substantive
covariates and matching transformations. The tree learner has little structural
advantage and must approximate a smooth logistic surface. Extra flexibility
does not create information.

This is one deterministic synthetic development population, not a Monte Carlo
comparison and not evidence about real lenders. Probabilities are tightly
concentrated near one, payment rows are clustered within applicants, and the
evaluation covers request-level accounting before constrained allocation.
Group neutrality is built into this baseline.

## Future Nonlinear Sensitivity

A separate future experiment should introduce controlled nonlinearities,
thresholds, and interactions into true risk while retaining the same observable
covariates. The traditional specification should remain simple, allowing a
clean test of flexible ML under known functional-form misspecification. That
extension is not implemented here.

