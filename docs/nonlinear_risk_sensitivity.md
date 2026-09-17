# Nonlinear True-Risk Sensitivity

## Motivation

This researcher-designed sensitivity asks whether flexible ML improves
conditional repayment-risk and expected-profit estimates when the traditional
lender observes the correct variables but imposes an oversimplified additive
logistic form. It is not evidence that real mortgage repayment risk has this
exact nonlinear structure.

The additive-logistic baseline, its configuration, policy IDs, and results were
preserved. The nonlinear experiment uses the same 10,000 applicant records,
contracts, IDs, groups, and applicant-level cohorts. Only hidden repayment truth
and the associated realized payment histories change.

## Frozen Nonlinear DGP

The world ID is `nonlinear_v1`. It starts with the six baseline transformed
covariates and their unchanged linear coefficients. Three terms were declared
before either model was fit:

```text
z_nonlinear = alpha_nl
  + 0.10*x_income + 0.45*x_credit + 0.08*x_employment
  + 0.15*x_assets - 0.30*x_dti - 0.20*x_ltv
  - 0.18*max(x_dti, 0)^2
  - 0.22*max(-x_credit, 0)*max(x_ltv, 0)
  + 0.18*tanh(x_assets)

rho_nonlinear = sigmoid(z_nonlinear)
```

The terms represent accelerating deterioration above DTI 0.45, an added
penalty when weak credit coincides with LTV above 0.80, and a bounded asset
buffer. Group is absent. No new substantive variable or latent information is
introduced.

Only `alpha_nl` was calibrated. On the deterministic 100,000-applicant
calibration population, the target mean full-repayment probability was
0.535548 and the solved intercept was `5.960906435206298`; the achieved mean
was `0.5355480000000521`. The frozen configuration fingerprint is
`a7262c73762dc755f5c13d1aeb57a3d3b9589447285921c562769544d4a9449c`.
Coefficients were not changed after model performance was observed.

## Comparability Across Worlds

| Quantity | Additive baseline | Nonlinear world |
|---|---:|---:|
| Mean rho | 0.992645 | 0.962906 |
| Median rho | 0.995317 | 0.996213 |
| rho p5 | 0.976764 | 0.810802 |
| rho p25 | 0.991093 | 0.986566 |
| rho p75 | 0.997626 | 0.998568 |
| rho p95 | 0.999089 | 0.999522 |
| Mean full-repayment probability | 0.535605 | 0.534870 |
| Median full-repayment probability | 0.569319 | 0.634228 |
| Full-repayment p5 | 0.059530 | approximately 0 |
| Full-repayment p95 | 0.896450 | 0.944252 |
| Mean true expected profit | $23,088 | $323 |
| Median true expected profit | $29,991 | $40,465 |
| Positive-profit share | 65.58% | 63.44% |

Matching the aggregate full-repayment mean does not preserve the distribution.
The nonlinear world has a much heavier lower-risk tail and higher upper
quantiles. For paired applicants, nonlinear minus additive rho had mean
-0.029739, SD 0.114424, p5 -0.165326, median 0.000495, and p95 0.001793.

## Paired Repayment Streams

A deterministic `10000 × 120` matrix of uniform draws was generated with seed
1,304,994. The same applicant-period uniforms were separately compared with
additive and nonlinear rho, with default absorbing in each world. This produces
matched counterfactual outcomes without reusing additive labels as nonlinear
labels. In this paired realization, additive and nonlinear full-repayment rates
were 53.05% and 53.08%.

## Model Specifications

`traditional_logit_nonlinear_world_v1` was re-estimated on nonlinear-world
`historical_train` payment rows. It retains the six baseline additive
transformations and receives no threshold, square, interaction, or group term.

`ml_histgb_nonlinear_world_v1` uses the same six raw observables as Prompt 12.
The unchanged eight-point grid was reselected on nonlinear validation log loss.
The selected settings were learning rate 0.03, 200 iterations, and 15 maximum
leaf nodes, with minimum leaf size 50 and L2 regularization 1.0.

At-risk payment rows were 488,890 train, 162,633 validation, and 159,060
evaluation. Evaluation data did not participate in calibration, fitting, or
selection.

## Probability Recovery

| Evaluation model | MAE | RMSE | Mean error | Correlation | Absolute-error p5 | p50 | p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Traditional | 0.013471 | 0.051301 | 0.011361 | 0.976971 | 0.000101 | 0.000793 | 0.083626 |
| HistGB | 0.018465 | 0.074066 | 0.006007 | 0.849198 | 0.000035 | 0.000581 | 0.120142 |

ML minus traditional absolute error had mean +0.004994 but median -0.000147;
ML was closer for 61.1% of applicants. Thus ML improved typical small errors
but incurred larger tail errors, leaving aggregate MAE and RMSE worse.

## Realized Payment Prediction

| Model | Brier | Log loss | ROC-AUC | Average precision |
|---|---:|---:|---:|---:|
| Traditional | 0.005685 | 0.030491 | 0.833565 | 0.998433 |
| HistGB | 0.005749 | 0.030526 | 0.830906 | 0.998406 |
| True-risk oracle | 0.005625 | 0.030121 | 0.835086 | 0.998448 |

The applicant-weighted mean absolute calibration-bin gap against hidden truth
was 0.012721 for traditional and 0.006692 for ML. Against realized frequencies,
the payment-row-weighted gaps were 0.000920 and 0.000973. Better aggregate bin
calibration for ML coexists with worse individual tail recovery.

## Expected-Profit Recovery

| Model | Mean error | MAE | RMSE | Wrong sign | False profitable | False unprofitable |
|---|---:|---:|---:|---:|---:|---:|
| Traditional | -$3,176 | $13,297 | $18,650 | 3.30% | 0.66% | 4.93% |
| HistGB | -$4,588 | $12,026 | $21,898 | 3.70% | 2.75% | 4.28% |

ML reduced profit MAE by $1,270, or 9.6%, but increased profit RMSE by $3,248
and the wrong-sign rate by 0.4 percentage points. The economic result is mixed,
not a clear improvement.

## Break-Even Disagreement

| Traditional | ML | n | Mean true profit | Median true profit | True positive-profit share |
|---|---|---:|---:|---:|---:|
| Nonpositive | Nonpositive | 776 | -$150,471 | -$126,738 | 4.77% |
| Nonpositive | Positive | 43 | $3,114 | $2,797 | 55.81% |
| Positive | Nonpositive | 19 | $28,324 | $8,855 | 84.21% |
| Positive | Positive | 1,162 | $92,035 | $79,227 | 99.83% |

The models disagreed on 62 requests (3.1%). ML corrected some traditional
nonpositive assessments, but its 19 opposite-direction disagreements usually
concerned truly positive-profit requests. These remain assessments, not funding
decisions.

## Neutral Group Audit

| Group | n | True rho | Traditional rho | Traditional risk MAE | ML rho | ML risk MAE | Traditional profit MAE | ML profit MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 986 | 0.951854 | 0.964571 | 0.014930 | 0.960246 | 0.020786 | $13,721 | $12,759 |
| B | 1,014 | 0.959662 | 0.969704 | 0.012052 | 0.963348 | 0.016208 | $12,884 | $11,313 |

Group is excluded everywhere except auditing. Differences are finite-sample
descriptions in a structurally group-neutral world, not proof of fairness.

## Baseline Versus Nonlinear Result

In the additive world, traditional/ML risk MAE was 0.000364/0.001341 and profit
MAE was $4,030/$14,038. In the nonlinear world, those pairs became
0.013471/0.018465 and $13,297/$12,026. Misspecification substantially weakened
both models. Flexible ML gained on median risk error, fraction closer, bin-level
truth calibration, and profit MAE, but not on risk MAE/RMSE, realized-label
metrics, profit RMSE, or sign classification.

The experiment therefore does not support a general claim that ML improved
under this misspecification. It shows that flexibility can help typical and
some economic errors while worsening tail-sensitive metrics. One deterministic
population, highly prevalent next-payment success, survival-selected payment
rows, and a small fixed grid limit generalization. No coefficients were retuned
to obtain a preferred result.

