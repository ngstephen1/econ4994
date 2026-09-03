# First Statistical Recovery Experiment

## 1. Objective

This experiment asks whether descriptive statistics and correctly specified
logistic regressions recover the disparity mechanisms deliberately encoded in
the synthetic mortgage data. It is an inference and recovery benchmark, not a
prediction competition and not evidence about real mortgage lenders.

## 2. Data-generating scenarios

The experiment uses the four validated scenarios at the `moderate` effect
level: `fair_baseline`, `direct_discrimination`, `upstream_inequality`, and
`mixed_mechanism`. White is the reference group and Black is the focal group.
The direct scenarios configure a Black treatment of -0.25 log odds. The
upstream scenarios shift income, credit score, and liquid-assets distributions
without adding those shifts directly to the approval equation.

## 3. Analysis sample

Each scenario is freshly generated with 100,000 applications, seed 4994, and
the frozen approval intercept `2.0362202310934663`. Descriptive tables use all
100,000 applications and all five configured race categories. Regressions use
the 79,096 White or Black applications so that `black = 0` unambiguously means
the White reference category. The same population and non-race random streams
are used across scenarios by design.

## 4. Outcome

The observed dependent variable is `approved`, coded 1 for approval and 0 for
denial. `approval_probability_true` is never used as the dependent variable or
as a predictor; it is retained only for synthetic ground-truth diagnostics.

## 5. Raw disparity estimand

The unadjusted estimand is the Black approval rate minus the White approval
rate. Its 95% confidence interval uses the unpooled Wald standard error for the
difference between two independent proportions:

    SE = sqrt[p_B(1-p_B)/n_B + p_W(1-p_W)/n_W]

The denial-rate difference is exactly the negative of the approval-rate
difference and is not treated as a separate statistical signal.

| Scenario | White approval | Black approval | Raw gap (pp) | 95% CI (pp) |
|---|---:|---:|---:|---:|
| Fair baseline | 80.071% | 80.030% | -0.041 | [-0.820, 0.739] |
| Direct discrimination | 80.071% | 77.071% | -3.000 | [-3.814, -2.187] |
| Upstream inequality | 80.071% | 73.455% | -6.616 | [-7.465, -5.767] |
| Mixed mechanism | 80.071% | 69.478% | -10.593 | [-11.474, -9.712] |

These are raw approval-rate disparities. They are not, by themselves, estimates
of discrimination.

## 6. Regression specifications

All models are statsmodels binary logits with an intercept:

- **Model 0:** `approved ~ black`
- **Model 1:** Model 0 plus transformed income, credit score, employment years,
  liquid assets, DTI, and LTV
- **Model 2 (main adjusted):** Model 1 plus indicator-coded loan purpose, loan
  type, and occupancy type
- **Model 3 (sensitivity):** Model 2 plus indicator-coded age group, sex, and
  ethnicity

Categorical references are Home purchase, Conventional, Principal residence,
age 35-44, Male, and Not Hispanic or Latino. Coefficients on the Model 3
demographics are not interpreted as causal effects. The primary models exclude
neighborhood minority share, true approval probability, denial reason,
property value, loan amount, and income-to-loan ratio.

## 7. Transformed variables

The analysis reads transformation parameters from the approval configuration
and matches the DGP units exactly:

    log_income = [ln(annual_income) - ln(95,000)] / 0.50
    credit_score_50 = (credit_score - 720) / 50
    employment_years_10 = (employment_years - 10) / 10
    log_liquid_assets = ln(liquid_assets + 1,000) - ln(36,000)
    dti_10pp = (debt_to_income_ratio - 0.36) / 0.10
    ltv_10pp = (loan_to_value_ratio - 0.80) / 0.10

## 8. Adjusted probability contrast

For each fitted model, every regression record is predicted twice while all
non-race covariates are held fixed: once with `black = 0` and once with
`black = 1`. The average of `p(Black) - p(White)` is the standardized adjusted
Black-White probability contrast. This probability-scale estimand is easier to
interpret than a log-odds coefficient.

## 9. Known synthetic ground truth and recovery

The true fixed-feature probability contrast is computed with the generator's
own approval equation rather than a duplicated approximation.

| Scenario | True log odds | Model 2 estimate | Coef. error | OR (95% CI) | True gap (pp) | Adjusted gap (pp) | Gap error (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fair baseline | 0.000 | -0.025 | -0.025 | 0.975 [0.921, 1.032] | 0.000 | -0.303 | -0.303 |
| Direct discrimination | -0.250 | -0.263 | -0.013 | 0.769 [0.728, 0.812] | -3.115 | -3.274 | -0.159 |
| Upstream inequality | 0.000 | 0.000 | 0.000 | 1.000 [0.948, 1.055] | 0.000 | 0.000 | 0.000 |
| Mixed mechanism | -0.250 | -0.272 | -0.022 | 0.762 [0.723, 0.803] | -3.199 | -3.478 | -0.280 |

The direct and mixed Model 2 estimates are within 0.022 log-odds units of the
configured -0.25 treatment. Their confidence intervals contain the configured
truth. The small fair-baseline deviation is compatible with sampling noise;
its confidence interval includes zero.

## 10. Scenario-by-scenario results

### Fair baseline

The raw gap was -0.041 percentage points and the Model 2 adjusted contrast was
-0.303 points. The Model 2 coefficient was -0.025 (95% CI -0.082 to 0.031,
`p = 0.377`). Both results are close to the zero ground truth.

### Direct discrimination

The raw gap was -3.000 points. The correctly adjusted Model 2 coefficient was
-0.263 and the adjusted contrast was -3.274 points, close to the -0.25 log-odds
and -3.115-point fixed-feature truths.

### Upstream inequality

The raw gap was -6.616 points, while the race-only log-odds coefficient was
-0.373. Once the DGP financial controls were included, the Model 2 coefficient
was 0.000015 and the adjusted contrast was 0.0002 points. Adjustment therefore
conditions away the deliberately modeled upstream path; this does not imply
that upstream inequality is economically or normatively unimportant.

### Mixed mechanism

The raw gap was -10.593 points. The race-only coefficient was -0.568, whereas
the Model 2 estimate was -0.272 and its adjusted probability contrast was
-3.478 points. Adjustment removed the modeled upstream association while
retaining approximately the configured direct component.

## 11. Coefficient paths

| Scenario | Model 0 | Model 1 | Model 2 | Model 3 |
|---|---:|---:|---:|---:|
| Fair baseline | -0.003 | -0.025 | -0.025 | -0.025 |
| Direct discrimination | -0.178 | -0.262 | -0.263 | -0.263 |
| Upstream inequality | -0.373 | 0.000 | 0.000 | -0.001 |
| Mixed mechanism | -0.568 | -0.272 | -0.272 | -0.273 |

Model 3 adds little in this DGP because age group, sex, and ethnicity receive no
direct treatment and are generated independently of race. This is a simulation
property, not a claim about real populations.

## 12. Fit and numerical diagnostics

All 16 models converged. Every matrix was full rank, with no duplicate or
zero-variance columns. Model 2 condition numbers ranged from 11.18 to 11.29,
and its largest standard error ranged from 0.098 to 0.103; no severe numerical
instability was detected. Log likelihood, AIC, BIC, McFadden pseudo-R-squared,
iteration count, and matrix diagnostics are retained in
`results/tables/logit_black_effects.csv`.

## 13. Non-collapsibility caveat

Logistic coefficients are non-collapsible. A Model 0 coefficient minus a Model
2 coefficient is not a mediation estimate, and coefficient shrinkage is not
"the amount of discrimination explained by controls." The standardized
probability contrasts help interpretation but also remain model-dependent.

## 14. Limits of synthetic evidence

Recovery here shows that these methods behave as expected under this known,
correctly specified DGP. It cannot establish that real-world discrimination
exists, that the simulated effect magnitudes are realistic, or that an adjusted
observational disparity is causal. The label "discrimination" applies here
only where the simulation explicitly encodes a direct protected-group term.

## 15. Next modeling step

The statistical benchmark is ready to serve as the reference for a carefully
scoped machine-learning benchmark. That later phase should compare predictive
and group-level behavior against the known scenario truth without treating AUC
as the research objective.
