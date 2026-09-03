# First Machine-Learning Benchmark

## 1. Research objective

This experiment asks whether models trained to imitate synthetic mortgage
approval decisions reproduce group differences in those decisions, including
when protected attributes are withheld from model inputs. It does not ask
whether machine learning can prove that a lender is discriminatory.

## 2. Connection to the statistical benchmark

Prompt 6 showed that a correctly specified statsmodels logit recovered the
configured direct Black effect after adjustment and conditioned away the
modeled upstream path. Prompt 7 uses the same four 100,000-row datasets, seed
4994, moderate treatments, and frozen intercept. It changes the estimand from a
conditional race coefficient to group differences in out-of-sample model
predictions. Those quantities are complementary, not interchangeable.

## 3. Synthetic scenarios

- `fair_baseline`: no upstream race shifts and no direct race term
- `direct_discrimination`: a -0.25 Black log-odds term, with comparable
  race-blind features
- `upstream_inequality`: Black shifts in income, credit score, and assets, but
  no direct race term
- `mixed_mechanism`: both the upstream shifts and direct term

The word discrimination is used only for the researcher-inserted direct term.

## 4. Feature regimes

### Race-blind primary model

The exact predictors are:

- `annual_income`
- `credit_score`
- `employment_years`
- `liquid_assets`
- `debt_to_income_ratio`
- `loan_to_value_ratio`
- `loan_purpose`
- `loan_type`
- `occupancy_type`

These are the observable fields that enter the known approval DGP. Race,
ethnicity, sex, age group, neighborhood minority share, identifiers, outcomes,
and ground-truth fields are excluded. Existing monthly debt, loan amount,
property value, income-to-loan ratio, neighborhood income, and local
unemployment are also excluded because they are components, redundant
derivatives, or explicit non-terms in this approval equation.

### Race-aware sensitivity model

This regime uses the same nine predictors and adds `race`. It is an
experimental sensitivity analysis, not a recommended operational lending
system. It tests whether direct race information improves recovery of labels
whose DGP explicitly contains a race term.

### Deferred proxy regime

The optional neighborhood-minority-share regime is deferred. Version 1 gives
that field no baseline race dependency and excludes it from underwriting, so
adding it here would expand the benchmark without cleanly testing the primary
direct-versus-upstream contrast. A later proxy/segregation scenario can activate
and evaluate that mechanism explicitly.

## 5. Train, validation, and test procedure

A NumPy seed-4994 permutation assigns exactly 60,000 applications to training,
20,000 to validation, and 20,000 to test. Assignment depends on row position,
not the target, and the shared application IDs receive identical partitions in
all scenarios. Every regime and model within a scenario uses the same rows.

Preprocessing and classifiers are fit only on training data. Continuous fields
are standardized; categorical fields use `OneHotEncoder(handle_unknown="ignore")`.
There is no imputer because these version-1 features have no missing values.
Validation log loss selects hyperparameters. Test data are used once after
selection and never influence tuning. The primary classification threshold is
0.50 for every record and group.

## 6. Models and hyperparameter policy

Three dependency-light sklearn families were evaluated:

- Logistic regression: `C` in `{0.1, 1.0, 10.0}`
- Random forest: 160 trees; `(max_depth, min_samples_leaf)` in
  `{(10,20), (None,20), (None,5)}`
- Histogram gradient boosting: `(learning_rate, max_iter, max_leaf_nodes)` in
  `{(0.05,150,15), (0.08,150,31), (0.05,250,31)}`

Random seeds are fixed, forests use square-root feature sampling, and histogram
boosting uses L2 regularization 1.0 without early stopping. This is a small
benchmark grid rather than a large search.

Selected parameters were stable: race-blind logistic selected `C=10` in every
scenario; race-aware logistic selected `C=1` except `C=0.1` in the mixed world.
Every histogram model selected `(0.05,150,15)`. Every forest selected 160 trees
and leaf size 20; only the upstream race-blind forest selected depth 10, while
the others selected unlimited depth.

## 7. Main model selection

Logistic regression had the lowest mean selected-candidate validation log loss
(0.38298), followed by histogram gradient boosting (0.38429) and random forest
(0.38827). Its probabilities also had the lowest errors against the known DGP.
It is therefore the main ML model. This choice was made from validation
performance and interpretability, not test accuracy alone.

## 8. Overall predictive performance

Values below are test ROC-AUC / Brier score / log loss.

| Scenario | Regime | Logistic | Random forest | HistGB |
|---|---|---:|---:|---:|
| Fair baseline | Race-blind | .8291 / .1180 / .3757 | .8268 / .1189 / .3784 | .8276 / .1184 / .3770 |
| Fair baseline | Race-aware | .8291 / .1180 / .3757 | .8261 / .1193 / .3797 | .8278 / .1183 / .3769 |
| Direct discrimination | Race-blind | .8287 / .1194 / .3796 | .8265 / .1203 / .3823 | .8272 / .1198 / .3809 |
| Direct discrimination | Race-aware | .8293 / .1192 / .3791 | .8257 / .1208 / .3838 | .8277 / .1196 / .3804 |
| Upstream inequality | Race-blind | .8285 / .1208 / .3835 | .8262 / .1219 / .3870 | .8271 / .1212 / .3848 |
| Upstream inequality | Race-aware | .8285 / .1208 / .3835 | .8256 / .1222 / .3876 | .8271 / .1212 / .3847 |
| Mixed mechanism | Race-blind | .8285 / .1225 / .3878 | .8267 / .1231 / .3900 | .8271 / .1227 / .3890 |
| Mixed mechanism | Race-aware | .8292 / .1223 / .3873 | .8269 / .1234 / .3908 | .8277 / .1226 / .3884 |

Accuracy ranged from 0.8261 to 0.8341. Average precision ranged from 0.9412 to
0.9466. Full accuracy, balanced accuracy, average precision, prevalence, and
prediction-rate columns are in `results/tables/ml_benchmark.csv`.

## 9. Predicted disparity results

The held-out observed gaps differ slightly from Prompt 6's full-sample gaps
because these use only the fixed test partition. Values are percentage points.

| Scenario | Observed label gap | Model | Race-blind mean gap | Race-aware mean gap | Race-blind threshold gap | Race-aware threshold gap |
|---|---:|---|---:|---:|---:|---:|
| Fair baseline | -0.149 | Logistic | 0.275 | 0.042 | 0.390 | 0.134 |
|  |  | Random forest | 0.196 | 0.007 | 0.320 | 0.193 |
|  |  | HistGB | 0.304 | 0.196 | 0.443 | 0.466 |
| Direct discrimination | -3.383 | Logistic | 0.280 | -2.747 | 0.250 | -2.912 |
|  |  | Random forest | 0.222 | -1.223 | 0.146 | -0.760 |
|  |  | HistGB | 0.320 | -2.024 | 0.615 | -1.823 |
| Upstream inequality | -6.866 | Logistic | -6.205 | -6.316 | -7.675 | -7.743 |
|  |  | Random forest | -5.781 | -6.148 | -6.043 | -5.873 |
|  |  | HistGB | -6.279 | -6.320 | -7.429 | -7.446 |
| Mixed mechanism | -10.680 | Logistic | -6.423 | -10.414 | -7.921 | -12.493 |
|  |  | Random forest | -6.139 | -8.580 | -6.683 | -8.569 |
|  |  | HistGB | -6.476 | -9.642 | -8.060 | -11.509 |

The probability-gap reproduction error is model mean-prediction gap minus the
observed label gap. For the main logistic model it was 0.424, 3.663, 0.661, and
4.257 points in the fair, direct, upstream, and mixed race-blind runs. It was
0.191, 0.637, 0.550, and 0.266 points when race was included. A smaller error
means closer reproduction of label disparity; it does not mean a model is more
fair.

## 10. Group audit

The audit retains race even where the model does not use it. For the main
logistic model, White test `n=13,471` and Black test `n=2,412` in every
scenario. Selected White/Black diagnostics are shown below as
`White / Black`.

| Scenario | Regime | Actual approval | Mean prediction | Predicted approval at .50 | ROC-AUC | Brier | TPR | FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Fair | Blind | .801/.799 | .798/.801 | .892/.896 | .828/.837 | .118/.118 | .953/.954 | .644/.661 |
| Fair | Aware | .801/.799 | .798/.799 | .892/.893 | .828/.837 | .118/.118 | .953/.953 | .643/.653 |
| Direct | Blind | .801/.767 | .795/.798 | .889/.891 | .828/.836 | .118/.129 | .951/.959 | .637/.667 |
| Direct | Aware | .801/.767 | .798/.771 | .891/.862 | .828/.836 | .118/.128 | .953/.943 | .643/.598 |
| Upstream | Blind | .801/.732 | .798/.736 | .892/.815 | .828/.823 | .118/.141 | .954/.917 | .644/.537 |
| Upstream | Aware | .801/.732 | .798/.735 | .892/.814 | .828/.823 | .118/.141 | .953/.917 | .644/.534 |
| Mixed | Blind | .801/.694 | .794/.729 | .886/.807 | .828/.817 | .118/.155 | .950/.918 | .632/.557 |
| Mixed | Aware | .801/.694 | .798/.694 | .891/.766 | .828/.817 | .118/.153 | .953/.892 | .643/.481 |

TPR and FPR here measure agreement with the simulated lender's decisions. They
do not establish equal opportunity, equalized odds, qualification, repayment,
or normative fairness because there is no independent repayment outcome.

## 11. Recovery of the synthetic true approval probability

For the main logistic model, overall MAE / RMSE against `p_true` were:

| Scenario | Race-blind | Race-aware |
|---|---:|---:|
| Fair baseline | .0122 / .0184 | .0124 / .0186 |
| Direct discrimination | .0148 / .0220 | .0123 / .0186 |
| Upstream inequality | .0124 / .0187 | .0125 / .0189 |
| Mixed mechanism | .0158 / .0236 | .0127 / .0191 |

Correlations with true probability ranged from 0.9938 to 0.9959 for logistic
models. In direct and mixed scenarios, withholding race increased Black
overprediction relative to truth: 2.70 and 3.28 points, respectively. Adding
race reduced those Black mean errors to 0.02 and -0.25 points. These metrics
evaluate recovery of the synthetic probability function; they do not use
`p_true` for training.

Binary prediction remains imperfect even for the true probability because
`approved` is a Bernoulli draw. The non-trained oracle's test Brier scores were
0.1175, 0.1186, 0.1203, and 0.1217 across the four scenarios. Its ROC-AUC values
were approximately 0.830–0.831. This is a reference for irreducible label
randomness, not a competing model.

## 12. Scenario interpretation

### Fair baseline

Both feature regimes remained near zero group disparity. Race did not improve
meaningful predictive performance, as expected when it has no DGP role.

### Direct discrimination

Race-blind models produced slightly positive mean-prediction gaps even though
the observed Black approval gap was negative. The direct cause was unavailable
and non-race features were comparable, so they could not recover the inserted
penalty. Race-aware models reproduced substantially more of the negative gap;
logistic regression came closest.

### Upstream inequality

Race-blind models reproduced substantial negative gaps because income, credit,
and assets differed by race and remained visible to the model. Adding race
changed logistic predictions by only about -0.11 points after those features
were known. Unequal predictions here reflect learning from group-correlated
financial patterns in this DGP, not a direct race term.

### Mixed mechanism

Race-blind models reproduced roughly the upstream portion but missed the
withheld direct component. Race-aware models moved closer to the total observed
and true-probability gaps. Logistic mean predictions changed from -6.423 to
-10.414 points when race was added.

## 13. Comparison with Prompt 6

- Fair: adjusted statistical and predictive results were near zero.
- Direct: statsmodels adjustment recovered the direct coefficient because race
  was explicitly included; race-blind ML could not reproduce that pathway.
- Upstream: the adjusted Black coefficient moved to zero after conditioning,
  while race-blind ML still generated unequal group predictions from the
  shifted financial inputs.
- Mixed: statsmodels isolated the direct component; race-blind ML learned the
  upstream component, while race-aware ML learned both more fully.

This contrast is substantive: a conditional coefficient and an unconditional
gap in model predictions answer different questions.

## 14. Threshold behavior and other cautions

At threshold 0.50, models predicted approval for roughly 88–91% of records even
though observed prevalence was about 79–80%. This is not a calibration failure:
many well-calibrated probabilities exceed 0.50, while some of their Bernoulli
outcomes are denials. It explains why threshold-based gaps can be larger than
mean-probability gaps and why accuracy alone is incomplete.

Tree ensembles offered no meaningful gain over logistic regression for this
logistic, mostly additive DGP. Race-aware forests and boosting also recovered
less of the direct disparity than race-aware logistic regression. Predictive
feature importance was deferred; it is not needed for this benchmark and would
not be causal evidence.

## 15. Limitations and future HMDA implications

This is one deterministic simulation run, not a Monte Carlo sampling study.
Its features and functional form are cleaner than observational lending data.
Synthetic recovery cannot establish real-world discrimination, and removing
race does not guarantee fair predictions when upstream features differ across
groups. Future HMDA work must use disparity language, prevent outcome leakage,
and acknowledge that true approval probabilities and causal mechanisms are
unknown.

## 16. Next step

The validated result tables can now support an interactive dashboard that calls
the same simulation and analysis APIs. Dashboard work should visualize these
estimands without relocating scientific logic into Streamlit.
