# Research Design

## 1. Research questions

The primary, accessible research question is:

> Do mortgage lending decisions differ across demographic groups, and can
> machine learning help detect those differences?

The more specific empirical question is:

> Are some demographic groups more likely to be denied mortgages after
> accounting for differences in borrower and loan characteristics?

The synthetic phase turns these questions into controlled experiments. Each
experiment will specify how demographic attributes, economic conditions, loan
requests, and approval decisions are related. The analysis can then be compared
with the mechanism that actually generated the data.

## 2. Why synthetic simulation is used first

Observed mortgage data can reveal disparities, but it cannot by itself reveal
which causal mechanism produced them. Differences may arise from observed
financial characteristics, direct differential treatment, upstream inequality,
geography, lender composition, omitted variables, measurement error, or several
mechanisms together.

Synthetic data provides known ground truth. The project can deliberately turn
direct and upstream pathways on or off, repeat samples under controlled
conditions, and measure when an analysis correctly detects or misses those
pathways. Synthetic results demonstrate statistical behavior under stated
assumptions; they do not establish that the same mechanism exists in the real
world.

## 3. Unit of observation

One row represents one synthetic mortgage application submitted by one
applicant to the simulated lending environment. It is an application record,
not a person-level panel and not a completed mortgage.

Information available before the decision includes:

- Protected or sensitive demographic attributes used to define groups for
  research and fairness auditing.
- Borrower financial characteristics known or measured during underwriting.
- Requested loan and property characteristics.
- Neighborhood or local-market context available at application time.
- Derived underwriting ratios computed from pre-decision information.

The outcomes are the true approval probability specified by the simulation,
the sampled binary approval decision, and an optional denial reason. These
outcomes must never be used as predictors of that same approval decision.

## 4. Variable taxonomy

The initial schema contains 24 variables:

- Identifier: application_id.
- Protected attributes: race, ethnicity, sex, and age_group.
- Borrower financial characteristics: annual_income, credit_score,
  employment_years, liquid_assets, existing_monthly_debt, and
  debt_to_income_ratio.
- Loan and property characteristics: loan_amount, property_value,
  loan_to_value_ratio, loan_purpose, loan_type, and occupancy_type.
- Neighborhood and context variables: neighborhood_income_index,
  neighborhood_minority_share, and local_unemployment_rate.
- Derived underwriting variable: income_to_loan_ratio.
- Outcomes and simulation truth: approval_probability_true, approved, and
  denial_reason.

The authoritative field-level table is in
[synthetic_schema.md](synthetic_schema.md).

Protected attributes are retained for descriptive analysis, estimating group
gaps, and fairness auditing. They should be excluded from the baseline
predictive underwriting feature set. A clearly labeled research model may
include a protected attribute when its purpose is to estimate or test a group
coefficient, and a scenario may intentionally insert a protected-attribute term
in the data-generating equation. Those are experimental uses, not a
recommendation for operational underwriting.

Neighborhood minority share is pre-decision information but is also a strong
candidate proxy for protected group and structural segregation. It should be
excluded from the initial baseline predictor set and introduced only in a
documented sensitivity or proxy-discrimination experiment.

## 5. Proposed synthetic schema

The schema is compact rather than a reproduction of HMDA. It retains variables
needed to represent repayment capacity, creditworthiness, accumulated
resources, leverage, the requested product, and local context.

Several fields have intentional relationships:

- Loan-to-value ratio is loan amount divided by property value.
- Income-to-loan ratio is annual income divided by loan amount.
- Debt-to-income ratio represents monthly debt obligations used in
  underwriting divided by gross monthly income. Version 1 combines existing
  debt with a non-persisted projected housing-payment approximation.
- High-DTI and high-LTV indicators are not persisted in the initial schema.
  They can be derived later for analyses that declare explicit thresholds.
- Denial reason is downstream of the decision. It may remain unimplemented or
  null in the first generator until a defensible reason-assignment mechanism is
  designed.

Allowed categories and field roles are documented in
[synthetic_schema.md](synthetic_schema.md). The version-1 distributions,
dependencies, bounds, coefficients, and treatment levels are documented in
[simulation_calibration.md](simulation_calibration.md).

## 6. Causal and data-generating process

The proposed causal order is:

    Protected attributes          Exogenous/local context
              \                       /
               \                     /
                v                   v
             Upstream socioeconomic conditions
                           |
                           v
            Income, credit, employment, assets, debt
                           |
                           v
              Requested loan and property features
                           |
                           v
                  Derived DTI/LTV/capacity ratios
                           |
                           v
                   Latent underwriting score
                           |
                           v
            True approval probability -> approval draw
                                          |
                                          v
                                  optional denial reason

    Protected attribute ----------------> latent underwriting score
                      optional direct-discrimination path

Three types of arrows must remain distinguishable:

### A. Underwriting relationships

Borrower capacity, creditworthiness, debt burden, leverage, and selected loan
features enter the latent underwriting score. Calling an arrow an underwriting
relationship describes its role in the simulation; it is not a legal or
normative conclusion that every possible use of that variable is fair.

### B. Structural or upstream inequality

A protected attribute can alter the distribution of neighborhood conditions or
economic resources, which then alter income, credit, assets, debt, or loan
characteristics. There is no protected-attribute term in the final approval
equation for an upstream-only scenario.

### C. Direct discrimination

A protected attribute can enter the latent underwriting score directly after
the modeled financial and loan characteristics are held fixed. This direct
path is present only when explicitly enabled.

The first focal protected attribute is race, with White as the reference and
Black as the initial comparison. Financial and loan distributions are equal
across race in fair_baseline except for sampling noise. Other protected
attributes remain available for auditing and receive no direct effect. Version
1 draws the demographic attributes independently as an explicit simplification.

## 7. Four core scenarios

### Scenario 1: Fair baseline

The focal group's representation may be unequal, but its categories share the
same financial, loan, and context distributions. The protected attribute does
not enter the approval equation. Estimated group effects should center on zero
across repeated samples, apart from sampling variation and predeclared model
error.

### Scenario 2: Direct discrimination

The financial and loan distributions remain comparable across focal groups,
but a configured demographic penalty enters the latent approval score.
Conditional analyses should detect a remaining group disparity after relevant
borrower and loan controls are included.

### Scenario 3: Upstream inequality

The focal protected attribute changes specified upstream distributions, such as
income, credit score, assets, debt burden, or neighborhood conditions. It does
not enter the approval equation directly. Raw approval rates should differ, but
the estimated group gap should shrink substantially when the complete and
correct set of mediating financial variables is included.

### Scenario 4: Mixed mechanism

Specified upstream group differences and a direct demographic penalty are both
active. Adjustment should remove the portion associated with modeled upstream
characteristics while leaving evidence of the direct path, subject to sampling
variation and correct model specification.

Formal scenario invariants and qualitative expectations are recorded in
[simulation_scenarios.md](simulation_scenarios.md).

## 8. Approval mechanism

Approval will use a logistic latent-score framework:

    z_i = alpha
        + beta_income * T_income(annual_income_i)
        + beta_credit * T_credit(credit_score_i)
        + beta_assets * T_assets(liquid_assets_i)
        + beta_employment * T_employment(employment_years_i)
        - beta_dti * T_dti(debt_to_income_ratio_i)
        - beta_ltv * T_ltv(loan_to_value_ratio_i)
        + selected categorical loan and context effects
        + delta(D_i)

    p_i = sigmoid(z_i) = 1 / (1 + exp(-z_i))

    approved_i ~ Bernoulli(p_i)

Here, delta(D_i) is zero in the fair and upstream-only scenarios. In a direct
or mixed scenario it is a configured category-specific log-odds effect relative
to a declared reference group. A penalty has a negative sign. Because the
logistic function is nonlinear, one log-odds penalty does not imply the same
percentage-point change for every applicant. The project should preserve both
the configured coefficient and a probability-scale ground-truth contrast.

A logistic mechanism is useful because it produces valid probabilities,
supports stochastic decisions among otherwise similar applicants, and provides
a transparent benchmark for later logistic-regression recovery. The true
approval probability is retained as approval_probability_true, information
that is available only because the data is synthetic.

Expected directional relationships are:

- Higher income, credit score, liquid assets, and employment stability should
  generally increase approval probability, conditional on the specification.
- Higher DTI and LTV should generally decrease approval probability.
- Loan amount and property value do not have unambiguous standalone signs after
  capacity and leverage are represented.
- Effects for loan purpose, loan type, occupancy, and local context require an
  explicit research justification before their signs are fixed.

Version-1 transforms and slopes are calibrated in
[simulation_calibration.md](simulation_calibration.md). The fair-baseline
intercept will be solved once against a target mean approval probability and
then frozen across scenarios. Coefficients are not estimated from HMDA and the
experimental direct and upstream effects are not empirical claims.

## 9. Raw, adjusted, and true disparity

These quantities answer different questions and must not be used
interchangeably.

### Raw disparity

For focal groups A and B, a simple probability-scale contrast is:

    mean(approved | A) - mean(approved | B)

It combines all pathways that make the groups differ in the simulated sample.
Approval-rate ratios may also be reported, with the favorable outcome and
reference group stated explicitly.

### Adjusted disparity

An adjusted disparity is an estimated conditional or standardized group
difference after including a declared control set. The project will compare:

- Demographic group only.
- Demographic group plus borrower controls.
- Demographic group plus borrower and loan controls.

The preferred probability-scale summary will be an average marginal effect or
standardized predicted-probability contrast, accompanied by the underlying
regression coefficient. It remains model-dependent and is not automatically a
causal estimate.

### True direct effect

The simulation's true direct effect is the exact category-specific term
delta(D_i) inserted into the latent score. It should be stored in run metadata.
A second ground-truth summary can average, over a fixed synthetic population,
the difference between approval probabilities calculated with the focal
category's direct term and with the reference term while holding all other
inputs fixed.

The log-odds parameter, its probability-scale average effect, the adjusted
regression estimate, and a raw approval-rate gap are different numerical
objects. Logistic coefficients are also non-collapsible, so coefficient changes
across nested models cannot by themselves be interpreted as the amount of
mediation. Probability-scale contrasts and known DGP quantities are essential.

## 10. Potential extension: bad controls / post-treatment controls

Controlling for every observable characteristic does not automatically identify
discrimination. Income, wealth, credit conditions, neighborhood, or even loan
choices may themselves reflect prior unequal treatment or structural
inequality. Conditioning on such mediators can remove part of the pathway the
researcher intends to measure and make a demographic coefficient appear small.

This is a future extension. It will require a clearly defined causal estimand
and simulations that label pre-treatment confounders, mediators, and any true
post-decision variables before model comparisons are interpreted causally.

## 11. Reproducibility strategy

Every future simulation run must accept and record:

- random_seed
- n_samples
- scenario
- focal protected attribute and reference category
- population shares and dependence assumptions
- variable distribution specifications
- approval-model transforms and coefficients
- direct and upstream scenario effects
- derived-variable definitions
- output locations and format

The resolved configuration must be serialized with each run. The planned YAML
shape is in [schema.yaml](../configs/simulation/schema.yaml); it remains a
design template. The canonical numerical calibration is in
[baseline.yaml](../configs/simulation/baseline.yaml), experimental treatments
are in [effect_sizes.yaml](../configs/simulation/effect_sizes.yaml), and
scenario files contain only mechanism switches and selected effect levels.

The implementation should use a local NumPy random generator initialized from
the recorded seed rather than global random state. Outputs should include the
resolved configuration, run identifier, timestamps, software version
information, and a code revision when available. Data should use Parquet;
summary tables may use CSV or Parquet and metadata should use YAML or JSON.

### Future generator validation

General checks:

- The same seed and resolved configuration produce identical data.
- A different seed changes stochastic draws.
- The row count is exactly n_samples and application identifiers are unique.
- Generated columns, data types, category levels, and conditional nulls match
  the schema.
- Approval probabilities are finite and in the closed interval from zero to
  one.
- The binary outcome contains only zero and one.
- DTI and LTV satisfy configured bounds; exceptions require an explicit
  scenario.
- Algebraic derived fields agree with their source columns.
- Directional perturbation tests recover the intended signs in the approval
  equation.
- Realized Bernoulli approval frequencies are consistent with stored true
  probabilities within sampling variation.

Scenario checks:

- Fair baseline: the focal protected attribute changes neither upstream
  distributions nor true approval probability conditional on generated inputs.
- Direct discrimination: the direct term enters the latent score exactly as
  configured, with upstream group shifts disabled.
- Upstream inequality: group shifts occur only in declared upstream
  distributions and no demographic term enters the approval equation.
- Mixed mechanism: both the declared upstream shifts and exact direct term are
  active.

## 12. Planned experiment sizes

Sample size will be configurable rather than embedded in source code:

- 10,000 rows for debugging and rapid validation.
- 100,000 rows for routine single-run experiments.
- 1,000,000 rows for selected large-sample or performance experiments.

Vectorized NumPy generation and efficient pandas dtypes make the first two sizes
comfortable on a typical laptop. One million rows is generally practical, but
the 25-column table plus temporary arrays, object-string columns, copies, and
model matrices can consume hundreds of megabytes or more. Categorical dtypes,
Parquet output, limited copies, and possibly chunked writing should be used.
Large Monte Carlo grids should begin at smaller sizes because repeated
million-row model fits can be substantially more expensive than generation.
The version-1 replication counts and child-seed policy are specified in
[simulation_calibration.md](simulation_calibration.md#19-planned-monte-carlo-sizes).

## 13. Transition to HMDA later

Only after the generator and analytical framework are validated will the
project map the compact synthetic concepts to 2024 HMDA fields. The HMDA stage
will use pre-decision predictors and exclude outcome-revealing or post-decision
fields from approval models. The synthetic approval probability and known
direct effect have no real-data equivalent.

Observed HMDA gaps will be described as raw or adjusted disparities, not
automatically as discrimination. Synthetic experiments can clarify how methods
behave under assumed mechanisms, but they cannot prove which mechanism
generated an observational HMDA disparity.

## Research cautions

- Synthetic data demonstrates implications of assumptions, not the existence
  of real-world discrimination.
- An observed disparity is not automatically evidence of discrimination.
- An adjusted disparity is not automatically a causal estimate.
- Protected attributes should be held out of baseline predictive underwriting
  models and retained for auditing, unless a clearly labeled experiment
  intentionally uses them.
- True discrimination exists in a synthetic scenario only because the
  data-generating process explicitly encodes it.
- Without a separate qualification or repayment outcome, true-positive-rate
  and false-positive-rate gaps describe agreement with the simulated lender's
  decision, not error relative to a normatively correct lending decision.
- A protected-attribute-blind classifier may produce similar prediction rates
  across groups in a direct-only scenario when no proxy exists, even though the
  simulated approval labels contain a direct penalty. Detection therefore
  requires group-aware auditing of labels, residuals, and calibration rather
  than reliance on prediction parity alone.
