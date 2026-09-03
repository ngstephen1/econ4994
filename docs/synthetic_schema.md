# Proposed Synthetic Mortgage Application Schema

## Scope and conventions

One row is one synthetic mortgage application. The initial schema contains 24
research-oriented variables rather than attempting to reproduce every HMDA
field.

“Pre-decision” means the value exists, or can be computed, before the simulated
approval draw. It does not imply that use of the variable is legally or
normatively appropriate. “ML use” describes the planned baseline predictive
model:

- Eligible: may be considered after the exact specification is declared.
- Audit only: excluded from baseline underwriting predictors but retained for
  disparity measurement and fairness auditing.
- Sensitivity only: excluded initially because the field may act as a proxy or
  create a causal-interpretation problem.
- Never: identifier, target, post-decision field, or hidden simulation truth.

Protected attributes may still appear in explicitly labeled statistical
research models that estimate group gaps. That is distinct from using them in
an operational approval predictor.

The public-HMDA compatibility class for every field is documented in
[simulation_calibration.md](simulation_calibration.md#16-synthetic-only-versus-hmda-compatible-variables).

## Field-level schema

| Variable | Type | Unit or allowed values | Meaning and causal role | Pre-decision | ML use | Protected / audit role |
|---|---|---|---|---|---|---|
| application_id | string | Unique, non-null identifier | Identifies an application; no substantive causal role | Administrative | Never | No |
| race | category | White; Black; Asian; American Indian or Alaska Native; Other / Multiracial | Protected group attribute and possible focal treatment in scenario paths | Yes | Audit only | Yes |
| ethnicity | category | Hispanic or Latino; Not Hispanic or Latino | Protected group attribute and possible focal treatment | Yes | Audit only | Yes |
| sex | category | Male; Female | Protected group attribute and possible focal treatment | Yes | Audit only | Yes; initial binary scope is a simplification |
| age_group | ordered category | 18–24; 25–34; 35–44; 45–54; 55–64; 65+ | Sensitive age band used for subgroup analysis; may constrain plausible employment history | Yes | Audit only | Yes / sensitive |
| annual_income | float | USD per year; nonnegative | Borrower repayment capacity; may be affected by upstream opportunity | Yes | Eligible | No; possible mediator |
| credit_score | integer | Synthetic score from 300 through 850 | Summary of modeled creditworthiness; may be affected by upstream conditions | Yes | Eligible | No; possible mediator |
| employment_years | float | Years; nonnegative and logically compatible with age band | Employment stability and labor-market history | Yes | Eligible | No; possible mediator |
| liquid_assets | float | USD; nonnegative | Liquid resources and down-payment/reserve capacity | Yes | Eligible | No; possible mediator |
| existing_monthly_debt | float | USD per month; nonnegative | Debt obligations existing before the proposed mortgage | Yes | Eligible | No; possible mediator |
| debt_to_income_ratio | float | Proportion from zero through 0.65 in version 1 | Existing monthly debt plus a non-persisted projected housing payment, divided by gross monthly income and winsorized | Yes | Eligible | No; derived mediator |
| loan_amount | float | USD; positive | Requested principal; depends partly on property choice, resources, and financing need | Yes | Eligible with specification caution | No |
| property_value | float | USD; positive | Value of the property associated with the application | Yes | Eligible with specification caution | No |
| loan_to_value_ratio | float | Proportion; positive with a configured plausible maximum | Loan amount divided by property value; measures leverage | Yes | Eligible | No; derived from loan fields |
| loan_purpose | category | Home purchase; refinance; home improvement | Purpose of the requested financing | Yes | Eligible after effects are justified | No |
| loan_type | category | Conventional; FHA; VA; USDA/RHS | Requested mortgage program or product family | Yes | Eligible after effects are justified | No |
| occupancy_type | category | Principal residence; second residence; investment property | Intended occupancy of the property | Yes | Eligible after effects are justified | No |
| neighborhood_income_index | float | Positive index relative to a reference level | Local socioeconomic context; may be generated downstream of structural inequality | Yes | Sensitivity initially | No; contextual mediator/proxy risk |
| neighborhood_minority_share | float | Proportion in the closed interval from zero to one | Neighborhood composition and potential segregation/proxy pathway | Yes | Sensitivity only | Audit/context; strong proxy risk |
| local_unemployment_rate | float | Proportion in the closed interval from zero to one | Local labor-market conditions that may affect financial stability | Yes | Eligible or sensitivity, depending on estimand | No; contextual mediator |
| income_to_loan_ratio | float | Annual income divided by loan amount; nonnegative | Derived capacity measure linking requested principal to income | Yes | Eligible, but avoid redundant specifications | No |
| approval_probability_true | float | Probability in the closed interval from zero to one | Exact DGP probability after all enabled effects; synthetic ground truth | No; hidden simulation truth | Never | Outcome/truth |
| approved | integer | 1 = approved; 0 = denied | Bernoulli draw from approval_probability_true; primary target | No; outcome | Target only | Outcome for group auditing |
| denial_reason | nullable category | Null for every version-1 record | Persisted but disabled until a later denial-reason mechanism is designed | No; post-decision | Never | Outcome diagnostic only |

## Variable relationships and consistency rules

- loan_to_value_ratio must equal loan_amount divided by property_value within
  numerical tolerance.
- income_to_loan_ratio must equal annual_income divided by loan_amount within
  numerical tolerance.
- approval_probability_true must be finite and bounded from zero through one.
- approved must be a binary Bernoulli draw using approval_probability_true.
- denial_reason must be null when approved equals one. Its mechanism is deferred
  and it may remain null for all rows in the first generator.
- Employment years must not contradict the declared age-band consistency rule.
- DTI construction must declare whether and how projected housing expense is
  combined with existing monthly debt.

## Baseline predictor policy

The baseline predictive underwriting feature set should exclude:

- application_id
- race
- ethnicity
- sex
- age_group
- neighborhood_minority_share
- approval_probability_true
- approved, except as the target
- denial_reason

Neighborhood income and unemployment require a declared causal estimand before
inclusion because conditioning on context can remove part of an upstream
pathway. High-DTI and high-LTV indicators may be computed later for analysis,
but are not persisted because they would be deterministic copies of information
already present in the continuous ratios.

## Version-1 calibration status

The schema remains fixed at 24 fields. Version-1 numerical decisions are in
[simulation_calibration.md](simulation_calibration.md) and the configuration
files under configs/simulation.

The numerical intercept remains a derived value rather than a researcher-chosen
constant: the future implementation will solve it on the fixed baseline
calibration population, save it in the resolved configuration, and reuse it
across scenarios. Empirical clipping rates and realized distribution summaries
must also be validated before experiments begin.
