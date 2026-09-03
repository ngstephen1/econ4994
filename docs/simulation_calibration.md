# Version 1 Simulation Calibration — Revision 2

## 1. Calibration goals

This document specifies the numerical assumptions for the first synthetic
mortgage-application experiments. The calibration is intended to be plausible,
transparent, testable, and easy to explain. It is not intended to reproduce the
U.S. mortgage market exactly or to estimate discrimination in real lending.

The version-1 design must:

- Produce economically coherent borrower, property, and loan relationships.
- Keep the four scenario mechanisms clean and separately configurable.
- Preserve a known logistic approval probability for every application.
- Make numerical choices traceable to either evidence or researcher judgment.
- Remain practical for samples from 10,000 through 1,000,000 rows.

The configuration is split across baseline.yaml, effect_sizes.yaml, and four
small files in configs/simulation/scenarios. The existing schema.yaml remains a
non-runnable design template.

Revision 2 changes only two researcher-chosen property-value dependency
parameters after the DTI-tail diagnosis: income elasticity is 0.90 rather than
0.55, and residual log SD is 0.18 rather than 0.30. The diagnosis and rejected
alternatives are documented in [dti_recalibration.md](dti_recalibration.md).

## 2. Empirical versus experimental assumptions

Every important numerical choice uses one of these labels:

| Label | Meaning |
|---|---|
| A — Empirically motivated approximation | Anchored to an authoritative aggregate statistic or established definition, then simplified for this synthetic application population |
| B1 — Researcher-chosen plausible calibration | Chosen for coherence, variation, and interpretability where no clean public estimate maps to the synthetic field |
| B2 — Researcher-chosen experimental treatment | Deliberately inserted direct or upstream effect used to test method recovery |

B2 values are not estimates of actual Black–White differences or real-world
discrimination. B1 values are also not empirical claims; they are transparent
modeling choices.

## 3. Focal demographic comparison

The first experiment uses:

| Item | Version-1 choice | Label | Reason |
|---|---|---|---|
| Focal protected attribute | race | Design choice | Directly matches the initial research experiment |
| Reference category | White | Design choice | Required comparison reference |
| Comparison category | Black | Design choice | Produces one clear, well-powered initial contrast |
| Direct effects on other races | Zero | B2 | Isolates the Black-versus-White treatment |
| Direct effects on ethnicity, sex, age | Zero | B2 | Prevents multiple estimands from being mixed |

All five race categories remain in the population. Ethnicity, sex, and age
group remain available for descriptive analysis and auditing.

## 4. Demographic shares

These are shares of a selected synthetic mortgage-application population, not
U.S. resident population shares.

| Attribute | Category shares | Label | Rationale |
|---|---|---|---|
| Race | White 0.67; Black 0.12; Asian 0.09; American Indian or Alaska Native 0.01; Other / Multiracial 0.11 | A | Broadly anchored to recent HMDA mortgage activity, with joint, missing, and combined race/ethnicity reporting simplified into this project's five-category race schema |
| Ethnicity | Hispanic or Latino 0.14; Not Hispanic or Latino 0.86 | A | Approximation informed by recent HMDA activity, separated from race for this simulation |
| Sex | Male 0.54; Female 0.46 | B1 | Balanced applicant mix with a modest male majority; public aggregates do not map cleanly to this two-category synthetic definition |
| Age group | 18–24 0.03; 25–34 0.24; 35–44 0.28; 45–54 0.21; 55–64 0.15; 65+ 0.09 | B1 | Concentrates applications in prime home-buying ages while retaining all schema groups |

Version 1 draws race, ethnicity, sex, and age group independently. This is a
deliberate simplification, not a claim about real demographic relationships.
Because age affects income and employment history, age groups will differ
financially. Race is independent of age in version 1, so those age mechanisms
do not create a race pathway in fair_baseline.

## 5. Financial distributions

Two correlated latent factors organize the population without creating a large
structural model:

- Financial stability F is standard normal.
- Neighborhood advantage N is standard normal.
- Their correlation is 0.20.

The correlation is B1. It creates sensible co-movement between local conditions
and financial resources while leaving substantial individual variation.

| Variable | Version-1 distribution or equation | Bounds | Label and rationale |
|---|---|---|---|
| annual_income | Conditional lognormal; reference median $95,000; log residual SD 0.40; coefficients 0.20 on F and 0.12 on N; age multipliers from 0.55 to 1.10 | $20,000–$500,000, clipped | A for level and skew; B1 for dependency coefficients. Mortgage-applicant income is right-skewed and selected above the general population |
| credit_score | Conditional normal centered at 720; residual SD 40; plus 25 points per unit F and 10 points per standardized log-income unit | 500–850, rounded and clipped | A for range and center; B1 for dependencies. Recent CFPB mortgage summaries report materially different medians across borrower and product groups |
| employment_years | Internal age drawn uniformly within age band; accumulated history equals an age-feasible maximum times a clipped employment fraction centered at 0.55, shifted 0.12 by F, residual SD 0.18 | 0 through min(age − 18, 45) | B1. It guarantees age compatibility without modeling job spells |
| liquid_assets | Conditional lognormal; reference median $35,000; log residual SD 0.70; income elasticity 0.65; coefficient 0.25 on F; age multipliers 0.20–1.80 | $1,000–$750,000, clipped | B1, with direction supported by Federal Reserve household-savings evidence |
| existing_monthly_debt | 15% zero; otherwise conditional lognormal with $650 positive median, log residual SD 0.65, income elasticity 0.35, and coefficient −0.12 on F | $0–$6,000, clipped | B1. The mixture allows debt-free applicants while retaining a right tail |
| debt_to_income_ratio | Back-end DTI from existing debt plus an internal projected housing payment, divided by gross monthly income | 0–0.65, winsorized | A for definition and broad level; B1 for tail handling |

Age-group income multipliers are:

| Age group | Income multiplier | Asset multiplier |
|---|---:|---:|
| 18–24 | 0.55 | 0.20 |
| 25–34 | 0.90 | 0.65 |
| 35–44 | 1.05 | 1.00 |
| 45–54 | 1.10 | 1.35 |
| 55–64 | 1.05 | 1.65 |
| 65+ | 0.85 | 1.80 |

These multipliers are B1. They encode life-cycle patterns transparently, not
causal estimates.

## 6. Dependency and correlation design

The generation order is:

    demographic draws
        -> correlated F and N latent factors
        -> neighborhood context
        -> income and employment history
        -> credit score, assets, and existing debt
        -> loan purpose/type/occupancy
        -> property value and LTV
        -> loan amount
        -> internal housing payment and DTI
        -> underwriting score and approval probability

Important dependencies are intentionally limited:

- Income rises with F, N, and middle working-age multipliers.
- Credit score rises with F and, modestly, with standardized log income.
- Assets rise with income, F, and age.
- Existing debt rises less than proportionally with income and falls with F.
- Property value rises with income and neighborhood income.
- LTV changes by product, purpose, and occupancy.

No race term enters any of these equations in fair_baseline or
direct_discrimination. Only effect_sizes.yaml may activate the three declared
Black upstream shifts.

## 7. DTI construction

Version 1 uses a transparent back-end DTI approximation:

    monthly_PI = loan_amount × 0.00648598

    monthly_tax_insurance = property_value × 0.0012

    projected_housing_payment = monthly_PI + monthly_tax_insurance

    raw_DTI = (existing_monthly_debt + projected_housing_payment)
              / (annual_income / 12)

    debt_to_income_ratio = min(max(raw_DTI, 0), 0.65)

The principal-and-interest factor is the standard 360-month amortization factor
at a 6.75% annual rate. The rate is an A approximation to the roughly 6.8%–7.1%
30-year environment reported by Freddie Mac during 2024. The tax-and-insurance
factor, 0.12% of property value per month, is B1.

projected_housing_payment is an internal intermediate and is not persisted.
This preserves the 24-field schema. The generator must report the share
winsorized at 0.65; if more than 5% hit the ceiling, the baseline inputs must be
recalibrated rather than accepting a large artificial boundary mass.

## 8. Loan and property generation

Categorical shares are:

| Variable | Shares | Label and rationale |
|---|---|---|
| loan_purpose | Home purchase 0.70; refinance 0.25; home improvement 0.05 | A approximation, deliberately retaining more refinance records than the high-rate 2023 origination mix so each purpose remains analyzable |
| loan_type | Conventional 0.73; FHA 0.16; VA 0.09; USDA/RHS 0.02 | A approximation to common HMDA product categories |
| occupancy_type | Principal residence 0.88; second residence 0.04; investment property 0.08 | B1 plausible application mix |

Property value is conditional lognormal:

    median base = $400,000
    income elasticity = 0.90
    neighborhood-income elasticity = 0.35
    residual log SD = 0.18
    bounds = $80,000–$2,000,000

The $400,000 anchor is A, informed by recent official home-price and HMDA loan
amount evidence. Elasticities, residual variation, and category multipliers are
B1. The revised elasticity and residual variation align property requests more
closely with income and reduce unrealistic loan/income tail combinations.
Refinance properties receive a 1.05 multiplier, home-improvement
properties 0.75, second residences 1.05, and investment properties 1.10.

Base LTV is a beta(4.0, 2.5) draw scaled to 0.45–0.98. Product offsets are 0.07
for FHA and 0.08 for VA and USDA/RHS. Refinance receives −0.08, home improvement
−0.35, second residence −0.03, and investment property −0.08. Final LTV is
clipped to 0.10–0.98. The broad level is A; beta shapes and offsets are B1.

The two exact identities are:

    loan_amount = property_value × loan_to_value_ratio

    income_to_loan_ratio = annual_income / loan_amount

Neither loan amount nor income-to-loan ratio is independently drawn. This
construction places the typical loan amount near the recent HMDA home-purchase
median while preserving meaningful variation.

## 9. Context-variable generation

All context variables depend on N but not race in version 1:

| Variable | Equation | Bounds | Label |
|---|---|---|---|
| neighborhood_income_index | 1.00 + 0.18N + normal residual with SD 0.12 | 0.40–2.00, clipped | A for index interpretation; B1 for equation |
| neighborhood_minority_share | logistic(−0.85 − 0.45N + normal residual with SD 0.75) | 0–1 by construction | B1 |
| local_unemployment_rate | 0.040 − 0.006N + normal residual with SD 0.010 | 0.015–0.150, clipped | A for 4.0% center; B1 for variation |

The 4.0% unemployment center matches the BLS 2024 annual U.S. unemployment
rate. The neighborhood income index uses 1.0 as area parity, consistent with
HMDA-style tract-to-area income ratios.

All three context variables are excluded from the initial approval equation.
Neighborhood minority share is never a baseline underwriting predictor. The
context fields may later enter explicitly labeled sensitivity analyses. Race
independence in fair_baseline prevents an accidental proxy path.

## 10. Approval equation

The version-1 latent score is:

    z_i = alpha
        + 0.20 x_income
        + 0.65 x_credit
        + 0.10 x_employment
        + 0.20 x_assets
        - 0.70 x_DTI
        - 0.40 x_LTV
        + purpose effect
        + loan-type effect
        + occupancy effect
        + direct race effect

    approval_probability_true = sigmoid(z_i)

    approved_i ~ Bernoulli(approval_probability_true_i)

The slopes are B1 and encode expected underwriting directions. The direct race
effect is B2 and is supplied only by an enabled scenario.

Categorical effects are:

| Variable | Reference | Other log-odds effects | Label |
|---|---|---|---|
| loan_purpose | Home purchase | Refinance +0.05; home improvement −0.15 | B1 |
| loan_type | Conventional | FHA −0.10; VA +0.05; USDA/RHS −0.05 | B1 |
| occupancy_type | Principal residence | Second residence −0.15; investment property −0.30 | B1 |

These small categorical terms provide heterogeneity without dominating credit,
DTI, or LTV. They are not empirical estimates.

The approval equation excludes loan amount, property value,
income_to_loan_ratio, and all context fields. Loan amount and property value
already determine LTV and housing burden; adding all raw components and ratios
would create unnecessary redundancy. Context is withheld because it may be a
mediator or proxy.

## 11. Predictor scaling and transforms

| Predictor | Transformation | Coefficient interpretation |
|---|---|---|
| annual_income | (ln(income) − ln(95,000)) / 0.50 | A one-unit transformed increase, approximately a 65% income ratio, adds 0.20 log-odds |
| credit_score | (score − 720) / 50 | 50 score points add 0.65 log-odds |
| employment_years | (years − 10) / 10 | 10 additional accumulated employment years add 0.10 log-odds |
| liquid_assets | (ln(assets + 1,000) − ln(36,000)) / 1.00 | One log unit of assets adds 0.20 log-odds |
| DTI | (DTI − 0.36) / 0.10 | A 10-percentage-point DTI increase subtracts 0.70 log-odds |
| LTV | (LTV − 0.80) / 0.10 | A 10-percentage-point LTV increase subtracts 0.40 log-odds |

Centering makes the intercept describe an applicant near the reference profile.
Log transforms prevent high-dollar tails from exerting linear leverage.

## 12. Target approval-rate calibration

The fair_baseline target mean approval probability is 0.80. This is an A
approximation for a synthetic sample restricted to resolved approvals and
denials across a mixed-purpose population. CFPB's 2023 report shows materially
different denial rates for home purchase and refinance applications, so 0.80
is a deliberately rounded target rather than a copied market statistic.

The intercept is not hand-picked:

1. Generate a fixed one-million-row fair-baseline calibration population with
   seed 499400.
2. Hold all slopes fixed.
3. Solve alpha by bisection on [−20, 20] until the mean stored probability is
   within 0.0005 of 0.80.
4. Save the numeric alpha and provenance in the calibration artifact.
5. Freeze that alpha for all scenarios and effect levels.

The intercept must not be re-tuned after a direct or upstream effect is enabled,
because doing so would partially erase the treatment's aggregate consequence.

After revision 2, the solved intercept is 2.0362202310934663 and the achieved
one-million-row fair-baseline mean is 0.8000000002705238. The previous intercept,
2.2422267822548747, remains recorded in the artifact provenance.

For a generated sample, the realized approval-rate diagnostic tolerance is:

    3 × sqrt(target × (1 − target) / n_samples) + 0.002

This recognizes Bernoulli variation while still detecting gross calibration
errors.

## 13. Direct-effect levels

Direct effects are B2 experimental treatments. White is zero; Black receives
the stated penalty; all other race categories receive zero.

| Level | Black log-odds effect | Odds ratio | Effect at p=0.50 | Effect at p=0.75 | Effect at target p=0.80 | Effect at p=0.90 |
|---|---:|---:|---:|---:|---:|---:|
| Mild | −0.10 | 0.9048 | −2.50 pp | −1.92 pp | −1.65 pp | −0.94 pp |
| Moderate | −0.25 | 0.7788 | −6.22 pp | −4.97 pp | −4.30 pp | −2.49 pp |
| Strong | −0.50 | 0.6065 | −12.25 pp | −10.47 pp | −9.19 pp | −5.48 pp |

For a baseline probability p and penalty delta, the treated probability is:

    sigmoid(logit(p) + delta)

The changing percentage-point effects demonstrate why a constant log-odds
penalty is not a constant approval-rate penalty. The scenario files use
moderate as the default first treatment; mild and strong are planned sweeps.

## 14. Upstream-effect levels

Upstream effects are also B2 experimental treatments. Only three variables are
shifted for Black applicants relative to White applicants. Other race
categories receive no shift.

| Level | annual_income log-location shift | Conditional income multiplier | credit_score location shift | liquid_assets log-location shift | Conditional asset multiplier |
|---|---:|---:|---:|---:|---:|
| Mild | −0.051293 | 0.95 | −10 points | −0.105361 | 0.90 |
| Moderate | −0.105361 | 0.90 | −25 points | −0.287682 | 0.75 |
| Strong | −0.223144 | 0.80 | −50 points | −0.693147 | 0.50 |

Income represents earning opportunity, credit score summarizes accumulated
credit conditions, and liquid assets represent accumulated resources. They are
upstream of the approval score and loan decision. The log-location shifts
multiply the conditional median before bounds are applied; the credit shift
changes the conditional location before rounding and clipping. Because assets
also depend on income, the income shift creates an additional indirect asset
change. The table's asset multiplier is therefore conditional on income,
financial stability, and age, not the final marginal group ratio.

Shifting only these three variables keeps the pathway interpretable. These
magnitudes are not claims about observed racial gaps.

## 15. Scenario parameter matrix

| Scenario | Upstream switch | Direct switch | Default level | Expected mechanism |
|---|---:|---:|---|---|
| fair_baseline | Off | Off | None | Sampling noise only |
| direct_discrimination | Off | On | Moderate direct | Conditional Black penalty |
| upstream_inequality | On | Off | Moderate upstream | Group gap through three financial mediators |
| mixed_mechanism | On | On | Moderate for both | Upstream and conditional pathways |

Every scenario inherits baseline.yaml. effect_sizes.yaml is a treatment library.
The scenario file only selects switches and effect levels. A future resolver
will merge in this order:

1. schema validation rules
2. baseline.yaml
3. effect_sizes.yaml level selected by the scenario
4. scenario override
5. explicit run-level seed or sample-size override

The complete resolved configuration must be saved with every future run.

## 16. Synthetic-only versus HMDA-compatible variables

This classification refers to the public modified HMDA data, not everything a
lender may collect internally.

| Variable | Class | Later HMDA relationship |
|---|---|---|
| application_id | Synthetic-only | Public HMDA excludes the universal loan identifier; a row ID can be created for analysis but is not equivalent |
| race | Directly or roughly HMDA-compatible | Public applicant race exists, but category construction and missing/joint cases require mapping |
| ethnicity | Directly or roughly HMDA-compatible | Public applicant ethnicity exists, with more missing/joint detail than this schema |
| sex | Directly or roughly HMDA-compatible | Public applicant sex exists; this two-category schema omits joint/not-provided values |
| age_group | Directly or roughly HMDA-compatible | Public age is disclosed in bins; the 65+ aggregation requires mapping |
| annual_income | Directly or roughly HMDA-compatible | Income relied on is public, with reporting and outlier caveats |
| credit_score | Synthetic-only in public HMDA | Lenders report relied-on scores, but applicant-level credit score is excluded from public disclosure |
| employment_years | Synthetic-only | No direct public HMDA field |
| liquid_assets | Synthetic-only | No direct public HMDA field |
| existing_monthly_debt | Synthetic-only | No direct public HMDA field |
| debt_to_income_ratio | Directly or roughly HMDA-compatible | Public DTI is available with reduced precision/categories and exemptions |
| loan_amount | Directly or roughly HMDA-compatible | Public loan amount exists with disclosure modifications |
| property_value | Directly or roughly HMDA-compatible | Public property value exists with disclosure modifications |
| loan_to_value_ratio | Directly or roughly HMDA-compatible | Public combined LTV is available when reported; definitions and missingness require care |
| loan_purpose | Directly or roughly HMDA-compatible | Public HMDA purpose categories require mapping |
| loan_type | Directly or roughly HMDA-compatible | Conventional, FHA, VA, and RHS/FSA concepts map closely |
| occupancy_type | Directly or roughly HMDA-compatible | Public occupancy type maps closely |
| neighborhood_income_index | Directly or roughly HMDA-compatible | Roughly maps to tract-to-area median family income percentage |
| neighborhood_minority_share | Directly or roughly HMDA-compatible | Roughly maps to tract minority population percentage |
| local_unemployment_rate | Synthetic-only | Requires an external geographic data merge; not a public HMDA application field |
| income_to_loan_ratio | Derived | Can be computed from income and loan amount where both are usable |
| approval_probability_true | Hidden simulation truth | No observational HMDA equivalent |
| approved | Hidden truth / outcome | Must be constructed from HMDA action_taken and a declared sample definition |
| denial_reason | Outcome | Public HMDA has coded denial reasons, but it is post-decision and never an approval predictor |

The synthetic schema will not be reduced merely because several variables are
unavailable in public HMDA. Their absence later is itself an important omitted-
variable and transportability limitation.

## 17. Tail handling and validity bounds

| Quantity | Validity rule |
|---|---|
| Monetary variables | Never negative; use the bounds in baseline.yaml |
| annual_income | Clip to $20,000–$500,000 |
| credit_score | Round to integer and clip to 500–850 |
| employment_years | Constrain to 0 through min(internal age − 18, 45) |
| liquid_assets | Clip to $1,000–$750,000 |
| existing_monthly_debt | Allow structural zero; clip positives at $6,000 |
| property_value | Clip to $80,000–$2,000,000 |
| LTV | Clip to 0.10–0.98, then derive loan amount exactly |
| DTI | Winsorize to 0–0.65 and monitor upper-bound mass |
| neighborhood_income_index | Clip to 0.40–2.00 |
| neighborhood_minority_share | Logistic construction guarantees 0–1 |
| local_unemployment_rate | Clip to 0.015–0.150 |

Revision 2 produces 4.2781% DTI upper-bound mass in the designated one-million-
row calibration population, below the unchanged 5% threshold.

Clipping is transparent and efficient but can create point masses. The future
generator must report clipping rates. Any baseline variable with more than 5%
of observations clipped requires recalibration or a move to truncation/rejection
sampling before scientific experiments proceed.

## 18. Reproducibility and seed policy

The canonical routine-run seed is 4994. The intercept-calibration seed is
499400 and is never reused for an experiment.

Future replicated experiments should create independent child streams using
NumPy SeedSequence from base seed 4994. Scenario, effect level, and replication
index must determine the child stream through a stable experiment manifest.
The resolved child seed and spawn key must be saved. Parallel execution must
not change stream assignment.

The same resolved configuration and child seed must reproduce identical data.
No scientific claim should rely on the canonical seed alone.

## 19. Planned Monte Carlo sizes

No simulations are run in this phase.

| Stage | Rows per replication | Replications | Purpose |
|---|---:|---:|---|
| Development | 10,000 | 5 per active scenario/level | Debug invariants and inspect distributions |
| Routine research | 100,000 | 100 per scenario/level | Estimate bias, variability, power, and false-positive behavior |
| Final high-confidence | 100,000 | 500 per selected scenario/level | Stabilize operating-characteristic estimates |
| Selected large-sample | 1,000,000 | 20 per selected scenario/level | Check large-sample recovery and computational scaling |

The final counts may be increased only after runtime benchmarking and a Monte
Carlo precision calculation. Datasets need not all be retained; configurations,
seeds, aggregate results, and selected audit samples are sufficient for
reproducibility.

## 20. Limitations

- Demographic independence is unrealistic and suppresses intersectional
  dependence.
- The application population is not a fitted sample from HMDA.
- The payment approximation uses one rate and term for every application.
- Clipping may create boundary masses.
- Employment history is a simplified stock, not observed job tenure.
- Assets and existing monthly debt are synthetic-only relative to public HMDA.
- The approval score is transparent but not a lender underwriting model.
- Adjusted differences are model-dependent and not automatically causal.
- Logistic coefficients are non-collapsible.
- Conditioning on income, credit, or assets removes an intentionally upstream
  pathway in the structural scenarios.
- TPR and FPR against approved measure prediction of lender labels, not a
  normative qualification outcome.
- Synthetic discrimination exists only because the experiment inserts it.
- Nothing in this calibration is evidence that real lenders discriminate.

## 21. Sources and calibration references

All sources were accessed September 3, 2026.

1. [2023 Mortgage Market Activity and Trends](https://files.consumerfinance.gov/f/documents/cfpb_2023-mortgage-market-activity-and-trends_2024-12.pdf),
   Consumer Financial Protection Bureau. Informed mortgage-applicant/borrower
   composition, loan amount, product mix, credit-score levels, DTI definition,
   mortgage-payment context, and the rounded approval-rate target.
2. [Summary of 2023 Data on Mortgage Lending](https://www.consumerfinance.gov/data-research/hmda/summary-of-2023-data-on-mortgage-lending/),
   Consumer Financial Protection Bureau. Informed race/ethnicity shares and
   denial-rate context.
3. [HMDA Getting It Right Guide, 2024](https://www.ffiec.gov/sites/default/files/data/hmda/2024Guide.pdf),
   Federal Financial Institutions Examination Council. Informed the mapping of
   synthetic variables to lender-reported HMDA concepts.
4. [HMDA reporting requirements FAQ](https://www.consumerfinance.gov/compliance/compliance-resources/mortgage-resources/hmda-reporting-requirements/home-mortgage-disclosure-act-faqs/),
   Consumer Financial Protection Bureau. Confirmed reporting treatment of
   income, property value, credit score, DTI, and combined LTV when relied on.
5. [Disclosure of Loan-level HMDA Data](https://files.consumerfinance.gov/f/documents/HMDA_Data_Disclosure_Policy_Guidance.Executive_Summary.FINAL.12212018.pdf),
   Consumer Financial Protection Bureau. Informed the conservative distinction
   between lender-reported fields and fields available in public modified HMDA,
   especially exclusion of applicant-level credit score.
6. [Regional and State Unemployment — 2024 Annual Averages](https://www.bls.gov/news.release/archives/srgune_03052025.pdf),
   U.S. Bureau of Labor Statistics. Informed the 4.0% unemployment center and
   plausible geographic variation.
7. [Economic, Housing and Mortgage Market Outlook — June 2024](https://www.freddiemac.com/research/forecast/20240620-economic-growth-in-line-historical-averages),
   Freddie Mac. Informed the 6.75% rounded mortgage-rate assumption used for
   the internal payment factor.
8. [New Residential Sales historical time series](https://www.census.gov/construction/nrs/data/series.html),
   U.S. Census Bureau and U.S. Department of Housing and Urban Development.
   Provided official home-price context for the property-value anchor.
9. [Report on the Economic Well-Being of U.S. Households in 2024](https://www.federalreserve.gov/publications/2025-economic-well-being-of-us-households-in-2024-overall-financial-well-being.htm),
   Board of Governors of the Federal Reserve System. Supported the direction of
   income, age, and savings relationships; it did not supply the synthetic
   liquid-asset distribution.
