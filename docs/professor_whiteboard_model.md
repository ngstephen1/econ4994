# Professor Whiteboard Lending Model — Step 1 Specification

Stage record: results, test counts, and later-work deferrals below describe
this step. See [current status](PROJECT_STATUS.md) and the
[documentation index](README.md) for the latest state across all three families.
The initial annual-rate suggestion is not the implemented per-period contract;
see [Step 6A accounting](professor_loan_economics.md) for the working economics.

Step 1 status: specification complete. This document records the initial
research design. Step 2 now implements only applicant characteristics and C;
see [population generation](professor_population_generation.md) for its exact
symbol-named output schemas and validation results. The broader proposed
schema below is a future design, not the current persisted dataset.
Step 3 now evaluates the frozen true-risk equation in a separate evaluator-only
artifact; see [true default risk](professor_true_default_risk.md).
Step 4 now adds the raw noisy target in separate bank-target and private-noise
tables; see [noisy risk signal](professor_noisy_risk_signal.md).
Step 5 now fits the specified raw-PN OLS with a fixed 80,000/20,000 split and
separate raw/bounded predictions; see [bank regression](professor_bank_regression.md).
Step 6A now values income-linked contracts on the held-out evaluation cohort;
see [loan economics](professor_loan_economics.md). Its working assumptions use
5% **per period**, the growing schedule `ell_i*(1+r)^t`, and PD as a constant
conditional per-period default probability. These supersede the initial
annual-rate suggestion/unresolved cash-flow interpretation for this step;
the saved risk equation and fitted model are unchanged.
Step 6B now selects one common **bounded-PD** cutoff using bank-expected profit
and 40% of saved evaluation requested principal; see
[lending threshold](professor_lending_threshold.md). This operational bounded
score supersedes Step 1's initial raw-score proposal. Exact ties are indivisible,
and decisions are frozen before evaluator truth is loaded. No unrestricted
allocation or oracle comparison is added. The initial proposals below
are historical design context; the linked implementation documents record the
actual schemas and choices made in Steps 2–6B.

This specification records the equations supplied in the Step 1 request. The
professor's original handwriting has not been independently inspected here.
The equations below are therefore the supplied **working transcriptions**, not
a claim that every symbol or economic interpretation has been confirmed.

## 1. Research objective

Study how a bank's imperfect information about borrower default risk affects
risk estimates, group lending disparities, expected profit, and credit
allocation under a capital constraint. The simulator knows the true risk;
the bank learns from a noisy risk measurement and four observable covariates.

The broader capstone question remains: "Do mortgage lending decisions differ
across demographic groups, and can machine learning help detect those
differences?" This model's initial specification concerns a traditional linear
risk estimator and threshold lending. Flexible-model comparisons and additional
discrimination mechanisms are outside Step 1.

## 2. Repository audit and preservation boundary

The Step 1 inspection found:

- Repository: `/Users/macbook/Hack/econ4994`.
- Branch: `v2-systemic-upstream`.
- HEAD: `7a7e590` — `Add professor-facing research review artifacts`.
- The working tree was clean before creating this document. HEAD matches the
  locally recorded `origin/v2-systemic-upstream`; no remote fetch was performed.
- README and project status document Version 1 approval research and Version 2
  repayment-risk estimation, economic portfolios, direct belief distortion,
  and systemic opportunity experiments.
- Project status reports five corrected replications and 360 validated
  portfolios for the systemic uncertainty checkpoint. The 50-replication study
  is incomplete; 19 legacy success records remain provisional and excluded.
- The latest documented test result is 284 passed with 32 warnings. Step 1
  does not establish a fresh test result or independently revalidate all
  historical experiments.

The audit read `README.md`, `docs/PROJECT_STATUS.md`, and
`docs/economic_lending_spec.md`, inventoried `src/fair_lending/`, `configs/`,
`experiments/`, and `tests/`, and inspected relevant configuration, repayment,
contract, profit, estimator, history, portfolio, provenance, and test code.

Some existing documentation retains earlier-phase language: the V2 spec still
says lending decisions are deferred and describes future replacement of subset
enumeration, although its later portfolio section and current `portfolio.py`
already implement population-scale allocation. Treat those passages as
historical context when assessing current capabilities. They are not edited
as part of this Step 1.

| Existing component | Reuse opportunity | Boundary for this model |
|---|---|---|
| NumPy, pandas, SciPy, statsmodels, PyYAML, PyArrow, Matplotlib, pytest | Existing dependencies support vectorized draws, OLS, diagnostics, exports, and tests. | No new framework is needed. |
| `economic_lending/config.py` | Named `SeedSequence` streams and canonical configuration fingerprinting are useful patterns or small utilities. | The V2 configuration loader validates a different DGP; it is not the new loader. |
| `economic_lending/recovery.py` | Atomic JSON writing and hashing are general utilities. | Its scientific-provenance function hashes V2-specific paths; it cannot certify the new source/configuration unchanged. |
| `economic_lending/traditional.py` | Transparent coefficient artifacts and observable-feature allowlists are useful design patterns. | Existing estimators target binary at-risk payments, use six V2 transforms, and exclude group. They do not implement `PN ~ Y + D + C + B`. |
| `economic_lending/lender_evaluation.py` | MAE, RMSE, signed-error, and group-audit ideas can guide evaluation. | Existing field names and repayment direction need a default-risk adapter or separate implementation. Binary-label scores need actual binary outcomes, which this specification has not defined. |
| `economic_lending/contracts.py`, `profit.py`, `repayment.py` | Separation of contracts, risk, and evaluator accounting is reusable. | Existing routines assume a specified payment schedule and conditional per-period repayment. Those meanings cannot be assigned to the new `true_PD` automatically. |
| `economic_lending/portfolio.py` | Budget-feasibility checks and diagnostic practices are useful references. | Its unrestricted binary knapsack does not implement a single common default-risk threshold. |
| Existing experiments and tests | Runner/source separation, seed tests, identity checks, and leakage tests provide templates. | Canonical runners, scientific parameters, prior results, and Monte Carlo recovery remain unchanged. |
| Version 1 analysis and dashboard | Descriptive gap reporting and presentation conventions remain useful. | Race-specific approval estimands, approval DGP, and dashboard APIs are not the new model. |

The new family should live under `fair_lending.professor_model`, with separate
configuration, artifact paths, schema, and model identifiers. It does not
replace either earlier version and is not yet labeled Version 3.

## 3. Source labels

| Label | Meaning | Items |
|---|---|---|
| Supplied professor-derived working equations | Current transcription supplied by the researcher; confirmation against the board remains possible. | Population distributions; the two sigmoid factors and their product; `C = 0.4 epsilon + omega`; `PN = 0.8 true_PD + eta`; regression with `Y, D, C, B`; threshold lending; 30 periods; income-linked lending; bank funds and profit objective. |
| Supplied working interpretations | Choices explicitly requested for the initial specification. | Second Normal argument means standard deviation; ordinary linear regression; `L_i = 30 ell_i`. |
| Researcher-chosen demonstration assumptions | Suggested economic settings, not professor-estimated parameters. | `theta = 0.10`; annual `r = 0.05`; funds equal 40% of total requested principal. |
| Proposed implementation conventions | Recommendations in this document, not implemented scientific decisions. | Separate lender/truth tables, explicit seeds and metadata, descriptive column names, and a separate bounded-prediction field. |
| Unresolved decisions | Information needed before the relevant later implementation. | Joint independence, risk-event horizon, period duration, payment schedule, recovery/costs/discounting, training/evaluation sampling, threshold selection and ties. |

## 4. Unit of observation and variable dictionary

One row in the proposed applicant table represents one synthetic applicant.
One row in each joined truth, signal, or estimate table refers to that same
applicant. The eventual contract baseline has one income-linked request per
applicant. A period-level payment schedule, if needed, is a separate table.

`log` means the natural logarithm. Monetary quantities use nominal synthetic
USD as a working unit convention; income must stay in the chosen dollar units
inside the stated log-risk equation.

| Symbol / proposed column | Type and scale | Meaning / equation | Information role |
|---|---|---|---|
| `applicant_id` | Unique identifier | Join key across logical tables. | Administrative only; never a predictor. |
| `Y` / `annual_income` | Positive float; USD/year | Annual income, with `log(Y) ~ Normal(11, 0.7)`. | Bank-observed predictor. |
| `log(Y)` / `log_annual_income` | Float; natural-log income | Derived value used by the true-risk equation. | Derivable from bank-observed income; not an additional primary OLS predictor. |
| `D` / `debt_to_income_ratio` | Float; ratio units | DTI drawn from `Normal(0.35, 0.16)`. | Bank-observed predictor; 0.35 means 35%. |
| `B` / `group_indicator` | Integer, 0 or 1 | Binary group membership, with `P(B=1)=0.15`. | Bank-observed predictor and audit attribute; B=0 is reference. |
| `epsilon` / `hidden_risk_factor` | Float; latent scale | Unobserved borrower risk, `Normal(0, 1)`. | Simulator/evaluator only. |
| `omega` / `credit_signal_noise` | Float; signal scale | Credit-measurement noise, `Normal(0, 0.2)`. | Simulator/evaluator only. |
| `C` / `credit_signal` | Float; signal scale | `0.4 epsilon + omega`. It is a noisy risk signal, not a 300–850 credit score. | Bank-observed predictor. |
| `P_income` / `income_default_component` | Float in (0,1) | `sigmoid(0.52 - 0.65 log(Y))`. | Simulator/evaluator component. |
| `P_other` / `other_default_component` | Float in (0,1) | `sigmoid(4 + 0.06 D + epsilon)`. | Simulator/evaluator component. |
| `true_PD` / `true_PD` | Float in (0,1) | Product `P_income * P_other`; default probability with horizon unresolved. | Hidden simulation truth. |
| `eta` / `noisy_risk_noise` | Float; risk-measurement scale | Additive target noise, `Normal(0, 0.1)`. | Simulator/evaluator only. |
| `PN` / `noisy_risk_measurement` | Unbounded float | `0.8 true_PD + eta`. | Observed fitting target; never a prediction-time covariate. |
| `beta0_hat` … `beta4_hat` | Fitted floats | OLS intercept and slopes for `Y, D, C, B`, in that order. | Bank model artifact; not applicant data. |
| `bank_PD` / `bank_PD_raw` | Unbounded float | OLS fitted risk measurement on the original target scale. | Bank-produced estimate; primary threshold score. |
| `bank_PD_bounded` | Float in [0,1] | Separately stored probability-valid economic input; proposed `clip(bank_PD_raw, 0, 1)`. | Proposed accounting convention; raw estimates remain available. |
| `prediction_out_of_bounds` | Boolean | Whether `bank_PD_raw < 0` or `bank_PD_raw > 1`. | Proposed estimator diagnostic. |
| `threshold` / `tau` | Scalar in raw-score units | Bank-selected default-risk cutoff. No value has been selected. | Policy parameter. |
| `x_i` / `approved` | Boolean / 0–1 | `1{bank_PD_raw_i <= tau}` for a threshold-feasible allocation. | Future lending decision, not repayment outcome. |
| `T` / `term_periods` | Positive integer; periods | 30 repayment periods. | Contract parameter; calendar duration unresolved. |
| `theta` / `income_loan_scale` | Positive scalar | Income-to-loan scaling parameter; suggested 0.10. | Researcher-chosen demonstration parameter. |
| `ell_i` / `income_linked_loan_unit` | Float; derived monetary scale | `theta * Y_i`. | Future contract quantity; not yet defined as a scheduled payment. |
| `L_i` / `requested_principal` | Positive float; USD | Working interpretation `30 ell_i`. | Future full-loan funding amount. |
| `r` / `annual_interest_rate` | Annual proportion | Suggested 0.05 annually. | Researcher-chosen contract parameter. |
| `A` / `bank_funds` | Nonnegative float; USD | Suggested `0.40 * sum_i L_i` over the allocation pool. | Future capital constraint. |
| `funded_principal_i` | Nonnegative float; USD | `x_i * L_i`. | Future allocation output. |
| `c_it` / `scheduled_payment` | Monetary amount per period | Contract receipt due at period t; formula unresolved. | Future contract schedule. |
| `S_it` / `payment_receipt_probability` | Probability | Probability that scheduled payment t is received, under a specified default event/timing model. | Future risk-to-cash-flow mapping. |
| `expected_profit_perceived_i` | Float; USD | Bank-expected receipts less principal and any explicitly defined costs. | Future policy assessment/objective. |
| `expected_profit_true_i` | Float; USD | Same accounting evaluated with hidden true risk. | Future evaluator-only assessment. |

The binary group is abstract; the request does not map it to an empirical race
or ethnicity. Including `B` in this model is intentional and differs from the
group-blind Version 2 baseline. It does not authorize an extra group term in
the simulator's true-risk equation.

## 5. Data-generating equations

For approximately 100,000 applicants, the supplied marginal distributions are:

```text
log(Y_i) ~ Normal(mean=11, standard_deviation=0.7)
D_i      ~ Normal(mean=0.35, standard_deviation=0.16)
B_i      ~ Bernoulli(probability=0.15)
epsilon_i ~ Normal(mean=0, standard_deviation=1)
omega_i   ~ Normal(mean=0, standard_deviation=0.2)
eta_i     ~ Normal(mean=0, standard_deviation=0.1)

C_i = 0.4 * epsilon_i + omega_i

sigmoid(z) = 1 / (1 + exp(-z))

P_income_i = sigmoid(0.52 - 0.65 * log(Y_i))
P_other_i  = sigmoid(4 + 0.06 * D_i + epsilon_i)
true_PD_i  = P_income_i * P_other_i

PN_i = 0.8 * true_PD_i + eta_i
```

These distributions specify marginals, not a complete joint distribution.
Independent primitive draws for `log(Y), D, B, epsilon, omega, eta` are a
proposed simple baseline assumption requiring confirmation in Step 2. Under
that assumption, `C` intentionally depends on `epsilon`, while `B` has no
structural path to true risk or the other applicant covariates.

The causal/information order is:

```text
log(Y) -> Y -----> P_income --+
D, epsilon -----> P_other ---+--> true_PD --> PN (with eta)
epsilon, omega -> C

historical Y, D, C, B + PN -> fitted OLS coefficients
new Y, D, C, B + coefficients -> bank_PD_raw
bank_PD_raw + threshold -> lending decisions
Y + theta + T -> requested principal
decisions + principal + risk-to-cash-flow mapping -> profit and funded capital
```

Higher income decreases the income component. Higher DTI and higher hidden
risk increase the other component. Higher `C` signals higher `epsilon` when
the noise assumptions hold. No coefficient, signal scale, or group treatment
is adjusted to produce a target result.

The product of the sigmoid factors is the defined risk function. Calling its
factors two independent default events would require an additional event model
that has not been supplied.

## 6. Bank information, noisy target, and regression

At prediction time the bank receives exactly `Y, D, B, C`. The simulator may
use hidden quantities to construct synthetic truth and `PN`, but those
quantities are withheld from the bank's input tables and model artifact.

For estimation, the bank additionally receives historical paired measurements
`(Y, D, C, B, PN)`. This is the minimal working interpretation needed to make
the specified regression possible. How a real bank would obtain such a noisy
risk measurement is not yet specified. `PN` is not a realized binary default
label or a payment-history target.

The declared initial estimator is ordinary least squares with an intercept:

```text
X_i = [1, Y_i, D_i, C_i, B_i]
beta_hat = argmin_beta sum over training i of (PN_i - X_i beta)^2

bank_PD_raw_i = beta0_hat
              + beta1_hat * Y_i
              + beta2_hat * D_i
              + beta3_hat * C_i
              + beta4_hat * B_i
```

Use annual income in levels in this regression. The true DGP uses log income;
replacing `Y` with `log(Y)` in the OLS specification would be a different
model. No logistic link is applied to OLS predictions. Any future numerical
centering must be algebraically reversed in the saved coefficient artifact.

The proposed predictor allowlist is `annual_income`, `debt_to_income_ratio`,
`credit_signal`, `group_indicator`, plus an intercept. The target is
`noisy_risk_measurement`. IDs, cohort labels, hidden factors, noise draws,
truth/components, future outcomes, and `PN` itself are excluded as predictors.

Both `PN` and raw fitted values may be outside [0,1]. Preserve `PN` as drawn
and `bank_PD_raw` as fitted. A proposed later accounting field is:

```text
bank_PD_bounded = min(1, max(0, bank_PD_raw))
```

Clipping is an explicit accounting convention, not a correction to the fit.
Its frequency and effect on economic values must be reported. Do not divide
the target or predictions by 0.8 automatically. With independent mean-zero
`eta`, `E[PN | Y,D,C,B] = 0.8 * E[true_PD | Y,D,C,B]`; OLS approximates this
scaled conditional expectation within a linear feature space.

The true-risk equation is nonlinear and contains a hidden factor. Consequently
this is a measurement-error and functional-form problem as well as an
estimation-error problem. Its fitted slopes do not have a one-to-one
coefficient-recovery comparison with the sigmoid-factor coefficients.
An estimated group slope or a group approval gap is not, by itself, evidence
that discrimination was encoded into the true-risk process.
Under the proposed independent-draw baseline, the population OLS group slope
is zero; a finite-sample fitted group slope need not equal zero exactly.

Training/evaluation design remains to be declared. Held-out applicant-level
evaluation is recommended; no split fractions, fitting sample, or seed have
been selected for this new model. Evaluation truth must not select OLS
settings or the bank's threshold.

## 7. Lending rule, requested capital, and economic objective

The working decision score is `bank_PD_raw`, so the supplied rule is:

```text
x_i(tau) = 1 if bank_PD_raw_i <= tau, otherwise 0
ell_i = theta * Y_i
L_i = 30 * ell_i
funded_principal_i = x_i(tau) * L_i

sum_i x_i(tau) * L_i <= A
```

Only approved loans consume funds. Let the sum range over the same allocation
pool for both the objective and budget. With the suggested demonstration
settings, `L_i = 3 * Y_i` and `A = 0.40 * sum_i L_i`. This is a loan-size rule,
not a specified payment schedule. Group membership does not alter theta,
principal, rate, or the threshold in the supplied baseline.

The bank's later optimization is over a **common threshold**, using perceived
expected profit:

```text
maximize over tau: sum_i x_i(tau) * expected_profit_perceived_i
subject to:       sum_i x_i(tau) * L_i <= A
```

The feasible policy set should include lending to nobody. A threshold that
violates the budget is infeasible; a separate arbitrary rationing rule or
unrestricted knapsack must not be silently appended. Boundary ties and the
threshold search domain need explicit treatment in the implementation step.
Even with risk ordered by threshold, varying income-linked principals can
affect aggregate capital and profit.

### Accounting template — not a finalized profit formula

The most specific defensible expression before payment/default timing is
resolved is:

```text
expected_profit_perceived_i = E_bank[total contract receipts_i] - L_i - k_i
expected_profit_true_i      = E_true[total contract receipts_i] - L_i - k_i
```

`k_i` denotes any later explicitly specified lending cost; no value or inclusion
decision is supplied. The following is a conditional accounting template:
if scheduled receipts are `c_it`, default is absorbing, and there is no recovery
or discounting, then

```text
expected_profit_perceived_i = sum_t c_it * S_bank_it - L_i - k_i
expected_profit_true_i      = sum_t c_it * S_true_it - L_i - k_i
```

If and only if `PD` is confirmed as constant conditional **per-period default
risk**, the corresponding survival would be `S_it = (1 - PD_i)^t`. If it is
whole-loan default risk, that formula is not justified without a timing model.
Neither interpretation is selected in Step 1.

The annual rate `r=0.05` cannot become a periodic rate until period duration
and the rate convention are specified. Neither `ell_i` nor `L_i` defines
scheduled bank receipts or interest accrual. V2's equal-payment simple-interest
rule, $1,500 cost, zero recovery, and 120-period survival must not be imported
as facts about this model. Discounting, funding costs, and recovery are also
unresolved. Unissued loans have zero lending cash flows; any fixed bank-level
cost would require a separate specification.

## 8. Proposed logical dataset schema

These are proposed future outputs, not files created in Step 1. Keep lender
inputs separate from evaluator truth, joined with a unique `applicant_id`.
For manual inspection, future CSV companions can accompany canonical Parquet
outputs; exact configuration and model coefficients should be saved in JSON
or YAML rather than repeated in every applicant row.

| Logical table | One row | Proposed columns | Availability / phase |
|---|---|---|---|
| `applicants` | Applicant | `applicant_id`, `annual_income`, `debt_to_income_ratio`, `group_indicator`, `credit_signal` | Bank-observed inputs; proposed Step 2 population. |
| `simulation_truth` | Applicant | `applicant_id`, `hidden_risk_factor`, `credit_signal_noise`, `noisy_risk_noise`, `income_default_component`, `other_default_component`, `true_PD` | Restricted simulator/evaluator artifact. Noise draws may instead be reproduced from metadata if a later persistence policy chooses that. |
| `training_signals` | Applicant with a fitting-target measurement | `applicant_id`, `noisy_risk_measurement` | Historical fitting target; retained raw. Training/evaluation roles live in a separate split manifest when defined. |
| `bank_risk_estimates` | Applicant scored by a declared model | `applicant_id`, `model_id`, `bank_PD_raw`, `bank_PD_bounded`, `prediction_out_of_bounds` | Later OLS step; no truth columns. Bounding rule saved in metadata. |
| `loan_requests` | Applicant's fixed request | `applicant_id`, `income_linked_loan_unit`, `requested_principal`, `term_periods`, `annual_interest_rate`, `payment_rule_id` | Deferred economic step; schedule rule is unresolved. |
| `payment_schedule` | Applicant-period | `applicant_id`, `period_index`, `scheduled_payment` | Deferred until the contract is specified. |
| `policy_assessments` | Applicant-policy assessment | `applicant_id`, `policy_id`, `bank_PD_raw`, `bank_PD_bounded`, `expected_receipts_perceived`, `expected_profit_perceived` | Deferred accounting; no hidden true risk or true profit. |
| `lending_decisions` | Applicant-policy decision | `applicant_id`, `policy_id`, `threshold`, `approved`, `funded_principal` | Deferred threshold allocation. |
| `economic_truth` | Applicant | `applicant_id`, `expected_receipts_true`, `expected_profit_true` | Deferred evaluator accounting after risk and contract meanings are settled. |
| `run_manifest` / `model_artifact` | Run or fitted model | Sample count, seed, resolved configuration, source/config hashes, dependency versions, schema ID, split IDs, coefficient order/values, prediction/bounding rule, threshold/budget provenance. | Reproducibility metadata; coefficient values only after fitting. |

No binary default outcome, repayment history, or denial reason is presently
defined. These should not be fabricated merely to resemble earlier datasets.
The true probability remains useful for evaluator risk-recovery diagnostics
without generating binary outcomes.

## 9. Working assumptions and interpretation checks

| Issue | Current specification / required decision |
|---|---|
| Sample size | Approximately 100,000 applicants, configurable; no data generated here. |
| Normal notation | All second arguments are standard deviations, as requested. In particular, omega SD is 0.2 and eta SD is 0.1, not their square roots. |
| Joint distribution | Independent primitive draws are proposed, not inferred from marginal distributions. Confirm before generation. |
| Income units and transformation | Annual synthetic USD; natural log in truth and raw annual income in OLS. Rescaling income without transforming coefficients changes the model. |
| DTI support | An untruncated Normal permits negative DTI. Retain the supplied working distribution in this design; document tail diagnostics before choosing any later truncation or clipping. |
| Group coding | `B=1` has probability 0.15; no empirical demographic labels or group-specific financial/risk shifts are supplied. |
| Credit signal direction | Higher C indicates higher hidden risk under independent measurement noise. It is not an ordinary credit-score scale. |
| Risk event | `true_PD` is called default risk, but conditional-period versus whole-loan horizon remains unresolved. This blocks a definitive multi-period profit formula. |
| Noisy target | PN is an unbounded continuous fitting measurement. Its observation/availability mechanism needs interpretation; it is not the bank's observable true risk. |
| 0.8 scale | Intentionally preserved. No automatic debiasing or calibration to truth. |
| OLS feature set | Intercept + Y + D + C + B; no substitution of log(Y), no extra group treatment. |
| Bounding | Preserve raw PN and raw fitted values. Separate clipping to [0,1] is proposed for later economic calculations and requires reporting. |
| Threshold | No cutoff chosen. Current written rule uses raw OLS score; using a bounded score instead changes threshold behavior at the boundaries and must be labeled. |
| Training/evaluation | New applicant-level split/cohort design and seed must be declared. Do not automatically copy V2's 60/20/20 split or fit on all evaluation applicants. |
| Economic demonstration | theta=0.10, annual r=0.05, and funds fraction=0.40 are supplied researcher suggestions. T=30 and L=30 ell are working professor-model interpretations. |
| Economic accounting | Clarify period calendar, payment schedule, default timing, recovery, transaction/funding costs, discounting, and allocation pool before economic implementation. |

### Algebraic implications to check in Step 2

These calculations use fixed reference values, not a generated population or
an estimated model:

- At `log(Y)=11`, income is approximately $59,874.14 per year.
- At `log(Y)=11`, `D=0.35`, and `epsilon=0`, the income factor is
  approximately 0.00131842, the other factor 0.98238098, and true_PD
  approximately 0.00129519 (0.1295%). This is a reference-point value, not
  the population mean default risk.
- At that point, the mean noisy signal before eta is approximately 0.00103615,
  while eta's SD is 0.1, or 10 percentage points. Signal recovery may therefore
  be difficult, and negative PN observations are permitted by construction.
- The supplied D distribution has approximately 1.4353% probability below
  zero. Clipping or redrawing would change the working distribution.
- With the suggested theta, the proposed principal is three times annual
  income. This does not determine monthly payments or interest receipts.

These implications justify diagnostics and professor clarification; they do
not justify retuning the equations. Probability bounds alone do not establish
an empirically realistic default-risk population.

## 10. Differences from existing Version 2

| Dimension | Existing V2 baseline | New professor model |
|---|---|---|
| Population | Calibrated dependent finances; two groups at 50/50; 10,000 development applicants. | Approximately 100,000 applicants; supplied lognormal income, Normal DTI, and Bernoulli group at 15%. |
| Truth | Logistic conditional repayment probability from six financial/contract transforms. | Product of two sigmoid default-risk factors including hidden epsilon. |
| Time meaning | Constant conditional per-period repayment; absorbing default; 120 stylized periods. | Default-risk horizon unresolved; 30 repayment periods. |
| Bank target | Realized binary payments on historical at-risk rows. | Continuous noisy risk measurement PN. |
| Initial estimator | Unregularized logistic regression. | Ordinary linear regression with intercept. |
| Information | Six observable financial/contract inputs; group excluded. | Y, D, C, B; noisy target available for fitting; epsilon and true_PD hidden. |
| Income transform | Declared centered log income. | Raw Y in bank regression; log Y in truth. |
| Loan contract | One property-related request; equal-payment simple-interest baseline. | Income-linked principal `L=30 theta Y`; receipt schedule unresolved. |
| Allocation | Arbitrary feasible binary subset maximizing perceived profit. | One common risk threshold, constrained by approved principal. |
| Mechanisms | Separate direct belief shifts and systemic opportunity treatment. | Neither treatment is inserted into the supplied baseline. |
| Artifacts | V2 scientific and recovery outputs already exist. | New schema, IDs, configuration, and output namespace; specification only so far. |

## 11. Recommended implementation structure

Use a small separate package without changing shared V1/V2 scientific code.
The following paths are recommendations, not directories created in Step 1:

```text
configs/professor_model/baseline.yaml
src/fair_lending/professor_model/
    __init__.py
    config.py          # declared parameters, units, validation, seed streams
    population.py      # Y, D, B, epsilon, C, PN and separate truth views
    risk.py            # exact two-factor true_PD equation
    bank.py            # later OLS allowlist, fit artifact, raw/bounded scoring
    lending.py         # later threshold search, budget, contract accounting
    validation.py      # identities, support diagnostics, information boundary
experiments/run_professor_model.py
tests/test_professor_model.py
data/synthetic/professor_model/<run_id>/
results/tables/professor_model/
results/metrics/professor_model/
```

The package's reusable logic belongs in source; the experiment script owns
orchestration. Do not create empty modules or configuration files before the
next implementation step. Reuse small general utilities where safe; otherwise
apply existing design patterns with new paths and allowlists. Hash the actual
new source, configuration, schema, and relevant dependency versions for run
provenance rather than relying on a Git revision alone.

## 12. Recommended Step 2 and validation plan

The original Step 1 recommendation included population, true risk, and the
noisy target. The subsequent Step 2 request narrows implementation to applicant
characteristics and C only: independent log_Y, D, B, epsilon, and omega, with
master seed 4994 and a validated 100,000-applicant population. True risk and
the noisy target remain deferred to separate steps. The current implementation
and tests are documented in `professor_population_generation.md`.

Future population tests should verify exact row count and unique IDs, same-seed
reproducibility, valid group coding, positive income, the exact C and PN
identities, exact two-factor truth, truth probabilities in [0,1], declared
noise scales, and the absence of hidden fields in lender input/target views.
Use diagnostics for Normal DTI tails and PN support instead of rejecting
values that the stated distributions explicitly allow. Distribution tests
need predeclared tolerances rather than exact sample means.

With independent primitive draws, changing B while fixing all other draws
must leave C, true_PD, and PN unchanged. Later OLS tests must enforce the four
predictors plus intercept, retain raw predictions, and verify transparent
coefficient reproduction. Later lending tests must check the common-threshold
rule, approved-only capital accounting, reject-all feasibility, ties, and
hand-checkable cash flows once the economic specification is resolved.

Before the economic step, obtain a confirmed default event and time horizon,
calendar period and annual-rate conversion, and scheduled receipt formula.
Step 1 ends with this document. Existing implementations, configurations,
portfolios, recovery records, and research artifacts are unchanged.
