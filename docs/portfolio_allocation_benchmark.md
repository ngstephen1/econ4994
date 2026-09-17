# Budget-Constrained Portfolio Allocation Benchmark

## Economic Question

This experiment asks how frozen repayment-risk estimation errors change credit
allocation and true expected lender profit when capital is scarce. It compares
a true-risk oracle, the traditional lender, and HistGradientBoosting in the
preserved additive-logistic and nonlinear risk worlds.

The oracle is an information benchmark, not a normative fairness benchmark.
Group is excluded from every risk model and allocation objective.

## Fixed-Request Optimization

The evaluation cohort contains 2,000 applicants and one indivisible requested
loan per applicant. Each policy solves:

```text
maximize    sum_i selected_i * perceived_expected_profit_i
subject to  sum_i selected_i * requested_principal_i <= K
            selected_i in {0, 1}
```

The no-loan action is always available. Nonpositive perceived-profit requests
are not funded merely to exhaust capital. Oracle perceived profit equals true
expected profit; the other policies use their already-frozen assessment files.
No model is retrained.

SciPy 1.17.1 `optimize.milp`, backed by HiGHS, solves the binary knapsack. All
18 primary runs terminated as optimal with reported relative MIP gap 0.0.
Runtimes ranged from approximately 0.004 to 1.082 seconds. Deterministic tests
verify the MILP against exhaustive subset enumeration on a hand-checkable case.

## Policy-Independent Budgets

Total requested principal is `$583,043,937.08`. The same absolute constraints
apply to every policy and both worlds:

| Regime | Fraction | Budget |
|---|---:|---:|
| Nonbinding | 1.00 | $583,043,937.08 |
| Moderate scarcity | 0.40 | $233,217,574.83 |
| Tight scarcity | 0.20 | $116,608,787.42 |

The moderate and tight budgets bind for every policy, with at most $1,674 left
unused. The 100% regime does not bind: utilization ranges from 59.79% to
68.25% because policies decline perceived nonpositive-profit requests.

## Additive-Logistic World

| Budget | Policy | Funded | Rate | True expected profit | Regret | Regret % | Realized profit |
|---|---|---:|---:|---:|---:|---:|---:|
| 100% | Oracle | 1,281 | 64.05% | $91.465m | $0 | 0% | $95.351m |
| 100% | Traditional | 1,295 | 64.75% | $91.256m | $0.208m | 0.228% | $96.480m |
| 100% | ML | 1,286 | 64.30% | $89.745m | $1.720m | 1.880% | $96.103m |
| 40% | Oracle | 736 | 36.80% | $73.958m | $0 | 0% | $78.711m |
| 40% | Traditional | 721 | 36.05% | $73.860m | $0.098m | 0.132% | $79.349m |
| 40% | ML | 736 | 36.80% | $72.766m | $1.192m | 1.612% | $78.584m |
| 20% | Oracle | 356 | 17.80% | $43.562m | $0 | 0% | $45.013m |
| 20% | Traditional | 344 | 17.20% | $43.529m | $0.033m | 0.075% | $45.764m |
| 20% | ML | 373 | 18.65% | $42.970m | $0.591m | 1.357% | $45.894m |

One realized draw sometimes gives a non-oracle policy higher realized profit.
That does not reverse expected-profit rankings; the oracle does not know future
payment draws.

## Nonlinear World

| Budget | Policy | Funded | Rate | True expected profit | Regret | Regret % | Realized profit |
|---|---|---:|---:|---:|---:|---:|---:|
| 100% | Oracle | 1,237 | 61.85% | $108.247m | $0 | 0% | $108.967m |
| 100% | Traditional | 1,181 | 59.05% | $107.482m | $0.765m | 0.707% | $109.111m |
| 100% | ML | 1,205 | 60.25% | $107.078m | $1.169m | 1.080% | $108.048m |
| 40% | Oracle | 758 | 37.90% | $86.639m | $0 | 0% | $88.653m |
| 40% | Traditional | 787 | 39.35% | $86.037m | $0.603m | 0.696% | $87.703m |
| 40% | ML | 761 | 38.05% | $85.657m | $0.983m | 1.134% | $85.333m |
| 20% | Oracle | 369 | 18.45% | $48.689m | $0 | 0% | $47.179m |
| 20% | Traditional | 396 | 19.80% | $48.239m | $0.450m | 0.925% | $48.054m |
| 20% | ML | 364 | 18.20% | $48.332m | $0.357m | 0.733% | $45.857m |

ML has lower regret than traditional only under nonlinear tight scarcity. At
40% and 100%, traditional retains higher true expected portfolio value. Prompt
13's lower ML profit MAE therefore does not uniformly translate into better
constrained allocations.

## Allocation Overlap

Jaccard overlap with the oracle:

| World | Budget | Traditional | ML |
|---|---|---:|---:|
| Additive | 100% | 0.966 | 0.909 |
| Additive | 40% | 0.938 | 0.849 |
| Additive | 20% | 0.928 | 0.774 |
| Nonlinear | 100% | 0.947 | 0.941 |
| Nonlinear | 40% | 0.882 | 0.868 |
| Nonlinear | 20% | 0.783 | 0.801 |

Traditional-versus-ML overlap falls as low as 0.689 in the nonlinear tight
regime. Scarcity makes ranking differences materially more visible even when
regret remains below 1% of oracle value.

At nonlinear 20%, traditional omits 33 oracle loans totaling $5.048m in true
expected profit and substitutes 60 loans totaling $4.598m. ML omits 43 oracle
loans totaling $5.049m and substitutes 38 totaling $4.692m. The smaller net
difference gives ML lower regret in that regime. These are allocation
disagreements, not false rejections.

## Truly Nonpositive Loans and Tail Risk

Oracle never funds a truly nonpositive-profit request. Under nonbinding capital:

- Additive traditional funds 29 such loans, contributing `$180,443` in true
  expected losses; additive ML funds 64, contributing `$932,437`.
- Nonlinear traditional funds 5, contributing `$103,347` in expected losses;
  nonlinear ML funds 21, contributing `$273,642`.

Under scarcity, nearly all funded loans are truly positive. The only exception
is one additive-world ML loan at 40%, with true expected loss `$1,545`.

The most negative funded request was `-$31,657` for additive traditional,
`-$96,199` for additive ML, and `-$68,288` for both nonlinear policies in the
nonbinding regime. Full mean, median, p5, p25, p75, p95, minimum, and aggregate
loss diagnostics are in `v2_portfolio_tail_diagnostics.csv`.

## Neutral Group Audit

Group B-minus-A funding-rate gaps range from +1.58 to +5.24 percentage points
across the 18 neutral-world portfolios. Corresponding per-applicant funded
principal and funded/requested-principal gaps are reported separately. Groups
remain absent from truth, prediction, objective values, constraints, costs, and
budgets. These finite-sample composition differences are neither discrimination
findings nor proof of fairness.

The audit distinguishes mean funded principal per applicant, including zeros,
from mean loan size among funded applicants.

## Exploratory Scarcity Curve

An additional predeclared 10%–100% curve confirms the primary results. In the
additive world, traditional regret stays between about `$26,106` and `$208,315`,
while ML regret grows from `$284,616` at 10% to `$1.720m` once capital becomes
nonbinding. In the nonlinear world, ML regret is below traditional at 10% and
20%, but traditional is lower from 30% onward. These points are exploratory and
do not replace the three primary budgets.

## Limitations and Next Step

This is a deterministic synthetic evaluation cohort with one fixed contract per
applicant and one realized repayment draw per world. Expected profit—not the
single realization—is primary. The oracle knows synthetic probabilities, not
future outcomes. No pricing, partial funding, multiple offers, capital charges,
diversification, risk aversion, or real-world operating constraints are modeled.

A later experiment can deliberately introduce separately configured direct or
systemic discrimination into estimated risk or allocation, then compare it with
this structurally neutral benchmark. Such mechanisms should remain distinct
from ordinary prediction error.

