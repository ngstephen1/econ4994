# Direct Discrimination Through Distorted Repayment Beliefs

## Research Question and Mechanism

This experiment asks how a researcher-inserted downward distortion in perceived
Group B repayment probability changes funding, portfolio composition, and true
expected lender profit. It is a synthetic direct-discrimination mechanism, not
an estimate of discrimination in actual mortgage markets.

Applicant characteristics, group assignments, requested contracts, true risk,
repayment draws, outcomes, transaction costs, and budgets remain identical to
the neutral control. Frozen traditional and HistGB estimates are not retrained.
Only the probability used by the lender changes:

```text
logit(rho_used_i) = logit(rho_base_i) - delta * I(group_i == B)
```

Group A retains `rho_used = rho_base`. The predeclared deltas and Group B
repayment-odds multipliers are:

| Delta | Odds multiplier |
|---:|---:|
| 0.00 | 1.000000 |
| 0.05 | 0.951229 |
| 0.10 | 0.904837 |
| 0.20 | 0.818731 |

These are controlled treatments, not approval coefficients, true-risk effects,
or estimates from HMDA.

## Multi-Period Amplification Before Allocation

At `delta=0.20`, Group B mean changes were:

| World | Base family | Mean rho change | Mean 120-period full-repayment change | Mean perceived-profit change |
|---|---|---:|---:|---:|
| Additive | Traditional | -0.001623 | -0.051249 | -$15,624 |
| Additive | HistGB | -0.001640 | -0.052693 | -$16,263 |
| Additive | True-risk reference | -0.001640 | -0.051167 | -$15,733 |
| Nonlinear | Traditional | -0.004594 | -0.032383 | -$11,437 |
| Nonlinear | HistGB | -0.005305 | -0.037262 | -$12,061 |
| Nonlinear | True-risk reference | -0.004644 | -0.036847 | -$11,875 |

A seemingly small conditional per-period change therefore produces a material
contract-horizon change. Perceived receipts and profit are recomputed from
`rho_used`; `rho_base` is preserved separately.

## Matched Controls and Portfolio Re-optimization

Each distorted policy is compared with its own `delta=0` allocation. All use
the exact Prompt 14 budgets: `$583,043,937.080`, `$233,217,574.832`, and
`$116,608,787.416`. The entire binary portfolio is re-optimized with
SciPy/HiGHS rather than removing Group B loans mechanically.

The experiment creates 24 probability policies—16 fitted-model policies and 8
evaluator-only true-risk references—and solves 72 portfolios. Every solve
terminated optimally with reported MIP gap zero. The true-risk reference is
stored separately from lender policy assessments.

## Funding Effects at Delta 0.20

Changes are relative to the same model's neutral portfolio. Percentage-point
effects are shown below.

| World | Budget | Model | Group A rate change | Group B rate change | Change in B-minus-A gap |
|---|---|---|---:|---:|---:|
| Additive | 100% | Traditional | 0.00 | -7.20 | -7.20 |
| Additive | 100% | HistGB | 0.00 | -6.41 | -6.41 |
| Additive | 40% | Traditional | +3.96 | -3.85 | -7.80 |
| Additive | 40% | HistGB | +3.55 | -3.85 | -7.40 |
| Additive | 20% | Traditional | +2.64 | -2.47 | -5.10 |
| Additive | 20% | HistGB | +2.64 | -3.16 | -5.79 |
| Nonlinear | 100% | Traditional | 0.00 | -2.47 | -2.47 |
| Nonlinear | 100% | HistGB | 0.00 | -3.85 | -3.85 |
| Nonlinear | 40% | Traditional | +2.23 | -1.97 | -4.20 |
| Nonlinear | 40% | HistGB | +3.25 | -2.86 | -6.11 |
| Nonlinear | 20% | Traditional | +1.52 | -1.28 | -2.80 |
| Nonlinear | 20% | HistGB | +1.62 | -1.68 | -3.30 |

Under nonbinding capital, Group A decisions do not change because the policy can
simply stop funding newly perceived-unprofitable Group B requests. Under scarce
capital, some Group A applicants become funded through portfolio spillovers.
Matched changes in per-applicant funded principal and funded/requested ratios
are retained in `v2_direct_group_effects.csv`.

## Economic Effects and Model Interaction

At `delta=0.20`, change in true expected portfolio profit relative to the same
fitted model's neutral policy was:

| World | Budget | Traditional | HistGB |
|---|---:|---:|---:|
| Additive | 100% | -$425,155 | -$801,675 |
| Additive | 40% | -$300,518 | -$202,156 |
| Additive | 20% | -$145,570 | -$255,088 |
| Nonlinear | 100% | -$762,029 | -$547,705 |
| Nonlinear | 40% | -$160,198 | -$44,994 |
| Nonlinear | 20% | -$31,501 | -$14,125 |

Effects are not monotonic for fitted models at weaker treatments. For example,
additive HistGB at 40% gains `$164,627` at delta 0.05, and nonlinear HistGB at
40% gains `$79,575`. The adverse group treatment accidentally offsets some
ordinary estimation/ranking error in those portfolios. This is an interaction
between prediction error, the distortion, and discrete optimization—not a
benefit or justification for discrimination.

Accordingly, total fitted-model regret is not mechanically decomposed into
ordinary error plus direct discrimination. The reporting table shows neutral
regret, distorted regret, and their incremental difference without claiming an
exact additive causal decomposition.

## Pure Mechanism Cost

The evaluator-only reference starts from `rho_true`, applies the same Group B
logit shift, and compares its allocation with the unchanged true-risk oracle.
Its cost is nonnegative in every run, as required by oracle optimality.

At delta 0.20:

| World | Budget | Pure cost | Percent of oracle profit | Oracle overlap |
|---|---:|---:|---:|---:|
| Additive | 100% | $615,638 | 0.673% | 0.947 |
| Additive | 40% | $211,201 | 0.286% | 0.912 |
| Additive | 20% | $144,300 | 0.331% | 0.845 |
| Nonlinear | 100% | $218,864 | 0.202% | 0.975 |
| Nonlinear | 40% | $152,505 | 0.176% | 0.931 |
| Nonlinear | 20% | $69,635 | 0.143% | 0.863 |

The cost rises with treatment strength. It ranges from `$1,702` at nonlinear
5%/tight to `$615,638` at additive 20%/nonbinding.

## Allocation Transitions and Overlap

At delta 0.20, same-model neutral Jaccard overlap ranges from 0.855 to 0.979.
In the additive 40% portfolios, traditional displaces 39 Group B loans and adds
39 Group A loans; HistGB displaces 39 Group B loans and adds 35 Group A loans.
In nonlinear 40%, the corresponding counts are 20/22 and 29/32.

Nonbinding portfolios show no Group A replacements because capital is not the
limiting factor. Group B funding falls while unchanged Group A positive-profit
loans remain funded.

## Economic Quality of Displaced and Replacement Loans

At delta 0.20, Group B loans displaced from additive tight portfolios have mean
true expected profit `$114,641` under traditional and `$87,207` under HistGB.
The corresponding newly funded Group A loans average `$104,633` and `$97,521`.

For nonlinear tight portfolios, displaced Group B loans average `$118,556`
and `$116,925`; Group A replacements average `$100,648` and `$123,349`.
These are selected portfolio subsets, not statements that either group is
inherently more or less creditworthy.

Complete counts, rho summaries, medians, and total true profit are in the
displaced- and replacement-borrower tables.

## Nonpositive True-Profit Loans

At delta 0.20, nonbinding portfolios fund fewer truly nonpositive loans than
their neutral versions because the treatment removes some Group B requests:

- Additive traditional: 15 loans, `$88,814` total expected loss.
- Additive HistGB: 47 loans, `$675,501` loss.
- Nonlinear traditional: 4 loans, `$101,468` loss.
- Nonlinear HistGB: 10 loans, `$74,890` loss.

No fitted-model distorted portfolio at 20% or 40% funds a truly nonpositive
request at delta 0.20. This does not economically justify Group B underfunding;
the matched neutral and oracle comparisons show profitable Group B loans are
also displaced.

## Interpretation and Limitations

The same direct log-odds treatment produces larger funding-gap changes in the
additive world for most specifications, while economic costs depend jointly on
the base model, budget, and discrete replacements. A smaller observed gap or
lower incremental regret does not by itself make a model fairer.

This is one deterministic synthetic applicant population with an intentionally
encoded direct mechanism. Delta is not empirically estimated. The model omits
pricing, partial funding, multiple offers, strategic response, legal standards,
and real institutional processes. No conclusion about actual mortgage-market
discrimination follows from the numerical magnitude.

