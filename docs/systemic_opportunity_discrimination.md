# Systemic discrimination through upstream opportunity

## Research question

Can an earlier discriminatory opportunity process create later group differences
in repayment risk and credit allocation even when the final lender does not use
group? This experiment isolates that upstream mechanism from Prompt 15's direct
distortion of lender beliefs.

This is a researcher-designed synthetic systemic-discrimination mechanism. It
does not establish that this exact opportunity process or treatment magnitude
exists in real mortgage markets.

## Explicit upstream decision

`opportunity_access` represents access to a prior stable economic or employment
opportunity. It is evaluator-side mechanism information and is never supplied to
the lender. For applicant *i* in systemic-strength world *s*,

```text
p_opportunity_i(s) = sigmoid(
    alpha_O
    + 0.50 * financial_stability_latent_i
    - s * I(group_i == B)
)

opportunity_access_i(s) = I(U_i <= p_opportunity_i(s))
```

The same `U_i` is used in every matched world. The predeclared strengths are
`0.00`, `0.10`, `0.20`, and `0.40`, corresponding to Group B opportunity-odds
multipliers of 1.0000, 0.9048, 0.8187, and 0.6703. Group A receives no logit
shift.

Only the neutral intercept was calibrated. On the 100,000-applicant seed-4994
calibration population, the 70% target implies
`alpha_O = 0.8971702496810331`. The frozen systemic configuration fingerprint is
`59cf4f50249620c678e77ba8da55c1cc6ac9b8cad8950554bc6bb578558df64b`.

## Causal ordering and downstream effects

```text
Group + pre-treatment latent characteristics
    -> opportunity access
    -> employment, income, and liquid assets
    -> property value, loan request, DTI, and LTV
    -> true repayment probability
    -> observed repayment history
    -> group-blind lender risk estimate
    -> perceived profit
    -> budget-constrained allocation
```

The effects were declared before lending evaluation and held fixed at every
strength:

- `+1.50` feasible employment years;
- `+0.06` in the annual-income log equation;
- `+0.10` in the liquid-assets log equation.

Employment remains capped by age-consistent feasible history. Income and assets
are regenerated inside the existing structural equations with common residuals;
property, requests, DTI, and LTV are then recomputed through their existing
dependencies. Credit score is unchanged. No post-hoc row multiplier is used.

## Matched worlds

Applicant ID, group, age, cohort, latent factors, every population residual,
opportunity uniform, contract terms, repayment uniforms, and absolute budgets
are common across worlds. Only descendants of changed opportunity access can
change. Group A opportunity probabilities, realizations, and all applicant and
contract characteristics are exactly invariant across `s`.

The 10,000-applicant development population contains 4,967 Group B applicants.
Observed access was:

| `s` | Group A access | Group B access | Group B switchers from `s=0` |
|---:|---:|---:|---:|
| 0.00 | 68.51% | 70.22% | 0 |
| 0.10 | 68.51% | 67.89% | 116 (2.34% of B) |
| 0.20 | 68.51% | 65.79% | 220 (4.43% of B) |
| 0.40 | 68.51% | 61.83% | 417 (8.40% of B) |

The 70% target applies to the mean neutral probability on the calibration
population; a finite development sample's realized binary access rate need not
equal exactly 70%.

## Financial and true-risk pathway

At `s=0.40`, the 417 opportunity switchers lost an average 1.481 employment
years, $8,295 annual income, and $9,677 liquid assets. Averaged across every
Group B applicant, the corresponding changes were -0.124 years, -$696, and
-$812.

The frozen true-risk equations remain group-blind and contain neither strength
nor opportunity access. Under the additive-logistic risk function, switchers'
mean conditional repayment probability fell by 0.000680, whole-loan repayment
probability by 0.02494, and true expected request profit by $8,185. Under
`nonlinear_v1`, the changes were -0.007003, -0.03266, and -$11,349. This larger
nonlinear response is an interaction with the already frozen risk function, not
a recalibration of the opportunity mechanism.

## Repayment histories and lender models

One seed-1604994 applicant-by-period uniform matrix is applied to every strength
and both risk families. A changed `rho_true` can therefore cause a different
default under the same underlying repayment draw.

Each world rebuilds its historical at-risk payment table and refits:

- the unregularized traditional logit using the six DGP-aligned transformed
  observables; and
- the Prompt 12 `HistGradientBoostingClassifier`, using the same six raw
  observables and validation-log-loss grid.

Both exclude group, opportunity access, strength, latent factors, and truth.
Evaluation data is not used for fitting or selection. For all systemic fitted
policies, `repayment_probability_base == repayment_probability_used`; direct
belief distortion is zero.

## Portfolio experiment

The oracle, traditional lender, and ML lender are reoptimized in each world with
SciPy/HiGHS under the fixed Prompt 14 budgets of $583,043,937.080,
$233,217,574.832, and $116,608,787.416. All 72 MILPs terminated optimally with
zero reported MIP gap and respected their budgets.

At `s=0.40`, the change in maximum attainable true expected portfolio profit was:

| Risk world | 100% budget | 40% budget | 20% budget |
|---|---:|---:|---:|
| Additive logistic | -$494,625 | -$142,641 | -$39,293 |
| Nonlinear | -$651,664 | -$156,596 | -$15,947 |

These are changes in the economic opportunity set under the synthetic upstream
process—not the narrow direct-discrimination cost used in Prompt 15.

Under the 40% budget at `s=0.40`, the change in the B-minus-A funding-rate gap
was -0.80 percentage points for the additive oracle and -0.70 points for the
nonlinear oracle. Traditional changes were -0.30 and -0.10 points; ML changes
were -1.70 and +0.01 points. Results need not be monotone across budgets: the
upstream process changes income, risk, requested principal, expected profit,
and profit per funded dollar simultaneously. In the tight regime, smaller loan
requests can sometimes increase the number of funded Group B applicants even
while their economic circumstances worsen.

Group A has no upstream treatment, but its final allocations can change because
all applicants compete for a shared budget. These changes are portfolio
spillovers, not treatment of Group A. The switcher-lending table traces
opportunity switchers through funded-both, neutral-only, systemic-only, and
unfunded-both states.

## Controlled descriptive audit

For every portfolio, Model U0 estimates `funded ~ group_B`. Model U1 adds the
six downstream controls used by the lender. U1 does not identify a causal
discrimination effect: its controls are descendants of the upstream mechanism,
so it intentionally conditions away part of the pathway.

All 72 U0 models were numerically identified. Deterministic portfolio boundaries
caused perfect or quasi-separation in 20 of 72 U1 models; those coefficients and
standard errors are recorded as missing with an explicit identification flag,
rather than reporting divergent estimates. This limitation is itself useful:
an exact allocation rule can make a conventional unpenalized funding logit
ill-posed.

## Main artifacts

- `results/tables/v2_systemic_opportunity_summary.csv`
- `results/tables/v2_systemic_financial_pathway.csv`
- `results/tables/v2_systemic_switcher_effects.csv`
- `results/tables/v2_systemic_switcher_lending.csv`
- `results/tables/v2_systemic_group_effects.csv`
- `results/tables/v2_systemic_portfolio_profit.csv`
- `results/tables/v2_systemic_allocation_overlap.csv`
- `results/tables/v2_systemic_oracle_reference.csv`
- `results/tables/v2_systemic_controlled_audit.csv`
- `results/tables/v2_systemic_additive_nonlinear_comparison.csv`

Evaluator mechanism data is separated from truth-free policy and decision
artifacts under `data/processed/economic_lending/v2_systemic_opportunity/`.

## Limitations

The opportunity construct and all downstream effects are synthetic. One
deterministic development population is not a Monte Carlo sampling analysis.
The final lender is group-blind by construction, but group-blindness alone does
not ensure equal allocation when earlier opportunities differ. Realized profit
is one paired stochastic illustration, not expected welfare. Finally, neither
raw nor downstream-adjusted group coefficients automatically identify causal
discrimination in observational data.
