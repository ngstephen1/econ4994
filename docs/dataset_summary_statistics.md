# Synthetic Dataset Summary Statistics

All statistics on this page were calculated from the four persisted
100,000-row, seed-4994, moderate-effect datasets. Monetary values are nominal
synthetic U.S. dollars. DTI, LTV, rates, shares, and probabilities are stored as
decimal proportions.

## Overall outcomes

| Scenario | Applications | Approvals | Denials | Approval rate | Mean true approval probability |
|:--|--:|--:|--:|--:|--:|
| Fair baseline | 100,000 | 80,054 | 19,946 | 0.8005 | 0.8015 |
| Direct discrimination | 100,000 | 79,702 | 20,298 | 0.7970 | 0.7979 |
| Upstream inequality | 100,000 | 79,272 | 20,728 | 0.7927 | 0.7933 |
| Mixed mechanism | 100,000 | 78,799 | 21,201 | 0.7880 | 0.7890 |

The fair baseline and direct-discrimination worlds share identical
pre-decision characteristics; only the configured direct race term changes
true probabilities and approval draws. The upstream and mixed worlds likewise
share pre-decision characteristics, while the mixed world adds the direct term.

## Fair baseline and direct-discrimination characteristics

The following pre-decision distributions are identical in these two scenarios.

| Variable | Mean | SD | p05 | Median | p95 | Minimum | Maximum |
|:--|--:|--:|--:|--:|--:|--:|--:|
| Annual income | 105,556.94 | 54,347.48 | 41,484.88 | 94,078.44 | 208,056.04 | 20,000.00 | 500,000.00 |
| Credit score | 719.79 | 50.08 | 637 | 720 | 802 | 514 | 850 |
| Employment years | 14.18 | 9.11 | 2.35 | 12.50 | 31.64 | 0.00 | 45.00 |
| Liquid assets | 57,361.45 | 65,027.35 | 6,764.63 | 37,425.81 | 174,529.60 | 1,000.00 | 750,000.00 |
| Existing monthly debt | 687.08 | 623.97 | 0.00 | 559.01 | 1,845.13 | 0.00 | 6,000.00 |
| Debt-to-income ratio | 0.4027 | 0.1120 | 0.2355 | 0.3915 | 0.6321 | 0.0641 | 0.6500 |
| Loan amount | 337,237.90 | 191,124.84 | 113,406.62 | 296,181.61 | 699,899.37 | 13,066.97 | 1,960,000.00 |
| Property value | 447,512.21 | 237,871.74 | 169,153.26 | 395,671.84 | 899,820.55 | 80,000.00 | 2,000,000.00 |
| Loan-to-value ratio | 0.7510 | 0.1293 | 0.5169 | 0.7654 | 0.9340 | 0.1000 | 0.9800 |
| Neighborhood income index | 1.0003 | 0.2153 | 0.6455 | 1.0006 | 1.3539 | 0.4000 | 1.9955 |
| Neighborhood minority share | 0.3244 | 0.1696 | 0.0923 | 0.2988 | 0.6426 | 0.0088 | 0.9471 |
| Local unemployment rate | 0.0401 | 0.0115 | 0.0206 | 0.0400 | 0.0592 | 0.0150 | 0.1009 |
| Income-to-loan ratio | 0.3409 | 0.1423 | 0.2103 | 0.3118 | 0.5563 | 0.1140 | 3.9466 |

True approval probability differs because it is an outcome-side quantity:

| Scenario | Mean | SD | p05 | Median | p95 | Minimum | Maximum |
|:--|--:|--:|--:|--:|--:|--:|--:|
| Fair baseline | 0.8015 | 0.2010 | 0.3498 | 0.8768 | 0.9880 | 0.0188 | 0.9998 |
| Direct discrimination | 0.7979 | 0.2031 | 0.3431 | 0.8737 | 0.9877 | 0.0176 | 0.9998 |

## Upstream-inequality and mixed-mechanism characteristics

The following pre-decision distributions are identical in these two scenarios.
Relative to fair baseline, aggregate shifts are modest because only Black
applications receive the upstream treatment.

| Variable | Mean | SD | p05 | Median | p95 | Minimum | Maximum |
|:--|--:|--:|--:|--:|--:|--:|--:|
| Annual income | 104,305.88 | 53,847.33 | 40,937.22 | 92,905.06 | 205,928.30 | 20,000.00 | 500,000.00 |
| Credit score | 716.58 | 50.82 | 633 | 717 | 800 | 500 | 850 |
| Employment years | 14.18 | 9.11 | 2.35 | 12.50 | 31.64 | 0.00 | 45.00 |
| Liquid assets | 55,339.69 | 63,366.18 | 6,397.76 | 35,851.99 | 169,463.07 | 1,000.00 | 750,000.00 |
| Existing monthly debt | 684.16 | 621.53 | 0.00 | 556.40 | 1,839.66 | 0.00 | 6,000.00 |
| Debt-to-income ratio | 0.4037 | 0.1123 | 0.2361 | 0.3924 | 0.6347 | 0.0641 | 0.6500 |
| Loan amount | 333,635.18 | 189,458.64 | 112,088.91 | 292,821.14 | 692,768.52 | 13,066.97 | 1,960,000.00 |
| Property value | 442,725.71 | 235,855.36 | 166,853.16 | 391,294.55 | 890,778.11 | 80,000.00 | 2,000,000.00 |
| Loan-to-value ratio | 0.7510 | 0.1293 | 0.5169 | 0.7654 | 0.9340 | 0.1000 | 0.9800 |
| Neighborhood income index | 1.0003 | 0.2153 | 0.6455 | 1.0006 | 1.3539 | 0.4000 | 1.9955 |
| Neighborhood minority share | 0.3244 | 0.1696 | 0.0923 | 0.2988 | 0.6426 | 0.0088 | 0.9471 |
| Local unemployment rate | 0.0401 | 0.0115 | 0.0206 | 0.0400 | 0.0592 | 0.0150 | 0.1009 |
| Income-to-loan ratio | 0.3405 | 0.1421 | 0.2100 | 0.3115 | 0.5557 | 0.1140 | 3.9466 |

True approval probability is:

| Scenario | Mean | SD | p05 | Median | p95 | Minimum | Maximum |
|:--|--:|--:|--:|--:|--:|--:|--:|
| Upstream inequality | 0.7933 | 0.2063 | 0.3336 | 0.8703 | 0.9874 | 0.0146 | 0.9998 |
| Mixed mechanism | 0.7890 | 0.2096 | 0.3219 | 0.8671 | 0.9872 | 0.0114 | 0.9998 |

## Outcomes by race and scenario

| Scenario | Race | n | Share | Approval rate | Denial rate | Mean true probability |
|:--|:--|--:|--:|--:|--:|--:|
| Fair baseline | White | 67,203 | 0.6720 | 0.8007 | 0.1993 | 0.8008 |
| Fair baseline | Black | 11,893 | 0.1189 | 0.8003 | 0.1997 | 0.8035 |
| Fair baseline | Asian | 9,108 | 0.0911 | 0.7970 | 0.2030 | 0.8046 |
| Fair baseline | American Indian or Alaska Native | 965 | 0.0097 | 0.8083 | 0.1917 | 0.7935 |
| Fair baseline | Other / Multiracial | 10,831 | 0.1083 | 0.8020 | 0.1980 | 0.8021 |
| Direct discrimination | White | 67,203 | 0.6720 | 0.8007 | 0.1993 | 0.8008 |
| Direct discrimination | Black | 11,893 | 0.1189 | 0.7707 | 0.2293 | 0.7724 |
| Direct discrimination | Asian | 9,108 | 0.0911 | 0.7970 | 0.2030 | 0.8046 |
| Direct discrimination | American Indian or Alaska Native | 965 | 0.0097 | 0.8083 | 0.1917 | 0.7935 |
| Direct discrimination | Other / Multiracial | 10,831 | 0.1083 | 0.8020 | 0.1980 | 0.8021 |
| Upstream inequality | White | 67,203 | 0.6720 | 0.8007 | 0.1993 | 0.8008 |
| Upstream inequality | Black | 11,893 | 0.1189 | 0.7345 | 0.2655 | 0.7344 |
| Upstream inequality | Asian | 9,108 | 0.0911 | 0.7970 | 0.2030 | 0.8046 |
| Upstream inequality | American Indian or Alaska Native | 965 | 0.0097 | 0.8083 | 0.1917 | 0.7935 |
| Upstream inequality | Other / Multiracial | 10,831 | 0.1083 | 0.8020 | 0.1980 | 0.8021 |
| Mixed mechanism | White | 67,203 | 0.6720 | 0.8007 | 0.1993 | 0.8008 |
| Mixed mechanism | Black | 11,893 | 0.1189 | 0.6948 | 0.3052 | 0.6978 |
| Mixed mechanism | Asian | 9,108 | 0.0911 | 0.7970 | 0.2030 | 0.8046 |
| Mixed mechanism | American Indian or Alaska Native | 965 | 0.0097 | 0.8083 | 0.1917 | 0.7935 |
| Mixed mechanism | Other / Multiracial | 10,831 | 0.1083 | 0.8020 | 0.1980 | 0.8021 |

Only Black receives the current direct and upstream experimental treatments.
The other race categories retain the same records and probabilities across
scenarios, apart from no scenario-specific change at all under the shared seed.

## White and Black financial comparison

| Scenario | Race | Mean income | Median income | Mean credit | Median assets | Mean DTI | Mean LTV | Mean true probability | Approval rate |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| Fair baseline | White | 105,391 | 93,880 | 719.56 | 37,459 | 0.4030 | 0.7511 | 0.8008 | 0.8007 |
| Fair baseline | Black | 105,281 | 93,488 | 720.57 | 37,248 | 0.4020 | 0.7497 | 0.8035 | 0.8003 |
| Direct discrimination | White | 105,391 | 93,880 | 719.56 | 37,459 | 0.4030 | 0.7511 | 0.8008 | 0.8007 |
| Direct discrimination | Black | 105,281 | 93,488 | 720.57 | 37,248 | 0.4020 | 0.7497 | 0.7724 | 0.7707 |
| Upstream inequality | White | 105,391 | 93,880 | 719.56 | 37,459 | 0.4030 | 0.7511 | 0.8008 | 0.8007 |
| Upstream inequality | Black | 94,762 | 84,139 | 693.54 | 26,087 | 0.4106 | 0.7497 | 0.7344 | 0.7345 |
| Mixed mechanism | White | 105,391 | 93,880 | 719.56 | 37,459 | 0.4030 | 0.7511 | 0.8008 | 0.8007 |
| Mixed mechanism | Black | 94,762 | 84,139 | 693.54 | 26,087 | 0.4106 | 0.7497 | 0.6978 | 0.6948 |

This table makes the mechanism distinction visible:

- Direct discrimination changes Black approval probabilities without changing
  Black pre-decision financial distributions.
- Upstream inequality changes the configured financial distributions without a
  direct race term in underwriting.
- The mixed scenario combines both pathways.

## Interpretation cautions

- These are synthetic distributions chosen for controlled method evaluation.
- They are not estimates of real U.S. mortgage applicants or lenders.
- A raw approval difference is a disparity, not automatically a causal estimate
  of discrimination.
- The true mechanism is known only because it was encoded by the simulation.
- `approval_probability_true` is used for evaluation and diagnostics, never as
  a training predictor.
