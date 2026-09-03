# Synthetic Dataset Inventory

## Generated files

| Scenario | Parquet file | Rows | Columns | Approx. in-memory size | Unique IDs |
|:--|:--|--:|--:|--:|--:|
| Fair baseline | `synthetic_fair_baseline_moderate_n100000_seed4994.parquet` | 100,000 | 24 | 19.74 MiB | 100,000 |
| Direct discrimination | `synthetic_direct_discrimination_moderate_n100000_seed4994.parquet` | 100,000 | 24 | 19.74 MiB | 100,000 |
| Upstream inequality | `synthetic_upstream_inequality_moderate_n100000_seed4994.parquet` | 100,000 | 24 | 19.74 MiB | 100,000 |
| Mixed mechanism | `synthetic_mixed_mechanism_moderate_n100000_seed4994.parquet` | 100,000 | 24 | 19.74 MiB | 100,000 |

All files are under `data/synthetic/`. Each has an adjacent
`.metadata.json` file containing its resolved configuration, seed, calibrated
intercept, configuration fingerprint, schema size, and provenance.

## Field inventory and completeness

The table below is computed from the fair-baseline dataset. The other scenarios
use the same persisted schema and missingness policy.

| Field | Stored dtype | Missing | Non-null unique values | Role |
|:--|:--|--:|--:|:--|
| `application_id` | object | 0 | 100,000 | Identifier; never a predictor |
| `race` | category | 0 | 5 | Protected attribute and audit group |
| `ethnicity` | category | 0 | 2 | Protected attribute and audit group |
| `sex` | category | 0 | 2 | Protected attribute and audit group |
| `age_group` | category | 0 | 6 | Protected/sensitive demographic attribute |
| `annual_income` | float64 | 0 | 99,843 | Borrower financial characteristic |
| `credit_score` | int16 | 0 | 324 | Borrower financial characteristic |
| `employment_years` | float64 | 0 | 97,193 | Borrower financial characteristic |
| `liquid_assets` | float64 | 0 | 99,857 | Borrower financial characteristic |
| `existing_monthly_debt` | float64 | 0 | 84,986 | Borrower financial characteristic |
| `debt_to_income_ratio` | float64 | 0 | 95,803 | Derived underwriting ratio |
| `loan_amount` | float64 | 0 | 100,000 | Requested loan amount |
| `property_value` | float64 | 0 | 99,837 | Property characteristic |
| `loan_to_value_ratio` | float64 | 0 | 98,635 | Derived underwriting ratio |
| `loan_purpose` | category | 0 | 3 | Loan characteristic |
| `loan_type` | category | 0 | 4 | Loan characteristic |
| `occupancy_type` | category | 0 | 3 | Loan/property characteristic |
| `neighborhood_income_index` | float64 | 0 | 99,756 | Context characteristic |
| `neighborhood_minority_share` | float64 | 0 | 100,000 | Audit/proxy-sensitivity field |
| `local_unemployment_rate` | float64 | 0 | 98,340 | Context characteristic |
| `income_to_loan_ratio` | float64 | 0 | 100,000 | Derived descriptive ratio |
| `approval_probability_true` | float64 | 0 | 100,000 | Synthetic ground truth; never a predictor |
| `approved` | int8 | 0 | 2 | Observed binary outcome |
| `denial_reason` | object | 100,000 | 0 | Disabled downstream field |

`denial_reason` being null is intentional. Version 1 does not invent denial
reasons without a separately designed assignment mechanism.

## Demographic composition

The same seed and demographic stream produce the same composition in all four
scenarios.

### Race

| Category | Count | Share |
|:--|--:|--:|
| White | 67,203 | 67.203% |
| Black | 11,893 | 11.893% |
| Other / Multiracial | 10,831 | 10.831% |
| Asian | 9,108 | 9.108% |
| American Indian or Alaska Native | 965 | 0.965% |

### Ethnicity

| Category | Count | Share |
|:--|--:|--:|
| Not Hispanic or Latino | 86,051 | 86.051% |
| Hispanic or Latino | 13,949 | 13.949% |

### Sex

| Category | Count | Share |
|:--|--:|--:|
| Male | 53,889 | 53.889% |
| Female | 46,111 | 46.111% |

### Age group

| Category | Count | Share |
|:--|--:|--:|
| 18–24 | 3,001 | 3.001% |
| 25–34 | 23,982 | 23.982% |
| 35–44 | 27,894 | 27.894% |
| 45–54 | 20,912 | 20.912% |
| 55–64 | 15,284 | 15.284% |
| 65+ | 8,927 | 8.927% |

Demographic attributes are independently generated in version 1. These shares
are synthetic calibration choices, not estimates of mortgage applicant
composition.

## Loan and property categories

### Loan purpose

| Category | Count | Share |
|:--|--:|--:|
| Home purchase | 69,900 | 69.900% |
| Refinance | 25,077 | 25.077% |
| Home improvement | 5,023 | 5.023% |

### Loan type

| Category | Count | Share |
|:--|--:|--:|
| Conventional | 72,759 | 72.759% |
| FHA | 15,965 | 15.965% |
| VA | 9,233 | 9.233% |
| USDA/RHS | 2,043 | 2.043% |

### Occupancy type

| Category | Count | Share |
|:--|--:|--:|
| Principal residence | 88,042 | 88.042% |
| Investment property | 7,977 | 7.977% |
| Second residence | 3,981 | 3.981% |

These categorical draws are shared across the four scenarios. Scenario
treatments do not change their distributions in version 1.

## Exact identities and bounds

Every persisted row satisfies:

```text
loan_amount = property_value × loan_to_value_ratio
income_to_loan_ratio = annual_income ÷ loan_amount
0 ≤ approval_probability_true ≤ 1
approved ∈ {0, 1}
```

Selected boundary diagnostics are:

| Scenario family | DTI at 0.65 ceiling | LTV at 0.98 ceiling | Existing debt equal to zero |
|:--|--:|--:|--:|
| Fair/direct | 4.198% | 1.365% | 14.984% |
| Upstream/mixed | 4.305% | 1.365% | 14.984% |

The DTI ceiling remains below the predeclared 5% validation limit.

## Storage and reproduction note

The Parquet and metadata files are generated artifacts and are ignored by Git.
They can be regenerated with the committed simulation code. This Markdown
inventory is tracked so that the dataset contents can be reviewed on GitHub
without committing approximately 55 MB of Parquet data.
