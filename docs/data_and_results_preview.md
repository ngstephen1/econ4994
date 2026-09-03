# Synthetic Data and Results Preview

This page provides small, human-readable excerpts from the generated Parquet
datasets and CSV result tables. The full generated artifacts are intentionally
excluded from Git because they can be reproduced from the committed code,
configuration, and seed.

## Full dataset files

Each dataset contains 100,000 mortgage applications and the same 24 persisted
fields:

- `data/synthetic/synthetic_fair_baseline_moderate_n100000_seed4994.parquet`
- `data/synthetic/synthetic_direct_discrimination_moderate_n100000_seed4994.parquet`
- `data/synthetic/synthetic_upstream_inequality_moderate_n100000_seed4994.parquet`
- `data/synthetic/synthetic_mixed_mechanism_moderate_n100000_seed4994.parquet`

The 24 fields are:

```text
application_id, race, ethnicity, sex, age_group, annual_income,
credit_score, employment_years, liquid_assets, existing_monthly_debt,
debt_to_income_ratio, loan_amount, property_value, loan_to_value_ratio,
loan_purpose, loan_type, occupancy_type, neighborhood_income_index,
neighborhood_minority_share, local_unemployment_rate,
income_to_loan_ratio, approval_probability_true, approved, denial_reason
```

`approved` is the observed Bernoulli outcome: 1 means approved and 0 means
denied. `approval_probability_true` is known only because the data are
synthetic. It is never used as a model input.

## Dataset snippets

The excerpts below follow the same five Black application IDs across all four
scenarios. This makes it possible to inspect how the configured mechanisms
change financial characteristics and true approval probabilities.

### Fair baseline

| application_id | race | annual_income | credit_score | liquid_assets | DTI | LTV | true approval probability | approved |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| APP000000011 | Black | 36,202.23 | 667 | 34,957.41 | 0.2772 | 0.9063 | 0.7419 | 1 |
| APP000000016 | Black | 58,239.64 | 690 | 88,876.58 | 0.6500 | 0.6336 | 0.5843 | 0 |
| APP000000021 | Black | 94,745.35 | 775 | 23,935.25 | 0.4251 | 0.7252 | 0.9253 | 1 |
| APP000000028 | Black | 120,067.70 | 702 | 81,132.53 | 0.4657 | 0.6913 | 0.8517 | 1 |
| APP000000029 | Black | 151,167.22 | 825 | 133,264.18 | 0.3731 | 0.7905 | 0.9823 | 1 |

### Direct discrimination

Financial characteristics remain equal to the fair baseline for these records,
but the configured Black direct term lowers the true approval probabilities.

| application_id | race | annual_income | credit_score | liquid_assets | DTI | LTV | true approval probability | approved |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| APP000000011 | Black | 36,202.23 | 667 | 34,957.41 | 0.2772 | 0.9063 | 0.6912 | 1 |
| APP000000016 | Black | 58,239.64 | 690 | 88,876.58 | 0.6500 | 0.6336 | 0.5226 | 0 |
| APP000000021 | Black | 94,745.35 | 775 | 23,935.25 | 0.4251 | 0.7252 | 0.9061 | 1 |
| APP000000028 | Black | 120,067.70 | 702 | 81,132.53 | 0.4657 | 0.6913 | 0.8172 | 1 |
| APP000000029 | Black | 151,167.22 | 825 | 133,264.18 | 0.3731 | 0.7905 | 0.9774 | 1 |

### Upstream inequality

There is no direct race term. Instead, the configured upstream treatment shifts
income, credit score, and liquid assets for Black applications.

| application_id | race | annual_income | credit_score | liquid_assets | DTI | LTV | true approval probability | approved |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| APP000000011 | Black | 32,581.99 | 640 | 24,482.63 | 0.2802 | 0.9063 | 0.6395 | 1 |
| APP000000016 | Black | 52,415.65 | 663 | 62,245.23 | 0.6500 | 0.6336 | 0.4693 | 0 |
| APP000000021 | Black | 85,270.77 | 748 | 16,763.20 | 0.4391 | 0.7252 | 0.8764 | 1 |
| APP000000028 | Black | 108,060.88 | 675 | 56,821.64 | 0.4859 | 0.6913 | 0.7582 | 1 |
| APP000000029 | Black | 136,050.43 | 798 | 93,332.35 | 0.3807 | 0.7905 | 0.9707 | 1 |

### Mixed mechanism

The upstream shifts and direct Black approval term are both active.

| application_id | race | annual_income | credit_score | liquid_assets | DTI | LTV | true approval probability | approved |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| APP000000011 | Black | 32,581.99 | 640 | 24,482.63 | 0.2802 | 0.9063 | 0.5801 | 1 |
| APP000000016 | Black | 52,415.65 | 663 | 62,245.23 | 0.6500 | 0.6336 | 0.4078 | 0 |
| APP000000021 | Black | 85,270.77 | 748 | 16,763.20 | 0.4391 | 0.7252 | 0.8466 | 1 |
| APP000000028 | Black | 108,060.88 | 675 | 56,821.64 | 0.4859 | 0.6913 | 0.7095 | 1 |
| APP000000029 | Black | 136,050.43 | 798 | 93,332.35 | 0.3807 | 0.7905 | 0.9627 | 1 |

An individual Bernoulli outcome need not change when its true probability
changes. The scenario comparisons are meaningful across the full samples, not
from five rows alone.

## Prompt 6 CSV excerpt: statistical disparities

Full file: `results/tables/race_approval_gaps.csv`

| Scenario | White n | Black n | White approval | Black approval | Raw gap (pp) | 95% CI (pp) |
|:--|--:|--:|--:|--:|--:|--:|
| Fair baseline | 67,203 | 11,893 | 0.8007 | 0.8003 | -0.0406 | [-0.8199, 0.7388] |
| Direct discrimination | 67,203 | 11,893 | 0.8007 | 0.7707 | -3.0003 | [-3.8139, -2.1866] |
| Upstream inequality | 67,203 | 11,893 | 0.8007 | 0.7345 | -6.6159 | [-7.4650, -5.7667] |
| Mixed mechanism | 67,203 | 11,893 | 0.8007 | 0.6948 | -10.5930 | [-11.4740, -9.7120] |

The raw gap is Black approval rate minus White approval rate. A negative value
means a lower Black approval rate in the synthetic sample.

## Prompt 7 CSV excerpt: main ML model

Full file: `results/tables/ml_benchmark.csv`

The excerpt shows the selected logistic-regression results. Probability gaps
and reproduction errors below are proportions; multiply by 100 for percentage
points.

| Scenario | Feature regime | Accuracy | ROC-AUC | Log loss | Brier | Observed gap | Mean-prediction gap | Threshold gap | Reproduction error | True-p MAE | True-p RMSE |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Fair baseline | Race-blind | 0.8339 | 0.8291 | 0.3757 | 0.1180 | -0.0015 | 0.0027 | 0.0039 | 0.0042 | 0.0122 | 0.0184 |
| Fair baseline | Race-aware sensitivity | 0.8340 | 0.8291 | 0.3757 | 0.1180 | -0.0015 | 0.0004 | 0.0013 | 0.0019 | 0.0124 | 0.0186 |
| Direct discrimination | Race-blind | 0.8317 | 0.8287 | 0.3796 | 0.1194 | -0.0338 | 0.0028 | 0.0025 | 0.0366 | 0.0148 | 0.0220 |
| Direct discrimination | Race-aware sensitivity | 0.8324 | 0.8293 | 0.3791 | 0.1192 | -0.0338 | -0.0275 | -0.0291 | 0.0064 | 0.0123 | 0.0186 |
| Upstream inequality | Race-blind | 0.8299 | 0.8285 | 0.3835 | 0.1208 | -0.0687 | -0.0620 | -0.0768 | 0.0066 | 0.0124 | 0.0187 |
| Upstream inequality | Race-aware sensitivity | 0.8299 | 0.8285 | 0.3835 | 0.1208 | -0.0687 | -0.0632 | -0.0774 | 0.0055 | 0.0125 | 0.0189 |
| Mixed mechanism | Race-blind | 0.8267 | 0.8285 | 0.3878 | 0.1225 | -0.1068 | -0.0642 | -0.0792 | 0.0426 | 0.0158 | 0.0236 |
| Mixed mechanism | Race-aware sensitivity | 0.8277 | 0.8292 | 0.3873 | 0.1223 | -0.1068 | -0.1041 | -0.1249 | 0.0027 | 0.0127 | 0.0191 |

## Other generated CSV tables

- `results/tables/descriptive_by_scenario.csv`
- `results/tables/logit_black_effects.csv`
- `results/tables/statistical_recovery.csv`
- `results/tables/ml_group_audit.csv`
- `results/tables/ml_hyperparameter_selection.csv`
- `results/tables/ml_selected_hyperparameters.csv`
- `results/tables/ml_race_regime_comparison.csv`
- `results/tables/ml_oracle_benchmark.csv`
- `results/tables/ml_calibration_bins.csv`
- `results/tables/ml_group_calibration_bins.csv`
- `results/tables/ml_split_summary.csv`
- `results/tables/statistical_ml_comparison.csv`

## Inspect locally

To print any Parquet or CSV excerpt:

```bash
python3 - <<'PY'
import pandas as pd

data = pd.read_parquet(
    "data/synthetic/synthetic_fair_baseline_moderate_n100000_seed4994.parquet"
)
results = pd.read_csv("results/tables/ml_benchmark.csv")

print(data.head(10).to_string(index=False))
print(results.head(10).to_string(index=False))
PY
```
