# Professor-Model Data and Result Guide

Paths below are relative to the repository root. On the author's machine that
root is `/Users/macbook/Hack/econ4994`. All records are synthetic.
These files are saved local artifacts, not files included in a fresh Git clone.

**Open first:** `applicants_bank_view.csv`, `loan_contracts.csv`,
`group_aware_vs_blind_portfolio.csv`, then `noise_full_summary.csv`.
The first three describe the canonical single-noise-draw analysis; the fourth
describes the complete 200-replication noise study.

## Applicant, signal, and prediction tables

Folder: `data/synthetic/professor_model/baseline/`.
Each main table has **100,000 rows**, one per applicant; join on `applicant_id`.

| Filename | Key columns and meaning | Information boundary |
|---|---|---|
| `applicants_bank_view.csv` | Y: annual income in dollars; D: DTI ratio; B: abstract 0/1 group; C: noisy credit signal | Bank-observed |
| `applicants_simulator_private.csv` | log_Y, epsilon, omega: private generating components | Evaluator only |
| `simulation_truth.csv` | P_income, P_other, true_PD: hidden probability product | Evaluator only |
| `training_signals.csv` | PN: continuous raw noisy regression target, including negatives | Historical bank measurement; not a default label |
| `risk_measurement_noise_private.csv` | eta: measurement-noise realization | Evaluator only |
| `applicant_splits.csv` | split: 80,000 training / 20,000 evaluation IDs | Fixed applicant-level split |
| `bank_risk_estimates.csv` | split, model_id, bank_PD_raw, bank_PD_bounded, prediction_out_of_bounds | Frozen group-aware estimate |
| `bank_risk_estimates_group_blind.csv` | Corresponding bank_PD_group_blind_raw/bounded fields | Frozen group-blind estimate |

All main tables above except `applicant_splits.csv` have matching `.parquet`
companions. Parquet preserves types efficiently; CSV is suitable for VS Code,
Excel, and Drive inspection. Raw probabilities and bounded probabilities must
not be confused: bounding is clipping, not calibration.

The 150-row audit-preview CSVs combine information for manual checks and are
**evaluator-only**, not lender training files or representative subsamples.

## Evaluation contracts, profit, and decisions

Folder: `data/processed/professor_model/`. Main tables have **20,000 rows**,
one per saved evaluation applicant; each has a `.parquet` companion.

| Filename | Important columns | Meaning |
|---|---|---|
| `loan_contracts.csv` | Y, ell_i, L_i, T, theta, r, payment_rule_id | One fixed income-linked request |
| `bank_expected_profit.csv` | bank_PD_bounded, expected_receipts_bank, expected_profit_bank | Group-aware perceived accounting |
| `bank_expected_profit_group_blind.csv` | bank_PD_group_blind_bounded, expected_receipts_bank, expected_profit_bank | Group-blind perceived accounting |
| `true_expected_profit.csv` | true_PD, expected_receipts_true, expected_profit_true | Evaluator-only potential loan value |
| `lending_decisions.csv` | optimal_tau, approved, requested_principal, funded_principal, decision_status | Canonical group-aware cutoff decision |
| `lending_decisions_group_blind.csv` | Same decision fields, group-blind score | Canonical group-blind cutoff decision |
| `oracle_lending_decisions.csv` | information_scope, true_PD, policy_id, cutoff and decision fields | Evaluator-only true-risk cutoff |

All have `applicant_id`. These are expected-value calculations and funding
decisions, not realized repayment histories. **No professor-model realized
default/outcome dataset has been generated.** Existing V2 repayment outcomes
belong to a different research family.

## Result CSVs to inspect

| Folder / filename | One row represents | Useful columns |
|---|---|---|
| `results/tables/professor_model/group_aware_vs_blind_portfolio.csv` | One canonical policy (3 rows) | policy, number_approved, true_expected_portfolio_profit, approval_gap_pp |
| `results/tables/professor_model/group_aware_vs_blind_group_audit.csv` | Policy × B group (6 rows) | approval_rate, funded_principal, n_applicants |
| `results/tables/professor_model/group_aware_vs_blind_risk.csv` | Policy × raw/bounded prediction × reference (8 rows) | MAE, RMSE, mean_signed_error |
| `results/tables/professor_model/bank_vs_oracle_summary.csv` | Canonical bank/oracle policy (2 rows) | funding, true profit, signed comparison, overlap |
| `results/tables/professor_model/noise_robustness_full/noise_full_results.csv` | Replication × policy (400 rows; 200 matched pairs) | replication, policy, beta_B_hat, approval_gap_pp, true_expected_portfolio_profit, true_profit_change_USD |
| `results/tables/professor_model/noise_robustness_full/noise_full_summary.csv` | Metric × scope (46 rows) | n_successful, n_defined, mean, median, sample_SD, MCSE_mean, p2.5, p97.5 |
| `results/tables/professor_model/noise_robustness_full/noise_full_coefficient_summary.csv` | Policy × coefficient (9 rows) | term, mean, sample_SD, MCSE_mean, quantiles |
| `results/tables/professor_model/noise_robustness_full/noise_full_group_outcome_summary.csv` | Policy × group × metric (16 rows) | B, metric, mean, dispersion |
| `results/tables/professor_model/noise_robustness_full/noise_full_tie_diagnostics.csv` | Replication × policy (400 rows) | zero_block_feasible, zero_block_principal_USD, reject_all_selected, maximum_budget_violation_USD |

Paired comparison fields are repeated on both policy rows in
`noise_full_results.csv`. Deduplicate by replication before counting paired
comparisons. Gaps ending `_pp` use percentage points; probability/rate fields
are proportions; monetary fields use dollars. Empirical quantiles describe
noise-outcome spread, not a confidence interval for a population causal effect.

## Metrics and complete inventory

`results/metrics/professor_model/noise_robustness_full/` contains the manifest,
immutable result/attempt records, bank freezes, evaluator signal reference,
and `latest_report.json`. The report verifies 200 successes, five pilot
reuses, 195 new replications, and zero failed attempts.

The [sharing manifest](../reports/professor_model/drive_upload/FILE_MANIFEST.json)
lists all **57 CSVs and 11 metadata JSONs** in the saved bundle, with exact
headers, row counts, byte sizes, and SHA-256 checksums. It also includes earlier
stage summaries and the five-replication pilot tables, which are retained as
historical evidence rather than substituted for Step 8B.

## Google Drive sharing

The current local archive is:
`reports/professor_model/drive_upload/csv_bundle.zip` (about 21.5 MB).
It preserves repository-relative folders and includes all 57 CSVs plus
11 metadata JSONs, about 72.9 MB uncompressed. Upload that ZIP or extract and
upload selected CSVs. The archive remains ignored by Git.

The committed [packaging script](../reports/professor_model/drive_upload/package_csv_bundle.py)
can copy existing files into a new bundle without generating data or fitting
models. Its default generated name differs from the author's renamed archive;
see [sharing instructions](../reports/professor_model/drive_upload/README.txt).
Do not overwrite the current archive or substitute hidden truth for bank data.
