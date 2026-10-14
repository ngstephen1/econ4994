# Research Documentation

Start with [current project status](PROJECT_STATUS.md) or the
[root README](../README.md). The latest active family is the professor's
imperfect-information lending model, completed through Step 8B.

Three research families coexist. Their groups, risk events, contracts, policy
rules, and Monte Carlo designs are different; do not combine their tables
without an explicit comparison design.

## Current professor model

| Topic | Document |
|---|---|
| Working equations and professor/researcher assumption labels | [Whiteboard specification](professor_whiteboard_model.md) |
| Applicant generation and information-separated schema | [Population generation](professor_population_generation.md) |
| Hidden two-factor default probability | [True default risk](professor_true_default_risk.md) |
| Raw noisy PN and private eta | [Noisy risk measurement](professor_noisy_risk_signal.md) |
| Group-aware OLS, split, raw/bounded predictions | [Bank regression](professor_bank_regression.md) |
| Contract payments, survival, and expected profit | [Loan economics](professor_loan_economics.md) |
| Restricted common-cutoff bank policy | [Lending threshold](professor_lending_threshold.md) |
| Evaluator-only true-risk cutoff | [Oracle comparison](professor_oracle_comparison.md) |
| Matched group-aware/group-blind policies | [Group-blind comparison](professor_group_blind_comparison.md) |
| Five-draw pilot (historical stage record) | [Noise robustness pilot](professor_noise_robustness_pilot.md) |
| Complete 200-noise-draw results and caveats | [Full noise robustness](professor_noise_robustness_full.md) |
| CSV/Parquet locations and sharing | [Data guide](professor_model_data_guide.md) |
| Eight slides, PDF, figures, and evidence | [Progress-report folder](../reports/professor_model/step8b_progress/README.md) |

Earlier professor documents describe what was implemented at each step.
Phrases such as “not yet added” and their test counts are stage-specific.
The initial annual-rate suggestion in Step 1 is not the implemented Step 6A
per-period contract; the later explicit accounting document governs results.

## Preserved Version 2: economic repayment and allocation

| Topic | Document |
|---|---|
| Conditional repayment event and economic schema | [V2 specification](economic_lending_spec.md) |
| Population and true repayment calibration | [Population calibration](economic_population_calibration.md) |
| Historical at-risk logit | [Traditional lender](traditional_lender_risk_model.md) |
| Flexible repayment estimator | [ML repayment benchmark](ml_repayment_risk_benchmark.md) |
| Nonlinear world and model misspecification | [Nonlinear risk sensitivity](nonlinear_risk_sensitivity.md) |
| Budget-constrained whole-request MILP | [Portfolio benchmark](portfolio_allocation_benchmark.md) |
| Direct distortion of repayment beliefs | [Direct belief discrimination](direct_belief_discrimination.md) |
| Upstream opportunity treatment | [Systemic opportunity](systemic_opportunity_discrimination.md) |
| Corrected five-replication checkpoint and incomplete 50-run study | [Monte Carlo recovery](systemic_monte_carlo_recovery.md) |

## Preserved Version 1: approval disparities

| Topic | Document |
|---|---|
| Research questions and causal design | [Research design](research_design.md) |
| 24-field schema and four worlds | [Schema](synthetic_schema.md), [scenarios](simulation_scenarios.md) |
| Assumptions and frozen parameters | [Calibration](simulation_calibration.md) |
| Generator implementation | [Generator design](generator_design.md) |
| DTI diagnosis and resolved tail warning | [DTI recalibration](dti_recalibration.md) |
| Regression recovery | [Statistical recovery](statistical_recovery.md) |
| Race-blind/race-aware prediction | [ML benchmark](ml_benchmark.md) |
| Complete 3,140-setting sensitivity study | [Sensitivity](sensitivity_analysis.md) |
| Seven-page V1 dashboard | [Dashboard guide](dashboard.md) |
| V1-only manual data previews and inventory | [Preview](data_and_results_preview.md), [inventory](dataset_inventory.md), [summary statistics](dataset_summary_statistics.md) |

## Navigation and archival policy

- [Project History](PROJECT_HISTORY.md) preserves the prior detailed status
  narrative, including historical test results and phase milestones.
- [V1/V2 run guide](running_v1_v2.md) retains earlier CLI entry points.
- Historical evidence remains in place so existing links and provenance
  references remain meaningful. “Cleanup” means labeling and indexing, not
  deleting scientific records.
- Full generated datasets/results remain local and ignored by Git; report
  excerpts and presentation artifacts are deliberately small, curated evidence.
