# Project Status

## Current phase

Phase 14 — deterministic systemic-opportunity experiment completed.
Prompt 17B — corrected five-replication checkpoint completed; full study not complete.

The legacy Monte Carlo directory contains 19 success records and one solver
failure (replication 13), representing 1,368 saved portfolios. These records
lack dirty-source fingerprints and are provisional. The interrupted retry was
stopped during recovery. The target remains 50 successful replications; 20 is
an interim checkpoint, not completion. No broader batch is authorized during
recovery. Replication 13 has now completed separately under the corrected
implementation: 72 feasible portfolios in 404.2 seconds, with provenance and
integrity checks passed. Summary-only resumption launched zero new runs.
The Prompt 17B focused suite passed 54 tests. The full suite passed 284 tests
with 32 existing deprecation warnings.
There are now five corrected validated replications, not 20 or 50.
See [the recovery audit](systemic_monte_carlo_recovery.md).

Prompt 17B added corrected replications 0, 1, 2, and 3 without selecting on
legacy outcomes. Together with replication 13, all five corrected records share
the same scientific, source-content, dependency, solver-setting, and schema
fingerprints. All 360 required portfolios were optimal within numerical
roundoff, with no time limits, failed portfolios, or budget violations. The
checkpoint is classified `A. READY_FOR_FULL_50`, but the remaining 45 runs were
not launched. Legacy records remain provisional and excluded from aggregation.

## Completed

- Research context established.
- Repository scaffold established.
- Proposed 24-variable synthetic data schema documented.
- Causal ordering and four core simulation scenarios documented.
- Design-only YAML configuration schema established.
- Version-1 demographic, financial, loan, context, and approval assumptions
  calibrated and documented.
- Direct and upstream mild/moderate/strong treatment levels defined.
- Configuration-integrity tests established.
- Version-1 vectorized generator implemented for all four scenarios.
- Frozen fair-baseline intercept calibration and reproducibility metadata
  implemented.
- Structured generator validation and automated behavioral tests implemented.
- Four 10,000-row seed-4994 validation datasets generated with one shared,
  one-million-row fair-baseline intercept calibration.
- DTI tail diagnosed on a deterministic 100,000-row population.
- Property-request dependencies minimally recalibrated without changing
  payment assumptions or experimental treatments.
- Approval intercept recalibrated once and frozen after the population change.
- Four deterministic 100,000-row, seed-4994 scenario datasets generated for
  the first statistical experiment.
- Overall and race-specific outcome summaries and unpooled 95% confidence
  intervals for the raw Black-White approval gap implemented.
- Four predeclared statsmodels logistic specifications implemented with exact
  DGP transformations and explicit categorical references.
- Standardized adjusted probability contrasts and generator-based direct-effect
  ground truth implemented.
- Statistical recovery tables, figures, diagnostics, and deterministic analysis
  tests completed.
- Two controlled ML feature regimes implemented: race-blind primary and
  race-aware sensitivity.
- Deterministic 60/20/20 application-level splitting and validation-only
  hyperparameter selection implemented.
- Logistic regression, random forest, and histogram gradient boosting evaluated
  across all four scenarios.
- Overall prediction, race-group audit, disparity-reproduction, calibration,
  true-probability recovery, and synthetic-oracle metrics completed.
- Five ML benchmark figures and deterministic leakage/splitting/behavior tests
  completed.
- Seven-page Streamlit research dashboard implemented around the existing
  simulation, statistical, and ML APIs.
- In-memory scenario generation, bounded custom treatments, active-sample
  statsmodels analysis, and a fixed-seed direct-effect sensitivity curve added.
- Saved benchmark exploration, race-blind/race-aware comparison, group audits,
  calibration, true-probability recovery, and a no-retraining global-threshold
  explorer added.
- Mechanism comparison, 24-field data dictionary, reproducibility metadata,
  CSV/Parquet/JSON downloads, controlled missing-artifact messages, caching, and
  cross-page state implemented.
- Dashboard support tests, Streamlit script execution checks, and local server
  route checks completed.
- Resumable sensitivity framework implemented with stable SHA-256 run identities,
  atomic per-run JSON, common-random-number seeds, and deterministic parallel
  execution.
- Direct-effect, upstream-strength, mixed-mechanism, sample-size, and detection
  designs completed across 3,140 unique replication/settings.
- All four statsmodels specifications and both validation-tuned logistic ML
  regimes evaluated for every run; raw, adjusted, predicted, and synthetic-truth
  quantities retained separately.
- Monte Carlo bias, RMSE, empirical intervals, coverage, detection, sign recovery,
  false positives, mechanism signatures, and result-derived thresholds completed.
- Thirteen sensitivity figures, tidy result families, a paper-ready summary, and
  a read-only seventh dashboard page completed.
- Version 1 approval-disparity benchmark preserved at Git tag
  `v1-approval-benchmark`.
- Version 2 economic lending package started separately under
  `src/fair_lending/economic_lending/`.
- Version 2 baseline schema, hidden conditional per-period repayment probability,
  period survival, survival-weighted expected receipts, expected profit,
  fixed-request allocation, and hand-checkable four-applicant fixture implemented.
- Version 2 repayment and cash-flow accounting reconciled: `rho_true` now means
  the probability of making the next payment conditional on no previous default,
  with default absorbing and full repayment probability equal to `rho_true ** T`.
- Version 2 applicant population and hidden true repayment DGP calibrated on a
  deterministic 100,000-row diagnostic sample, then persisted as a 10,000-row
  development dataset with train, validation, and evaluation cohorts.
- Absorbing realized repayment histories, cohort-safe outputs, named random
  streams, population/risk/economic validation, and group-invariance audits
  implemented.
- Traditional lender at-risk payment histories implemented without post-default
  rows or applicant-level cohort leakage.
- Unregularized logistic risk estimation fitted on `historical_train`, with
  held-out validation/evaluation metrics, hidden-truth recovery, oracle
  comparison, group audit, and perceived-profit diagnostics completed.
- Applicant-weighted logit and linear-probability sensitivities documented;
  the primary model remains the probability-valid unweighted logit.
- Truth-free evaluation `policy_assessments.parquet` and transparent
  coefficient-only model metadata generated for `traditional_logit_v1`.
- A single histogram-gradient-boosting repayment-risk model was fit to the same
  six substantive observables and at-risk payment target as the frozen
  traditional lender.
- Eight predeclared HistGB settings were compared using historical-validation
  log loss only; evaluation data remained outside model selection.
- Traditional, flexible-model, and true-risk-oracle probability recovery,
  realized-label performance, calibration, group audits, and expected-profit
  diagnostics were completed.
- The traditional logit outperformed HistGB under the correctly specified
  additive-logistic baseline, as anticipated by the research design.
- Truth-free `policy_assessments.parquet` now contains separate
  `traditional_logit_v1` and `ml_histgb_v1` evaluation rows.
- A separate `nonlinear_v1` true-risk world was frozen without changing the
  additive baseline, applicant population, contracts, information set, or group
  neutrality.
- The nonlinear intercept was calibrated on 100,000 applicants to the baseline
  mean full-repayment probability before models were evaluated.
- Matched common-uniform repayment histories, re-estimated traditional logit,
  validation-selected HistGB, probability recovery, economic diagnostics, and
  cross-world tables were completed.
- HistGB was closer for 61.1% of nonlinear-world applicants and reduced profit
  MAE by 9.6%, but had worse aggregate risk MAE/RMSE, profit RMSE, and wrong-sign
  rate; the sensitivity therefore produced a mixed result rather than a general
  ML advantage.
- True-risk oracle, frozen traditional, and frozen HistGB assessments were
  allocated across the same 2,000 evaluation requests using exact SciPy/HiGHS
  binary MILP under common 100%, 40%, and 20% budgets.
- All 18 primary portfolios solved with reported zero optimality gap; the 40%
  and 20% constraints bound for every policy, while the 100% regime left funds
  unused rather than funding nonpositive perceived-profit requests.
- True expected portfolio value, economic regret, realized-profit illustration,
  allocation overlap, disagreement consequences, tail losses, and neutral group
  audits were completed across both risk worlds.
- ML achieved lower regret than traditional only in the nonlinear 20% regime;
  prediction improvements did not translate uniformly into allocation gains.
- A separate direct-belief-distortion family now applies predeclared Group B
  repayment-log-odds shifts of 0, 0.05, 0.10, and 0.20 after frozen traditional,
  HistGB, or evaluator true-risk probabilities are formed.
- True risk, applicant characteristics, contracts, repayment outcomes, base
  predictions, and Prompt 14 budgets remain unchanged across treatments.
- Twenty-four matched probability policies and 72 exact portfolios were
  evaluated with difference-in-gap, group-level spillover, displaced/replacement
  borrower, overlap, true-profit, and incremental-regret diagnostics.
- The true-risk mechanism reference produced nonnegative economic cost in all
  treatments; fitted-model interactions occasionally offset ordinary estimation
  error at mild strengths, without making the discriminatory treatment benign.
- Structural tests confirm that group does not enter baseline applicant
  characteristics, repayment truth, profit, or allocation.
- A separate `systemic_opportunity_v1` family now models an explicit upstream
  opportunity decision using only group and pre-treatment latent stability.
- The neutral opportunity intercept was calibrated once to a 70% probability
  target; strengths of 0.00, 0.10, 0.20, and 0.40 and modest employment,
  income, and asset effects were frozen before lending evaluation.
- Matched population residuals, opportunity uniforms, repayment uniforms,
  cohorts, contracts, and absolute budgets support paired counterfactuals;
  Group A applicant characteristics are exactly invariant.
- Both additive and nonlinear true-risk functions remain group-blind, and each
  systemic world refits the same group-blind traditional logit and HistGB model.
- Seventy-two oracle/traditional/ML portfolios solved optimally with zero
  reported MIP gap. Financial pathways, opportunity switchers, Group A
  portfolio spillovers, allocation gaps, profit changes, and oracle regret are
  reported separately.
- The controlled funding audit explicitly flags 20 U1 specifications with
  perfect or quasi-separation instead of interpreting divergent coefficients.

## Resolved calibration finding

The original one-million-row population placed 10.7893% at the 0.65 DTI
ceiling. Revision 2 reduces this to 4.2781% while retaining the unchanged 5%
threshold. The original and revised diagnostics are documented in
`docs/dti_recalibration.md`.

## Statistical recovery finding

In the 100,000-row runs, Model 2 estimated Black log-odds coefficients of
-0.263 and -0.272 in the direct and mixed scenarios, respectively, against the
configured -0.25 direct effect. The upstream-only coefficient moved from
-0.373 unadjusted to approximately zero with the DGP controls. Full results and
caveats are documented in `docs/statistical_recovery.md`.

## Machine-learning benchmark finding

Logistic regression was selected as the main ML model using mean validation log
loss. In the direct scenario, its race-blind prediction gap was approximately
zero while its race-aware gap was -2.747 percentage points. In the upstream
scenario, its race-blind gap was already -6.205 points because the shifted
financial features were available. In the mixed scenario, adding race changed
the gap from -6.423 to -10.414 points. These are disparities reproduced from
synthetic mechanisms, not claims about real lenders. Full results are in
`docs/ml_benchmark.md`.

## Sensitivity and Monte Carlo finding

Across 50 replications at 100,000 rows, Model 2 recovered the configured
moderate direct effect of -0.25 at a mean of -0.249 log odds. Moderate upstream
inequality generated a mean raw gap of -6.93 percentage points while the Model
2 adjusted contrast remained approximately zero. The fair-baseline Model 2
false-positive rate was 2%, with 98% interval coverage. The complete design,
thresholds, signatures, numerical policy, and limits are documented in
`docs/sensitivity_analysis.md`.

## Next

- Keep direct-belief and upstream-opportunity mechanisms separate unless a
  later prompt explicitly defines a combined experiment.
- Decide whether Prompt 17 should add repeated-seed uncertainty, an alternative
  upstream mechanism, or capstone synthesis before extending the DGP.
- Consolidate the Version 1 synthetic findings into capstone-ready narrative
  and tables.
- Preserve the distinction between label prediction, true-probability recovery,
  disparity reproduction, and normative fairness.
- Continue deferring HMDA work until the planned synthetic workflow is complete.
