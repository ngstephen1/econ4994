# Project History — Stage Records

This is the preserved status narrative recorded through professor-model Step 8B.
For the concise current state, see [Project Status](PROJECT_STATUS.md) and the
[documentation index](README.md). Test counts and statements about work “not
started” below apply to the stages where they were recorded, not necessarily
the present repository. Scientific results and their caveats are retained.

---

# Project Status

## Current phase

Professor's lending model — a separate research family:

- Step 1: **Complete** — repository audit and research specification.
- Step 2: **Complete** — 100,000 applicant characteristics and noisy credit
  signals generated with seed 4994, information-separated CSV/Parquet exports,
  and successful structural/reproducibility validation.
- Step 3: **Complete** — deterministic evaluator-only true-risk components
  and true_PD calculated for the existing 100,000 applicants; strict bounds,
  exact equations, preserved input checksums, and three diagnostic plots validated.
- Step 4: **Complete** — raw PN and private eta generated from reserved child
  [5], with exact information-separated schemas, unchanged original inputs,
  shared preview IDs, and CSV/Parquet agreement validated.
- Step 5: **Complete** — unregularized raw-PN OLS with intercept and exactly
  Y, D, C, B fitted on 80,000 training applicants; 20,000 held out using
  reserved child [6]. Model/predictions frozen before hidden-truth evaluation.
- Step 6A: **Complete** — 30-period income-linked contracts and survival-weighted
  bank/true expected receipts and profits calculated for the 20,000 saved
  evaluation applicants. Per-period r=0.05 and conditional per-period default
  are explicit working assumptions; bank/evaluator artifacts remain separate.
- Step 6B: **Complete** — one common bounded-PD cutoff maximizes saved bank
  expected profit under 40% of evaluation requested principal. Exact ties are
  indivisible; decisions/candidates are saved before evaluator truth loads.
- Step 7A: **Complete** — evaluator-only true-risk cutoff benchmark using the
  identical saved capital budget, contracts, and evaluation applicants. The
  original bank decisions and cutoff remain frozen; only the oracle is optimized.
- Step 7B: **Complete** — one group-blind raw-PN OLS fit on the same saved
  80,000 training IDs, with 20,000 held out, identical contracts/capital, and
  a separate exact common-cutoff policy frozen before truth evaluation.
  Original group-aware bank and oracle models/decisions remain unchanged.
- Step 8A: **Complete** — five-replication noise-only robustness pilot on the
  frozen population/split/contracts/budget. Independent seed-499408 eta
  children [0]–[4], ten reused-specification OLS fits, and ten exact common-cutoff
  policies validated; canonical PN, fits, allocations, and oracle remain frozen.
- Step 8B: **Complete** — exactly 200 validated noise-only replications on
  the same frozen population/split/contracts/capital/oracle: five exact pilot
  reuses and 195 new replications. All 400 policy cells passed arithmetic,
  exact-cutoff, indivisible-tie, budget, and provenance checks, with no failed
  attempts. No new raw applicant datasets were persisted.
- Professor-model flexible-model comparisons: **Not started**.

The professor-model focused suite passed **344 tests** with no warnings,
including 24 new Step 8B tests. The current full suite
finished with **624 passed, 4 failed, and 32 deprecation warnings** in 138.93s
(executed alongside part of the Monte Carlo run).
There are no new professor-model failures. The four
failures are existing systemic-checkpoint artifact lookups: saved successes
reference Git revision `8a9f71f`, while the current loader uses HEAD `7a7e590`
as part of its run ID. Read-only checks confirmed identical scientific source
and dependency fingerprints and valid saved identities for all five corrected
success records. Recovery code/results were not changed to bypass these tests.
The professor's contract, expected-profit accounting, and restricted common
cutoff policy are implemented, with a separate evaluator true-risk cutoff
benchmark. Neither policy is unrestricted portfolio optimization.
See [population generation](professor_population_generation.md),
[true default risk](professor_true_default_risk.md), and
[the professor specification](professor_whiteboard_model.md).
The raw target/noise implementation is documented in
[noisy risk signal](professor_noisy_risk_signal.md).
The fitted equation, held-out results, and exact export paths are documented
in [bank regression](professor_bank_regression.md).
The contract, accounting assumptions, and actual economic results are in
[loan economics](professor_loan_economics.md).
The common-cutoff search, funded portfolio, and post-decision group audit are
documented in [lending threshold](professor_lending_threshold.md).
The matched evaluator-only benchmark is documented in
[oracle comparison](professor_oracle_comparison.md).
The matched group-aware/group-blind fit, risk audit, cutoff, and allocation
comparison are documented in
[group-blind comparison](professor_group_blind_comparison.md).
The fixed-population five-noise-draw pilot and its limits are documented in
[noise robustness pilot](professor_noise_robustness_pilot.md).
The completed 200-replication study, descriptive noise distributions, tie
diagnostics, immutable artifacts, and interpretation limits are documented in
[full noise robustness](professor_noise_robustness_full.md).

Step 4 retained 49,453 negative PN measurements (49.453%). Noise SD is 182.65
times scaled-truth SD; the observed PN/true_PD correlation is 0.003684.
All 31 registered Step 2/3 data, configuration, and source artifacts remained
unchanged during Step 4. That step did not add regression or lending.

Step 5 retained all existing inputs and configurations byte-for-byte: 45 files
in the independent pre-step snapshot and 47 original artifacts registered by
the prior manifests passed preservation checks. Exact deterministic refitting
reproduced coefficients, split labels, and all 100,000 predictions.
Held-out PN R² is 0.0000006933; OLS has only marginally lower RMSE and slightly
worse MAE than the training-mean PN reference. Raw hidden-risk MAE is
0.0005264084, versus 0.0004903232 for the separately bounded predictions.
Evaluation includes 923 negative raw predictions (4.615%); all are preserved,
with 923 bounded-zero ties. Bounding is not calibration. No new threshold,
default outcome, repayment history, lending, profit, flexible ML, or Monte Carlo
implementation was added for this research family.

Step 6A preserves all 61 pre-step data/configuration/source/result snapshot
files and all 64 registered originals. The bank regression was not refitted,
and the existing 80,000/20,000 split and raw/bounded predictions are unchanged.
Total evaluation requested principal is $4,565,218,051.87. Mean expected profit
is $291,416.11 under bank beliefs and $292,188.17 under truth; profit MAE is
$4,400.06. All 20,000 requests have positive expected profit under both views,
with zero profit-sign disagreement. This follows from the stylized growing
payment schedule and low PD, not parameter retuning. The break-even default
probability is 4.76190476% per period; the maximum absolute accounting residual
at break-even is $1.862645149e-9. These are potential per-request values,
not approvals or portfolio results. No threshold, budget, allocation, new
risk model, or discrimination treatment was introduced.

Step 6B uses an unrounded budget of $1,826,087,220.7466326 and selects bounded
PD cutoff 0.0009333057709409261. It approves 3,900/20,000 applicants (19.5%),
funding $1,825,953,398.73 at 99.99267165% budget utilization. Unused capital is
$133,822.01: the constraint is not exactly exhausted, but the next indivisible
risk-group expansion would exceed it. All 923 clipped-zero scores are funded
together; the selected boundary itself is unique. The bank estimates portfolio
profit at $2,393,797,751.16; post-freeze true value is $2,366,074,476.85, an
overestimate of $27,723,274.31, not an oracle-regret measure. B=1 minus B=0
approval disparity is -14.38570492 percentage points; this is not proof of
intentional discrimination. All 19,079 candidate cutoffs including reject-all
were evaluated, with 2,979 feasible. Independent rational-arithmetic enumeration
reproduced the optimum. All 76 pre-step snapshot files and 80 registered
original artifacts remain unchanged; no refitting or new mechanisms were added.

Step 7A selects true-risk cutoff 0.0008343666409349597 under the unchanged
$1,826,087,220.7466326 budget. The oracle approves 3,406 applicants versus the
bank's frozen 3,900, with true expected profits of $2,370,206,703.87 versus
$2,366,074,476.85. The signed oracle-minus-bank comparison is +$4,132,227.02,
or 0.1743403650% of oracle profit. Because their scores differ, the cutoff
portfolios are different feasible families: nonnegative regret is observed
here, not a universal guarantee or unrestricted optimality bound. The overlap
is 2,516 approvals, Jaccard 52.5261%. B=1 minus B=0 approval gaps are -14.3857
points for bank and +0.7387 for oracle; bank-minus-oracle gap difference is
-15.1244 points, descriptive rather than an identified discrimination effect.
All 89 pre-step snapshot files and 94 registered originals remain unchanged.
No bank refit/reoptimization, new risk model, group-blind fit, ML comparison,
intervention, pricing change, unrestricted allocation, or Monte Carlo was added.

Step 7B selects group-blind cutoff 0.0009481206961177917 under the same exact
$1,826,087,220.7466326 budget. It approves 3,802/20,000 applicants (19.01%)
versus the frozen group-aware bank's 3,900 (19.50%) and oracle's 3,406
(17.03%). B=0 and B=1 approval rates become 18.986447% and 19.141914%:
the B=1 minus B=0 gap moves from -14.385705 to +0.155468 percentage points,
a +14.541172-point change. There are 3,443 approvals in common, 457
group-aware-only approvals (all B=0), and 359 group-blind-only approvals
(all B=1); funded-set Jaccard overlap is 80.840573%.
Group-blind true expected profit is $2,366,599,763.61, up $525,286.77;
the signed oracle-minus-blind difference is +$3,606,940.25. Raw hidden-risk
MAE falls from 0.0005264084 to 0.0004910068, but both fits have near-zero
held-out PN R-squared. These are single-population descriptive comparisons,
not an isolated causal B-coefficient effect or unrestricted optimal regret.
All 109 pre-step snapshot files and 108 registered prior artifacts remain
byte-identical. No regeneration, old-model refit, bank/oracle reoptimization,
flexible ML, new mechanism, policy type, or Monte Carlo was added.

Step 8A completed exactly five independent eta replications, preserving all
262 pre-step scientific source/configuration/test and professor-data/result
files byte-for-byte. The group-aware B coefficient is negative in four draws
and positive in one. Aware approval gaps range from -1.331062 to +19.220990
percentage points (mean +11.564049); blind gaps range from +0.058694 to
+0.975591 points (mean +0.601002). Excluding B changes true profit by
+$20,498.20 to +$770,409.00 across these draws (mean +$308,239.49).
All ten exact cutoffs passed independent rational-arithmetic enumeration and
budget validation. Every zero-score block was feasible, including the
replication-3 aware block that uses 99.0207% of capital. Oversized-block
reject-all behavior is covered by deterministic tests. The saved oracle
remains at 3,406 approvals and $2,370,206,703.87 true expected profit.
Mean replication runtime was 2.655 seconds; successful-record resume and
summary-only reporting launched zero new replications. Results are immutable
compact JSON plus four CSVs and two figures, not five new raw datasets.
This is conditional measurement-noise sensitivity on one population, not
formal robustness/significance evidence or a general fairness conclusion.
Step 8B had not been launched at pilot completion; V1/V2 scientific and
recovery code remains unchanged.

Step 8B completed 200/200 independent eta replications (master seed 499408),
with five validated pilot reuses and 195 new draws. All 285 original
scientific/data/result files, including pilot outputs, remain byte-identical.
The initial invocation took 641.545 seconds; resumed reporting reused all
200 records without new fits. The aware B coefficient has mean 0.000063278,
SD 0.001063127, and MCSE 0.000075174, with 105 positive and 95 negative draws.
Aware approval gaps average +8.839521 pp (SD 36.206101), versus +0.365663 pp
(SD 0.599209) blind; 131/200 draws reach the original 14.385705 pp absolute
aware gap. Blind absolute gaps are smaller in 183/200 draws (91.5%), not all.
Blind true expected profit is higher in 141/200 draws, lower in 47, and equal
in 12. Median blind-minus-aware profit is +$651,806.68; the +$24,616,229.75
mean is dominated by two aware-reject-all/blind-lending cases above $2.36B,
which remain in the primary summaries. Reject-all occurs in 13 aware and
11 blind policy cells due to oversized indivisible clipped-zero blocks.
The evaluator-only B signal coefficient is 0.00000384199494513, with maximum
OLS decomposition residual 2.25514e-17. These are conditional noise-only
synthetic outcomes, not real-world discrimination evidence, a guarantee of
group-blind fairness/profit, or unrestricted oracle-optimality comparisons.

The frozen true-risk equation produces mean PD 0.142208%, median 0.127824%,
and no applicant above 1%. The marginal DTI correlation is approximately zero
and slightly negative, while its fixed-feature derivative is positive. These
diagnostics are recorded without changing the coefficients or adding floors.
Step 2 data/configuration/metadata and all V1/V2 scientific work are preserved.

### Preserved Version 2 status

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
