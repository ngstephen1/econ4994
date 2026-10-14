# Professor Model — Step 7B Group-Aware vs Group-Blind OLS

Stage record: results, test counts, and later-work deferrals below describe
this step. See [current status](PROJECT_STATUS.md) and the
[documentation index](README.md) for the latest state across all three families.

## Objective and matched research design

How does excluding binary group B from the bank's risk regression change
prediction accuracy, lending, group approval differences, and true expected
profit? This step fits **one additional OLS model**, not a new population or
discrimination mechanism. The original group-aware model and true-risk oracle
remain frozen, including their cutoffs and decisions.

The same 100,000 saved applicants, raw PN measurements, and authoritative split
are reused: **80,000 training and 20,000 evaluation**. No split is redrawn.
The training-ID fingerprint and PN mean agree with the original model; hashes
verify the unchanged underlying PN file, including negative targets. No
correction for PN's 0.8 scale, feature engineering, tuning, or outcome-selected
refitting is performed.

## Model specification and information boundary

The original equation remains:

```text
group_aware_raw = 0.0030197849371854635
               - 1.0913735436754443e-08 * Y
               - 0.0023144630522321961 * D
               - 0.00040562204042523186 * C
               + 0.00071052614427191425 * B
```

The new `professor_bank_ols_group_blind_v1` fits raw PN with unregularized
`statsmodels.OLS(...).fit(method="pinv")`, using exactly `[1, Y, D, C]`:

```text
bank_PD_group_blind_raw = 0.003128176268420078
                       - 1.0915403719340988e-08 * Y
                       - 0.0023182617012250323 * D
                       - 0.0004049883329481245 * C

bank_PD_group_blind_bounded = clip(bank_PD_group_blind_raw, 0, 1)
```

Y remains annual synthetic dollars in levels, D a ratio, and C on its original
continuous scale. The design has full rank 4 and condition number
650,874.35, reflecting the unequal predictor units. No sign constraints or
rescaling are imposed. B, IDs, split labels, PN, hidden factors, and true risk
are not predictors. B is retained only for later auditing in this new fit.

Raw predictions and out-of-range flags are separate from bounded predictions.
There are **883 negative evaluation predictions (4.415%)**, none above one,
and 883 exact clipped-zero scores. Clipping is not calibration.

Bank loaders read observables, PN, saved split/predictions, contracts, and the
saved budget. Private files can be byte-hashed for preservation but are not
numerically read at this stage. The new coefficient artifact, all predictions,
expected profits, candidate cutoffs, and decisions are saved and checksummed
**before** evaluator truth and oracle decisions are numerically loaded.
The evaluator loader verifies the freeze, source contents, configuration,
original evidence, and every frozen bank artifact. It cannot reselect a bank
cutoff. Decision artifacts contain neither B nor hidden truth.

## Held-out prediction results

All errors below use raw probability/measurement units, not percentages.
Each row uses the identical 20,000 evaluation IDs. Bounded predictions are
included as an explicit accounting sensitivity, not a separately fitted model.

| Policy | Prediction | PN MAE | PN RMSE | PN R² |
|---|---|---:|---:|---:|
| Group-aware | Raw | 0.0793871070 | 0.0996671572 | 0.0000006933 |
| Group-blind | Raw | 0.0793835782 | 0.0996639250 | 0.0000655515 |
| Group-aware | Bounded | 0.0793855101 | 0.0996669100 | 0.0000056534 |
| Group-blind | Bounded | 0.0793829388 | 0.0996643655 | 0.0000567129 |

Both models have essentially zero explanatory power for individual noisy PN
measurements. Slightly improved PN errors here are not a general ranking.

| Policy | Prediction | Hidden true-PD MAE | RMSE | Mean error (estimate − truth) | Correlation |
|---|---|---:|---:|---:|---:|
| Group-aware | Raw | 0.0005264084 | 0.0007221061 | +0.0000495850 | 0.560342 |
| Group-blind | Raw | 0.0004910068 | 0.0006765862 | +0.0000488698 | 0.589987 |
| Group-aware | Bounded | 0.0004903232 | 0.0006452314 | +0.0000856701 | 0.581123 |
| Group-blind | Bounded | 0.0004559405 | 0.0005959274 | +0.0000839361 | 0.622077 |

For example, blind bounded MAE 0.0004559405 is **0.04559405 percentage
points**, not 0.04559405% relative error. Hidden truth is evaluation-only.

The risk-group CSV reports mean truth, mean prediction, MAE, RMSE, and signed
error for both groups under both prediction versions. In raw units:

| Policy | B | n | Mean true PD | Mean prediction | MAE | Mean error |
|---|---:|---:|---:|---:|---:|---:|
| Group-aware | 0 | 16,970 | 0.001432238 | 0.001374953 | 0.000477965 | −0.000057285 |
| Group-aware | 1 | 3,030 | 0.001415046 | 0.002063172 | 0.000797725 | +0.000648126 |
| Group-blind | 0 | 16,970 | 0.001432238 | 0.001481881 | 0.000490415 | +0.000049644 |
| Group-blind | 1 | 3,030 | 0.001415046 | 0.001459580 | 0.000494322 | +0.000044535 |

Paired raw predictions change by +0.0001069286 on average for B=0 and
−0.0006035915 for B=1. Overall mean absolute paired change is 0.0001821731.
Bounded changes and their ranges are saved separately. B removal changes
all fitted coefficients, not just a single fixed-coefficient contribution.

## Identical economics and common-cutoff search

Saved contracts retain T=30, theta=0.10, r=0.05 **per period**,
ell=theta·Y, principal L=30·ell, and scheduled payment ell·(1+r)^t.
No costs, recovery, discounting, or realized defaults are added. With p equal
to the new bounded prediction:

```text
expected_receipts_bank = sum[t=1..30] (1-p)^t * ell * (1+r)^t
expected_profit_bank = expected_receipts_bank - L
approved = 1{p <= tau}
```

The budget is read from the original bank selection JSON, not rounded or
recomputed: **$1,826,087,220.7466326**. The existing exact binary-float prefix
algorithm is reused through a local observable-only schema adapter; no old
feature allowlists or scientific modules are changed. Exact ties are
indivisible; no applicants are skipped and no requests are partially funded.
Reject-all is a separate null cutoff, not tau=0.

The new optimum is **tau=0.0009481206961177917**. All 19,119 candidates
including reject-all are evaluated, with 2,921 feasible. All 883 clipped-zero
applicants are funded together. The next prefix requires $1,826,107,830.54,
above the unchanged budget, leaving $309,176.56 unused. Independent
`fractions.Fraction` enumeration reproduced the exact cutoff and objective.

## Three frozen-policy portfolios — evaluator only

| Policy | Cutoff | Approved | Approval rate | Funded principal | Utilization | Perceived profit | True expected profit |
|---|---:|---:|---:|---:|---:|---:|---:|
| Group-aware | 0.0009333057709409261 | 3,900 | 19.50% | $1,825,953,398.73 | 99.992672% | $2,393,797,751.16 | $2,366,074,476.85 |
| Group-blind | 0.0009481206961177917 | 3,802 | 19.01% | $1,825,778,044.19 | 99.983069% | $2,392,136,130.18 | $2,366,599,763.61 |
| True-risk oracle | 0.0008343666409349597 | 3,406 | 17.03% | $1,825,849,955.62 | 99.987007% | Not a bank estimate | $2,370,206,703.87 |

Mean true PD among approved applicants is 0.07939238% for group-aware,
0.07725445% for group-blind, and 0.06661497% for oracle. Blind true expected
profit is **$525,286.77 higher** than aware profit in this population.
Signed oracle-minus-policy true-profit differences are **+$4,132,227.02**
(aware) and **+$3,606,940.25** (blind), respectively 0.1743403650% and
0.1521782993% of oracle profit. These are expected, not realized, values.

Different score orderings generate different feasible cutoff families.
Nonnegative oracle-minus-bank differences are observed here, **not guaranteed**
in general. This is not unrestricted optimal regret or a universal
value-of-information bound.

## Group lending differences

Denominators are all evaluation applicants within each B group; funded
principal per applicant includes zero for rejections.

| Policy | B | Approved | Approval rate | Funded principal per applicant |
|---|---:|---:|---:|---:|
| Group-aware | 0 | 3,679 | 21.679434% | $98,344.57 |
| Group-aware | 1 | 221 | 7.293729% | $51,830.39 |
| Group-blind | 0 | 3,222 | 18.986447% | $90,627.90 |
| Group-blind | 1 | 580 | 19.141914% | $94,990.94 |
| Oracle | 0 | 2,871 | 16.918091% | $90,368.16 |
| Oracle | 1 | 535 | 17.656766% | $96,469.38 |

B=1 minus B=0 approval gaps are −14.38570492 pp (aware), +0.15546752 pp
(blind), and +0.73867493 pp (oracle). Blind minus aware gap change is
**+14.54117244 pp**: B=1 approval rises **11.84818482 pp**, while B=0
approval falls **2.69298763 pp**. The funded-principal gap per applicant
changes from −$46,514.18 to +$4,363.04, a **+$50,877.22** change.

## Who gains or loses funding?

| Allocation group | Applicants | Mean true PD | Mean potential true profit |
|---|---:|---:|---:|
| Approved by both | 3,443 | 0.07616962% | $638,367.77 |
| Group-aware only | 457 | 0.10367243% | $367,996.17 |
| Group-blind only | 359 | 0.08765854% | $469,915.14 |
| Rejected by both | 15,741 | 0.15997499% | $210,214.69 |

Jaccard funded-set overlap is **80.840573%**. All 457 losses are in B=0;
all 359 gains are in B=1. The overlap CSV includes all four categories for
overall/B=0/B=1, with null means for empty cells, not fabricated zero risks.
Potential profits for rejected applicants are not profits earned by a policy.
No changed rejection is automatically labeled unfair.

## Interpretation and limits

The original +0.0007105261 B slope contributes **+0.07105261 percentage
points** to raw estimated default risk at fixed other covariates. The true
risk DGP contains no B term. This finite-sample coefficient can arise when
fitting extremely noisy PN in one synthetic realization; it is not a measured
discriminatory preference. Including B can change risk ordering and hence
funding under scarcity. This comparison also re-estimates other slopes,
changes clipped-zero ties, and reoptimizes the cutoff. It does **not** isolate
the entire effect of that coefficient or identify a causal discrimination
effect. Coefficient-level counterfactuals and repeated-seed stability are
future separately specified experiments, not implemented here.

Removing B does not guarantee equal approval rates, calibrated risk, or higher
profit. Other observed variables and request sizes still affect decisions.
Large nominal profit levels reflect the stylized growing payment schedule,
low PD, and absence of costs/discounting, not empirical mortgage profitability.
No flexible ML, new DGP, discrimination treatment, fairness constraint,
Monte Carlo, pricing optimization, or new policy type is added.

## Files and reproduction

All paths below are relative to `/Users/macbook/Hack/econ4994/`.

| Location | New files / purpose |
|---|---|
| `configs/professor_model/` | `group_blind_comparison.yaml`: fixed specification and authoritative split/budget references. |
| `src/fair_lending/professor_model/` | `group_blind.py`, `group_blind_io.py`: reusable fit, exact-search adapter, evaluator comparisons, preservation and freeze lifecycle. |
| `experiments/` | `run_professor_group_blind_comparison.py`: orchestration only. |
| `tests/` | `test_professor_group_blind.py`: deterministic unit/integration tests using isolated toy files. |
| `data/synthetic/professor_model/baseline/` | `bank_risk_estimates_group_blind.csv`, `.parquet`: one row per original applicant, including original split, model ID, raw/bounded scores and out-of-bounds flag. |
| `data/processed/professor_model/` | `bank_expected_profit_group_blind.csv`, `.parquet`; `lending_decisions_group_blind.csv`, `.parquet`: one row per original evaluation applicant, bank-only. `group_blind_comparison_metadata.json`: evaluator provenance. |
| `results/tables/professor_model/` | `group_blind_regression_coefficients.csv`, `group_blind_threshold_candidates.csv`; `group_aware_vs_blind_risk.csv`, `group_aware_vs_blind_risk_group_audit.csv`, `group_aware_vs_blind_prediction_changes.csv`, `group_aware_vs_blind_portfolio.csv`, `group_aware_vs_blind_group_audit.csv`, `group_aware_vs_blind_allocation_overlap.csv`. |
| `results/metrics/professor_model/` | `group_blind_regression_model.json`, `group_blind_lending_selection.json`, `group_blind_freeze_checkpoint.json`, `group_blind_comparison_validation.json`. |
| `results/figures/professor_model/` | `group_aware_vs_blind_group_approval.png`, `group_aware_vs_blind_true_profit.png`: rendered and visually inspected for labels, units, and clipping. |

Model artifacts contain transparent coefficients, allowlists, units, solver,
training IDs/target fingerprint, and separate bounding convention. Metadata
records full configuration, actual source hashes, dependency versions, Git
revision/dirty state, zero new random draws, and all original/output checksums.
Generated data and results retain the existing Git-ignore policy.

```bash
python3 experiments/run_professor_group_blind_comparison.py
# Replaces only Step 7B outputs, never prior bank/oracle outputs:
python3 experiments/run_professor_group_blind_comparison.py --overwrite
python3 -m pytest -q tests/test_professor_group_blind.py
python3 -m pytest -q tests/test_professor_*.py
python3 -m pytest -q
```

## Validation

The 35 new deterministic tests cover feature allowlists/units, raw targets,
training/evaluation isolation, fixed-coefficient invariance to B and private
fields, prediction replay, raw/bounded separation, exact accounting, tied
cutoffs/reject-all/full funding, independent rational enumeration, risk/group
metric arithmetic, signed profit comparisons, overlap, original checksums,
overwrite protection, tamper detection, and freeze-before-truth gating.
All **285 professor-model tests passed without warnings**.

The complete repository suite finished with **565 passed, 4 failed, and 32
existing deprecation warnings** in 89.29 seconds. There are no new Step 7B
failures. The same four failures remain in
`tests/test_systemic_corrected_checkpoint.py`:

- `test_corrected_replication_has_exactly_72_unique_complete_cells`
- `test_replication13_provenance_is_internally_consistent`
- `test_legacy_comparison_is_diagnostic_and_never_authorizes_reuse`
- `test_interim_summary_uses_five_corrected_records_only`

They fail to locate saved V2 records through current-HEAD-derived identities
(`7a7e590`, versus saved `8a9f71f`); unrelated recovery code/results were not
changed or bypassed. Warnings are existing scikit-learn/SciPy L-BFGS-B
`disp`/`iprint` deprecations (26 ML benchmark and six sensitivity warnings).

All **109 pre-step data/configuration/source/result snapshot files** and
**108 registered prior artifacts** remain byte-identical, including the
original group-aware model, bank decisions, oracle decisions, and capital
budget. All three new CSV/Parquet pairs agree exactly using round-trip float
parsing. Saved model replay and all new artifact checksums passed independent
verification. Existing V1/V2 code, scientific settings, portfolios, and
Monte Carlo recovery remain unchanged. No commit or push is performed.
