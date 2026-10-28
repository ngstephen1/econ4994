# Professor Model — Step 9A Revised Risk Scenarios

## Purpose and stopping point

Reconcile the professor's two revised diagram branches using the original
frozen Step 5/7B OLS fits, plus an analytical control that isolates the manual
adjustment. This step produces **risk estimates and diagnostics only**.
No population/noise generation, regression fitting, profit experiment,
cutoff search, loan selection, or Monte Carlo was performed.

The revised equations are the researcher's supplied transcription of the
professor's diagram. The original diagram was not independently inspected.
Earlier experiments and the completed Step 8B noise study remain unchanged.

## Original versus revised comparisons

Step 7B compared a group-aware OLS with a newly fitted group-blind OLS, each
followed by its own lending cutoff. Step 8B repeated that comparison across
200 noise draws. Those comparisons changed multiple fitted coefficients.

Step 9A uses **the canonical single-noise-draw fits**, not a selected Monte
Carlo draw or average coefficient. The revised professor diagram specifies:

```text
DIRECT:
PD_direct_raw = beta0_blind + betaY_blind*Y + betaD_blind*D
              + betaC_blind*C - 0.1*B

NO DIRECT:
PD_no_direct_raw = beta0_aware + betaY_aware*Y + betaD_aware*D
                 + betaC_aware*C + betaB_aware*B

ANALYTICAL CONTROL (not a third professor-written branch):
PD_blind_reference_raw = beta0_blind + betaY_blind*Y
                       + betaD_blind*D + betaC_blind*C
```

| Scenario ID | Base | Manual adjustment | Interpretation |
|---|---|---|---|
| `professor_no_direct_group_aware` | Frozen group-aware OLS | 0 | Professor's no-manual-adjustment branch, still includes fitted B |
| `professor_direct_minus_0p1_B` | Frozen group-blind OLS | **−0.1 × B** | Professor's direct branch, applied after fitting |
| `group_blind_no_adjustment_reference` | Frozen group-blind OLS | 0 | Matched control for the manual adjustment |

Direct versus control changes only the specified adjustment. Direct versus
no-direct also changes the fitted regression specification. These comparisons
do not measure identical effects. Neither is an empirical causal estimate.

## Terminology, sign, and units

The revised diagram calls its fitted quantities PN. The existing schema
continues to reserve **PN for the noisy regression target**:
`PN = 0.8*true_PD + eta`. It is neither overwritten nor redefined here.
New fitted scenario fields use `estimated_PD_*`.

The adjustment is exactly **−0.1 raw probability units = −10 percentage
points for B=1**. It is not a log-odds shift, 0.1%, relative percentage change,
or estimated coefficient. It lowers perceived default risk and therefore
**favors B=1 in the risk assessment**, rather than imposing an adverse risk
penalty on that group. Actual lending effects remain uncomputed.

“No direct discrimination” is the professor's branch label for **no manually
imposed adjustment**. It does not imply that B is excluded, that the fitted
B coefficient is zero, or that the policy is normatively fair. This is an
explicitly specified synthetic mechanism, not a real-world discrimination
estimate. Conditional per-period default retains its existing meaning.

## Frozen fitted coefficients and verified provenance

| Term | Group-aware: `professor_bank_ols_v1` | Group-blind: `professor_bank_ols_group_blind_v1` |
|---|---:|---:|
| Intercept | 0.0030197849371854635 | 0.003128176268420078 |
| Y (annual dollars, levels) | −1.0913735436754443e−08 | −1.0915403719340988e−08 |
| D (DTI ratio) | −0.002314463052232196 | −0.0023182617012250323 |
| C (original signal) | −0.00040562204042523186 | −0.0004049883329481245 |
| B (binary) | +0.0007105261442719142 | Excluded from base fit |

Canonical JSON artifacts under `results/metrics/professor_model/`:

- `bank_regression_model.json`, SHA-256
  `99c948d71f186f5def7f350ca4a2f0b79ff8110fe95e5993937743dd15bd8f0e`.
- `group_blind_regression_model.json`, SHA-256
  `e065880c4099502efd03c1b86eb496970fe7faad99f4c8a22f88a6075ce50df7`.

Their configuration fingerprints are respectively
`e163db2a69dca2b5d448d9f98b1f0e40602504e88e7a5ae45ab5211296e31d48` and
`99860a1efa103f3406a0179d315403929fa8020613f00d9da451727704c7d8d5`.
Coefficient CSVs agree exactly with those JSON values. Recorded configuration,
source-content hashes, freeze checkpoints, raw/bounded prediction pairs,
CSV/Parquet values, and cohort identities were checked across **133 files**.
Source verification uses originally recorded paths, not an enlarged directory
glob or current Git HEAD as a substitute for scientific provenance.

Both models have the same training-ID fingerprint
`244ee8e6a074836e654451ea423de0dbe317aaa7ec721b793eb17bbb77d44c95`
and evaluation-ID fingerprint
`454e526e12d96a9a779a850c537fb6f4b181775468e2fa508d61f7c878c2ef72`.
Replaying their coefficients on evaluation observables reproduces the saved
raw predictions exactly, without fitting. Scenario bases use those saved raw
values as authoritative, not rounded CSV displays.

## Sample and information-access boundary

Each scenario contains the **same 20,000 evaluation applicants**: 16,970 B=0
and 3,030 B=1. The 80,000 training applicants are not rescored for the new
scenario output; their saved IDs only validate the model/split provenance.
All three scenarios yield 60,000 unique scenario/applicant rows.

The numerical loader reads only original bank observables (ID, Y, D, B, C),
saved split labels, coefficient/prediction artifacts, original contracts,
and the saved capital value. It does **not** parse PN, epsilon, omega, eta,
true_PD, profits, outcomes, or private applicant tables. Hashing private file
bytes for preservation is separate from using their contents as predictors.
Even the evaluator-only preview contains no hidden truth or PN.

## Output schema and exact identities

One scenario row contains:

| Field | Meaning |
|---|---|
| `applicant_id`, `scenario_id` | Unique applicant/scenario key |
| `base_model_id`, `B` | Frozen source fit and binary treatment/audit group |
| `estimated_PD_base_raw` | Original frozen OLS raw prediction |
| `group_adjustment` | −0.1 for direct B=1; zero otherwise |
| `estimated_PD_adjusted_raw` | Base raw plus adjustment, including negatives |
| `estimated_PD_bounded` | `clip(adjusted_raw, 0, 1)` |
| `clipped_below_zero`, `clipped_above_one` | Strict raw <0 / >1 flags |

All applicant-level probability fields retain **raw units**. Summary columns
ending `_pp` are 100 times their raw values; `_share` fields are fractions.
Exact boundary zeros/ones are counted separately from actual clipping of
values outside [0,1]. No values are discarded. Clipping is not calibration.
New B audit columns use int64 for portable CSV/Parquet agreement, preserving
all original B values and leaving the original int8 Parquet bytes unchanged.

## Observed risk distributions

Displayed risk means use percentage-point units (100 × raw probability).
The CSVs also report medians, raw minima/maxima, adjustment means, both
clipping counts, exact bounded-zero/one counts, and shares, without rounding.

| Scenario | B | n | Mean adjusted raw PD (pp) | Mean bounded PD (pp) | Negative raw / bounded zero | Zero share |
|---|---:|---:|---:|---:|---:|---:|
| No-direct, group-aware | 0 | 16,970 | 0.137495 | 0.141377 | 851 / 851 | 5.0147% |
| No-direct, group-aware | 1 | 3,030 | 0.206317 | 0.208395 | 72 / 72 | 2.3762% |
| Direct, blind minus 0.1B | 0 | 16,970 | 0.148188 | 0.151574 | 734 / 734 | 4.3253% |
| Direct, blind minus 0.1B | 1 | 3,030 | **−9.854042** | **0** | **3,030 / 3,030** | **100%** |
| Blind analytical control | 0 | 16,970 | 0.148188 | 0.151574 | 734 / 734 | 4.3253% |
| Blind analytical control | 1 | 3,030 | 0.145958 | 0.150140 | 149 / 149 | 4.9175% |

Exactly **3,030 B=1 rows** receive −0.1. All their adjusted raw predictions
are negative and all become bounded zero. The mean adjustment for that group
is −10 pp; its overall-scenario mean is −1.515 pp because B=1 comprises
15.15% of the evaluation cohort. No scenario has a raw value above one or a
bounded-one value.

Across all applicants, negative/raw-bounded-zero counts are 923 (4.615%) for
no-direct, 3,764 (18.82%) for direct, and 883 (4.415%) for the control.
For B=1 the direct raw range is approximately −10.532623 to −9.649604 pp.
Thus clipping erases all within-B=1 score differences in the direct branch.
The large treatment is preserved as specified, not retuned to avoid saturation.

## Zero-score capital diagnostic, not an allocation

The saved budget is exactly **$1,826,087,220.7466326** from
`results/metrics/professor_model/lending_selection.json`.
Principal comes from the original evaluation `loan_contracts.parquet`.

| Scenario | Zero-score applicants | B=0 / B=1 zeros | Requested principal in zero group | Share of budget | Entire zero group feasible? |
|---|---:|---:|---:|---:|---|
| No-direct, aware | 923 | 851 / 72 | $709,224,665.32 | 38.8385% | Yes |
| Direct | 3,764 | 734 / 3,030 | **$1,276,685,167.37** | **69.9137%** | **Yes** |
| Blind reference | 883 | 734 / 149 | $691,901,043.03 | 37.8898% | Yes |

Feasibility compares exact rational values of the saved binary floats,
without cent rounding, numerical budget slack, or a shrunken budget.
Displayed sums use `math.fsum`. No loan is approved or rejected here, and
these principals are **requested**, not capital actually consumed. Feasibility
alone does not establish the later profit-maximizing cutoff or group funding.

## Artifacts and reproduction

All paths are repository-relative:

| Location | Files / purpose |
|---|---|
| `configs/professor_model/revised_scenarios.yaml` | Exact branch definitions, sign/scale, cohorts, and saved capital |
| `src/fair_lending/professor_model/revised_scenarios.py` | Risk-only transformation, identities, summaries, preview, capital diagnostic |
| `src/fair_lending/professor_model/revised_scenarios_io.py` | Frozen provenance/read allowlists, exclusive export, metadata, one figure |
| `experiments/build_professor_revised_scenarios.py` | Orchestration without training/generation/optimization |
| `tests/test_professor_revised_scenarios.py` | Deterministic toy tests, no fitting in the new test fixtures |
| `data/processed/professor_model/revised_scenarios/` | `scenario_risk_estimates.csv` / `.parquet` (60,000 rows); `scenario_risk_preview_EVALUATOR_ONLY.csv` (48 rows, eight unique IDs from each B group across three scenarios); `scenario_risk_metadata.json` |
| `results/tables/professor_model/` | `revised_scenarios_risk_summary.csv` (3 rows), `revised_scenarios_group_summary.csv` (6), `revised_scenarios_zero_tie_diagnostics.csv` (3) |
| `results/metrics/professor_model/revised_scenarios_validation.json` | Identity, preservation, export, and zero-tie checks |
| `results/figures/professor_model/revised_scenarios_raw_bounded_risk.png` | One two-panel boxplot by scenario/group; separate x scales, full min–max whiskers |

The preview uses the first eight sorted evaluation IDs within each B group,
not outcome-selected applicants. The figure was rendered and visually
inspected; all raw negatives remain visible and bounded B=1 direct risk
collapses to zero. Summary headers/units and exact joins follow the scientific
CSV-validation guidance; no workbook or additional framework was introduced.

```bash
python3 experiments/build_professor_revised_scenarios.py
python3 -m pytest -q tests/test_professor_revised_scenarios.py
python3 -m pytest -q tests/test_professor_*.py
python3 -m pytest -q
```

The runner refuses to overwrite any existing Step 9A output. There is no
overwrite/refit/regenerate switch. Read saved outputs to inspect the completed
run. Generated data/results remain under the existing Git-ignore policy.

## Validation and preservation results

- New Step 9A tests: **35 passed, no warnings** (5.15 seconds).
- Complete professor-model suite: **379 passed, no warnings** (15.41 seconds).
- Full repository suite: **659 passed, 4 failed, 32 existing deprecation
  warnings** (107.54 seconds).

The four failures are unchanged V2 artifact lookups in
`tests/test_systemic_corrected_checkpoint.py`: complete 72-cell coverage,
replication-13 provenance, legacy-comparison isolation, and the five-record
interim summary. The loader constructs filenames using current Git HEAD,
which differs from the revision in the saved corrected records. No V2 code,
test, or recovery artifact was altered or suppressed. The 32 warnings are
the existing scikit-learn/SciPy L-BFGS-B `disp`/`iprint` deprecations, 26 from
the ML benchmark tests and six from sensitivity tests. This is not an
all-green full repository.

Tests cover the exact sign/scale and scenario mapping, both frozen reference
identities, B=0 invariance, exact B=1 subtraction, raw/bounded separation,
boundary versus clipping counts, summary units/denominators, deterministic
row order, complete cells, forbidden columns, corrupt references, source/model
fingerprints, no-fitting/no-optimization hooks, numerical read allowlists,
preview IDs, exact-budget/rounding edges, lossless exports, preservation,
tampered-input rejection, and refusal to overwrite completed outputs.

All **905 pre-step scientific source/configuration/test and professor-model
data/result files** remain byte-identical. All 133 frozen-reference checks
passed, as did coefficient replay and CSV/Parquet equality. No prior
population, true-risk, PN, model, bank/oracle decision, Step 8A/8B result,
V1/V2 implementation, or systemic recovery artifact was overwritten.
Nothing was staged, committed, or pushed.

## Recommendation for Step 9B

Proceed only under a separate lending-comparison request. Use the same saved
contracts, exact capital, common-cutoff/tie rule, and all three frozen Step 9A
scores. Freeze scenario lending decisions before evaluator truth is used.
Report direct versus blind control separately from direct versus aware;
portfolio-level effects can include capital competition and spillovers.
Do not infer approvals/profit from the risk summaries, equate bounded zero
with true zero risk, or label these researcher-defined effects empirical
discrimination. No Step 9B computations were launched.
