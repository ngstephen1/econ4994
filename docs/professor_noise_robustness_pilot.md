# Professor Model — Step 8A Noise-Only Robustness Pilot

Stage record: results, test counts, and later-work deferrals below describe
this step. See [current status](PROJECT_STATUS.md) and the
[documentation index](README.md) for the latest state across all three families.

**Step 8A: five-replication pilot completed; subsequently extended by the
[completed Step 8B study](professor_noise_robustness_full.md).**

## Research question

Are the large group approval differences stable, or can random noise in the
bank's risk measurements produce very different results? This pilot conditions
on **one fixed synthetic population** and varies only the measurement noise.
It does not estimate uncertainty over new applicants, new splits, contracts,
or true-risk worlds. The canonical Step 4 realization and Steps 5–7B policies
are preserved as historical references, not counted among these five draws.

## What stays fixed, and what changes

The 100,000 saved applicant IDs, Y, D, B, C, hidden epsilon/omega, and true_PD
remain unchanged. So do the original **80,000/20,000 train/evaluation split**,
20,000 evaluation contracts, payment schedules, requested principals, true
expected profits, exact capital budget, and saved oracle allocation/value.
No population generation, true-risk calculation, or oracle optimization runs.

Only eta is redrawn; PN and all bank estimates/decisions derived from PN
can change. The simulator computes:

```text
eta_ir ~ Normal(mean=0, SD=0.1)
PN_ir = 0.8 * saved_true_PD_i + eta_ir
```

PN stays raw: no clipping, sigmoid, recentering, division by 0.8, or discarded
negative targets. SD=0.1 is 10 percentage points in probability units. In the
canonical data, noise SD was 182.65 times scaled-truth SD, motivating this
experiment; that ratio does not itself prove a fitted model cannot learn.

## Random streams and provenance

`SeedSequence(499408).spawn(5)` defines exactly five independent PCG64 streams.
Replication indices 0–4 use spawn keys `[0]`, `[1]`, `[2]`, `[3]`, `[4]`.
Draws are attached in ascending lexicographic applicant-ID order. Neither
canonical seed 4994 child [5] nor any population/split stream is advanced.

Before any new draws, an immutable manifest recorded **262 original-file
SHA-256 hashes**: professor datasets/results plus existing scientific source,
configuration, experiment, and test files across the research repository.
This inventory exactly matches the independent pre-edit snapshot. All 262
files remained byte-identical. HMDA files were not read or modified.

The manifest also records actual source-content hashes, configuration,
dependencies/Python, Git revision/dirty state, and experiment fingerprint:

```text
5601ef2a8fbc57675d7d63ab3f313eda405f26af28cf0e6a2658025b30e37fd3
```

Each replication records its stream, eta digest, PN/ID fingerprints, both
coefficient artifacts, prediction/profit/decision/candidate fingerprints,
threshold diagnostics, evaluator metrics, and runtime. The model IDs identify
the same estimator families **within that replication**, not replacements of
the canonical Step 5/7B fits. No new applicant-level datasets are persisted;
the fixed originals plus stream information permit reconstruction.

## Bank models and information boundary

Both fits reuse the existing unregularized `statsmodels.OLS(...).fit(method="pinv")`:

- Group-aware: `PN_r ~ 1 + Y + D + C + B`.
- Group-blind: `PN_r ~ 1 + Y + D + C`.

Y stays in annual dollar levels, D in ratios, and C on its original signal
scale. There are no new predictors, transformations, tuning, or split draws.
Only training IDs and their PN_r labels enter parameter estimation. The bank
receives observables and PN, never eta, epsilon, omega, components, true_PD,
or true profit. The simulator necessarily uses saved truth to construct PN;
that is separate from bank fitting. Evaluation PN does not select settings.

Raw and bounded predictions remain separate. Bounding is the existing
`clip(raw, 0, 1)` accounting convention, **not probability calibration**.
Both bank allocations are frozen to an immutable compact checkpoint before
true-profit/oracle evaluation. Evaluator code verifies those frozen identities
and cannot change the models or cutoffs. Even though raw rows are not saved,
their exact fingerprints and deterministic reconstruction identify the sets.

## Economics, cutoff rule, and independent validation

Contracts remain T=30, theta=0.10, r=0.05 **per period**, with:

```text
ell = 0.10 * Y
L = 30 * ell
payment_t = ell * (1+r)^t
expected_receipts_bank = sum[t=1..30] (1-bank_PD_bounded)^t * payment_t
expected_profit_bank = expected_receipts_bank - L

approved_i(tau) = 1{bank_PD_bounded_i <= tau}
maximize sum_i approved_i * expected_profit_bank_i
subject to sum_i approved_i * L_i <= 1826087220.7466326
```

The unrounded budget is read from the frozen Step 6B selection; it is not
recomputed. Both policies reuse the unchanged exact binary-float prefix
search. Equal stored scores are indivisible; no skipping or partial financing
is permitted. Reject-all is a separate null cutoff, never tau=0. If the
zero-score block is too expensive, it is not split to manufacture lending.

An independent `fractions.Fraction` audit checked **every candidate** for
each of the ten canonical bank policies, not merely a re-call to the same
search. It agreed on exact feasibility, perceived objective, selected cutoff,
and all full-loan decisions. All ten audits passed with **zero budget
violations**. Small deterministic fixtures also exercise oversized zero blocks,
ties, negative-profit expansions, reject-all, and incorrect selections.

Budget residual means exact funded principal minus available capital. The
JSON records the selected residual and maximum residual among feasible
candidates, both nonpositive, plus maximum positive violation (zero). Rounded
display totals are not used to authorize spending.

## Five-replication results

Approval gaps and changes below are **B=1 minus B=0 percentage points**;
change means group-blind gap minus group-aware gap.

| Replication / spawn key | Runtime (s) | Aware B coefficient | Aware gap (pp) | Blind gap (pp) | Gap change (pp) | Aware approved | Blind approved |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 / [0] | 2.691 | −0.000245466210 | +9.974737 | +0.754000 | −9.220737 | 3,436 | 3,410 |
| 1 / [1] | 2.654 | +0.000163089858 | −1.331062 | +0.975591 | +2.306653 | 3,414 | 3,412 |
| 2 / [2] | 2.707 | −0.000352490864 | +18.167957 | +0.651723 | −17.516234 | 3,940 | 3,863 |
| 3 / [3] | 2.500 | −0.000324092906 | +11.787624 | +0.058694 | −11.728930 | 5,597 | 5,594 |
| 4 / [4] | 2.722 | −0.000297224470 | +19.220990 | +0.565004 | −18.655986 | 4,758 | 4,716 |

There is **one positive and four negative B coefficients**, so the sign
changes. No zero coefficients occur. All draws were predeclared; none was
selected or discarded based on its result. The original canonical B slope
was positive and its aware approval gap was negative; this five-draw pilot
does not reproduce that direction consistently.

Runtime measures target creation through fitting, threshold audits, bank
freeze, and evaluation, excluding common input loading, final record publication,
and consolidated reporting. Total measured replication runtime is 13.274 seconds,
mean 2.655 seconds. This is not the multi-hour V2 systemic experiment.

### Group rates and allocation overlap

| Replication | Aware B=0 (%) | Aware B=1 (%) | Blind B=0 (%) | Blind B=1 (%) | Approved by both | Aware only | Blind only | Jaccard (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 15.668827 | 25.643564 | 16.935769 | 17.689769 | 3,195 | 241 | 215 | 87.5103 |
| 1 | 17.271656 | 15.940594 | 16.912198 | 17.887789 | 3,353 | 61 | 59 | 96.5448 |
| 2 | 16.947555 | 35.115512 | 19.216264 | 19.867987 | 3,478 | 462 | 385 | 80.4162 |
| 3 | 26.199175 | 37.986799 | 27.961108 | 28.019802 | 5,295 | 302 | 299 | 89.8066 |
| 4 | 20.878020 | 40.099010 | 23.494402 | 24.059406 | 4,272 | 486 | 444 | 82.1223 |

Denominators remain 16,970 for B=0 and 3,030 for B=1. The replication CSV
also records within-pair changes in each group's approval rate. Removing B
does not force equal group outcomes or preserve the number of funded loans.

### Risk prediction — evaluator only

All errors are in raw probability units, not percent or relative error. PN
RMSE below uses raw predictions. JSON records also retain bounded-PN errors,
raw/bounded means, signed errors, correlations, range counts, and coefficients.

| Replication | Model | PN RMSE | Raw true-PD MAE | Raw true-PD RMSE | Bounded true-PD MAE | Bounded true-PD RMSE |
|---|---|---:|---:|---:|---:|---:|
| 0 | Aware | 0.10021232 | 0.00054278 | 0.00075126 | 0.00049108 | 0.00065309 |
| 0 | Blind | 0.10021157 | 0.00054193 | 0.00074589 | 0.00049092 | 0.00064855 |
| 1 | Aware | 0.09961505 | 0.00064457 | 0.00113585 | 0.00043677 | 0.00051655 |
| 1 | Blind | 0.09961615 | 0.00064302 | 0.00113484 | 0.00043525 | 0.00051366 |
| 2 | Aware | 0.09928033 | 0.00043495 | 0.00061933 | 0.00040804 | 0.00055863 |
| 2 | Blind | 0.09927984 | 0.00042151 | 0.00060694 | 0.00039547 | 0.00054702 |
| 3 | Aware | 0.09970258 | 0.00106985 | 0.00133965 | 0.00089019 | 0.00110494 |
| 3 | Blind | 0.09970218 | 0.00106646 | 0.00133498 | 0.00088931 | 0.00110410 |
| 4 | Aware | 0.10006307 | 0.00047638 | 0.00065185 | 0.00046844 | 0.00063914 |
| 4 | Blind | 0.10006239 | 0.00046820 | 0.00064419 | 0.00046071 | 0.00063194 |

### True expected portfolio value

The frozen oracle remains at **3,406 approvals**, cutoff
**0.0008343666409349597**, and true profit **$2,370,206,703.87** for every
replication. Oracle decisions are read, never reoptimized.

| Replication | Aware true profit ($) | Blind true profit ($) | Blind − aware ($) | Oracle − aware ($) | Oracle − blind ($) |
|---|---:|---:|---:|---:|---:|
| 0 | 2,369,668,531.41 | 2,369,863,850.57 | +195,319.16 | +538,172.45 | +342,853.29 |
| 1 | 2,369,993,954.26 | 2,370,014,452.47 | +20,498.20 | +212,749.60 | +192,251.40 |
| 2 | 2,366,443,900.02 | 2,367,214,309.03 | +770,409.00 | +3,762,803.84 | +2,992,394.84 |
| 3 | 2,355,001,513.50 | 2,355,079,750.58 | +78,237.08 | +15,205,190.36 | +15,126,953.28 |
| 4 | 2,361,652,655.94 | 2,362,129,389.96 | +476,734.01 | +8,554,047.92 | +8,077,313.91 |

Differences use unrounded values before display rounding. Bank-perceived
portfolio profit, issued borrowers' mean true PD, funded capital, utilization,
unused capital, and group rates are retained per model in each result JSON.
All five blind-minus-aware profit changes are positive here; five observations
do not establish a general model ranking or statistical significance.

The signed oracle differences are not floored at zero. Different score
orderings generate different cutoff-feasible portfolio families, so this
reference is **not unrestricted optimal regret** or a guaranteed upper bound.

### Zero-score blocks and cutoff behavior

Zero counts also equal negative-raw counts in all ten policies; none has a raw
prediction above one. Each policy has one tied score group, at bounded zero.

| Replication | Aware zeros | Blind zeros | Aware zero-block capital ($) | Blind zero-block capital ($) | Both feasible? | Aware cutoff | Blind cutoff | Feasible candidates aware / blind |
|---|---:|---:|---:|---:|---|---:|---:|---|
| 0 | 1,456 | 1,436 | 1,027,197,351.18 | 1,020,630,896.32 | Yes | 0.0004907386512242839 | 0.0004916469942358908 | 1,982 / 1,976 |
| 1 | 2,598 | 2,598 | 1,528,065,345.91 | 1,528,648,592.21 | Yes | 0.0003596647908360154 | 0.0003563658599999149 | 818 / 816 |
| 2 | 742 | 728 | 606,934,665.18 | 606,280,643.40 | Yes | 0.0009084052349843867 | 0.0009160919151720815 | 3,200 / 3,137 |
| 3 | 5,525 | 5,504 | 1,808,203,764.72 | 1,804,991,428.28 | Yes | 0.000009945427449874535 | 0.000014626857639292003 | 74 / 92 |
| 4 | 350 | 318 | 324,397,281.67 | 306,489,810.10 | Yes | 0.0010109421795449504 | 0.001015261652501874 | 4,410 / 4,400 |

Candidate counts include reject-all. No canonical replication required
reject-all, but deterministic tests cover the case where it is the only
feasible policy. Replication 3's aware zero block alone uses **99.0207%** of
the budget; only 72 more applicants are funded. Clipping and indivisible ties
therefore materially shape the policy family even without a budget violation.

## Descriptive five-draw summary

Sample SD uses ddof=1; these are descriptive pilot summaries, not confidence
intervals, significance tests, or estimates over applicant populations.

| Metric | Mean | Median | Minimum | Maximum | Sample SD |
|---|---:|---:|---:|---:|---:|
| Aware B coefficient | −0.000211237 | −0.000297224 | −0.000352491 | +0.000163090 | 0.000212941 |
| Aware gap (pp) | +11.564049 | +11.787624 | −1.331062 | +19.220990 | 8.232559 |
| Blind gap (pp) | +0.601002 | +0.651723 | +0.058694 | +0.975591 | 0.339739 |
| Gap change (pp) | −10.963047 | −11.728930 | −18.655986 | +2.306653 | 8.393937 |
| Blind − aware true profit ($) | +308,239.49 | +195,319.16 | +20,498.20 | +770,409.00 | 312,425.28 |

## Interpretation and recommendation for Step 8B

Changing only measurement noise was sufficient to reverse the canonical
group-aware gap's direction and generate large changes in its magnitude.
This demonstrates conditional noise sensitivity, not a formal conclusion
that noise explains a particular fraction of all possible disparities.
The group-blind gaps were much smaller in these five draws, but equal group
approval is neither imposed nor guaranteed.

The simulator has no structural B term. A fitted B coefficient is not a
measured discriminatory preference. Conditional on one finite population,
sample feature correlations and nonlinear misspecification can also make the
noiseless finite-sample linear projection's B slope nonzero. The zero line in
the figure is a structural reference, not an assertion that the conditional
finite-sample OLS mean is exactly zero. No extra noiseless fit was added.

The population, true-risk equation, PN scaling, and economic assumptions
remain researcher-defined. Growing receipts at 5% per period without costs
or discounting are not a calibrated mortgage contract. Synthetic results do
not establish real-world discrimination or normative fairness.

The pilot is technically ready for a **separately authorized, predeclared
conditional-noise study**, subject to reviewing these accounting/noise
assumptions with the professor. Step 8B should specify a replication target,
precision goals, coefficient/gap sign frequencies, paired profit effects,
and zero-block/reject-all reporting before execution. New-population or
split variation would be a distinct experiment. Keep this five-run namespace
immutable and handle unrelated V2 checkpoint lookup failures in a separate task.
Observed 2.655 seconds per replication suggests roughly 133 seconds of
replication computation for 50 draws, plus IO/reporting overhead; this is an
extrapolation, not a measured full-study runtime. **No Step 8B run is launched.**

## Files and reproducibility commands

New implementation files:

- `configs/professor_model/noise_robustness_pilot.yaml`
- `src/fair_lending/professor_model/noise_pilot.py`
- `src/fair_lending/professor_model/noise_pilot_io.py`
- `experiments/run_professor_noise_pilot.py`
- `tests/test_professor_noise_pilot.py`
- This document. Only `docs/PROJECT_STATUS.md` is modified from prior work.

All result paths below are relative to `/Users/macbook/Hack/econ4994/`:

| Folder | Files / meaning |
|---|---|
| `results/metrics/professor_model/noise_robustness_pilot/` | `manifest.json`; `bank_freeze_00.json` through `bank_freeze_04.json`; `replication_00.json` through `replication_04.json`; `summary_metadata.json`. Exactly five immutable successful result files, plus compact freezes and provenance. |
| `results/tables/professor_model/` | `noise_pilot_replications.csv`: five paired run summaries; `noise_pilot_coefficients.csv`: 45 model-term rows; `noise_pilot_group_approval.csv`: 20 replication/policy/group rows; `noise_pilot_summary.csv`: mean/median/min/max/sample-SD per numeric metric, with defined-value counts. |
| `results/figures/professor_model/` | Exactly two figures: `noise_pilot_group_coefficient.png` (zero reference) and `noise_pilot_approval_gaps.png` (paired gaps in pp). Both rendered and visually inspected. |

The 12 JSON files total about 197 KB; no five new 100,000-row datasets are
saved. The original eta/PN, models, decisions, and oracle files are unchanged.
Existing generated-output Git-ignore rules remain in effect.

```bash
# New execution: refuses to replace an existing pilot manifest/results.
python3 experiments/run_professor_noise_pilot.py
# Validate/reuse existing successes; only missing indices could run.
python3 experiments/run_professor_noise_pilot.py --resume
# Strictly load/consolidate saved records: no noise draws, fitting or search.
python3 experiments/run_professor_noise_pilot.py --summarize-only
python3 -m pytest -q tests/test_professor_noise_pilot.py
python3 -m pytest -q tests/test_professor_*.py
python3 -m pytest -q
```

Actual resume reused all five successes and launched **zero new replications**.
Summary-only reproduced all table/figure hashes without fitting. Provenance
checks reject changed source, configuration, dependencies, or original files;
record integrity checks reject malformed or incomplete successes. JSON
publication is atomic and exclusive: existing immutable files are never replaced.

## Validation

The 35 new deterministic tests cover streams/formula/raw-target retention,
fixed applicant/truth/split/contract/budget/oracle identity, source/input
preservation, both feature allowlists, evaluation-target isolation, prediction
replay and raw/bounded separation, exact cutoff/tie/budget accounting, oversized
zero blocks, freeze-before-truth evaluation, immutable/tamper-aware persistence,
group/profit arithmetic, deterministic five-run summaries, and read-only
summarization. All **320 professor-model tests passed without warnings**.

The complete suite finished with **600 passed, 4 failed, and 32 existing
deprecation warnings** in 96.37 seconds. No Step 8A tests failed. The four
unchanged failures are in `tests/test_systemic_corrected_checkpoint.py`:

- `test_corrected_replication_has_exactly_72_unique_complete_cells`
- `test_replication13_provenance_is_internally_consistent`
- `test_legacy_comparison_is_diagnostic_and_never_authorizes_reuse`
- `test_interim_summary_uses_five_corrected_records_only`

The current-HEAD (`7a7e590`) lookup does not find the older saved V2 record
identities (`8a9f71f`). Unrelated recovery code/results were neither modified
nor bypassed. The warnings are existing scikit-learn/SciPy L-BFGS-B `disp` and
`iprint` deprecations: 26 ML benchmark warnings and six sensitivity warnings.
The full repository is therefore **not all-green**, despite a passing
professor-model pilot. Nothing was staged, committed, or pushed.
