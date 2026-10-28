# Professor Model — Step 10B Fixed-Coefficient Comparison

## Prespecified design and decisions

Step 10B evaluates one coefficient intervention under the existing stylized
professor model. It does not fit a model or change a borrower population.
Following explicit scope clarification, it includes the **original canonical
realization plus all 200 saved Step 8B noise draws**. The canonical result is
reported separately, never pooled into the 200-draw distribution. All cases
use the same 20,000 evaluation applicants, original split and contracts.

| Policy | Construction | What is allowed to change |
|---|---|---|
| W: `frozen_aware` | Existing aware coefficients, predictions and saved cutoff | Nothing; replay and verify fingerprints |
| Z: `aware_B_zero` | `raw_Z = raw_W - saved_beta_B*B`, then the existing [0,1] clipping | Only the saved B contribution; its downstream assessments and optimum common cutoff |
| F: `frozen_blind` | Existing fully refitted blind model and saved cutoff | Nothing; this remains the historical refit reference |

The current canonical B slope is approximately +0.0007105261443, or
+0.07105261443 percentage points. Z uses the full saved coefficient, not this
rounded display value. Each saved noise case uses **that case's own** B
coefficient and sign. Removing a negative B coefficient increases B=1 raw
risk; a benefit is not assumed. The intercept and Y/D/C slopes remain fixed,
with no recentering, rescaling, new transforms or interactions.

The authoritative operation subtracts from the saved/replayed raw W score.
An independent coefficient replay with B set to zero must agree within
absolute raw-score tolerance 1e−15; B=0 raw scores remain exactly unchanged.
B=0 approvals can nevertheless change because of competition for capital.
The counterfactual ID `professor_aware_B_zero_v1` describes an intervention,
not a newly fitted OLS model.

### Research-specification review

| Question | Authoritative existing convention | Step 10B decision |
|---|---|---|
| Rate/calendar | `configs/professor_model/loan_economics.yaml`, Step 6A report and contract code: T=30, r=0.05 per period, ell=0.1Y, L=30ell, payment_t=ell(1+r)^t | Preserve exactly; no calendar duration, annual-rate conversion or mortgage-realism claim |
| Risk event | Same YAML and `profit.payment_survival`: constant conditional per-period default, absorbing default | Receipts use (1−p)^t; no change to true risk or event timing |
| PN scale | Saved model artifacts: raw PN target, no target transform or prediction rescaling | Preserve 0.8 scaling; no correction, calibration or new labels |
| Adjustment magnitude | Step 10A preregistration and the explicit Step 10B request: remove the saved aware B contribution | Use each exact fitted beta_B; do not insert the separate −0.1B manual treatment |
| Clipping/floor | Original aware/blind models use `clip(raw,0,1)`; Step 9C's D is a separately labeled zero-only positive-floor sensitivity | Use original clipping only. No change to Step 9C D, its floor or any saved policy |
| Ties/approval | `lending_threshold.yaml` and `lending.choose_cutoff`: score<=tau, exact ties indivisible, full requests, reject-all, exact prefix sums and lowest-cutoff objective tie rule | Preserve semantics and exact capital $1,826,087,220.7466326; optimize Z only |

These explicit working conventions support a conditional model comparison.
The professor's final calendar/payment interpretation and external realism
remain unresolved scientific questions, not reasons to silently choose new
economic assumptions. No contradiction requiring a different baseline was
found. Neither PN nor the true-risk equation is recalibrated.

### Prediction diagnostics without noise reconstruction

For the canonical case, raw/bounded predictions are compared with **saved**
evaluation PN and hidden true_PD after the new decision freeze. For the 200
draws, the user selected true-risk recovery only: individual PN/eta vectors
are not regenerated. All three policies' noise-study rows explicitly record
`PN_metrics_status=unavailable_no_noise_replay`; canonical PN errors are not
substituted for draw-specific PN errors. Historical W/F PN results remain
available in Step 8B but are not a three-policy PN comparison.

## Implementation, information boundary and provenance

The new `fixed_coefficient.py` and `fixed_coefficient_io.py` modules isolate
the intervention and persistence. The runner is
`experiments/run_professor_fixed_coefficient.py`; its frozen configuration is
`configs/professor_model/fixed_coefficient.yaml`. Old scientific modules are
not edited. The exact additive `[1,Y,D,C,B]` model contract is checked;
unexpected interactions, offsets, transforms or settings fail explicitly.

W/F predictions are replayed from coefficients. Decisions are reconstructed
with each policy's **saved cutoff**, then their exact prediction, assessment
and decision fingerprints are checked. No W/F cutoff search runs. Existing
cash-flow functions, the exact prefix search and independent Fraction audit
are reused for Z. Legacy model IDs appear only in private compatibility
adapters; exported Z artifacts retain the intervention ID.

The bank stage reads observable covariates, coefficients, contracts, saved
cutoffs and **bank-only** historical freeze payloads. Prior preservation maps
bind all source record/freeze bytes without parsing evaluator outcomes.
The full historical draw record, saved true profit/PD and canonical PN target
are released only after the new Z policy is saved and hash-frozen. True
profits are read from the original artifact; no new repayment outcomes exist.
The evaluator cannot select new parameters or policies.

Current source contents, configuration, dependency versions, input identities
and historical freeze/record bindings define a SHA-256 experiment identity.
Output directories are versioned by that identity, so source changes cannot
silently reuse old successes. Individual case results, bank freezes and
attempt records are immutable. Invalid records are retained and excluded;
resume retries rather than counting them. Stage-specific failures retain
their exception, traceback and runtime. Raw noise-study applicant rows are
released after each case; only the canonical Z rows are exported.

There are 201 declared cases and 603 expected policy cells. A full completion
claim requires all of them, with exactly 200 paired noise observations for
each comparison. No failed or invalid case is silently dropped. The five
pilot draws are already part of the 200 saved draws, not five extra cases.

## Estimands and reporting conventions

For each policy m, `V_m = sum(approved_m * saved_true_profit)` and
`G_m = 100*(approval_rate_B1 − approval_rate_B0)`. The three signed
comparisons are **Z−W**, **Z−F**, and historical **F−W**. Positive profit
difference means higher true expected value for the first policy; a group-gap
sign has no automatic normative interpretation. The historical F−W comparison
must still show 47/200 negative profit differences.

Policy/group tables distinguish perceived profit, true expected profit,
approval rates, capital use, raw/bounded risk errors, approved risk and tie
diagnostics. Pairwise allocation categories are disjoint. Potential profit
for unfunded applicants is labeled hypothetical, not earned. Empty approved
sets have undefined risk means/Jaccard where appropriate, not fabricated zeros.

Noise summaries include mean, median, sample SD (ddof=1), MCSE of the mean,
empirical 2.5/97.5 percentiles, extrema, and exact positive/negative/zero
counts. Every summary includes its denominator. The percentiles describe
conditional noise variation, not population confidence intervals. Extreme
outcomes and reject-all cases remain in the primary results.

The raw Z−W contrast isolates a coefficient contribution. Its allocation
response also passes through clipping, ties, capital competition and cutoff
optimization. This is not a causal mediation decomposition, evidence of
intentional discrimination, or a fairness proof. The frozen true-risk oracle
is not reoptimized or relabeled as an unrestricted benchmark.

## Canonical verified results

These are one saved noise realization, not the 200-draw means. All values
are from frozen decisions and saved evaluator truth; stored exports are
unrounded.

| Quantity | W: frozen aware | Z: B contribution removed | F: frozen blind |
|---|---:|---:|---:|
| Cutoff | 0.0009333057709409261 | 0.0008422626987943261 | 0.0009481206961177917 |
| Approvals | 3,900 | 3,802 | 3,802 |
| B=0 approval rate | 21.679434% | 18.986447% | 18.986447% |
| B=1 approval rate | 7.293729% | 19.141914% | 19.141914% |
| B=1 minus B=0 gap | −14.385705 pp | +0.155468 pp | +0.155468 pp |
| Funded principal | $1,825,953,398.73 | $1,826,028,908.10 | $1,825,778,044.19 |
| Unused capital | $133,822.01 | $58,312.64 | $309,176.56 |
| Perceived expected profit | $2,393,797,751.16 | $2,397,568,368.62 | $2,392,136,130.18 |
| True expected profit | $2,366,074,476.85 | $2,366,927,525.69 | $2,366,599,763.61 |
| Raw true-PD MAE (probability units) | 0.0005264084 | 0.0004790734 | 0.0004910068 |
| Bounded true-PD MAE (probability units) | 0.0004903232 | 0.0004389398 | 0.0004559405 |
| Raw held-out PN RMSE (measurement units) | 0.0996671572 | 0.0996640822 | 0.0996639250 |
| Mean true PD among approved | 0.079392% | 0.077234% | 0.077254% |

Z−W true profit is **+$853,048.84**; Z−F is **+$327,762.07**. The historical
F−W difference remains **+$525,286.77**. Z−W changes the group gap by
+14.541172 pp and reduces total approvals by 98. It shares 3,443 approvals
with W, gains 359 and loses 457, with Jaccard 0.8084057290.

Z and F have the same group counts (3,222 B=0, 580 B=1) and gap, but **not
the same portfolio**: 3,801 shared approvals, one Z-only request, one F-only
request, and Jaccard 0.9994740994. Aggregate group metrics can conceal
economically meaningful allocation differences.

Z leaves every B=0 raw score and B=0 raw-risk MAE unchanged, yet B=0 approval
falls because the common capital constraint reallocates lending. B=1 raw
MAE falls from 0.0007977251 to 0.0004852831 in this realization. These
findings do not imply improvement across every noise draw, probability
calibration, fairness, or real-world discrimination.

## Completed 200-draw comparison

**Execution status: run successfully.** All 201 cases completed, with 603
policy cells, 1,206 policy/group cells and 2,412 allocation-overlap cells.
There were zero failed attempts and zero missing/invalid cases. The canonical
case and draws 0, 8 and 199 were validated first; resume reused those four
successes while completing the remaining 197 cases. Cumulative successful-case
runtime was **339.19 seconds**. This is measured case runtime, not an isolated
hardware benchmark or a future runtime estimate.

Each following comparison uses **200 paired saved noise draws**, excluding
the canonical case. Dollar differences retain the first-minus-second sign.

| True-profit difference | Mean | Median | Positive / negative / equal |
|---|---:|---:|---:|
| Z−W | +$12,820,599.42 | +$592,951.23 | 138 / 50 / 12 |
| Z−F | −$11,795,630.33 | $0.00 | 69 / 90 / 41 |
| F−W, historical reference | +$24,616,229.75 | +$651,806.68 | 141 / 47 / 12 |

| True-profit difference | Sample SD | MCSE of mean | Empirical p2.5 / p97.5 | Minimum / maximum |
|---|---:|---:|---:|---:|
| Z−W | $290,340,530.95 | $20,530,175.83 | −$4,654,113.83 / +$7,151,617.92 | −$2,362,393,298.96 / +$2,369,952,982.90 |
| Z−F | $374,830,824.38 | $26,504,541.77 | −$515,516.73 / +$440,096.88 | −$2,365,919,379.46 / +$2,369,952,982.90 |
| F−W | $235,673,271.08 | $16,664,616.81 | −$4,279,050.08 / +$7,122,265.83 | −$6,816,666.05 / +$2,365,919,379.46 |

Z has higher/lower/equal true profit relative to W in **69%/25%/6%** of
draws, and relative to F in **34.5%/45%/20.5%**. Neither removal nor full
blind refitting universally dominates. The original 47/200 blind-profit
disadvantages are reproduced exactly rather than assigned to Z.

### Allocation, recovery and tail mechanisms

Mean approval gaps are +8.839521 pp (W), +0.356329 pp (Z), and +0.365663 pp
(F). The mean Z−W gap change is −8.483192 pp; Z−F is −0.009334 pp. Z's
absolute gap is smaller than W's in 183/200 draws, larger in five and equal
in twelve. These are descriptive synthetic group differences, not an agreed
fairness score or an empirical causal estimate.

Mean raw true-PD MAE is 0.0007440443 (W), 0.0006936813 (Z), and
0.0006792367 (F). Corresponding bounded MAEs are 0.0006873190,
0.0006455605, and 0.0006329107. These are raw probability units, so multiply
by 100 for percentage-point errors. Canonical prediction rankings are not
the same as average rankings across noise draws.

Reject-all occurs in **13 W**, **12 Z**, and **11 F** cases. Each count
matches that policy's oversized indivisible zero-block count. The new mean
profit differences are tail-sensitive in both directions:

- Draw 64: W/F reject all, but Z funds 3,701 applicants, yielding
  +$2,368,412,746.59 relative to either reference.
- Draw 154: W/F reject all, but Z funds 3,454, yielding +$2,369,952,982.90.
- Draw 89: Z rejects all while W/F fund 4,331/4,161; Z−W is
  −$2,362,393,298.96 and Z−F is −$2,363,904,428.10.
- Draw 92: Z and W reject all, while F funds 3,861; Z−F is
  −$2,365,919,379.46.

All cases remain in the primary results. The empirical central percentiles
omit some rare extremes; read them together with medians, extrema and sign
counts. MCSEs of the mean are larger than the magnitudes of the new mean
contrasts. Do not use the mean signs alone as strong model-ranking evidence.

Removing the group contribution preserves the aware intercept, whereas full
blind refitting also changes the intercept and other slopes. Clipping and
capital then act on different score distributions. Thus nearly equal group
gaps can coexist with very different economic outcomes at discontinuities.
The raw coefficient intervention is isolated; the resulting portfolio effect
still includes all the preserved downstream policy mechanisms.

## Artifacts and reproduction

Repository root: `/Users/macbook/Hack/econ4994/`.
Experiment fingerprint:
`cd7f3ee268cfc4d04ab906f0dbb6caab0de730aa21fdd96743c7eae815efebc7`.

| Directory relative to repository | Exact files / meaning |
|---|---|
| `results/tables/professor_model/fixed_coefficient/cd7f3ee268cfc4d0/` | `canonical_policies.csv` (3 policies), `canonical_groups.csv` (6 rows), `canonical_paired.csv` (3 contrasts), `canonical_overlap.csv` (12 allocation-category rows) |
| Same tables directory | `noise_policies.csv` (600 policy/draw rows, including prediction/tie diagnostics), `noise_groups.csv` (1,200 group rows), `noise_paired.csv` (600 paired-contrast rows), `noise_overlap.csv` (2,400 allocation-category rows), `noise_summary.csv` (24 metric/contrast summaries with n=200) |
| `data/processed/professor_model/fixed_coefficient/cd7f3ee268cfc4d0/40de5435c9f44a42a8638d2c02d37013/` | Canonical Z only: `counterfactual_predictions.csv` / `.parquet`, `counterfactual_profit.csv` / `.parquet`, `counterfactual_decisions.csv` / `.parquet` (20,000 rows each), and `counterfactual_candidates.csv` (all score-prefix endpoints and reject-all) |
| `results/metrics/professor_model/fixed_coefficient/` | `preservation_manifest.json`: 1,018 original-file hashes and allowed status/index edits |
| `results/metrics/professor_model/fixed_coefficient/cd7f3ee268cfc4d0/` | `manifest.json`, `latest_report.json`, `final_validation.json`; immutable `records/`, `freezes/`, `attempts/`, each containing 201 case files |

All export locations and hashes, including exact UUID-named case files, are
listed in `final_validation.json`. Model metadata stays transparent JSON;
no pickle, new fitted model, copied population, new noise vector, new figure,
or flexible-model pipeline is produced.

```bash
# Declared canonical + all saved draws, resuming verified successes:
python3 experiments/run_professor_fixed_coefficient.py --resume

# Does not create manifests, cases or consolidated files:
python3 experiments/run_professor_fixed_coefficient.py --summarize-only

# Explicit diagnostic subsets retain an incomplete full-study status:
python3 experiments/run_professor_fixed_coefficient.py --case canonical --resume
python3 experiments/run_professor_fixed_coefficient.py --case noise --replication 8 --resume
```

## Validation, deviations and preserved work

The new focused suite passed **38 tests, no warnings, in 4.04 seconds**.
The narrow dependency set (bank regression, blind comparison, loan economics,
lending threshold, plus the new tests) passed **190 tests, no warnings, in
7.53 seconds**. No unrelated full scientific suite was rerun. The previous
717 passed / 4 failed / 32 warnings full-suite record remains historical;
the four V2 checkpoint lookups were not suppressed, repaired or relabeled.

Tests cover signed/zero B coefficients, unchanged originals, B=0 exact
equality, coefficient replay tolerance, unsupported model specifications,
clipping without a positive floor, exact budgets and oversized ties,
forbidden reference optimization, malformed IDs/records, probability/profit/
group/overlap arithmetic, evaluator freezing, tamper detection, retained
failures, source invalidation, deterministic resume and read-only summary.

Independent saved-file checks validated all 201 coefficient blocks, 400 frozen
noise baseline cells against the original results, all 24 summary rows
(192 independently recomputed statistics plus sign counts), and allocation
partition/profit arithmetic. The canonical exported scores, clipping,
decision mask, original-dollar budget and candidate coverage were checked
separately. All 201 cases resumed with scoring disabled; API and CLI
summary-only checks left all **622 existing Step 10B artifact files** unchanged.
Canonical CSV/Parquet exports agree exactly. All 1,018 pre-step files remain
byte-identical, except that status/index were explicitly excluded as the two
authorized additive documentation edits.

No scientific deviations, failed attempts or invalid draws occurred. Scope
was explicitly expanded from the earlier canonical-first proposal to include
the 200 saved fits; PN replay was explicitly declined. These decisions do
not alter the preserved Step 10A documents or their historical evidence-index
core. Current Step 10B evidence is appended under the index's `extensions`
field. Old source-directory resume guards were not invoked or bypassed;
recorded bank-only freezes and source-content identities support this reuse.

## Limits and single recommended next step

This is conditional evidence on one synthetic population and split. Z is
not a refitted estimator, calibrated probability model, normative fairness
policy, or proof of real-world discrimination. Removing B's fitted term is
not the professor's separate −10 pp score adjustment. The stylized growing
payment schedule, per-period event/rate convention, low risk scale and absent
costs/discounting limit economic interpretation. No guaranteed profit gain,
population confidence interval or unrestricted oracle bound follows.

**Next step:** review the coefficient-only versus full-refit results and the
rare zero-block/capital discontinuities with the professor before defining
another experiment. More noise, different economics, new tie rules, new
populations, ML and HMDA remain separate decisions. No further study is run
automatically. The working tree retains dependent uncommitted Steps 9A–10A
work; Step 10B changes are left unstaged/uncommitted and nothing is pushed.
