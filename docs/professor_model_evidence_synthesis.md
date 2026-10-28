# Professor Model — Step 10A Evidence Synthesis and Research Design

Audit date: **2026-10-28**. Audience: professor review and capstone writing.
This is an audit of existing evidence, not a new simulation or fitted model.
Step 9C is already implemented and is not repeated. The proposed Step 10B
below has **not been implemented or run**. The researcher selected a
canonical-fit comparison first; extending it across saved noise draws is a
separate possible follow-up.

## Executive summary

- The canonical OLS has almost zero held-out PN R², but nonzero recovery of
  hidden risk. These are different prediction questions. Neither noisy PN nor
  clipped OLS output is an empirically calibrated default probability.
- On the fixed population, 200 noise draws produce an aware B coefficient
  with 105 positive and 95 negative signs. Aware approval gaps range from
  −71.591 to +89.134 pp. This is conditional noise sensitivity, not variation
  across independent borrower populations.
- The canonical blind refit changes the approval gap from −14.385705 to
  +0.155468 pp and raises true expected profit by **+$525,286.77**. That
  comparison changes the intercept and other slopes as well as dropping B;
  it does not isolate the fitted B coefficient.
- The separate manual **−0.1B** adjustment favors B=1 in perceived default
  risk. Clipping maps all 3,030 B=1 scores to zero; the selected gap becomes
  +88.450206 pp. Step 9C's declared positive floor preserves the 3,764-person
  shared tie and all 4,990 approvals. This rules out a need for the *exact
  zero level* in that one sensitivity, not the importance of ties generally.
- Blind minus aware true profit is positive/negative/zero in 141/47/12 of
  200 draws. Its +$24.616M mean is dominated by two reject-all discontinuities;
  the median is +$651,806.68. There is no guaranteed blind profit advantage.
- The contract is stylized, and the oracle optimizes a score-specific cutoff
  family, not all feasible portfolios. Hash integrity does not establish
  economic realism, fairness, or empirical discrimination.
- Recommend a **single frozen-fit B-coefficient-zero intervention** compared
  with the saved aware and fully refitted blind policies. Keep the present
  economics conditional, require professor review, and do not run it in 10A.

## Research model, assumptions, and estimands

The professor-derived equations are supplied working transcriptions, not an
independent reading of the handwriting. The population has 100,000 applicants,
80,000 training and 20,000 evaluation IDs. B is an abstract binary group, not
an assigned empirical race. Independent primitives use Normal second
parameters as **standard deviations**:

```
log(Y) ~ N(11, SD 0.7)        D ~ N(0.35, SD 0.16)
B ~ Bernoulli(0.15)          epsilon ~ N(0, SD 1)
omega ~ N(0, SD 0.2)        C = 0.4*epsilon + omega

true_PD = sigmoid(0.52 - 0.65*log(Y)) * sigmoid(4 + 0.06*D + epsilon)
eta ~ N(0, SD 0.1)          PN = 0.8*true_PD + eta
```

The saved baseline has no structural B term in true risk and no B-dependent
upstream treatment. Finite-sample group associations can nevertheless exist.
Normal DTI is retained, including negative draws, rather than silently
recalibrated. The bank observes Y, D, C, B and training PN. It does not observe
epsilon, true_PD, or true profit. The primary OLS uses `[1,Y,D,C,B]`, with Y
in dollar **levels**, whereas truth uses log income and a hidden factor.
Blind OLS uses `[1,Y,D,C]`. Both target raw PN without division by 0.8.
Consequently this is not a correctly specified direct regression on true_PD;
there is no one-to-one recovery target for the sigmoid coefficients.

Under the specified mean-zero independent noise,
`E[PN|Y,D,C,B] = 0.8 E[true_PD|Y,D,C,B]`. The raw OLS projection and bounded
`clip(raw,0,1)` accounting scores remain distinct. PN is a continuous noisy
measurement, not an observed default or payment label. This family does not
provide realized defaults against which to claim observed default calibration.

The implemented accounting interprets PD as **conditional per-period default
before that period's payment**, with absorbing default. T=30, theta=0.10 and
r=0.05 **per period**, with no calendar period asserted:

```
ell_i = theta*Y_i; L_i = T*ell_i; payment_it = ell_i*(1+r)^t
R_i(p) = sum[t=1..T] payment_it*(1-p)^t
Pi_i(p) = R_i(p) - L_i
```

There is no discounting, funding/transaction cost, prepayment, or recovery.
The growing schedule is not mortgage amortization. The exact saved capital
is **$1,826,087,220.7466326**, 40% of the original evaluation requested
principal. Each policy maximizes perceived expected profit over its own
common-score prefixes `score <= tau`, with indivisible exact ties, full
requests, no skipping, reject-all, and the original exact arithmetic/tie rule.

| Quantity | Definition / unit | Inference boundary |
|---|---|---|
| PN prediction error | Estimate versus held-out PN; raw measurement units | Measures prediction of the noisy target, not true default recovery |
| Hidden-risk recovery | Estimate versus saved true_PD; MAE/RMSE in raw probability units | Evaluator-only; ranking/correlation does not imply calibration |
| Perceived profit | Sum of funded requests' Pi at bounded bank scores; USD | Bank objective under its beliefs |
| True expected profit | Sum of the same funded requests' **saved** true-profit values; USD | Expected synthetic value, not realized returns |
| Group approval gap | `100*(approval_rate_B1 - approval_rate_B0)`; pp | Descriptive group contrast unless a specific simulated intervention is defined |
| Within-group risk error | MAE and signed estimate-minus-true_PD error within each B group | Different from approval disparity; neither metric alone defines fairness |
| Noise uncertainty | Distribution across 200 eta draws, conditional on fixed applicants/split | Not a new-population interval or real-world discrimination confidence interval |

For unit checks, 0.0005 probability error is **0.05 pp**; a probability of
0.0014 is **0.14%**. A −0.1 adjustment is **−10 pp**, not −0.1%, and is not
a log-odds coefficient. All monetary comparisons below retain their signs.

## Evidence matrix: implemented Steps 5–9C

Numbers are display-rounded; linked CSV/JSON artifacts retain precision.
Source IDs link to exact report/result/validation paths below.

| Step / question | Frozen inputs and comparison | Main quantitative evidence | Supports | Does not establish | Sources |
|---|---|---|---|---|---|
| 5: Can OLS predict PN and recover hidden risk? | Saved applicants/PN; 80k train, 20k evaluation; aware OLS versus training-mean reference | Held-out PN R²=6.932707e−7; raw true-PD MAE=0.0005264084, correlation=0.560342; 923 negative evaluation predictions | Negligible PN prediction gain over the mean, with some hidden-risk ranking signal | No recoverable true-risk information, calibrated PD, or a structural interpretation of B | [E05](#e05-step-5-risk-estimation) |
| 6A: What do these requests imply economically? | Same 20k contracts and scores; perceived versus true per-request values; no lending yet | Mean perceived/true profit $291,416.11/$292,188.17; profit MAE $4,400.06; 100% profitable in both views; break-even PD 4.76190476% per period | Internally consistent potential cash-flow accounting | Realistic mortgage economics or profits actually earned | [E06A](#e06a-step-6a-contracts-and-profit) |
| 6B: Which common cutoff maximizes perceived profit under capital? | Original aware score, contracts and fixed budget; all 19,079 candidates | 3,900 approvals; cutoff 0.0009333057709409261; utilization 99.992672%; gap −14.385705 pp; true profit $2,366,074,476.85 | Exact optimum within that score's feasible cutoff family | Unrestricted portfolio optimality, causal discrimination, or a fully spent budget | [E06B](#e06b-step-6b-bank-lending) |
| 7A: How does the bank compare with true-risk cutoff lending? | Original bank unchanged; identical pool/contracts/capital; true-risk oracle optimized separately | Oracle 3,406 approvals; true profit $2,370,206,703.87; oracle minus bank +$4,132,227.02 (0.174340% of oracle profit) | A matched, evaluator-only restricted-policy benchmark | A universal nonnegative regret or value-of-information bound across different rankings | [E07A](#e07a-step-7a-oracle) |
| 7B: What changes when B is excluded and OLS refitted? | Identical PN/split/contracts/capital; saved aware versus blind refit | Blind 3,802 approvals; gap +0.155468 pp; blind minus aware true profit +$525,286.77; 457 aware-only and 359 blind-only approvals | Material allocation change under full refitting | A fixed-coefficient B effect, guaranteed fairness, or guaranteed profit gain | [E07B](#e07b-step-7b-blind-refit) |
| 8A: Is the canonical direction stable under new eta? | Fixed population/split/truth/economics; five predeclared noise draws, two fits per draw | B signs: 1 positive/4 negative; aware gaps −1.331 to +19.221 pp; all ten threshold audits passed | Feasibility and initial evidence of noise sensitivity | A separate independent sample of five beyond the later 200; population uncertainty | [E08A](#e08a-step-8a-pilot) |
| 8B: How variable are coefficients and lending over 200 eta draws? | Same fixed population; five pilot reuses plus 195 new draws, 400 policy cells | B signs 105/95; blind absolute gap smaller in 183/200; profit advantage 141/200, disadvantage 47/200; aware/blind reject-all 13/11 | Conditional noise-driven variability and cutoff discontinuities | A stable directional group effect, general model ranking, or independent-population inference | [E08B](#e08b-step-8b-full-noise-study) |
| 9A: What do the three frozen score branches mean? | Original aware and blind fits; aware no-manual-adjustment, blind−0.1B, unadjusted blind | 923/3,764/883 bounded zeros; all 3,030 B=1 direct scores negative; no upper clipping; direct tie needs 69.913701% of capital | Exact score construction and a capital-feasible zero block | Lending outcomes at this risk-only step or an isolated fitted-B intervention | [E09A](#e09a-step-9a-revised-risks) |
| 9B: How does the manual adjustment affect lending? | Same blind base for direct/control; all applicants/contracts/budget fixed; aware/oracle unchanged | Direct 4,990 approvals; gap +88.450206 pp; direct minus blind true profit −$7,332,311.99; +2,450 B=1 gains and −1,262 B=0 losses | A controlled total policy response to the manual adjustment with existing clipping/cutoffs | Separate allocation shares attributable to adjustment, clipping, ties, or refitting | [E09B](#e09b-step-9b-revised-lending) |
| 9C: Does replacing exact zeros by a small positive score change allocation? | Frozen A versus D; floor from q=.01 of strictly positive blind control; E unchanged | Floor 0.00013923734234194752; same 3,764-person tie, 4,990 approvals and true profit; D minus A perceived profit −$7,854,813.35 | Persistence under this zero-only floor sensitivity | That ties/clipping are irrelevant, or a complete additive mechanism decomposition | [E09C](#e09c-step-9c-mechanism-audit) |

### Canonical within-group risk recovery

E07B's group-risk CSV compares raw estimates with hidden true_PD on the same
evaluation IDs. Errors below are explicitly converted to **percentage points**,
not percentages of true risk:

| Model | B | Applicants | Raw MAE (pp) | Mean raw error (estimate minus truth, pp) |
|---|---:|---:|---:|---:|
| Aware | 0 | 16,970 | 0.047796469 | −0.005728478 |
| Aware | 1 | 3,030 | 0.079772515 | +0.064812604 |
| Blind refit | 0 | 16,970 | 0.049041494 | +0.004964387 |
| Blind refit | 1 | 3,030 | 0.049432166 | +0.004453454 |

The canonical aware model overestimates B=1 risk more strongly; the blind refit
has similar raw errors across groups in this draw. These are held-out recovery
diagnostics, distinct from both noisy-PN accuracy and the group funding gap.
They do not isolate one fitted coefficient or establish fairness. Bounded
errors are stored separately and must not be substituted for raw errors
without identifying the transformation.

### Noise-study denominators and tails

The 400-row full-study table has two policy rows per replication; paired
contrasts repeat on those rows. Paired statistics use **200**, not 400,
observations. The five pilot replications are included in the 200; they are
not 205 independent draws. The canonical original PN realization is a
separate historical comparison, not one of those 200 streams.

From the saved full-study rows, the aware B slope has mean 0.000063278,
sample SD 0.001063127, and 105/95 positive/negative signs. Aware/blind mean
gaps are +8.839521/+0.365663 pp; their SDs are 36.206101/0.599209 pp.
These are distributions across noise draws, not standard errors for one fit.

Blind minus aware true profit has mean **+$24,616,229.75**, median
**+$651,806.68**, and minimum **−$6,816,666.05**. Replications 8 and 92
contribute +$2,361,202,289.11 and +$2,365,919,379.46, respectively, when the
aware policy rejects all and blind lends. Retain both in primary summaries;
do not present the mean as the typical gain or silently trim outliers.
The 13 aware/11 blind reject-all cases exactly match the corresponding
oversized zero-block cases in the saved diagnostics. No partial tie funding
or budget violation is inferred or introduced.

The saved evaluator-only signal projection gives B=0.00000384199494513.
The recorded OLS signal/noise decomposition passes with maximum residual
2.25514e−17. This is a finite-population linear projection, not the
nonlinear DGP's structural group coefficient and not an operational bank fit.

## Scientific-claim audit

| ID / claim | Verdict and saved evidence | Defensible inference boundary |
|---|---|---|
| C01: PN contains little recoverable individual-level signal for these OLS models | **Supported narrowly.** E05 PN R²≈0 and little improvement over the training mean; noise/scaled-signal SD ratio 182.652 from the saved signal validation. Raw hidden-risk correlation is nevertheless 0.560342. | Say the specified models predict individual noisy PN poorly. Do not equate this with zero learnable true-risk signal or impossibility for all methods. Averaging 80k training labels and predicting hidden conditional risk are different from predicting a new eta draw. |
| C02: The aware coefficient and group gaps vary materially with noise | **Supported.** E08B: 105/95 coefficient signs, aware gap range −71.591 to +89.134 pp, SD 36.206 pp. | Conditional on one population, split, OLS design, contract and policy. No inference to new populations or discriminatory intent. |
| C03: Excluding B materially changes allocation | **Supported as a full-refit comparison.** E07B: gap change +14.541172 pp, 457 lost/359 gained approvals relative to aware. | Both models' intercepts/slopes differ. The comparison changes scores, clipping, rankings, perceived profits and selected cutoffs; it is not the isolated effect of beta_B. |
| C04: −0.1B creates a disparity reversal because of clipping/ties | **Mechanism facts supported; attribution requires qualification.** E09A–C verify raw adjustment, negative scores, clipping, the shared block and resulting approvals. | A versus E uses the same blind base and identifies a deterministic total response to that imposed score adjustment under the frozen policy. It does not partition that response into separate causal clipping/tie shares. A versus the aware branch additionally combines a base-fit change with the manual intervention. |
| C05: Blind lending can increase true profit, but not always | **Supported.** Canonical +$525,286.77; E08B positive/negative/zero in 141/47/12 draws. | Do not claim dominance, typical +$24.616M gains, or a general economic advantage for group-blind models. |
| C06: Oracle is a restricted cutoff benchmark | **Supported by code and artifacts.** `oracle.optimize_oracle` delegates the common-cutoff search using true-PD ordering and saved true profits (E07A). | The rule/capital match, but different scores generate different sets of feasible cutoff portfolios. Neither family generally contains the other; signed differences need not be nonnegative. It is not an unrestricted knapsack optimum. |
| C07: Positive profit does not validate economic realism | **Supported.** E06A: all requests profitable; the break-even PD is 4.76190476% per period, above the entire evaluation true-PD range (maximum about 0.862209%). | Results depend on growing payments, 5% per period, low PD, and absent costs/discounting. Do not describe them as observed mortgage returns or a realistic lending calibration. |
| C08: A lower approval gap alone does not prove fairness | **Not established as a normative claim.** E07B/E08B provide descriptive gaps and group errors, not an agreed fairness objective. | Abstract B, risk errors, loan amounts, opportunity and welfare are distinct. No fairness intervention or normative ranking was defined. |
| C09: These results establish intentional or real-world causal discrimination | **Not supported.** Baseline truth is group-neutral; E09A adds an explicit synthetic score treatment only. | A known policy-parameter intervention can have a conditional simulated effect; an observed B-group contrast, fitted coefficient, or empirical discrimination claim does not follow. No real mortgage borrowers were studied. |

## Step 9C: what A and D actually identify

Implementation was checked in `mechanism_audit.build_variants`,
`optimize_policy`, `tie_diagnostics`, the IO freeze path, the YAML, and saved
CSV/Parquet/selection artifacts, rather than inferred from names.

Let b_i be the **frozen blind raw prediction**. A is
`clip(b_i - 0.1*B_i, 0, 1)`. D begins with A and uses
`where(A == 0, f, A)`, where f is the linear-interpolated first percentile
of **strictly positive frozen blind-control bounded evaluation scores**.
This is not `max(A,f)`: the 161 positive A scores below f stay unchanged.
B keeps raw negatives for diagnosis; C uses only `max(raw_direct,0)` and
equals A because no raw score exceeds one. Neither B nor C becomes a policy.

f=0.00013923734234194752, derived from 19,117 positive control values.
A's 3,764 zero scores (734 B=0; 3,030 B=1) become one equally sized positive
tie in D; there is no collision with an existing positive score. D therefore
changes some risk values and their ordering relative to lower positives,
as well as bank-perceived receipts/profit. It does **not** undo the manual
adjustment, change fitted coefficients, or untie the block.

The block's requested principal is $1,276,685,167.37 (69.913701% of capital).
In D, reaching it requires funding the 161 lower scores too: a 3,925-person
prefix costing $1,365,351,773.15 (74.769253%), still feasible. The saved
candidate tables select the same higher cutoff
**0.0006623478981412278** for A and D. They fund the same 4,990 IDs,
use $1,826,025,513.85, and leave $61,706.89 unused. The next prefix is
capital-infeasible. Selected boundary size is one; the large shared tie is
inside the funded prefix, not at its final boundary.

A/D Jaccard is 1.0 with no exclusive approvals and 15,010 common rejections.
Both have gap +88.450206 pp and true expected profit $2,359,267,451.62.
D lowers perceived profit by $7,854,813.35 but does not change the maximizing
prefix. In A, estimated B=1 risk is zero while mean true B=1 PD is 0.141505%.
Thus zero is a model output, not absence of actual synthetic default risk.

The identified sensitivity is **one zero-only score replacement with all
other mechanisms fixed**, followed by the same policy optimization. Its
zero allocation contrast neither decomposes the original treatment effect
nor proves that any positive floor or tie policy would behave identically.
Source/output hashes prove the saved files match their recorded implementation;
they do not validate a causal interpretation of the implementation.

## Discrepancies, ambiguities, and unresolved questions

No numerical contradiction was found in the **selected claims and saved rows
checked here**. This audit did not refit models, reselect cutoffs, independently
revalidate every historical scientific result, or assert all code is correct.
The following issues must remain visible:

| Issue | Evidence / consequence | Treatment in this synthesis |
|---|---|---|
| Near-zero PN fit versus nonzero true-risk recovery | E05/E07B distinguish targets and show nonzero hidden-risk correlations | Narrow C01; do not claim that no signal is recoverable |
| Direct versus no-direct branch changes the base model | E09A uses aware for no-manual adjustment, blind for manual −0.1B | Use blind E as the manual-adjustment control; label aware comparisons combined |
| Positive-floor name can imply all scores are floored or ties broken | E09C changes exact zeros only and retains the shared block | State the actual `where` rule and prefix cost; do not call it a tie-breaking experiment |
| Oracle CSV uses legacy `economic_regret` field names | E07A records signed oracle-minus-bank value; code/doc warn about different feasible families | Preserve files; interpret as signed restricted-cutoff differences |
| Initial annual-rate / unresolved-horizon text versus implemented economics | Step 1 is historical; Step 6A explicitly adopts r=5% per period and conditional per-period PD | Document the later working interpretation; professor confirmation of calendar/event/schedule realism remains outstanding |
| Very high noise and 0.8 scaling | Mean true PD≈0.142208% versus noise SD=10 pp; target projects scaled risk | Preserve both; request review rather than silently rescale or recalibrate |
| Count field `new_noise_fits=195` can be read as 195 model fits | E08B reports 195 new replications, each fitting two operational banks | There were 390 new operational fits in 8B, 400 including the pilot; the separate evaluator projection is not a bank |
| Current source-inventory growth can block old CLI resume | Eight 9A–9C modules are additional to the Step 8B recorded source inventory; `prepare_manifest` compares a fresh full-directory identity | Recorded Step 8B paths/contents, originals and result hashes still match. Static inspection predicts an identity guard failure on an expanded-tree resume; no CLI was invoked or guard bypassed |
| Historical test totals differ | Last Step 9C record: 437 professor tests passed; full 717 passed/4 failed/32 warnings | Historical stage totals are not fresh Step 10A tests. Four known V2 lookup failures remain separate and untouched |

PN's measurement interpretation, the per-period event/calendar relationship,
the professor's exact contract, and whether −0.1 was intended as an absolute
probability change all need substantive review. A clean hash cannot answer
those questions. Older stage-specific deferrals should not be read as the
current project status.

## Recommended next experiment: preregistration for review, not execution

### Why this experiment first?

| Candidate | What it would clarify | Priority / reason |
|---|---|---|
| Fixed-coefficient B removal versus saved full blind refit | Separates one fitted coefficient contribution from all-other-coefficient/refit changes on the canonical evidence | **First proposed experiment.** Answers the main unresolved attribution in 7B without new data, fitting, economics or noise |
| Rate, event horizon, payment schedule and cost review | Economic meaning and external plausibility | **Review prerequisite**, followed by a separately specified model family if changes are requested; do not mix new economics into the coefficient audit |
| Alternative clipping/tie or capital rules | Which policy discontinuities generate reject-all and large gaps | Later, separately predeclared; these change the policy set and are not a clean coefficient comparison |
| Noise-scale or bounded-estimator comparisons | Measurement strength, scale/calibration and functional form | Later; changes in target scale or estimator would confound the immediate attribution question |
| Repeat the controlled intervention over the 200 saved fits | Conditional noise stability of the new comparison | Follow-up only after the canonical audit is understood; no new noise or refits are needed in principle, but new policy evaluations still require a separate request |

### Frozen inputs and three policies

Use only the existing canonical fits and original 20,000 evaluation IDs:

| Policy | Raw score | Estimation / decision action in proposed Step 10B |
|---|---|---|
| W: saved group-aware | `sW_i = saved_aware_raw_i` | Read its saved coefficients, scores, cutoff and decisions; never refit or reoptimize this reference |
| Z: B coefficient set to zero | `sZ_i = saved_aware_raw_i - saved_beta_B*B_i` | Change only the B contribution; retain the aware intercept and Y/D/C slopes, with no recentering or refit |
| F: saved full blind refit | `sF_i = saved_blind_raw_i` | Reuse the existing blind fit, cutoff and decisions; it is a historical full-refit reference, not a new fit |

The saved aware B slope is about **+0.0007105261443**, or +0.07105261443 pp.
Removing it is not the professor's separate **−0.1B** intervention. Use the
saved full-precision coefficient, never the display-rounded number.
The authoritative Z operation is subtraction from the saved raw score.
Check agreement with replay of the coefficient vector with B set to zero
within absolute raw-score tolerance 1e−15; B=0 saved raw scores must remain
bit-identical. No Z score, profit, or decision has been computed for 10A.

Preserve all raw scores; apply the **same** `[0,1]` clipping to Z. Only Z gets
new perceived profits and a newly optimized common cutoff, under the unchanged
contract, budget, exact ties, reject-all and objective tie rule. Do not add a
fixed-tau policy, tie breaking, alternative floor, group cutoff, or fairness
constraint to this initial study. Group B can affect Z construction but does
not enter the downstream optimizer's input view. B=0 *scores* stay fixed;
B=0 approvals may change through capital competition.

All original applicants, splits, PN, private factors, true risk, fits,
coefficients, raw/bounded predictions, contracts, bank assessments,
aware/blind/oracle decisions, Steps 8A/8B records, and Steps 9A–9C outputs
must remain byte-identical. Use a dedicated new artifact namespace; no
overwrite or adopting old records under a new identity.

### Exact comparisons and outcomes

For policy m, define `G_m = 100*(rate_m,B1 - rate_m,B0)` and
`V_m = sum_i approved_m,i * saved_true_profit_i`.

- **Primary isolated-coefficient comparison:** Z minus W. Report
  `Delta_G_coef = G_Z-G_W` (pp), `Delta_V_coef = V_Z-V_W` (USD), approval
  counts, group-specific rate changes, capital use and overlap.
- **Full-refit comparison:** F minus W, using the frozen existing reference.
- **Incremental refit comparison:** F minus Z. On the raw scale this reflects
  changes in the intercept and Y/D/C slopes after dropping B. Group-gap and
  profit differences add arithmetically across these three policies, but that
  is not a causal mediation decomposition or an order-invariant attribution
  of clipping, ties and capital.
- Report raw/bounded score changes, MAE/RMSE and signed errors versus held-out
  PN and evaluator true_PD, overall and within B groups. These metrics evaluate
  a fixed intervention; do not select its parameters using truth or errors.
- For each portfolio report perceived and true expected profit, mean true PD
  among approved, approvals/rates, funded principal/utilization/unused funds,
  unique scores, clipped counts, tied-block sizes/cost/feasibility, selected
  boundary ties and all candidates. Pairwise overlap uses four disjoint
  categories and Jaccard; unfunded profit remains explicitly hypothetical.
- Oracle may be displayed only as the unchanged evaluator reference. All
  differences retain signs. No sampling confidence interval or significance
  claim is justified by this single deterministic intervention alone.

The exact raw identity is `sZ-sW = -saved_beta_B*B`; the raw full-refit change
also includes `(blind_intercept-aware_intercept) + sum_j(delta_beta_j*X_j)`.
This identity is not a numerical result for the unrun experiment. It isolates
an estimated *policy parameter*, not the causal effect of changing an
applicant's group or proof of discrimination. Allocation contrasts include
the ensuing clipping, budget competition and optimal-cutoff response.

### Information boundary, validation and go/no-go

The future bank stage may read saved observable covariates/groups, fit
coefficients, raw predictions, contracts and exact capital. It must not read
epsilon, eta, true_PD, true profit, future events or oracle decisions. Freeze
Z scores, assessments, candidates, selection and decisions before releasing
saved truth/oracle to the evaluator. Truth is for assessment only; no parameter
or decision tuning is allowed.

Required tests: original-file preservation and reference identity; exactly
one coefficient contribution changed; B=0 raw equality; the raw contrast and
coefficient replay tolerance; unchanged IDs/contracts; valid bounded scores;
no hidden-field leakage or refit; exact original-dollar budget feasibility;
indivisible ties including oversized-zero reject-all; independent candidate
enumeration; decision freeze/tamper rejection; arithmetic for group gaps,
profits and overlap; deterministic reruns; and transparent raw/bounded exports.
Do not require a desired gap sign, profit advantage, or approval count.

**Go** only after professor/researcher approval of this specific canonical
contrast and conditional accounting, a separate implementation instruction,
valid frozen inputs and passing focused tests. **No-go** on ambiguous intended
B treatment, changed/missing provenance, leakage, infeasible/split ties,
modified original references or proposed parameter retuning. Four existing V2
lookup failures are disclosed separately; they neither validate this future
study nor authorize repairs. The present status is **proposed, not run**.

## Professor-review questions

1. Was −0.1 intended as **10 percentage points of default risk**? Is its
   large magnitude relative to approximately 0.14% mean true PD intentional?
2. Is the next question the fitted B contribution with all other coefficients
   frozen, rather than changing a whole estimator or inserting a new treatment?
3. Does PN represent a noisy scaled risk measurement in the intended model,
   including its 0.8 multiplier, negative values and SD=0.1?
4. Are PD's conditional per-period event and r=5% per period intended? What
   calendar duration, growing payment schedule, costs and discounting should
   later economic interpretations use?
5. Should identical clipped scores be indivisible for lending, including
   reject-all when the lowest block exceeds capital? Changing this is a
   separately defined policy comparison, not an implementation fix.
6. What normative group outcome is relevant beyond a raw approval gap?
   Group B is abstract, and equality of rates alone has not been established
   as the professor's fairness objective.

## Evidence ledger and exact source paths

All paths resolve from this document in `/Users/macbook/Hack/econ4994/docs/`.
The [machine-readable evidence index](../results/metrics/professor_model/professor_model_evidence_index.json)
records current hashes, roles, step IDs, locators and verification limits.
Large datasets are referenced, not copied into the synthesis.

### E05: Step 5 risk estimation

[Report](professor_bank_regression.md),
[held-out metrics](../results/tables/professor_model/bank_regression_evaluation_metrics.csv),
[group errors](../results/tables/professor_model/bank_regression_evaluation_groups.csv),
[coefficient artifact](../results/metrics/professor_model/bank_regression_model.json),
[validation](../results/metrics/professor_model/bank_regression_validation.json),
[metadata](../data/synthetic/professor_model/baseline/bank_regression_metadata.json).
Noise scale and mean true PD also use
[signal validation](../results/metrics/professor_model/noisy_signal_validation.json) and
[risk distribution](../results/tables/professor_model/true_risk_distribution.csv).

### E06A: Step 6A contracts and profit

[Report](professor_loan_economics.md),
[summary](../results/tables/professor_model/loan_economics_summary.csv),
[validation](../results/metrics/professor_model/loan_economics_validation.json),
[contracts](../data/processed/professor_model/loan_contracts.parquet),
[saved true profit](../data/processed/professor_model/true_expected_profit.parquet),
[metadata](../data/processed/professor_model/loan_economics_metadata.json).

### E06B: Step 6B bank lending

[Report](professor_lending_threshold.md),
[portfolio](../results/tables/professor_model/lending_portfolio_summary.csv),
[evaluator summary](../results/tables/professor_model/lending_evaluator_summary.csv),
[selection](../results/metrics/professor_model/lending_selection.json),
[validation](../results/metrics/professor_model/lending_validation.json),
[metadata](../data/processed/professor_model/lending_metadata.json).

### E07A: Step 7A oracle

[Report](professor_oracle_comparison.md),
[comparison](../results/tables/professor_model/bank_vs_oracle_summary.csv),
[group rates](../results/tables/professor_model/bank_vs_oracle_group_audit.csv),
[validation](../results/metrics/professor_model/oracle_comparison_validation.json),
[metadata](../data/processed/professor_model/oracle_comparison_metadata.json).

### E07B: Step 7B blind refit

[Report](professor_group_blind_comparison.md),
[risk comparison](../results/tables/professor_model/group_aware_vs_blind_risk.csv),
[group risk errors](../results/tables/professor_model/group_aware_vs_blind_risk_group_audit.csv),
[portfolio](../results/tables/professor_model/group_aware_vs_blind_portfolio.csv),
[group rates](../results/tables/professor_model/group_aware_vs_blind_group_audit.csv),
[overlap](../results/tables/professor_model/group_aware_vs_blind_allocation_overlap.csv),
[blind coefficient artifact](../results/metrics/professor_model/group_blind_regression_model.json),
[validation](../results/metrics/professor_model/group_blind_comparison_validation.json),
[metadata](../data/processed/professor_model/group_blind_comparison_metadata.json).

### E08A: Step 8A pilot

[Report](professor_noise_robustness_pilot.md),
[five paired rows](../results/tables/professor_model/noise_pilot_replications.csv),
[validation/summary](../results/metrics/professor_model/noise_robustness_pilot/summary_metadata.json),
[manifest](../results/metrics/professor_model/noise_robustness_pilot/manifest.json).

### E08B: Step 8B full noise study

[Report](professor_noise_robustness_full.md),
[400 policy rows](../results/tables/professor_model/noise_robustness_full/noise_full_results.csv),
[summary](../results/tables/professor_model/noise_robustness_full/noise_full_summary.csv),
[coefficients](../results/tables/professor_model/noise_robustness_full/noise_full_coefficient_summary.csv),
[ties](../results/tables/professor_model/noise_robustness_full/noise_full_tie_diagnostics.csv),
[completion/validation](../results/metrics/professor_model/noise_robustness_full/latest_report.json),
[manifest](../results/metrics/professor_model/noise_robustness_full/manifest.json),
[signal projection](../results/metrics/professor_model/noise_robustness_full/signal_reference_EVALUATOR_ONLY.json).
The existing `replications/replication_000.json` through `replication_199.json`
and their recorded bank freeze files provide per-draw evidence.

### E09A: Step 9A revised risks

[Report](professor_revised_scenarios.md),
[group risk summary](../results/tables/professor_model/revised_scenarios_group_summary.csv),
[zero blocks](../results/tables/professor_model/revised_scenarios_zero_tie_diagnostics.csv),
[frozen scores](../data/processed/professor_model/revised_scenarios/scenario_risk_estimates.parquet),
[validation](../results/metrics/professor_model/revised_scenarios_validation.json),
[metadata](../data/processed/professor_model/revised_scenarios/scenario_risk_metadata.json).

### E09B: Step 9B revised lending

[Report](professor_revised_lending_comparison.md),
[portfolio](../results/tables/professor_model/revised_lending/portfolio_comparison.csv),
[group rates](../results/tables/professor_model/revised_lending/group_outcome_audit.csv),
[overlap](../results/tables/professor_model/revised_lending/pairwise_allocation_overlap.csv),
[validation](../results/metrics/professor_model/revised_lending/validation.json),
[manifest](../results/metrics/professor_model/revised_lending/run_manifest.json).

### E09C: Step 9C mechanism audit

[Report](professor_mechanism_audit.md),
[variant definitions](../results/tables/professor_model/mechanism_audit/variant_definitions.csv),
[score diagnostics](../results/tables/professor_model/mechanism_audit/risk_clipping_diagnostics.csv),
[tie/capital audit](../results/tables/professor_model/mechanism_audit/tie_capital_diagnostics.csv),
[portfolio](../results/tables/professor_model/mechanism_audit/portfolio_comparison.csv),
[groups](../results/tables/professor_model/mechanism_audit/group_outcome_audit.csv),
[overlap](../results/tables/professor_model/mechanism_audit/pairwise_allocation_overlap.csv),
[candidates](../results/tables/professor_model/mechanism_audit/cutoff_candidates.csv),
[floor derivation](../results/metrics/professor_model/mechanism_audit/positive_floor_derivation.json),
[bank freeze](../results/metrics/professor_model/mechanism_audit/bank_freeze_checkpoint.json),
[validation](../results/metrics/professor_model/mechanism_audit/validation.json),
[manifest](../results/metrics/professor_model/mechanism_audit/run_manifest.json).

## Step 10A checks and preservation

Only documentation and the small evidence index are new. The existing
`PROJECT_STATUS.md` receives a short additive entry; unrelated history,
earlier reports, source, configurations and results remain unchanged.
Read-only checks cover source/output/preservation hashes, exact Step 8B
replication-policy coverage and paired denominators, saved summary arithmetic,
Step 9C raw/bounded/zero-only identities and decision masks, group/profit
aggregates, metadata locators, and documentation links. No figure was recreated.

The last completed scientific tests remain the Step 9C record: **33 new tests
passed; 437 professor tests passed; full suite 717 passed, 4 existing V2
failures, 32 warnings**. They were not rerun or relabeled as Step 10A tests.
Four `test_systemic_corrected_checkpoint.py` lookups remain unresolved, not
suppressed or repaired. Targeted checks passed for 100 indexed evidence items,
200 saved noise records and freeze hashes, all 46 full-study summary rows
(552 statistics), nine configuration-file identities, and 108 local links
including ten evidence anchors. All **1,016 pre-existing snapshotted files**
remain byte-identical; only the expressly permitted existing status document
is edited. The evidence index records these checks and their limitations.
No applicants, targets, models, profits, cutoffs, portfolios, noise replications,
figures, ML/HMDA code, or discrimination mechanisms are created by Step 10A.
No staging, commits or pushes are performed.
