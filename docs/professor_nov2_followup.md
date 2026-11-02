# November 2 Professor Follow-up

## Predeclared study

This protocol implements the professor's corrected November 2 specification.
The declared run completed and validated on **November 2, 2026**. Earlier
studies, including the historical **minus 0.1B** treatment, remain unchanged.

The deliverable is one matched realization in each of two synthetic worlds,
with three methods per world and two group bars per method. It is not an
independent-population study or a new 200-noise-draw experiment.

### Matched population and economic assumptions

Reuse the 100,000 original applicants, the saved 80,000 training / 20,000
evaluation split, original measurement noise eta, and existing contracts.
The exact shared capital budget is **$1,826,087,220.7466326**. The working
contract retains T=30, theta=0.10, r=0.05 **per period**, growing scheduled
payments, conditional per-period default, and no recovery, discounting or
funding cost. These are stylized synthetic assumptions, not conventional
mortgage accounting or annual realized bank profits.

The primary allocation rule remains the existing common cutoff on bounded
PD: full requests, indivisible exact ties, and a separate reject-all action.
Raw-score ordering remains a separate historical sensitivity.

### Scenario definitions

| World | Population | OLS and ML features |
|---|---|---|
| Scenario 1: no systemic differences | Original bank, private, target and truth tables | Y, D, C |
| Scenario 2: systemic differences | epsilon_2 = epsilon_1 + B; C_2 = 0.4 epsilon_2 + omega; same eta | Y, D, C, B |

The +1 epsilon mean shift for B=1 is a researcher-selected demonstration
assumption, chosen before fitting or inspecting the new lending results.
The epsilon variance remains one. Keep Y, D, B, omega, IDs, split, contracts
and capital matched, with B=0 exactly unchanged. Recalculate Scenario 2 truth
using the original frozen equation:

```
true_PD = sigmoid(0.52 - 0.65 * log(Y)) * sigmoid(4 + 0.06 * D + epsilon)
PN = 0.8 * true_PD + eta
```

The epsilon factor is already near saturation, so an epsilon-only treatment
may produce a modest true-risk change. C must change consistently with epsilon;
the systemic shift is consequently partly visible through the noisy signal.
Comparisons between worlds change both the upstream risk distribution and the
permitted feature set. They do not isolate the epsilon pathway alone.

### Three methods per world

1. **OLS control:** original frozen Y/D/C model in Scenario 1; a training-only
   Y/D/C/B refit in Scenario 2.
2. **OLS + penalty:** exactly the corresponding control raw score **+0.01B**,
   then clip to [0,1]. This means one percentage point of perceived default
   probability, not a relative one-percent increase or a log-odds coefficient.
3. **Gradient boosting:** `HistGradientBoostingRegressor`, trained on the same
   scenario-specific observable features and raw continuous PN target, with
   no manual penalty.

The primary models in Scenario 2 include B and are not group-blind. Higher
estimated default risk disfavors an applicant in the ordering; the earlier
slide wording saying that *lower* estimated PD disfavors B=1 is corrected.

### ML selection protocol

Inside the existing 80,000 training IDs, assign 64,000 fitting and 16,000
validation IDs using sorted applicant IDs and PCG64 with
`SeedSequence(4994, spawn_key=(1102,0))`. Reuse these assignments across worlds.

Evaluate exactly four combinations: `max_leaf_nodes` in {3,7} crossed with
`max_iter` in {25,75}. Fix squared-error loss, learning rate 0.05,
`min_samples_leaf=1000`, `l2_regularization=10`, `early_stopping=False`, and
random state 4994. Select by raw-PN validation MSE; exact ties prefer fewer
leaves and then fewer iterations. Refit the winner on all 80,000 training IDs.
Do not select by evaluation PN, hidden PD, true profit, or group gaps.

### Information boundary and output definitions

The simulator prepares private truth and observed PN. Lender functions receive
only declared observables, PN, administrative split IDs, and observable
contracts. IDs/split/PN are not prediction features. Keep raw scores and clipped
accounting probabilities separately. Freeze every model, score, perceived
assessment, cutoff and decision before the evaluator loads private true profits.

Approval bars use approved / all evaluation applicants within the group.
Profit bars use **total true expected bank profit contributed by funded loans
in that group**. Their two group contributions sum to portfolio profit.
Companion tables include profit per eligible applicant because B=0 and B=1
have different sample sizes. Rejected requests contribute zero to funded
profit; their potential profit is not a realized loss.

Compare penalty and control decisions, including displaced and newly funded
loans, without assuming that the fitted control is true-profit optimal.
Lower B approval, a profit loss, a positive fitted B coefficient and ML
superiority are hypotheses, not required outcomes or tuning targets.

### Deliverables and validation

Use a new `nov2_followup` namespace for bank/private data, prediction and
decision tables, six portfolio rows, twelve group rows, diagnostic tables,
model artifacts, metadata and four six-bar figures. Refuse silent overwrites.
The updated 14-slide deck preserves its earlier pages as a clearly labeled
historical appendix and replaces the two scenario result pages with editable
approval/profit charts. The original supplied PDF and prior deliveries remain
unchanged.

Validate exact replay and paired equations, B=0 invariance, feature/target
allowlists, deterministic training-only selection, estimated-risk cash flows,
exact budget/cutoff/tie behavior, independent rational enumeration, group
aggregation, plotting units, saved hashes and protected original artifacts.
Run new tests, the relevant reused dependencies, then the professor-model
suite. The four previously recorded V2 artifact-identity failures remain a
separate known issue, not repaired or suppressed by this study.

## Execution and results

The run used the declared settings without parameter retuning. All six
policies passed exact cutoff and independent rational-arithmetic checks.
Each method has 20,000 held-out applicants: 16,970 B=0 and 3,030 B=1.
Approval percentages below use each group's full eligible sample.

| Scenario | Method | B=0 approval | B=1 approval | B=1 minus B=0 gap | True portfolio profit | Capital used |
|---|---|---:|---:|---:|---:|---:|
| 1: no systemic | OLS control | 18.9864% | 19.1419% | +0.1555 pp | $2,366.600M | 99.9831% |
| 1: no systemic | OLS +0.01B | 25.1797% | 0.0000% | -25.1797 pp | $2,362.779M | 99.9881% |
| 1: no systemic | Gradient boosting | 23.4001% | 24.1254% | +0.7253 pp | $1,701.223M | 72.4065% |
| 2: systemic | OLS control | 21.7266% | 7.1617% | -14.5649 pp | $2,365.853M | 99.9869% |
| 2: systemic | OLS +0.01B | 25.1797% | 0.0000% | -25.1797 pp | $2,362.674M | 99.9839% |
| 2: systemic | Gradient boosting | 27.5840% | 27.6568% | +0.0728 pp | $2,050.145M | 87.2452% |

### Corrected penalty and allocation displacement

The positive manual penalty funds **no B=1 applicants** in either world.
Relative to control, it removes 580 B=1 approvals and adds 1,051 B=0
approvals in Scenario 1. The removed requests contribute $373.249M of
potential true expected bank profit, while the replacement requests contribute
$369.428M. Net true portfolio profit decreases by **$3,821,117.03**.

In Scenario 2, it removes 217 B=1 approvals and adds 586 B=0 approvals.
The removed/replacement true-profit contributions are $201.686M/$198.507M,
and net true profit decreases by **$3,179,162.14**. These are matched portfolio
differences, not realized losses or a general theorem that every fitted
control must outperform a distorted score. The displaced loans are truly
profitable under the unchanged synthetic accounting.

### Systemic risk magnitude and model interpretation

Across all 15,070 B=1 applicants, mean true conditional per-period PD rises
from **0.14228508%** to **0.14475891%**, a **0.00247384 percentage-point**
increase. The mean credit signal increases by 0.4, as required by the
structural signal equation. The modest risk response is consistent with the
near-saturated second sigmoid and was not enlarged after inspecting results.
All B=0 bank, private, PN and truth rows remain exactly unchanged.

Scenario 2's fitted OLS B coefficient is **+0.0008949665345** raw PN units
(about +0.08950 percentage points), before adding the separate +1 pp penalty.
Its sign and size are properties of this noisy fitted sample. They are not
an identified causal discrimination coefficient or a direct measure of the
upstream treatment. Scenario 1 excludes B, whereas Scenario 2 includes B;
the change in their control outcomes therefore combines information-set and
population changes.

### ML results and indivisible score groups

Training-only PN validation selects **three leaves and 25 iterations** in
both worlds, from the four settings fixed in the protocol. The final models
are refit on the same 80,000 training applicants. No evaluation outcomes or
hidden truth enter selection.

Gradient boosting approves more applicants than either OLS policy here, but
has lower true portfolio profit and leaves substantial capital unused. Every
request has positive perceived profit, but its next whole tied-score block
would require cumulative funding of **$3.145B** in Scenario 1 and **$2.785B**
in Scenario 2, exceeding the $1.826B budget. The declared common-cutoff rule
cannot split these exact prediction ties or skip applicants. The largest ML
tied groups contain 8,204 and 7,128 applicants, respectively. This makes
coarse prediction values economically consequential under this policy class.
The unused capital is a validated consequence, not a failed optimization.

Bounded hidden-PD recovery MAE is about 0.000456 for control versus 0.000631
for ML in Scenario 1, and 0.000491 versus 0.000531 in Scenario 2. These errors
are **probability units**. ML does not achieve general prediction, profit or
fairness superiority in this single matched run. A small approval gap is not
by itself a complete fairness assessment.

### Exact outputs and reproduction

All paths below are relative to the repository root. New datasets have
CSV/Parquet companions; result tables are CSV. Earlier datasets are not
replaced.

| Folder | Main files and meaning |
|---|---|
| `data/synthetic/professor_model/nov2_followup/<scenario>/` | `bank`, `targets`, `private`, `truth`, `noise_private` as `.csv` and `.parquet`; bank and simulator information remain separate |
| `data/processed/professor_model/nov2_followup/` | `predictions`, `assessments`, `decisions`, `loan_contracts` with CSV/Parquet companions; original and inner split CSVs; separately labeled evaluator-only true-profit tables |
| `results/tables/professor_model/nov2_followup/` | `portfolio_results.csv` (six rows), `group_results.csv` (12), `risk_diagnostics.csv`, `penalty_comparisons.csv`, `systemic_risk_audit.csv`, `ml_validation.csv`, `cutoff_candidates.csv` (76,743 candidates) |
| `results/metrics/professor_model/nov2_followup/` | Frozen config, model metadata, selections, preservation inventory, bank freeze, validation, and final run manifest |
| `results/models/professor_model/nov2_followup/` | Two fitted HistGradientBoostingRegressor joblib artifacts; exact feature order and dependencies in JSON metadata |
| `results/figures/professor_model/nov2_followup/` | Four six-bar charts, each as PNG and vector PDF: approval rates and true profit for each scenario |
| `reports/professor_model/nov2_followup/delivery/` | Updated 14-slide PowerPoint and matching read-only PDF handout; slides 9–14 are historical appendix |

```bash
# First execution only: refuses existing output files.
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python3 experiments/run_professor_nov2_followup.py

# Read and verify the completed evidence without fitting, writing or selecting.
python3 experiments/run_professor_nov2_followup.py --summarize-only
```

The scientific run took about 21 seconds in the recorded local environment,
excluding tests, independent audits and presentation production. The final
manifest binds 57 output artifacts plus 132 registered canonical content
identities, configuration/source contents, dependency versions, Git revision
and dirty-worktree state. Both worlds share the original noise realization.

### Validation evidence

- **50 new deterministic tests passed**, no warnings, in 3.55 seconds.
- New plus relevant regression/risk/accounting/cutoff tests: **234 passed**,
  no warnings, in 7.88 seconds.
- Fresh professor-model suite: **584 passed**, no warnings, in 25.91 seconds.
- All six cutoffs pass independent rational enumeration of every candidate.
- CSV/Parquet values round-trip exactly; all original Scenario 1 control
  decisions match the frozen group-blind decisions.
- Reloaded serialized ML models reproduce saved evaluation predictions
  exactly. Independent coefficient replay differs from saved OLS predictions
  by at most 3.47e-18, below the predeclared 1e-15 tolerance.
- Summary-only execution changes no saved file contents or mtimes and starts
  zero new computations. Generated simulator truth/noise and bank artifacts
  are hash-bound before evaluator release.
- The pre-edit inventory contains **1,846 existing files**. All scientific
  originals remain byte-identical; only the two explicitly allowed current
  status/guide documents are updated for the new work.

The unrelated full repository suite was not rerun. Its last recorded result
remains **717 passed, four existing V2 checkpoint lookup failures, and 32
warnings**. Those four failures were neither modified nor suppressed here.

### Next research decision

Review the corrected two-scenario figures with the professor. Decide whether
the present one-realization results and explicit tie/capital explanation are
sufficient for the final slides, or whether a separately specified robustness
study is required. The scientific execution added no additional noise draw,
risk-equation retuning, raw-order policy, HMDA application or external message.
Git publication is a separate repository operation.
