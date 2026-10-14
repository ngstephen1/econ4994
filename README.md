<p align="center">
  <img src="docs/assets/fair-lending-logo.svg" alt="Fair Lending Lab: economics, simulation, equity, and reproducible research" width="100%">
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img alt="Python 3.11+" src="https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white"></a>
  <a href="docs/PROJECT_STATUS.md"><img alt="Professor model: Step 8B complete" src="https://img.shields.io/badge/professor_model-Step_8B_complete-861F41?style=flat-square"></a>
  <a href="#validation-status"><img alt="Latest full suite: 624 passed, 4 failed" src="https://img.shields.io/badge/latest_suite-624_passed%2C_4_failed-E5751F?style=flat-square"></a>
  <a href="LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/license-MIT-253746?style=flat-square"></a>
</p>

<p align="center">
  <strong>Lending Under Imperfect Information</strong><br>
  Risk Estimation, Group Disparities, and Credit Allocation<br>
  Nguyen Phan Nguyen · Virginia Tech ECON 4994
</p>

A synthetic-first capstone combining economics, statistics, machine learning,
and reproducible research. The current professor-inspired model studies how
noisy risk measurements affect lending under a capital constraint. Earlier
approval and economic-lending experiments remain separate and preserved.

## Menu

| Start here | Run and explore | Evidence |
|---|---|---|
| [Current professor model](#current-professor-model) | [Setup](#setup) | [Research slides](reports/professor_model/step8b_progress/oct15_progress.pptx) |
| [Research-family guide](#three-separate-research-families) | [Professor-model workflow](#professor-model-workflow) | [Eight-page report](reports/professor_model/step8b_progress/professor_model_progress_8_slides.pdf) |
| [Current status](docs/PROJECT_STATUS.md) | [Preserved V1/V2 workflows](docs/running_v1_v2.md) | [CSV/data guide](docs/professor_model_data_guide.md) |
| [Documentation index](docs/README.md) | [Dashboard](#dashboard-version-1) | [Evidence manifest](reports/professor_model/step8b_progress/evidence_manifest.json) |
| [Research safeguards](#research-safeguards) | [Validation status](#validation-status) | [Detailed project history](docs/PROJECT_HISTORY.md) |

## Current professor model

**Steps 1–8B are complete.** One fixed population has 100,000 applicants,
with 80,000 training and 20,000 evaluation IDs. The bank fits raw noisy risk
measurements using ordinary least squares:

```text
true_PD = sigmoid(0.52 − 0.65 log(Y)) × sigmoid(4 + 0.06 D + epsilon)
PN = 0.8 true_PD + eta,    eta ~ Normal(0, SD=0.1)

Group-aware: PN ~ intercept + Y + D + C + B
Group-blind: PN ~ intercept + Y + D + C
```

Y is annual income, D is a DTI ratio, B is an abstract binary group, and C is
a noisy credit signal. The bank cannot observe epsilon or true_PD.
Raw predictions remain separate from clipped probabilities used in accounting.

The working contract uses 30 periods, theta=0.10, **r=0.05 per period**,
principal `L=30 theta Y`, and growing scheduled payments. PD means constant
conditional per-period default. Each lender selects a common bounded-PD
cutoff to maximize perceived expected profit under the unchanged
**$1,826,087,220.7466326** capital budget. Exact ties are indivisible and
reject-all is allowed. This is a stylized contract and restricted cutoff
policy, not conventional mortgage amortization or unrestricted knapsack.

### What the 200-noise-replication study found

Only eta varies; applicants, true risk, split, contracts, capital, and the
true-risk oracle remain fixed. All **200 replications** completed:
five validated pilot reuses and 195 additional replications, yielding
400 policy-level rows.

| Finding | Result |
|---|---:|
| Smaller absolute approval gap under group-blind lending | 183/200 (91.5%) |
| Aware absolute gap at least the original 14.385705 pp magnitude | 131/200 (65.5%) |
| Mean aware / blind approval gap, B=1 minus B=0 | +8.840 / +0.366 pp |
| Mean / median blind-minus-aware true expected profit | +$24.616M / +$651,807 |
| Blind true profit higher / lower / equal | 70.5% / 23.5% / 6.0% |
| Reject-all under aware / blind policies | 13/200 / 11/200 |

Two differences above **+$2.36 billion** dominate the profit mean. They arise
when an oversized clipped-zero block forces aware reject-all while the blind
policy funds loans. Both cases remain in all primary summaries and plots.
These results condition on one synthetic population; 91.5% does not prove
fairness, and group-blind lending does not always improve profit.

Read [full noise robustness](docs/professor_noise_robustness_full.md) for
SDs, MCSEs, empirical outcome spreads, provenance, and the evaluator-only
coefficient decomposition. The [report folder](reports/professor_model/step8b_progress/README.md)
contains eight slides, the matching PDF, build source, previews, and evidence.
The PDF is image-backed; editable objects remain in the PowerPoint.

## Three separate research families

| Family | Question and methods | Status and starting point |
|---|---|---|
| Version 1: approval disparities | Known direct/upstream/mixed approval mechanisms, logistic recovery, ML, dashboard, 3,140-run sensitivity study | Preserved at `v1-approval-benchmark`. [Research design](docs/research_design.md) |
| Version 2: economic lending | Conditional repayment histories, traditional logit versus HistGB, expected profit, binary portfolio allocation, direct belief and upstream opportunity mechanisms | Deterministic experiments complete. Corrected systemic checkpoint has five replications; its 50-replication study remains incomplete. [V2 specification](docs/economic_lending_spec.md), [recovery audit](docs/systemic_monte_carlo_recovery.md) |
| Professor model: imperfect information | Two-factor hidden default risk, continuous PN targets, aware/blind OLS, common cutoffs, fixed-population noise sensitivity | Steps 1–8B complete, including 200 noise replications. [Model overview](docs/professor_whiteboard_model.md) |

The professor model has **no flexible ML comparison or inserted discrimination
intervention**. Its abstract B groups are not V1's White/Black categories.
The completed V1 Monte Carlo study, incomplete V2 systemic study, and
completed professor-model noise study must not be conflated.

## Setup

Python 3.11+ is required.

```bash
git clone https://github.com/ngstephen1/econ4994.git
cd econ4994
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -c "import fair_lending; print('fair_lending import OK')"
```

### Professor-model workflow

These are **manual reproduction commands**, not instructions to overwrite
existing evidence. Generation/fitting steps refuse existing outputs without
an explicit overwrite flag. Do not run them merely to browse saved results.

| Step | Command | Documentation |
|---|---|---|
| 2: applicants | `python3 experiments/generate_professor_population.py` | [Population](docs/professor_population_generation.md) |
| 3: hidden truth | `python3 experiments/generate_professor_true_risk.py` | [True risk](docs/professor_true_default_risk.md) |
| 4: raw PN | `python3 experiments/generate_professor_noisy_signal.py` | [Noisy measurement](docs/professor_noisy_risk_signal.md) |
| 5: aware OLS | `python3 experiments/fit_professor_bank_regression.py` | [Bank regression](docs/professor_bank_regression.md) |
| 6A: contracts/profit | `python3 experiments/calculate_professor_loan_economics.py` | [Economics](docs/professor_loan_economics.md) |
| 6B: bank cutoff | `python3 experiments/run_professor_lending_threshold.py` | [Cutoff](docs/professor_lending_threshold.md) |
| 7A: true-risk reference | `python3 experiments/run_professor_oracle_comparison.py` | [Oracle](docs/professor_oracle_comparison.md) |
| 7B: blind comparison | `python3 experiments/run_professor_group_blind_comparison.py` | [Aware versus blind](docs/professor_group_blind_comparison.md) |
| 8A: five-draw pilot | `python3 experiments/run_professor_noise_pilot.py` | [Pilot stage record](docs/professor_noise_robustness_pilot.md) |
| 8B: 200-draw study | `python3 experiments/run_professor_noise_full.py` | [Full study](docs/professor_noise_robustness_full.md) |

Use the full-study `--resume` option only when continued computation is intended.
Its `--summarize-only` option reads saved records without fitting or drawing noise,
but regenerates summary files. Resume requires matching scientific provenance.
The full-study wrapper validates pilot reuse; an old direct pilot resume can
reject the enlarged source inventory, as explained in the full-study document.

### Dashboard (Version 1)

```bash
python3 -m streamlit run dashboard/app.py
```

Open `http://localhost:8501`. This seven-page dashboard explores the **V1
approval framework**, not the professor model. Saved benchmark pages need
locally generated artifacts. See [dashboard guide](docs/dashboard.md) and
[V1/V2 commands](docs/running_v1_v2.md).

## Data and results

| Location | Contents |
|---|---|
| `data/synthetic/professor_model/baseline/` | 100,000 bank/private applicant rows, truth, raw PN, split, aware/blind predictions, CSV/Parquet companions and metadata |
| `data/processed/professor_model/` | 20,000 evaluation contracts, bank/true profit, aware/blind/oracle lending decisions |
| `results/tables/professor_model/` | Canonical and stage-specific CSV summaries |
| `results/tables/professor_model/noise_robustness_full/` | Five CSV tables for the 200-noise-draw study |
| `results/metrics/professor_model/noise_robustness_full/` | Immutable replication records, attempts, freezes, manifest and completion report |
| `reports/professor_model/step8b_progress/` | Shareable report, small preview, charts, build code and evidence |
| `reports/professor_model/drive_upload/` | CSV-bundle generator, inventory, instructions; large ZIP stays local |

Full generated datasets, numerical experiment outputs, and sharing ZIPs are
**not committed**. A fresh clone contains scientific code, configurations,
documentation, tests, and curated presentation artifacts, not all local data.
The [data guide](docs/professor_model_data_guide.md) explains the key files,
information boundary, and Google Drive sharing workflow.

## Validation status

Latest recorded full suite: **624 passed, 4 failed, 32 existing warnings**.
Professor-model suite: **344 passed with no warnings**.

```bash
python3 -m pytest -q tests/test_professor_*.py
python3 -m pytest -q
```

The four failures are existing V2 corrected-checkpoint lookups: the loader
uses current Git HEAD in run IDs, while saved records use an earlier revision.
They are not suppressed or repaired by changing scientific results.
Artifact-dependent tests also require matching local evidence. The repository
is not all-green. See [current status](docs/PROJECT_STATUS.md).

## Research safeguards

- Synthetic evidence cannot establish real-world mortgage discrimination.
- Observed or adjusted disparities are not automatically causal effects.
- Hidden truth/noise belongs to the evaluator, never lender features.
- V1/V2 baseline models exclude protected attributes; the professor model's
  group-aware OLS intentionally includes B as a controlled comparison.
- Clipping preserves probability bounds but does not calibrate predictions.
- Signed oracle comparisons across different cutoff rankings are not
  unrestricted optimal-regret bounds or fairness guarantees.
- Keep empirical noise-outcome spreads, MCSEs, and real-world uncertainty distinct.
- Preserve stage-specific results and provenance instead of silently retuning.

## Next research decisions

Review noise scale, the stylized contract, and indivisible clipped-zero cutoffs
with the professor. Predeclare a matched B-coefficient intervention before
making broader mechanism claims. Alternative noise levels or independent
populations require separate experiments. Resolve V2 checkpoint identity
tests separately. No HMDA work is implemented for the professor family.

## License

[MIT](LICENSE).
