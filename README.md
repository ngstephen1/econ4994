<p align="center">
  <img src="docs/assets/fair-lending-logo.svg" alt="Fair Lending Lab — Economics, Simulation, Equity, and Reproducible Research Software" width="100%">
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img alt="Python 3.11+" src="https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white"></a>
  <a href="docs/PROJECT_STATUS.md"><img alt="Phase: Version 2 systemic opportunity" src="https://img.shields.io/badge/phase-v2_systemic_opportunity-159A9C?style=flat-square"></a>
  <a href="tests"><img alt="Tests: 230 passing" src="https://img.shields.io/badge/tests-230_passing-2E7D32?style=flat-square"></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-E87722?style=flat-square"></a>
</p>

<p align="center">
  <strong>A synthetic-first study of fairness and disparities in mortgage lending decisions.</strong><br>
  Virginia Tech ECON 4994 · Economics of Diversity, Equity, and Inclusion
</p>

---

## Menu

| Start here | Run the project | Understand the research | Browse outputs |
|---|---|---|---|
| [Research question](#the-research-question) | [Quick start](#quick-start) | [How the generator works](#how-the-generator-works) | [Repository map](#repository-map) |
| [Four experimental worlds](#four-experimental-worlds) | [Launch the dashboard](#launch-the-interactive-dashboard) | [Calibration snapshot](#calibration-snapshot) | [Documentation index](#documentation-index) |
| [Current capabilities](#what-is-implemented) | [Run the tests](#verify-the-installation) | [Research safeguards](#research-safeguards) | [Data preview](docs/data_and_results_preview.md) |
| [Project status](docs/PROJECT_STATUS.md) | [Reproduce the benchmarks](#reproduce-the-research-benchmarks) | [Methodology](docs/research_design.md) | [Results summaries](docs/dataset_summary_statistics.md) |

### New here? Follow one of these paths

| Your goal | Recommended path |
|---|---|
| Understand the project in five minutes | Read the [research question](#the-research-question), [four worlds](#four-experimental-worlds), and [research safeguards](#research-safeguards). |
| Explore without reading the code first | Complete [Quick start](#quick-start), then [launch the dashboard](#launch-the-interactive-dashboard). |
| Inspect the generated data manually | Open the [data and results preview](docs/data_and_results_preview.md), [dataset inventory](docs/dataset_inventory.md), and [summary statistics](docs/dataset_summary_statistics.md). |
| Reproduce the scientific results | Run the [statistical and ML benchmarks](#reproduce-the-research-benchmarks), then read their linked methodology documents. |
| Understand or modify the implementation | Start with the [repository map](#repository-map), then read [generator design](docs/generator_design.md) and the tests beside each module. |
| See what is finished and what comes next | Open [PROJECT_STATUS.md](docs/PROJECT_STATUS.md) and the [roadmap](#roadmap). |

## Version Guide

| Version | Status | Main question | Entry point |
|---|---|---|---|
| Version 1 approval benchmark | Preserved at Git tag `v1-approval-benchmark` | How do approval disparities appear under direct, upstream, mixed, and fair synthetic mechanisms? | [Research design](docs/research_design.md) |
| Version 2 economic lending model | Upstream systemic-opportunity experiment completed | How do repayment risk, expected profit, discrimination mechanisms, and budget-constrained allocation shape lending? | [Economic lending spec](docs/economic_lending_spec.md) · [Systemic experiment](docs/systemic_opportunity_discrimination.md) |

## The research question

> Are some demographic groups more likely to be denied mortgages after
> accounting for differences in borrower and loan characteristics?

This capstone combines economics, simulation, statistics, machine learning,
fairness evaluation, and reproducible research software. Prediction is used as
a tool for understanding disparity mechanisms—not as an end in itself.

## Why synthetic data comes first

An observed mortgage gap can reflect financial characteristics, direct
differential treatment, upstream inequality, geography, omitted variables, or
several mechanisms at once. Observational data does not reveal which mechanism
is true by itself.

This project begins with controlled synthetic worlds where the complete
data-generating process is known. Statistical and machine-learning methods can
therefore be evaluated against ground truth before the framework is applied to
real HMDA data.

### Four experimental worlds

| Scenario | Upstream group differences | Direct race term | Intended signal |
|---|:---:|:---:|---|
| `fair_baseline` | No | No | Sampling variation only |
| `direct_discrimination` | No | Yes | Conditional direct disparity |
| `upstream_inequality` | Yes | No | Disparity through financial mediators |
| `mixed_mechanism` | Yes | Yes | Both pathways operating together |

White is the reference group and Black is the initial focal comparison. Direct
and upstream effects are researcher-inserted experimental treatments—not
estimates of real-world discrimination.

## What is implemented

- A vectorized, seed-controlled 24-field mortgage-application generator.
- Config-driven demographic, financial, context, property, and loan processes.
- Correlated latent factors without persisting unobserved variables.
- Fair, direct, upstream, and mixed scenarios with explicit ground truth.
- A logistic approval mechanism with stored true approval probabilities.
- One-time fair-baseline intercept calibration, frozen across scenarios.
- Structured schema, distribution, identity, clipping, and scenario validation.
- Deterministic DTI diagnostics and a documented population recalibration.
- Parquet output with adjacent configuration and reproducibility metadata.
- A self-contained automated test suite.
- Descriptive Black-White gaps with independent-proportion confidence intervals.
- Four nested statsmodels logit specifications matched to the known DGP.
- Standardized adjusted probability contrasts and direct-effect recovery tables.
- Three reproducible figures for the first 100,000-row statistical experiment.
- Race-blind and race-aware ML regimes with leakage-safe sklearn pipelines.
- Logistic, random-forest, and histogram-gradient-boosting benchmarks.
- Held-out group audits, disparity-reproduction metrics, probability recovery,
  calibration bins, and a non-trained synthetic oracle reference.
- A seven-page Streamlit research dashboard for in-memory simulation, active-sample
  regression, saved ML benchmark exploration, and cross-mechanism comparison.
- Bounded custom synthetic treatments, a fixed-seed sensitivity curve, dataset
  exports, and a no-retraining global-threshold explorer.
- A resumable 3,140-run Monte Carlo framework spanning direct, upstream, mixed,
  sample-size, and detection experiments with common-random-number seeds.
- Monte Carlo bias, RMSE, coverage, detection, false-positive, mechanism-signature,
  and result-derived threshold summaries with 13 research figures.
- A read-only dashboard page for exploring precomputed sensitivity results.
- A separate Version 2 economic lending baseline package for conditional
  per-period repayment truth, survival-weighted cash flows, expected profit,
  and budget-constrained fixed-request allocation.
- A calibrated Version 2 applicant population with group-invariant finances,
  a declared logistic true-risk DGP, absorbing repayment histories, and
  applicant-level train/validation/evaluation cohorts.
- A traditional lender logit estimated from at-risk historical payment rows,
  with truth recovery, oracle comparison, survival-selection diagnostics,
  group auditing, and perceived expected-profit assessment.
- One controlled histogram-gradient-boosting benchmark using the identical
  observable information set, payment target, and applicant-level cohorts.
- Validation-only ML selection, paired probability recovery, calibration,
  group auditing, break-even disagreement, expected-profit error, and a
  true-risk-oracle reference.
- A separate nonlinear true-risk sensitivity on the identical applicants and
  contracts, with frozen DTI curvature, credit-LTV interaction, asset-buffer
  terms, paired repayment draws, and cross-world model comparison.
- Exact population-scale binary portfolio allocation for oracle, traditional,
  and ML policies under common nonbinding, moderate, and tight capital budgets.
- True expected-profit regret, realized-profit illustration, overlap,
  allocation disagreements, tail losses, and neutral group audits.
- A matched direct-discrimination experiment that shifts Group B perceived
  repayment log odds after frozen model estimation while preserving true risk,
  outcomes, contracts, budgets, and neutral controls.
- A separate matched systemic-opportunity experiment in which an upstream
  group penalty changes opportunity access, employment, income, and assets
  while the final traditional and ML lenders remain group-blind.
- Additive and nonlinear true-risk comparisons, opportunity-switcher pathways,
  fixed-budget oracle/fitted allocations, Group A portfolio spillovers, and a
  controlled descriptive funding audit for the systemic experiment.

HMDA analysis remains reserved for a later phase. Prediction-error metrics
relative to lender decisions are not presented as proof of normative fairness.

## Quick start

Python 3.11 or newer is required.

```bash
git clone https://github.com/ngstephen1/econ4994.git
cd econ4994

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

### Verify the installation

```bash
python3 -c "import fair_lending; print('fair_lending import OK')"
python3 -m pytest -q
```

If both commands succeed, the package, scientific dependencies, and test
environment are available.

### First synthetic dataset

Generate and validate a 10,000-row fair-baseline dataset:

```bash
python3 -m fair_lending.simulation.generate \
  --scenario fair_baseline \
  --effect-level moderate \
  --rows 10000 \
  --seed 4994
```

The command writes:

```text
data/synthetic/
├── synthetic_fair_baseline_moderate_n10000_seed4994.parquet
└── synthetic_fair_baseline_moderate_n10000_seed4994.metadata.json

results/
├── metrics/validation_synthetic_fair_baseline_moderate_n10000_seed4994.json
└── tables/summary_synthetic_fair_baseline_moderate_n10000_seed4994.csv
```

Generated data and results are ignored by Git. The metadata contains the
resolved configuration, config fingerprint, random-stream assignments,
intercept provenance, package version, timestamp, and dataset shape.

> On a fresh clone, the first generation run creates the local one-million-row
> calibration artifact automatically. Later runs reuse it;
> `--recalibrate-intercept` deliberately rebuilds it.

### Version 2 economic population

Generate the deterministic 10,000-applicant Version 2 development population:

```bash
python3 experiments/run_economic_population_calibration.py
```

This writes separate applicant, loan-option, hidden-truth, and realized-outcome
Parquet files under `data/synthetic/economic_lending/v2_baseline/`. The
100,000-row calibration check can be rerun without saving raw calibration data:

```bash
python3 experiments/run_economic_population_calibration.py --calibration --no-save
```

See [Version 2 population calibration](docs/economic_population_calibration.md)
for the dependency structure, exact true-risk formula, results, and limitations.

Fit and evaluate the traditional lender model after generating the population:

```bash
python3 experiments/run_traditional_lender.py
```

See [Traditional lender risk model](docs/traditional_lender_risk_model.md) for
the information boundary, fitted coefficients, oracle comparison, and economic
diagnostics.

Run the frozen systemic-opportunity experiment after the Version 2 dependencies
are installed:

```bash
python3 experiments/run_systemic_opportunity.py
```

See [Systemic opportunity discrimination](docs/systemic_opportunity_discrimination.md)
for the upstream equation, matched-world design, pathway results, allocation
effects, controlled-audit caveat, and limitations.

Compare the frozen traditional lender with the flexible repayment-risk model:

```bash
python3 experiments/run_ml_repayment_benchmark.py
```

The benchmark searches eight predeclared HistGB settings on the historical
validation cohort and reserves evaluation outcomes for final comparison. Under
the current additive-logistic truth, the traditional model performs better,
which is an expected and informative result. See the
[V2 ML repayment-risk benchmark](docs/ml_repayment_risk_benchmark.md).

Run the frozen nonlinear misspecification sensitivity:

```bash
python3 experiments/run_nonlinear_risk_benchmark.py
```

This does not overwrite the additive world. It writes separate nonlinear data,
metrics, tables, models, and figures. See
[Nonlinear risk sensitivity](docs/nonlinear_risk_sensitivity.md).

Allocate the frozen evaluation policies under common capital constraints:

```bash
python3 experiments/run_portfolio_allocation_benchmark.py
```

The runner uses SciPy/HiGHS binary MILP and does not retrain either risk model.
See [Portfolio allocation benchmark](docs/portfolio_allocation_benchmark.md).

Run the direct belief-distortion experiment:

```bash
python3 experiments/run_direct_belief_distortion.py
```

This creates separate treatment artifacts and does not overwrite the neutral
portfolio benchmark. See
[Direct belief discrimination](docs/direct_belief_discrimination.md).

### Reproduce the research benchmarks

Run the deterministic four-scenario statistical recovery experiment:

```bash
python3 experiments/run_statistical_recovery.py
```

This generates one 100,000-row, seed-4994 dataset per scenario, fits Models
0–3, and writes compact tables, figures, and experiment metadata under
`results/`. See [Statistical recovery](docs/statistical_recovery.md) for the
specifications, observed results, and interpretation cautions.

Run the controlled machine-learning benchmark:

```bash
python3 experiments/run_ml_benchmark.py
```

This reuses the same scenario datasets, makes one stable 60/20/20 split, tunes
three model families on validation log loss, and evaluates race-blind and
race-aware sensitivity regimes on identical test records. See
[ML benchmark](docs/ml_benchmark.md).

Run the full resumable sensitivity and Monte Carlo design:

```bash
python3 experiments/run_sensitivity.py --experiment all --resume --workers 2
```

The routine evaluates 3,140 unique replication/settings and writes each result
atomically, so interrupted work can resume safely. Add `--quick` for a 16-run
installation and pipeline check. See
[Sensitivity analysis](docs/sensitivity_analysis.md) for the grids, estimands,
observed results, and interpretation limits.

### Launch the interactive dashboard

After generating the statistical and ML benchmark artifacts, run:

```bash
python3 -m streamlit run dashboard/app.py
```

The dashboard can generate new samples in memory, including bounded custom
treatments, and rerun the validated statistical specifications. Its ML page uses
saved held-out benchmark probabilities, so moving the classification threshold
does not retrain models. See [Dashboard guide](docs/dashboard.md).

The dashboard opens at `http://localhost:8501` by default. Start with
**Research Overview** for the concepts or **Simulation Lab** to generate an
interactive sample. Generated benchmark files are intentionally ignored by Git.
Run the statistical and ML commands before opening their result pages, and run
the sensitivity command before opening **Sensitivity Experiments** on a fresh
clone.

## How the generator works

```text
protected attributes + exogenous context
                    ↓
       upstream economic conditions
                    ↓
 income · credit · employment · assets · debt
                    ↓
       property and loan application
                    ↓
          DTI · LTV · capacity ratios
                    ↓
       latent underwriting score
                    ↓
    true approval probability → approval draw
```

The generator uses separate NumPy `SeedSequence` streams for demographics,
latent factors, context, financial variables, loans, and approval draws. The
same resolved configuration, seed, and code revision reproduce the same data.

## Calibration snapshot

The current population calibration resolved an excessive DTI boundary mass by
strengthening the income–property dependency without changing mortgage payment
assumptions or experimental treatment sizes.

| Check | Result |
|---|---:|
| Fair-baseline target mean approval probability | `0.8000` |
| Achieved one-million-row calibration mean | `0.8000000003` |
| Original DTI ceiling mass | `10.7893%` |
| Recalibrated DTI ceiling mass | `4.2781%` |
| Predeclared maximum | `5.0000%` |

See [DTI recalibration](docs/dti_recalibration.md) for the diagnosis, rejected
candidates, before/after distributions, and intercept provenance.

## Repository map

| Path | Purpose |
|---|---|
| `README.md` | Entry point, setup instructions, navigation, and project overview |
| `configs/simulation/` | Baseline calibration, treatment library, and scenario switches |
| `configs/economic_lending/` | Standalone Version 2 population, contract, risk, cohort, and randomness calibration |
| `src/fair_lending/simulation/` | Generator, approval DGP, calibration, diagnostics, and validation |
| `src/fair_lending/analysis/` | Descriptive estimands, statsmodels logits, and standardized contrasts |
| `src/fair_lending/models/` | ML features, splitting, pipelines, evaluation, and race-group audit |
| `src/fair_lending/sensitivity/` | Monte Carlo grids, seeds, evaluation, persistence, summaries, thresholds, and figures |
| `src/fair_lending/economic_lending/` | Version 2 conditional repayment-risk, cash-flow, profit, and fixed-request allocation baseline |
| `src/fair_lending/dashboard/` | Reusable dashboard data, simulation, export, chart, and state services |
| `experiments/` | Reproducible statistical, ML, and sensitivity experiment entry points |
| `dashboard/` | Multipage Streamlit research interface |
| `data/synthetic/` | Generated Parquet datasets; ignored by Git |
| `data/external/` | Future external inputs, including HMDA; ignored by Git |
| `results/` | Generated metrics, tables, and figures; ignored by Git |
| `tests/` | Generator, statistical, ML, dashboard, and reproducibility tests |
| `notebooks/` | Exploration only; never the source of core research logic |
| `docs/` | Research design, assumptions, calibration, results, and project status |

### Common entry points

| Task | File or command |
|---|---|
| Generate one synthetic dataset | `python3 -m fair_lending.simulation.generate --help` |
| Run statistical recovery | `python3 experiments/run_statistical_recovery.py` |
| Run the ML benchmark | `python3 experiments/run_ml_benchmark.py` |
| Run the sensitivity design | `python3 experiments/run_sensitivity.py --experiment all --resume --workers 2` |
| Generate the Version 2 population | `python3 experiments/run_economic_population_calibration.py` |
| Fit the Version 2 traditional lender | `python3 experiments/run_traditional_lender.py` |
| Compare traditional and flexible V2 risk estimates | `python3 experiments/run_ml_repayment_benchmark.py` |
| Run the matched nonlinear-risk sensitivity | `python3 experiments/run_nonlinear_risk_benchmark.py` |
| Run the V2 portfolio allocation benchmark | `python3 experiments/run_portfolio_allocation_benchmark.py` |
| Run the direct belief-distortion experiment | `python3 experiments/run_direct_belief_distortion.py` |
| Run the systemic-opportunity experiment | `python3 experiments/run_systemic_opportunity.py` |
| Launch the dashboard | `python3 -m streamlit run dashboard/app.py` |
| Run all tests | `python3 -m pytest -q` |
| Check current phase | [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md) |

> `data/` and `results/` contain locally generated artifacts and are ignored by
> Git. Their `.gitkeep` files preserve the directory layout on a fresh clone.

## Research safeguards

> **A disparity is not automatically evidence of discrimination.**

- Synthetic discrimination exists only where the researcher explicitly inserts
  a direct treatment.
- Adjusted disparities remain model-dependent and are not automatically causal
  estimates.
- Race, ethnicity, and sex are audit/treatment attributes, not ordinary
  baseline underwriting predictors.
- Variables downstream of inequality can become bad controls and hide part of
  the pathway being studied.
- Predictive accuracy and normative fairness are different questions.
- Synthetic findings cannot establish that real-world lenders discriminate.

## Documentation index

| Document | Contents |
|---|---|
| [Project status](docs/PROJECT_STATUS.md) | Current phase, completed work, and next steps |
| [Research design](docs/research_design.md) | Questions, causal structure, and estimands |
| [Synthetic schema](docs/synthetic_schema.md) | Exact 24-field schema and variable roles |
| [Simulation scenarios](docs/simulation_scenarios.md) | Four worlds and expected qualitative behavior |
| [Simulation calibration](docs/simulation_calibration.md) | Numerical assumptions and evidence labels |
| [Generator design](docs/generator_design.md) | Architecture, random streams, DGP, and validation |
| [DTI recalibration](docs/dti_recalibration.md) | Tail diagnosis, candidates, and selected revision |
| [Statistical recovery](docs/statistical_recovery.md) | First descriptive and logistic recovery experiment |
| [ML benchmark](docs/ml_benchmark.md) | Predictive performance, disparity reproduction, and true-probability recovery |
| [Dashboard guide](docs/dashboard.md) | Pages, controls, caching, artifacts, launch instructions, and cautions |
| [Sensitivity analysis](docs/sensitivity_analysis.md) | Monte Carlo design, recovery, detection, thresholds, signatures, and limits |
| [Economic lending spec](docs/economic_lending_spec.md) | Version 2 repayment event, period survival, cash-flow, profit, schema, and allocation baseline |
| [Economic population calibration](docs/economic_population_calibration.md) | Version 2 applicant dependencies, true-risk DGP, cohorts, realized outcomes, and validation results |
| [Traditional lender risk model](docs/traditional_lender_risk_model.md) | At-risk histories, traditional logit, oracle comparison, survival selection, and perceived profit |
| [V2 ML repayment-risk benchmark](docs/ml_repayment_risk_benchmark.md) | Identical-information traditional-versus-HistGB probability and profit comparison |
| [Nonlinear risk sensitivity](docs/nonlinear_risk_sensitivity.md) | Frozen nonlinear DGP, matched repayment streams, model recovery, and economic consequences |
| [Portfolio allocation benchmark](docs/portfolio_allocation_benchmark.md) | Oracle/traditional/ML allocation, scarcity, regret, overlap, tails, and neutral group audit |
| [Direct belief discrimination](docs/direct_belief_discrimination.md) | Log-odds treatment, matched group effects, spillovers, mechanism cost, and model interaction |
| [Systemic opportunity discrimination](docs/systemic_opportunity_discrimination.md) | Explicit upstream opportunity, matched descendants, group-blind lending, switchers, spillovers, and portfolio effects |
| [Data and results preview](docs/data_and_results_preview.md) | Human-readable dataset and CSV excerpts for manual review |
| [Dataset inventory](docs/dataset_inventory.md) | File sizes, schema completeness, categories, and validation identities |
| [Dataset summary statistics](docs/dataset_summary_statistics.md) | Numeric distributions and group outcomes across all four scenarios |

## Roadmap

- [x] Research design and repository scaffold
- [x] Synthetic schema and scenario definitions
- [x] Calibrated, validated synthetic generator
- [x] DTI diagnosis and population recalibration
- [x] Descriptive and statistical disparity analysis
- [x] Machine-learning evaluation
- [x] Synthetic sensitivity and Monte Carlo experiments
- [x] Preserve Version 1 approval benchmark at `v1-approval-benchmark`
- [x] Start Version 2 economic lending baseline
- [x] Calibrate the Version 2 applicant population and true repayment DGP
- [x] Estimate traditional lender repayment risk from historical payment data
- [x] Compare traditional and flexible repayment-risk estimation
- [x] Test both estimators in a separate nonlinear true-risk world
- [x] Compare frozen policies under common budget-constrained allocation
- [x] Isolate direct discrimination through post-model repayment-belief distortion
- [ ] Research-relevant fairness analysis
- [x] Interactive Streamlit dashboard
- [ ] Carefully scoped 2024 HMDA application

## Later HMDA application

After the synthetic experiments are complete, the framework may be applied to
2024 Home Mortgage Disclosure Act data. Real-data findings will use terms such
as **disparity**, **adjusted disparity**, or **demographic gap**. An observed
HMDA difference will not automatically be labeled discrimination.

## License

Released under the [MIT License](LICENSE).
