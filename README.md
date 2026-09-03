<p align="center">
  <img src="docs/assets/fair-lending-logo.svg" alt="Fair Lending Lab — Economics, Simulation, Equity, and Reproducible Research Software" width="100%">
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img alt="Python 3.11+" src="https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white"></a>
  <a href="docs/PROJECT_STATUS.md"><img alt="Phase: ML benchmark" src="https://img.shields.io/badge/phase-ML_benchmark-159A9C?style=flat-square"></a>
  <a href="tests"><img alt="Tests: 63 passing" src="https://img.shields.io/badge/tests-63_passing-2E7D32?style=flat-square"></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-E87722?style=flat-square"></a>
</p>

<p align="center">
  <strong>A synthetic-first study of fairness and disparities in mortgage lending decisions.</strong><br>
  Virginia Tech ECON 4994 · Economics of Diversity, Equity, and Inclusion
</p>

---

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

The Streamlit dashboard and HMDA analysis are intentionally reserved for later
phases. Prediction-error metrics relative to lender decisions are not presented
as proof of normative fairness.

## Quick start

Python 3.11 or newer is required.

```bash
git clone https://github.com/ngstephen1/econ4994.git
cd econ4994

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
pytest
```

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
| `configs/simulation/` | Baseline calibration, treatment library, and scenario switches |
| `src/fair_lending/simulation/` | Generator, approval DGP, calibration, diagnostics, and validation |
| `src/fair_lending/analysis/` | Descriptive estimands, statsmodels logits, and standardized contrasts |
| `src/fair_lending/models/` | ML features, splitting, pipelines, evaluation, and race-group audit |
| `experiments/` | Reproducible research and calibration entry points |
| `data/synthetic/` | Generated Parquet datasets; ignored by Git |
| `data/external/` | Future external inputs, including HMDA; ignored by Git |
| `results/` | Generated metrics, tables, and figures; ignored by Git |
| `tests/` | Configuration, generator, scenario, and calibration tests |
| `notebooks/` | Exploration only; never the source of core research logic |
| `dashboard/` | Future Streamlit interface |
| `docs/` | Research design, assumptions, calibration, and project status |

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

## Documentation

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
- [ ] Research-relevant fairness analysis
- [ ] Interactive Streamlit dashboard
- [ ] Carefully scoped 2024 HMDA application

## Later HMDA application

After the synthetic experiments are complete, the framework may be applied to
2024 Home Mortgage Disclosure Act data. Real-data findings will use terms such
as **disparity**, **adjusted disparity**, or **demographic gap**. An observed
HMDA difference will not automatically be labeled discrimination.

## License

Released under the [MIT License](LICENSE).
