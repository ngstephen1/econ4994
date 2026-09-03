# Fairness and Disparities in Mortgage Lending Decisions

This repository supports a Virginia Tech ECON 4994 undergraduate research
capstone in the Economics of Diversity, Equity, and Inclusion minor. The
project studies whether mortgage lending decisions differ across demographic
groups after accounting for borrower and loan characteristics.

## Research objective

The project combines economics, simulation, statistics, machine learning,
fairness evaluation, and reproducible research software. Prediction is a tool
for studying disparity mechanisms; maximizing predictive performance is not
the primary research objective.

## Synthetic-first design

The first stage will construct synthetic mortgage-lending environments with
known data-generating processes. These controlled simulations will be used to
evaluate whether statistical and machine-learning methods recover deliberately
specified mechanisms, including a fair baseline, direct discrimination,
upstream inequality, and a mixed mechanism.

The version-1 synthetic generator and its validation layer are implemented.
Statistical models, machine-learning models, fairness analyses, and research
experiments have not yet been implemented. The repository scaffold, variable
schema, causal order, scenario definitions, numerical calibration, and
generator behavior are documented and represented in configuration files.

## Later HMDA application

After the synthetic experiments are implemented and validated, the framework
may be applied to 2024 Home Mortgage Disclosure Act (HMDA) data. Observed
differences in HMDA will be described as disparities or adjusted disparities,
not automatically interpreted as discrimination. HMDA data is not part of the
current implementation phase and is not stored in this repository.

## Repository structure

```text
configs/simulation/       Baseline calibration, treatment library, and scenarios
data/synthetic/           Generated synthetic data (ignored by Git)
data/processed/           Generated processed data (ignored by Git)
data/external/            External data, including future HMDA inputs (ignored)
src/fair_lending/         Reusable research package
  simulation/             Synthetic data-generating processes
  analysis/               Descriptive and statistical analysis
  models/                 Predictive model training and evaluation
  fairness/               Research-relevant fairness measures
  utils/                  Shared, lightweight utilities
experiments/              Reproducible experiment entry points
notebooks/                Exploratory notebooks, not the source of core logic
dashboard/                Future Streamlit interface
results/metrics/          Generated metrics (ignored by Git)
results/figures/          Generated figures (ignored by Git)
results/tables/           Generated tables (ignored by Git)
docs/                     Research design and project status
tests/                    Automated tests
```

Empty generated-data and output directories are retained with `.gitkeep`
files. Their generated contents are excluded from version control.

## Setup

Python 3.11 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
pytest
```

Project dependencies are declared in `pyproject.toml`. The requirements file
installs the package in editable mode together with development dependencies.

After installation, a small reproducible dataset can be generated with:

```bash
python3 -m fair_lending.simulation.generate \
  --scenario fair_baseline \
  --effect-level moderate \
  --rows 10000 \
  --seed 4994
```

The command writes Parquet data and adjacent metadata, then saves structured
validation output under `results/`. The first normal run reuses the calibrated
intercept artifact; pass `--recalibrate-intercept` only when deliberately
rebuilding that one-million-row fair-baseline calibration artifact.

## Current status

The research context, repository scaffold, simulation design, version-1
calibration, generator, and initial generator-validation workflow are
established. The original DTI-tail issue has been diagnosed and recalibrated
below the predeclared 5% ceiling-mass threshold, and the canonical validation
runs pass their structural checks. The synthetic DGP is ready for the next
modeling phase. See
[docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md),
[docs/research_design.md](docs/research_design.md),
[docs/synthetic_schema.md](docs/synthetic_schema.md), and
[docs/simulation_scenarios.md](docs/simulation_scenarios.md). Numerical
assumptions and their evidence labels are in
[docs/simulation_calibration.md](docs/simulation_calibration.md).
Implementation details are in
[docs/generator_design.md](docs/generator_design.md), and the population change
is documented in [docs/dti_recalibration.md](docs/dti_recalibration.md).
