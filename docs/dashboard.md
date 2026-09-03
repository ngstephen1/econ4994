# Interactive Synthetic Fair-Lending Dashboard

## Purpose

The Streamlit dashboard is a presentation and sensitivity-analysis layer around
the validated synthetic generator, statistical recovery analysis, and
machine-learning benchmark. It helps users ask how direct race effects and
upstream economic inequality produce different observed, adjusted, and
predicted Black–White approval gaps.

The dashboard does not estimate real-world discrimination and is not a lending
tool. Its direct and upstream mechanisms are researcher-controlled synthetic
treatments.

## Architecture

The user interface is intentionally thin:

```text
dashboard/app.py and dashboard/pages/
                    ↓
src/fair_lending/dashboard/ presentation services
                    ↓
existing simulation · analysis · model-evaluation APIs
```

Core scientific behavior remains in `src/fair_lending/simulation/`,
`src/fair_lending/analysis/`, and `src/fair_lending/models/`. The dashboard
does not duplicate the approval equation or regression specifications.

Reusable dashboard services provide:

- deterministic in-memory generation and bounded treatment overrides;
- centralized statistical/ML result loading and controlled missing-file errors;
- the existing statsmodels model sequence and standardized contrast;
- consistent percentage-point, percentage, and currency formatting;
- Plotly figures, exports, and cross-page session-state keys.

## Pages

1. **Research Overview** separates the main and statistical research questions
   and describes the four simulated worlds in accessible language.
2. **Simulation Lab** generates an active sample, compares White and Black
   distributions and approval rates, exposes synthetic ground truth, runs one
   lightweight sensitivity curve, and constructs downloads.
3. **Statistical Analysis** fits Models 0–3 to the active sample and displays
   the Black coefficient, odds ratio, confidence interval, p-value,
   standardized probability contrast, and coefficient path.
4. **ML Benchmark** displays saved Prompt 7 performance, race-blind versus
   race-aware gaps, group audits, probability recovery, calibration, and a
   no-retraining global-threshold explorer.
5. **Mechanism Comparison** combines saved Prompt 6 and Prompt 7 results in a
   four-scenario mechanism matrix and result-dependent interpretation.
6. **Methodology** explains the DGP, frozen intercept, statistical and ML
   methods, limitations, future HMDA phase, and all 24 persisted fields.
7. **Sensitivity Experiments** reads the precomputed Monte Carlo summaries,
   figures, thresholds, and modal mechanism signatures. It never launches or
   resumes simulation work from the web interface.

## Interactive simulations

The Simulation Lab accepts:

- one of the four fixed scenarios;
- mild, moderate, or strong effect level;
- 1,000, 5,000, 10,000, 50,000, or 100,000 applications;
- a nonnegative integer random seed.

Generation occurs only after the user selects **Generate simulation**. The
default is 10,000 rows with seed 4994. Data remain in memory and are stored in
Streamlit session state so they can be used on the statistical page without
regeneration.

### Custom treatment controls

The advanced mode exposes only four bounded experimental parameters:

- Black direct effect from -0.50 through 0.00 log odds;
- conditional Black income multiplier from 0.70 through 1.00;
- Black credit-score location shift from -75 through 0 points;
- conditional Black liquid-assets multiplier from 0.30 through 1.00.

Overrides are applied to a deep copy of the resolved configuration. They never
modify `baseline.yaml`, `effect_sizes.yaml`, or scenario YAML files. Custom
settings continue to use the frozen fair-baseline intercept, because
recalibrating each scenario would partially offset the experimental treatment.

## Benchmark mode versus live mode

The Simulation Lab and Statistical Analysis pages are live: they operate on the
active, possibly custom, generated sample. The ML page is benchmark mode: it
reads the validated Prompt 7 tables and held-out probabilities rather than
retraining models during Streamlit reruns.

Optional live logistic training was deliberately deferred. This keeps the
dashboard stable and preserves a clear distinction between exploratory custom
simulation and the predeclared ML benchmark.

## Threshold explorer

`experiments/run_ml_benchmark.py` now saves
`results/metrics/ml_test_predictions.parquet`. The file contains the held-out
application identifier, race, simulated decision, synthetic true probability,
and predicted probability for each scenario/model/feature-regime combination.
Moving the dashboard threshold from 0.20 through 0.80 recomputes classifications
and group metrics from those probabilities. It does not refit or retune a model,
and it applies one threshold to every group.

## Caching and session state

- `st.cache_data` keys deterministic generated data by the immutable simulation
  request, including custom treatments.
- Static result tables and held-out predictions are cached after loading.
- The current data, metadata, summary, request, and statistical results use
  explicit session-state keys shared across pages.
- Changing scenario, effect level, row count, seed, or a custom parameter creates
  a distinct generation request. Merely changing a plot or threshold does not
  regenerate data or retrain models.

## Reproducibility and exports

Every live simulation exposes:

- scenario, effect level, sample size, and seed;
- configured direct and upstream treatment values;
- frozen approval intercept;
- configuration fingerprint and random-stream strategy;
- whether the run was persisted (`false` for dashboard runs).

The active sample can be downloaded as CSV or Parquet. A one-row summary CSV and
the complete resolved reproducibility configuration JSON are also available.
Download construction is in memory; the dashboard does not automatically add
datasets to `data/synthetic/`.

## Run locally

After the repository setup in the README:

```bash
python3 -m streamlit run dashboard/app.py
```

The default local address is `http://localhost:8501`.

## Required generated artifacts

Generated results are intentionally ignored by Git. The benchmark and
mechanism pages expect the following after running the documented experiments:

```bash
python3 experiments/run_statistical_recovery.py
python3 experiments/run_ml_benchmark.py
python3 experiments/run_sensitivity.py --experiment all --resume --workers 2
```

The centralized loader supports:

- `race_approval_gaps.csv`
- `logit_black_effects.csv`
- `statistical_recovery.csv`
- `ml_benchmark.csv`
- `ml_group_audit.csv`
- `ml_race_regime_comparison.csv`
- `ml_oracle_benchmark.csv`
- `ml_calibration_bins.csv`
- `ml_group_calibration_bins.csv`
- `statistical_ml_comparison.csv`
- `ml_test_predictions.parquet`
- `sensitivity_direct_summary.csv`
- `sensitivity_upstream_summary.csv`
- `sensitivity_mixed_summary.csv`
- `sensitivity_sample_size_summary.csv`
- `monte_carlo_detection.csv`
- `monte_carlo_coverage.csv`
- `mechanism_signatures.csv`
- `detection_thresholds.csv`
- `sensitivity_paper_summary.csv`

When an artifact is absent, the relevant page gives the experiment command
needed to recreate it rather than exposing a raw `FileNotFoundError`.

## Interpretation cautions

- An observed outcome gap does not identify its mechanism.
- A race-blind model can reproduce upstream disparity because shifted financial
  variables are available to it.
- A race-blind model can miss a direct effect when race is withheld and no
  modeled proxy carries that signal.
- Statistical adjustment can condition away an upstream pathway; this does not
  show that upstream inequality is unimportant.
- Logistic coefficients are non-collapsible, so coefficient shrinkage is not a
  formal mediation estimate.
- Group error rates measure agreement with simulated lender decisions. No
  independent repayment or normative qualification outcome exists.
- Synthetic findings illustrate behavior under chosen assumptions and cannot
  prove that real-world discrimination exists.

## Current limitations and future HMDA integration

The dashboard does not include live ML training, race-specific thresholds,
SHAP, neural networks, fairness-constrained optimization, or HMDA data. A later
HMDA phase may reuse presentation and analysis patterns, but its observational
results must use disparity language because synthetic ground truth will no
longer be available.
