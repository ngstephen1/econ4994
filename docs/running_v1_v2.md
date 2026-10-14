# Preserved Version 1 and Version 2 Run Guide

These workflows are separate from the [current professor model](../README.md#current-professor-model).
Use the [root setup instructions](../README.md#setup) first. Commands are
manual reproduction entry points, not a recommendation to recompute completed
or expensive experiments. Consult each stage's documentation before writing
outputs; keep original evidence and frozen scientific parameters intact.

## Version 1: approval disparities

| Purpose | Command | Scope |
|---|---|---|
| Generate a development sample | `python3 -m fair_lending.simulation.generate --scenario fair_baseline --effect-level moderate --rows 10000 --seed 4994` | First generation may build the one-million-row calibration if the frozen artifact is absent |
| Statistical recovery | `python3 experiments/run_statistical_recovery.py` | Four deterministic 100,000-row scenarios |
| Prediction benchmark | `python3 experiments/run_ml_benchmark.py` | Existing race-blind/race-aware feature regimes |
| Quick sensitivity smoke check | `python3 experiments/run_sensitivity.py --quick` | Small development design |
| Full saved sensitivity design | `python3 experiments/run_sensitivity.py --experiment all --resume --workers 2` | 3,140 unique runs; expensive; requires compatible provenance |
| Interactive exploration | `python3 -m streamlit run dashboard/app.py` | Seven-page V1 dashboard at localhost:8501 |

See [generator design](generator_design.md), [statistical recovery](statistical_recovery.md),
[ML benchmark](ml_benchmark.md), [sensitivity](sensitivity_analysis.md), and
[dashboard](dashboard.md). The dashboard does not implement V2 or the professor model.

## Version 2: repayment risk and economic allocation

| Stage | Manual entry point | Documentation |
|---|---|---|
| Calibrated development population | `python3 experiments/run_economic_population_calibration.py` | [Calibration](economic_population_calibration.md) |
| Traditional historical-payment logit | `python3 experiments/run_traditional_lender.py` | [Traditional lender](traditional_lender_risk_model.md) |
| Flexible repayment estimator | `python3 experiments/run_ml_repayment_benchmark.py` | [ML benchmark](ml_repayment_risk_benchmark.md) |
| Matched nonlinear true-risk world | `python3 experiments/run_nonlinear_risk_benchmark.py` | [Nonlinear sensitivity](nonlinear_risk_sensitivity.md) |
| Oracle/traditional/ML whole-request portfolios | `python3 experiments/run_portfolio_allocation_benchmark.py` | [Portfolio allocation](portfolio_allocation_benchmark.md) |
| Direct belief distortion | `python3 experiments/run_direct_belief_distortion.py` | [Direct mechanism](direct_belief_discrimination.md) |
| Upstream opportunity treatment | `python3 experiments/run_systemic_opportunity.py` | [Systemic mechanism](systemic_opportunity_discrimination.md) |

The corrected systemic repeated-seed study is **not complete**. Its five-run,
360-portfolio checkpoint is separate from provisional legacy successes.
Do not launch its remaining 45 runs without a new decision; first address the
current revision-dependent checkpoint lookup failures. See [recovery audit](systemic_monte_carlo_recovery.md).

## Data and tests

Existing V1/V2 CSV summaries are under `results/tables/`; V2 applicant,
loan-option, hidden-truth, and outcome tables are under
`data/synthetic/economic_lending/`, with policy artifacts under
`data/processed/economic_lending/`. Generated files are not in a fresh clone.

Run `python3 -m pytest -q` for the full suite. Some tests require compatible
local artifacts; the [current status](PROJECT_STATUS.md) reports the observed
failures rather than promising an all-green fresh clone.
