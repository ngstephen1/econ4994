"""Deterministic tests for repeated-seed systemic-opportunity uncertainty."""

from __future__ import annotations

import json
import itertools

import numpy as np
import pandas as pd
import pytest

from fair_lending.economic_lending.config import load_economic_config
from fair_lending.economic_lending.ml import ML_FEATURES
from fair_lending.economic_lending.portfolio import solve_fixed_request_portfolio
from fair_lending.economic_lending.repayment import true_repayment_probabilities
from fair_lending.economic_lending.systemic import (
    draw_systemic_population_inputs,
    generate_systemic_world,
    load_systemic_config,
)
from fair_lending.economic_lending.systemic_mc import (
    MC_ANALYSIS_VERSION,
    PRIMARY_ESTIMANDS,
    expected_record_counts,
    load_systemic_mc_config,
    load_valid_replication,
    monte_carlo_summary,
    neutral_budgets,
    records_to_frames,
    replication_id,
    replication_seeds,
    summarize_scalar,
    write_replication_atomic,
)
from fair_lending.economic_lending.traditional import (
    TRADITIONAL_LENDER_FEATURES,
    build_policy_assessments,
)


@pytest.fixture(scope="module")
def setup_worlds():
    baseline = load_economic_config()
    systemic = load_systemic_config()
    seeds = replication_seeds(499417, 0)
    draws = draw_systemic_population_inputs(600, baseline, systemic, seed=seeds.population_seed)
    worlds = {strength: generate_systemic_world(draws, baseline, systemic, strength) for strength in [0.0, .1, .2, .4]}
    return baseline, systemic, seeds, draws, worlds


def test_scientific_parameters_are_identical_across_replications() -> None:
    config = load_systemic_mc_config()
    first = load_systemic_config()
    second = load_systemic_config()
    assert first["metadata"]["config_fingerprint"] == second["metadata"]["config_fingerprint"] == config["systemic_config_fingerprint"]


def test_top_level_replication_seeds_are_independent() -> None:
    seeds = [replication_seeds(499417, index) for index in range(20)]
    assert len({seed.replication_seed for seed in seeds}) == 20
    assert len({seed.population_seed for seed in seeds}) == 20


def test_same_replication_seed_reproduces_population_exactly() -> None:
    baseline = load_economic_config(); systemic = load_systemic_config(); seed = replication_seeds(499417, 3)
    first = draw_systemic_population_inputs(100, baseline, systemic, seed=seed.population_seed)
    second = draw_systemic_population_inputs(100, baseline, systemic, seed=seed.population_seed)
    pd.testing.assert_frame_equal(first, second, check_exact=True)


def test_different_replication_seeds_change_population() -> None:
    baseline = load_economic_config(); systemic = load_systemic_config()
    first = draw_systemic_population_inputs(100, baseline, systemic, seed=replication_seeds(499417, 1).population_seed)
    second = draw_systemic_population_inputs(100, baseline, systemic, seed=replication_seeds(499417, 2).population_seed)
    assert not first.equals(second)


def test_common_random_numbers_are_preserved_across_strengths(setup_worlds) -> None:
    _, _, _, _, worlds = setup_worlds
    columns = ["financial_stability_latent", "creditworthiness_latent", "opportunity_uniform"]
    for strength in [.1, .2, .4]:
        pd.testing.assert_frame_equal(worlds[0.0]["mechanism"][columns], worlds[strength]["mechanism"][columns])


def test_group_a_dgp_and_true_risk_are_invariant(setup_worlds) -> None:
    baseline, _, _, _, worlds = setup_worlds
    a = worlds[0.0]["applicants"]["group"].astype(str).eq("A")
    truth0 = true_repayment_probabilities(worlds[0.0]["applicants"], worlds[0.0]["loan_options"], baseline["true_risk"])
    for strength in [.1, .2, .4]:
        pd.testing.assert_frame_equal(worlds[0.0]["applicants"].loc[a].reset_index(drop=True), worlds[strength]["applicants"].loc[a].reset_index(drop=True))
        truth = true_repayment_probabilities(worlds[strength]["applicants"], worlds[strength]["loan_options"], baseline["true_risk"])
        np.testing.assert_array_equal(truth0.loc[a, "repayment_probability_per_period_true"], truth.loc[a, "repayment_probability_per_period_true"])


def test_group_b_opportunity_access_is_monotone_for_matched_draws(setup_worlds) -> None:
    _, _, _, _, worlds = setup_worlds
    b = worlds[0.0]["applicants"]["group"].astype(str).eq("B")
    access = [worlds[strength]["mechanism"].loc[b, "opportunity_access"].to_numpy() for strength in [0.0, .1, .2, .4]]
    for first, second in zip(access, access[1:]):
        assert (second <= first).all()


def test_s_zero_world_reconstructs_exactly(setup_worlds) -> None:
    baseline, systemic, _, draws, worlds = setup_worlds
    repeated = generate_systemic_world(draws, baseline, systemic, 0.0)
    for key in ["applicants", "loan_options", "mechanism"]:
        pd.testing.assert_frame_equal(worlds[0.0][key], repeated[key], check_exact=True)


def test_budgets_are_constructed_from_neutral_evaluation_only(setup_worlds) -> None:
    _, _, _, _, worlds = setup_worlds
    neutral = worlds[0.0]
    ids = neutral["applicants"].loc[neutral["applicants"]["cohort"].astype(str).eq("evaluation"), "applicant_id"]
    loans = neutral["loan_options"][neutral["loan_options"]["applicant_id"].isin(ids)]
    budgets = neutral_budgets(loans, {"all": 1.0, "forty": .4, "twenty": .2})
    assert budgets["all"] == pytest.approx(loans.requested_principal.sum())
    assert budgets["forty"] == pytest.approx(.4 * budgets["all"])


def test_same_absolute_budget_can_be_reused_for_treated_worlds(setup_worlds) -> None:
    _, _, _, _, worlds = setup_worlds
    ids = worlds[0.0]["applicants"].loc[worlds[0.0]["applicants"]["cohort"].astype(str).eq("evaluation"), "applicant_id"]
    neutral = worlds[0.0]["loan_options"][worlds[0.0]["loan_options"].applicant_id.isin(ids)]
    budget = neutral_budgets(neutral, {"forty": .4})["forty"]
    treated = worlds[.4]["loan_options"][worlds[.4]["loan_options"].applicant_id.isin(ids)]
    assert budget == pytest.approx(.4 * neutral.requested_principal.sum())
    assert budget != .4 * treated.requested_principal.sum()
    assert worlds[.4]["loan_options"].requested_principal.sum() != worlds[0.0]["loan_options"].requested_principal.sum()


def test_lender_feature_sets_remain_group_blind() -> None:
    forbidden = {"group", "opportunity_access", "systemic_strength"}
    assert forbidden.isdisjoint(TRADITIONAL_LENDER_FEATURES)
    assert forbidden.isdisjoint(ML_FEATURES)


def test_direct_belief_delta_remains_zero() -> None:
    assert load_systemic_config()["direct_belief_delta"] == 0.0


def test_systemic_policy_base_equals_used(setup_worlds) -> None:
    _, _, _, _, worlds = setup_worlds
    loans = worlds[0.0]["loan_options"].head(20)
    prediction = pd.DataFrame({"applicant_id": loans.applicant_id, "repayment_probability_traditional": .995})
    policy = build_policy_assessments(loans, prediction)
    np.testing.assert_array_equal(policy.repayment_probability_base, policy.repayment_probability_used)


def test_solver_feasibility_and_optimality() -> None:
    loans = pd.DataFrame({"applicant_id": [1, 2, 3], "option_id": ["a", "b", "c"], "requested_principal": [60., 50., 40.]})
    assessments = pd.DataFrame({"applicant_id": [1, 2, 3], "option_id": ["a", "b", "c"], "expected_profit_perceived": [12., 10., 7.]})
    result = solve_fixed_request_portfolio(loans, assessments, 100., world_id="w", budget_id="b", policy_id="p")
    assert result.solver_metadata["success"]
    assert result.solver_metadata["optimality_gap"] == 0
    assert result.decisions.funded_amount.sum() <= 100


def test_large_dollar_milp_uses_equivalent_scaled_budget_constraint() -> None:
    loans = pd.DataFrame({"applicant_id": [1, 2, 3], "option_id": ["a", "b", "c"], "requested_principal": [310_000_000., 290_000_000., 10_000_000.]})
    assessments = pd.DataFrame({"applicant_id": [1, 2, 3], "option_id": ["a", "b", "c"], "expected_profit_perceived": [31_000_000., 28_000_000., 900_000.]})
    result = solve_fixed_request_portfolio(loans, assessments, 300_000_000., world_id="w", budget_id="large", policy_id="p")
    assert result.solver_metadata["budget_constraint_scaling"] == "integer_cents_gcd"
    assert result.decisions.approved.tolist() == [False, True, True]
    assert result.decisions.funded_amount.sum() <= 300_000_000. + 1e-6
    assert result.solver_metadata["optimality_gap"] == 0


def test_matched_effect_arithmetic() -> None:
    assert (.37 - .46) - (.42 - .44) == pytest.approx(-.07)
    assert 75_000_000 - 75_200_000 == -200_000


def test_monte_carlo_sd_and_mcse_calculation() -> None:
    result = summarize_scalar([1., 2., 3., 4.])
    expected_sd = np.std([1., 2., 3., 4.], ddof=1)
    assert result["sd"] == pytest.approx(expected_sd)
    assert result["mcse"] == pytest.approx(expected_sd / 2)


def test_empirical_quantiles_are_calculated() -> None:
    result = summarize_scalar(range(1, 101))
    assert result["p2_5"] == pytest.approx(pd.Series(range(1, 101)).quantile(.025))
    assert result["p97_5"] == pytest.approx(pd.Series(range(1, 101)).quantile(.975))


def test_sign_stability_follows_mean_sign() -> None:
    negative = summarize_scalar([-3., -2., -1., 1.])
    positive = summarize_scalar([-1., 1., 2., 3.])
    assert negative["sign_stability"] == .75
    assert positive["sign_stability"] == .75


def _complete_record(run_id: str, fingerprint: str, revision: str, counts: dict[str, int]) -> dict:
    design = load_systemic_mc_config()["design"]
    worlds = list(itertools.product(design["risk_worlds"], design["systemic_strengths"]))
    models = [(*w, p) for w in worlds for p in design["policies"]]
    cells = [(*m, b) for m in models for b in design["budget_fractions"]]
    rows = [{**dict(zip(["risk_world", "systemic_strength", "family", "budget_id"], c)),
             **{k: 0.0 for k in PRIMARY_ESTIMANDS}, "replication_index": 0,
             "bank_budget": 100., "total_principal_funded": 90., "a_funding_rate": .5, "b_funding_rate": .5,
             "solver_success": True, "solver_mip_gap": 0.,
             "solver_metadata": {"budget_feasible": True, "budget_residual_dollars": -10., "budget_constraint_numerical_guard_dollars": 0.}}
            for c in cells]
    return {"status": "success", "run_id": run_id, "analysis_version": MC_ANALYSIS_VERSION,
            "config_fingerprint": fingerprint, "code_revision": revision, "replication_index": 0,
            "n_applicants": 10000, "cohort_counts": {"historical_train": 6000, "historical_validation": 2000, "evaluation": 2000},
            "budgets": {b: 100. for b in design["budget_fractions"]},
            "runs": rows, "switcher_lending": [r for r in rows if r["systemic_strength"] != 0],
            "model_performance": [{**dict(zip(["risk_world", "systemic_strength", "family"], m)),
                                   **{k: 0. for k in ["risk_mae", "risk_rmse", "profit_mae", "profit_rmse"]}} for m in models],
            "ml_selections": [dict(zip(["risk_world", "systemic_strength"], w)) for w in worlds],
            "controlled_audits": [{**r, "model": m} for r in rows for m in ["U0_race_only", "U1_downstream_controls"]]}


def test_failure_records_are_retained_atomically(tmp_path) -> None:
    path = tmp_path / "failure.json"
    record = {"status": "failed_model", "run_id": "x", "failure_message": "test"}
    write_replication_atomic(record, path)
    assert json.loads(path.read_text())["status"] == "failed_model"


def test_resume_skips_valid_completed_record(tmp_path) -> None:
    config = load_systemic_mc_config(); counts = expected_record_counts(config)
    record = _complete_record("id", config["metadata"]["config_fingerprint"], "rev", counts)
    path = tmp_path / "record.json"; write_replication_atomic(record, path)
    loaded = load_valid_replication(path, expected_run_id="id", expected_config_fingerprint=config["metadata"]["config_fingerprint"], expected_revision="rev", expected_counts=counts)
    assert loaded is not None
    assert {k: v for k, v in loaded.items() if k != "record_checksum"} == record


@pytest.mark.parametrize("contents", ["{broken", json.dumps({"status": "success", "run_id": "id"})])
def test_corrupt_or_incomplete_replication_is_rejected_for_rerun(tmp_path, contents) -> None:
    config = load_systemic_mc_config(); path = tmp_path / "record.json"; path.write_text(contents)
    loaded = load_valid_replication(path, expected_run_id="id", expected_config_fingerprint=config["metadata"]["config_fingerprint"], expected_revision="rev", expected_counts=expected_record_counts(config))
    assert loaded is None


def test_stable_run_identity_uses_config_seed_index_and_revision() -> None:
    seed = replication_seeds(499417, 2)
    first = replication_id("config", seed, "revision")
    assert first == replication_id("config", seed, "revision")
    assert first != replication_id("config", replication_seeds(499417, 3), "revision")


def test_summary_reconstructs_without_raw_applicant_rows() -> None:
    rows = []
    for replication in [0, 1]:
        for strength in [0., .2]:
            rows.append({"replication_index": replication, "risk_world": "additive_logistic_baseline", "systemic_strength": strength, "family": "traditional", "budget_id": "moderate_40pct", **{name: float(replication + strength) for name in ["delta_a_funding_rate", "delta_b_funding_rate", "delta_funding_gap", "delta_principal_gap", "delta_funded_requested_gap", "delta_true_expected_portfolio_profit", "oracle_regret", "delta_oracle_regret", "delta_realized_portfolio_profit"]}})
    record = {"status": "success", "run_id": "one", "replication_index": 0, "timing": {}, "raw_rows_persisted": False, "runs": rows, "pathway": [], "switcher_lending": [], "model_performance": [], "ml_selections": [], "controlled_audits": []}
    frame = records_to_frames([record])["runs"]
    summary = monte_carlo_summary(frame)
    assert not summary.empty
    assert "annual_income" not in frame.columns
