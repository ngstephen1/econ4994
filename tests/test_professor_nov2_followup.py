"""Deterministic November follow-up tests; toy fixtures are not research runs."""
import numpy as np
import pandas as pd
import pytest

from fair_lending.professor_model import group_blind
from fair_lending.professor_model import nov2_followup as study
from fair_lending.professor_model.contracts import make_loan_contracts
from fair_lending.professor_model.profit import expected_cashflows
from fair_lending.professor_model.risk import compute_true_risk, load_true_risk_config


@pytest.fixture(scope="module")
def toy():
    rng = np.random.default_rng(4994)
    n = 80
    ids = np.array([f"toy_{i:03d}" for i in range(n)])
    log_y = rng.normal(11, .7, n)
    epsilon, omega = rng.normal(size=n), rng.normal(0, .2, n)
    b = np.tile([0, 1], n // 2).astype("int8")
    bank = pd.DataFrame({"applicant_id": ids, "Y": np.exp(log_y),
                         "D": rng.normal(.35, .16, n), "B": b,
                         "C": .4 * epsilon + omega})
    private = pd.DataFrame({"applicant_id": ids, "log_Y": log_y,
                            "epsilon": epsilon, "omega": omega})
    truth = compute_true_risk(bank.merge(private, on="applicant_id"), load_true_risk_config())
    eta = pd.DataFrame({"applicant_id": ids, "eta": rng.normal(0, .1, n)})
    targets = pd.DataFrame({"applicant_id": ids, "PN": .8 * truth.true_PD + eta.eta})
    splits = pd.DataFrame({"applicant_id": ids,
                           "split": ["train"] * 64 + ["evaluation"] * 16})
    fitted_view = bank.merge(splits, on="applicant_id").merge(targets, on="applicant_id")
    fit = group_blind.fit_group_blind(fitted_view.loc[:, group_blind.FIT_COLUMNS])
    predictions = group_blind.risk_estimates(fitted_view, fit)
    contracts = make_loan_contracts(bank.iloc[64:], T=30, theta=.1, r=.05)
    return dict(bank=bank, private=private, truth=truth, eta=eta, targets=targets,
                splits=splits, frozen_model=fit, frozen_predictions=predictions,
                contracts=contracts)


def world(toy, scenario):
    return study.construct_world(toy["bank"], toy["private"], toy["targets"],
                                 toy["eta"], toy["truth"], scenario)


def inner(toy):
    return study.make_inner_splits(toy["splits"], fit_rows=48, validation_rows=16)


def fit(toy, scenario=study.SCENARIOS[0], targets=None):
    w = world(toy, scenario)
    return study.fit_scenario_models(
        w["bank"], w["targets"] if targets is None else targets,
        toy["splits"], inner(toy), scenario,
        frozen_blind_model=toy["frozen_model"],
        frozen_blind_predictions=toy["frozen_predictions"])


def test_scenario_one_exactly_replays_all_saved_views(toy):
    w = world(toy, study.SCENARIOS[0])
    for key in ("bank", "private", "targets", "truth"):
        pd.testing.assert_frame_equal(w[key], toy[key], check_exact=True)


def test_systemic_shift_and_credit_equation_match_declaration(toy):
    w = world(toy, study.SCENARIOS[1])
    np.testing.assert_array_equal(w["private"].epsilon,
                                  toy["private"].epsilon + toy["bank"].B)
    np.testing.assert_allclose(w["bank"].C,
        .4 * w["private"].epsilon + toy["private"].omega, rtol=0, atol=1e-15)
    np.testing.assert_array_equal(w["targets"].PN, .8 * w["truth"].true_PD + toy["eta"].eta)


def test_systemic_world_preserves_group_zero_bits(toy):
    w = world(toy, study.SCENARIOS[1]); mask = toy["bank"].B.eq(0)
    for key in ("bank", "private", "targets", "truth"):
        pd.testing.assert_frame_equal(w[key].loc[mask], toy[key].loc[mask], check_exact=True)


def test_systemic_risk_rises_for_same_treated_people_without_equation_change(toy):
    w = world(toy, study.SCENARIOS[1]); mask = toy["bank"].B.eq(1)
    assert (w["truth"].loc[mask, "true_PD"] > toy["truth"].loc[mask, "true_PD"]).all()
    np.testing.assert_array_equal(w["truth"].P_income, toy["truth"].P_income)


@pytest.mark.parametrize("column", ["Y", "D", "B", "applicant_id"])
def test_systemic_keeps_other_applicant_characteristics(toy, column):
    w = world(toy, study.SCENARIOS[1])
    pd.testing.assert_series_equal(w["bank"][column], toy["bank"][column], check_exact=True)


def test_world_construction_does_not_mutate_inputs(toy):
    originals = {k: v.copy(deep=True) for k, v in toy.items() if isinstance(v, pd.DataFrame)}
    world(toy, study.SCENARIOS[1])
    for k, original in originals.items():
        pd.testing.assert_frame_equal(toy[k], original, check_exact=True)


def test_inner_split_is_deterministic_and_excludes_evaluation(toy):
    a, b = inner(toy), inner(toy)
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    assert set(a.applicant_id) == set(toy["splits"].query("split == 'train'").applicant_id)
    assert a.inner_split.value_counts().to_dict() == {"fit": 48, "validation": 16}
    assert a.applicant_id.is_unique


def test_inner_split_seed_changes_assignment(toy):
    a = inner(toy)
    b = study.make_inner_splits(toy["splits"], fit_rows=48, validation_rows=16, seed=4995)
    assert not a.equals(b)


def test_inner_split_is_input_order_invariant(toy):
    a = inner(toy)
    b = study.make_inner_splits(toy["splits"].iloc[::-1], fit_rows=48, validation_rows=16)
    pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_inner_split_rejects_incomplete_training_count(toy):
    with pytest.raises(ValueError):
        study.make_inner_splits(toy["splits"], fit_rows=47, validation_rows=16)


@pytest.fixture(scope="module")
def toy_fit(toy):
    return fit(toy)


def test_prediction_cells_are_complete_and_unique(toy_fit, toy):
    p = toy_fit["predictions"]
    assert len(p) == 3 * len(toy["contracts"])
    assert not p.duplicated(["applicant_id", "scenario_id", "method_id"]).any()
    assert set(p.method_id) == set(study.METHODS)
    assert p.bounded_PD.between(0, 1).all()
    np.testing.assert_array_equal(p.bounded_PD, np.clip(p.raw_PD, 0, 1))


def test_frozen_control_predictions_replay_exactly(toy_fit, toy):
    actual = toy_fit["predictions"].query("method_id == 'ols_control'").sort_values("applicant_id")
    expected = toy["frozen_predictions"].query("split == 'evaluation'").sort_values("applicant_id")
    np.testing.assert_array_equal(actual.raw_PD, expected[group_blind.RAW])


def test_penalty_is_exact_positive_one_point_before_clipping(toy_fit, toy):
    p = toy_fit["predictions"]
    control = p.query("method_id == 'ols_control'").sort_values("applicant_id")
    penalty = p.query("method_id == 'ols_penalty'").sort_values("applicant_id")
    b = toy["bank"].set_index("applicant_id").loc[control.applicant_id, "B"].to_numpy()
    np.testing.assert_array_equal(penalty.raw_PD, control.raw_PD.to_numpy() + .01 * b)
    np.testing.assert_array_equal(penalty.bounded_PD, np.clip(penalty.raw_PD, 0, 1))


def test_ml_grid_is_predeclared_and_selected_without_truth(toy_fit):
    g = toy_fit["validation_grid"]
    assert len(g) == 4
    assert set(zip(g.max_leaf_nodes, g.max_iter)) == {(3, 25), (3, 75), (7, 25), (7, 75)}
    params = toy_fit["ml_model"].get_params()
    assert params["early_stopping"] is False
    assert params["loss"] == "squared_error"
    assert params["min_samples_leaf"] == 1000


@pytest.mark.parametrize("column", ["epsilon", "eta", "true_PD", "expected_profit_true"])
def test_lender_rejects_hidden_columns(toy, column):
    w = world(toy, study.SCENARIOS[0]); w["bank"][column] = 0
    with pytest.raises(ValueError):
        study.fit_scenario_models(w["bank"], w["targets"], toy["splits"], inner(toy),
            study.SCENARIOS[0], frozen_blind_model=toy["frozen_model"],
            frozen_blind_predictions=toy["frozen_predictions"])


def test_evaluation_targets_cannot_change_model_or_predictions(toy_fit, toy):
    changed = toy["targets"].copy()
    eval_ids = set(toy["splits"].query("split == 'evaluation'").applicant_id)
    changed.loc[changed.applicant_id.isin(eval_ids), "PN"] = 1000.
    actual = fit(toy, targets=changed)
    pd.testing.assert_frame_equal(actual["predictions"], toy_fit["predictions"], check_exact=True)
    pd.testing.assert_frame_equal(actual["validation_grid"], toy_fit["validation_grid"], check_exact=True)


def test_deterministic_fit_reproduces_every_prediction(toy_fit, toy):
    actual = fit(toy)
    pd.testing.assert_frame_equal(actual["predictions"], toy_fit["predictions"], check_exact=True)


def test_primary_feature_sets_match_scenario(toy_fit, toy):
    assert toy_fit["model_metadata"]["features"] == ["Y", "D", "C"]
    assert toy_fit["ml_model"].n_features_in_ == 3
    s2 = fit(toy, study.SCENARIOS[1])
    assert s2["model_metadata"]["features"] == ["Y", "D", "C", "B"]
    assert s2["ml_model"].n_features_in_ == 4
    assert s2["ols_model"]["feature_order"] == ["Y", "D", "C", "B"]


def test_core_rejects_unrecognized_scenario(toy):
    with pytest.raises(ValueError): world(toy, "invented_world")


def test_policy_accounting_reuses_validated_survival_sum(toy_fit, toy):
    p = toy_fit["predictions"].query("method_id == 'ols_control'")
    result = study.optimize_policy(toy["contracts"], p, float(toy["contracts"].L_i.sum()))
    a = result["assessments"].sort_values("applicant_id")
    c = toy["contracts"].sort_values("applicant_id")
    receipts, profit = expected_cashflows(c, p.sort_values("applicant_id").bounded_PD.to_numpy())
    np.testing.assert_array_equal(a.expected_receipts_bank, receipts)
    np.testing.assert_array_equal(a.expected_profit_bank, profit)
    assert result["decisions"].funded_principal.sum() <= c.L_i.sum()


def test_oversized_clipped_tie_remains_indivisible(toy):
    c = toy["contracts"].iloc[:2].copy()
    p = pd.DataFrame({"applicant_id": c.applicant_id, "scenario_id": study.SCENARIOS[0],
        "method_id": "ols_control", "raw_PD": [-.2, -.1], "bounded_PD": [0., 0.]})
    result = study.optimize_policy(c, p, float(c.L_i.max()))
    assert not result["decisions"].approved.any()
    assert result["selection"]["number_approved"] == 0


def test_zero_budget_selects_reject_all(toy_fit, toy):
    p = toy_fit["predictions"].query("method_id == 'ols_control'")
    result = study.optimize_policy(toy["contracts"], p, 0.)
    assert not result["decisions"].approved.any()


def test_ml_selection_uses_mse_then_predeclared_complexity_tie_rule(toy_fit):
    grid = toy_fit["validation_grid"]
    winner = grid.sort_values(["validation_PN_MSE", "max_leaf_nodes", "max_iter"]).iloc[0]
    selected = grid.loc[grid.selected]
    assert len(selected) == 1
    assert (selected.iloc[0].max_leaf_nodes, selected.iloc[0].max_iter) == (
        winner.max_leaf_nodes, winner.max_iter)


@pytest.mark.parametrize("column", ["true_PD", "PN", "B", "expected_profit_true"])
def test_cutoff_rejects_unapproved_input_columns(toy_fit, toy, column):
    p = toy_fit["predictions"].query("method_id == 'ols_control'").assign(**{column: 0.})
    with pytest.raises(ValueError):
        study.optimize_policy(toy["contracts"], p, 0.)


def test_clipping_validation_rejects_tampered_scores(toy_fit, toy):
    p = toy_fit["predictions"].query("method_id == 'ols_control'").copy()
    p.loc[p.index[0], "bounded_PD"] = .9
    with pytest.raises(ValueError, match="clip"):
        study.optimize_policy(toy["contracts"], p, 0.)


def test_exact_full_budget_includes_entire_profitable_tie(toy):
    import math
    applicants = toy["bank"].iloc[:2].copy()
    applicants["Y"] = [1000., 2000.]
    c = make_loan_contracts(applicants, T=30, theta=.1, r=.05)
    p = pd.DataFrame({"applicant_id": c.applicant_id, "scenario_id": study.SCENARIOS[0],
        "method_id": "ols_control", "raw_PD": [0., 0.], "bounded_PD": [0., 0.]})
    budget = math.fsum(c.L_i)
    result = study.optimize_policy(c, p, budget)
    assert result["decisions"].approved.all()
    assert result["selection"]["total_funded_principal"] == budget
    assert result["selection"]["exact_budget_exhausted"]


@pytest.fixture(scope="module")
def evaluated(toy_fit, toy):
    from fair_lending.professor_model.profit import assess_true_profit
    import math
    budget = .4 * math.fsum(toy["contracts"].L_i)
    policies = {m: study.optimize_policy(toy["contracts"],
        toy_fit["predictions"].query("method_id == @m"), budget) for m in study.METHODS}
    ids = toy["contracts"].applicant_id.tolist()
    truth = toy["truth"].set_index("applicant_id").loc[ids, ["true_PD"]].reset_index()
    profit = assess_true_profit(toy["contracts"], truth)
    audit = toy["bank"].set_index("applicant_id").loc[ids, ["B"]].reset_index()
    target = toy["targets"].set_index("applicant_id").loc[ids].reset_index()
    result = study.evaluate_policies(policies, toy["contracts"], audit, profit, target)
    return result, policies, profit, audit, target


def test_group_aggregation_uses_all_eligible_applicants(evaluated):
    result, policies, _, audit, _ = evaluated
    assert len(result["groups"]) == 6 and len(result["portfolio"]) == 3
    for row in result["groups"].itertuples():
        denominator = int(audit.B.eq(row.B).sum())
        assert row.n_applicants == denominator
        assert row.approval_rate == row.number_approved / denominator
        assert row.true_profit_per_eligible_applicant_USD == row.true_expected_portfolio_profit_USD / denominator


def test_group_profit_and_funded_principal_sum_to_portfolio(evaluated):
    result, _, _, _, _ = evaluated
    for row in result["portfolio"].itertuples():
        g = result["groups"].query("method_id == @row.method_id")
        assert g.number_approved.sum() == row.number_approved
        assert g.true_expected_portfolio_profit_USD.sum() == pytest.approx(
            row.true_expected_portfolio_profit_USD, rel=0, abs=1e-6)
        assert g.funded_principal_USD.sum() == pytest.approx(row.funded_principal_USD, rel=0, abs=1e-6)


def test_penalty_displacement_profit_identity(evaluated):
    result, _, _, _, _ = evaluated
    p = result["portfolio"].set_index("method_id")
    c = result["penalty_comparisons"].set_index("scope")
    expected = p.loc["ols_penalty", "true_expected_portfolio_profit_USD"] - p.loc["ols_control", "true_expected_portfolio_profit_USD"]
    assert c.loc["overall", "true_profit_change_USD"] == pytest.approx(expected, rel=0, abs=1e-6)
    assert c.loc["overall", "true_profit_change_USD"] == (
        c.loc["overall", "true_profit_gained_USD"] - c.loc["overall", "true_profit_lost_USD"])
    assert c.loc[["B=0", "B=1"], "net_approval_change"].sum() == c.loc["overall", "net_approval_change"]


def test_evaluator_does_not_fit_or_optimize(evaluated, toy, monkeypatch):
    result, policies, profit, audit, target = evaluated
    def forbidden(*args, **kwargs):
        raise AssertionError("Evaluator must not fit or select portfolios")
    monkeypatch.setattr(study, "choose_cutoff", forbidden)
    monkeypatch.setattr(study, "fit_scenario_models", forbidden)
    again = study.evaluate_policies(policies, toy["contracts"], audit, profit, target)
    for key in result:
        pd.testing.assert_frame_equal(result[key], again[key], check_exact=True)


def test_frozen_config_matches_declared_scientific_parameters():
    from fair_lending.professor_model import nov2_followup_io as io
    config = io.load_config()
    assert config["direct_penalty"] == .01
    assert config["epsilon_shift"] == 1.
    assert config["budget"] == study.EXACT_BUDGET
    assert config["scenario_features"] == {key: list(value) for key, value in study.FEATURES.items()}
    assert config["ml"] == study.ML_SETTINGS


@pytest.mark.parametrize("parameter,value", [("direct_penalty", -.1), ("epsilon_shift", 2.),
    ("budget", 1.), ("signal_scale", 1.)])
def test_config_rejects_scientific_retuning(tmp_path, parameter, value):
    import yaml
    from fair_lending.professor_model import nov2_followup_io as io
    config = io.load_config(); config[parameter] = value
    p = tmp_path / "changed.yaml"; p.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="predeclared"):
        io.load_config(p)


def test_missing_summary_is_read_only_and_never_launches_work(tmp_path, monkeypatch):
    from fair_lending.professor_model import nov2_followup_io as io
    from pathlib import Path
    def forbidden(*args, **kwargs):
        raise AssertionError("Read-only summary attempted work or writes")
    monkeypatch.setattr(io, "run_experiment", forbidden)
    monkeypatch.setattr(study, "fit_scenario_models", forbidden)
    monkeypatch.setattr(study, "construct_world", forbidden)
    monkeypatch.setattr(Path, "mkdir", forbidden)
    with pytest.raises(FileNotFoundError, match="starts no work"):
        io.summarize_only(root=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_hash_verification_detects_tampering(tmp_path):
    from fair_lending.professor_model import nov2_followup_io as io
    p = tmp_path / "source.txt"; p.write_text("original")
    hashes = {"source.txt": io.file_hash(p)}
    io.verify_hashes(hashes, root=tmp_path)
    p.write_text("changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        io.verify_hashes(hashes, root=tmp_path)


@pytest.mark.parametrize("path", ["../outside.txt", "/outside.txt"])
def test_hash_verification_rejects_escape_paths(tmp_path, path):
    from fair_lending.professor_model import nov2_followup_io as io
    with pytest.raises(ValueError, match="repository"):
        io.verify_hashes({path: "irrelevant"}, root=tmp_path)


def test_dataset_roundtrip_and_overwrite_protection(tmp_path):
    from fair_lending.professor_model import nov2_followup_io as io
    frame = pd.DataFrame({"applicant_id": ["a", "b"], "raw_PD": [-.001, .03],
                          "approved": [False, True]})
    stem = tmp_path / "bank_data"
    io.write_pair(stem, frame)
    before = stem.with_suffix(".csv").read_bytes()
    pd.testing.assert_frame_equal(pd.read_parquet(stem.with_suffix(".parquet")), frame, check_exact=True)
    pd.testing.assert_frame_equal(pd.read_csv(stem.with_suffix(".csv"), float_precision="round_trip"), frame, check_exact=True)
    with pytest.raises(FileExistsError): io.write_pair(stem, frame)
    assert stem.with_suffix(".csv").read_bytes() == before


def test_figure_cells_must_be_complete_and_unique(evaluated, tmp_path):
    from fair_lending.professor_model.nov2_followup_figures import save_figures
    groups = evaluated[0]["groups"]
    with pytest.raises(ValueError, match="twelve"):
        save_figures(groups, tmp_path / "figures")
    assert not (tmp_path / "figures").exists()
