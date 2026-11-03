"""Small-penalty selection tests use toy loans, never a production search."""
from copy import deepcopy
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fair_lending.professor_model import nov2_followup as parent
from fair_lending.professor_model import five_percent_penalty as study
from fair_lending.professor_model.contracts import make_loan_contracts


@pytest.fixture
def toy():
    ids = [f"prediction_{i:02d}" for i in range(8)]
    bank = pd.DataFrame({"applicant_id": ids, "Y": [1000.] * 8})
    audit = pd.DataFrame({"applicant_id": ids, "B": [0] * 6 + [1] * 2})
    contracts = make_loan_contracts(bank, T=30, theta=.1, r=.05)
    splits = pd.DataFrame({"applicant_id": ["training_00", "training_01", *ids],
                           "split": ["train"] * 2 + ["evaluation"] * 8})

    def predictions(scenario, group_b=(-.004, .002), method="ols_control"):
        raw = np.array([.001, .002, .003, .004, .005, .006, *group_b])
        return pd.DataFrame({"applicant_id": ids, "scenario_id": scenario,
            "method_id": method, "raw_PD": raw, "bounded_PD": np.clip(raw, 0, 1)}).loc[:, parent.PREDICTION_COLUMNS]

    controls = {study.SCENARIOS[0]: predictions(study.SCENARIOS[0]),
                study.SCENARIOS[1]: predictions(study.SCENARIOS[1], (.0015, .008))}
    return {"contracts": contracts, "audit": audit, "splits": splits,
            "predictions": predictions, "controls": controls,
            "audits": {s: audit.copy() for s in study.SCENARIOS}, "budget": 15000.}


def select(toy, **kwargs):
    return study.select_common_penalty(toy["contracts"], toy["controls"],
        toy["audits"], toy["budget"], **kwargs)


def test_candidate_sequence_matches_professor_successor_design():
    assert study.CANDIDATES == (.005, .0025, .00125, .000625, .0003125, .00015625)


def test_adjustment_adds_exact_delta_before_clipping_and_preserves_group_a(toy):
    control = toy["controls"][study.SCENARIOS[0]]
    adjusted = study.adjust_penalty(control, toy["audit"], .005)
    np.testing.assert_array_equal(adjusted.raw_PD, control.raw_PD + .005 * toy["audit"].B)
    np.testing.assert_array_equal(adjusted.bounded_PD, np.clip(adjusted.raw_PD, 0, 1))
    np.testing.assert_array_equal(adjusted.loc[:5, "raw_PD"], control.loc[:5, "raw_PD"])
    assert adjusted.method_id.eq("ols_penalty").all()


def test_negative_and_above_one_scores_are_clipped_after_adjustment(toy):
    control = toy["predictions"](study.SCENARIOS[0], (-.02, .998))
    result = study.adjust_penalty(control, toy["audit"], .005)
    assert result.iloc[6].raw_PD == -.02 + .005
    assert result.iloc[6].bounded_PD == 0.
    assert result.iloc[7].raw_PD == .998 + .005
    assert result.iloc[7].bounded_PD == 1.


def test_adjustment_is_input_order_invariant_and_does_not_mutate(toy):
    original = toy["controls"][study.SCENARIOS[0]].copy(deep=True)
    audit = toy["audit"].copy(deep=True)
    a = study.adjust_penalty(original, audit, .005)
    b = study.adjust_penalty(original.iloc[::-1], audit.iloc[::-1], .005)
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    pd.testing.assert_frame_equal(original, toy["controls"][study.SCENARIOS[0]], check_exact=True)
    pd.testing.assert_frame_equal(audit, toy["audit"], check_exact=True)


@pytest.mark.parametrize("delta", [0., -.1, True, float("nan"), float("inf")])
def test_adjustment_rejects_nonpositive_or_invalid_penalties(toy, delta):
    with pytest.raises(ValueError):
        study.adjust_penalty(toy["controls"][study.SCENARIOS[0]], toy["audit"], delta)


@pytest.mark.parametrize("column", ["true_PD", "expected_profit_true", "epsilon", "PN"])
def test_adjustment_rejects_evaluator_and_target_columns(toy, column):
    control = toy["controls"][study.SCENARIOS[0]].assign(**{column: 0.})
    with pytest.raises(ValueError):
        study.adjust_penalty(control, toy["audit"], .005)


def test_adjustment_rejects_hidden_group_audit_columns(toy):
    with pytest.raises(ValueError):
        study.adjust_penalty(toy["controls"][study.SCENARIOS[0]],
                             toy["audit"].assign(true_PD=.5), .005)


@pytest.mark.parametrize("approved_a,approved_b,expected", [
    (4, 1, True), (3, 1, False), (2, 1, False),
    (6, 2, False), (0, 0, False), (5, 0, False)])
def test_criterion_uses_strict_rates_not_raw_counts(toy, approved_a, approved_b, expected):
    decisions = pd.DataFrame({"applicant_id": toy["audit"].applicant_id,
        "approved": [True] * approved_a + [False] * (6 - approved_a)
                    + [True] * approved_b + [False] * (2 - approved_b)})
    result = study.group_criterion(decisions, toy["audit"])
    assert result["criterion_met"] is expected
    assert (result["n_A"], result["n_B"], result["approved_A"], result["approved_B"]) == (
        6, 2, approved_a, approved_b)
    assert result["rate_A"] == approved_a / 6
    assert result["rate_B"] == approved_b / 2


def test_criterion_rejects_non_boolean_decisions(toy):
    decisions = pd.DataFrame({"applicant_id": toy["audit"].applicant_id, "approved": [1] * 8})
    with pytest.raises(ValueError): study.group_criterion(decisions, toy["audit"])


def test_criterion_requires_both_groups_and_complete_ids(toy):
    decisions = pd.DataFrame({"applicant_id": toy["audit"].applicant_id, "approved": [True] * 8})
    with pytest.raises(ValueError):
        study.group_criterion(decisions.iloc[:-1], toy["audit"])
    with pytest.raises(ValueError):
        study.group_criterion(decisions, toy["audit"].assign(B=0))


def test_selector_waits_until_both_worlds_pass_and_stops_at_first_success(toy):
    calls = []
    def optimizer(contracts, predictions, budget):
        calls.append((predictions.scenario_id.iloc[0], predictions.method_id.iloc[0]))
        return parent.optimize_policy(contracts, predictions, budget)
    result = select(toy, optimizer=optimizer)
    assert result["status"] == "criterion_met"
    assert result["selected_delta"] == .0025
    assert len(calls) == 4
    assert all(method == "ols_penalty" for _, method in calls)
    attempts = result["attempts"]
    assert attempts.delta.tolist() == [.005, .005, .0025, .0025]
    assert attempts.loc[attempts.delta.eq(.005), "criterion_met"].tolist() == [True, False]
    assert not attempts.loc[attempts.delta.eq(.005), "selected"].any()
    assert attempts.loc[attempts.delta.eq(.0025), "selected"].all()
    assert set(result["selected_policies"]) == set(study.SCENARIOS)


def test_selector_stops_after_first_candidate_when_both_pass(toy):
    controls = {s: toy["predictions"](s) for s in study.SCENARIOS}
    result = study.select_common_penalty(toy["contracts"], controls, toy["audits"], toy["budget"])
    assert result["status"] == "criterion_met"
    assert result["selected_delta"] == .005
    assert len(result["attempts"]) == 2
    assert len(result["attempt_records"]) == 2


def test_no_success_retains_all_attempts_without_zero_or_extra_candidates(toy):
    controls = {s: toy["predictions"](s, (.1, .2)) for s in study.SCENARIOS}
    result = study.select_common_penalty(toy["contracts"], controls, toy["audits"], toy["budget"])
    assert result["status"] == "criterion_unmet"
    assert result["selected_delta"] is None
    assert result["selected_policies"] == {}
    assert len(result["attempts"]) == 12
    assert len(result["attempt_records"]) == 12
    assert result["attempts"].delta.tolist() == [v for v in study.CANDIDATES for _ in study.SCENARIOS]
    assert not result["attempts"].selected.any()


def test_selector_does_not_mutate_frozen_control_scores(toy):
    before = deepcopy(toy["controls"])
    select(toy)
    for scenario in study.SCENARIOS:
        pd.testing.assert_frame_equal(before[scenario], toy["controls"][scenario], check_exact=True)


def test_selector_has_no_truth_profit_or_model_fitting_interface(toy, monkeypatch):
    parameters = set(inspect.signature(study.select_common_penalty).parameters)
    assert not parameters.intersection({"truth", "true_PD", "profit", "true_profit", "targets", "models"})
    def forbidden(*args, **kwargs):
        raise AssertionError("Penalty search must not fit or evaluate hidden profit")
    monkeypatch.setattr(parent, "fit_scenario_models", forbidden)
    monkeypatch.setattr(parent, "evaluate_policies", forbidden)
    assert select(toy)["status"] == "criterion_met"


def test_selector_produces_deterministic_attempts_and_selected_masks(toy):
    a, b = select(toy), select(toy)
    assert a["selected_delta"] == b["selected_delta"]
    pd.testing.assert_frame_equal(a["attempts"], b["attempts"], check_exact=True)
    assert a["attempt_records"] == b["attempt_records"]
    for scenario in study.SCENARIOS:
        pd.testing.assert_frame_equal(a["selected_policies"][scenario]["decisions"],
                                      b["selected_policies"][scenario]["decisions"], check_exact=True)


@pytest.mark.parametrize("candidates", [(), (0.,), (-.1,), (True,), (float("nan"),),
                                      (.001, .002), (.005, .005)])
def test_candidate_sequence_rejects_invalid_or_non_descending_values(toy, candidates):
    with pytest.raises(ValueError): select(toy, candidates=candidates)


def test_selector_requires_exact_two_worlds(toy):
    with pytest.raises(ValueError):
        study.select_common_penalty(toy["contracts"], {study.SCENARIOS[0]: toy["controls"][study.SCENARIOS[0]]},
                                    toy["audits"], toy["budget"])


def test_selector_rejects_optimizer_budget_changes(toy):
    def wrong_budget(contracts, predictions, budget):
        p = parent.optimize_policy(contracts, predictions, budget)
        p["selection"]["available_bank_capital"] = budget + 1
        return p
    with pytest.raises(ValueError, match="budget"):
        select(toy, optimizer=wrong_budget)


def test_existing_budget_and_indivisible_ties_remain_enforced(toy):
    controls = {s: toy["predictions"](s).assign(raw_PD=-.01, bounded_PD=0.) for s in study.SCENARIOS}
    result = study.select_common_penalty(toy["contracts"], controls, toy["audits"], toy["budget"], candidates=(.005,))
    assert result["status"] == "criterion_unmet"
    assert result["attempts"].approved_A.eq(0).all()
    assert result["attempts"].approved_B.eq(0).all()
    assert all(record["checks"]["independent_exact_enumeration_passed"]
               for record in result["attempt_records"])


def test_prediction_cohort_is_disjoint_and_fingerprinted(toy):
    predictions = pd.concat([toy["predictions"](s, method=m)
        for s in study.SCENARIOS for m in study.METHODS], ignore_index=True)
    result = study.validate_prediction_cohort(toy["splits"], toy["contracts"], predictions, toy["audit"])
    assert result["training_rows"] == 2 and result["prediction_rows"] == 8
    assert result["training_prediction_disjoint"]
    assert result["prediction_only_contracts_and_audit"]


def test_training_applicant_cannot_enter_contracts_or_prediction_profit_cohort(toy):
    c = toy["contracts"].copy(); c.loc[0, "applicant_id"] = "training_00"
    with pytest.raises(ValueError, match="prediction cohort"):
        study.validate_prediction_cohort(toy["splits"], c,
            toy["controls"][study.SCENARIOS[0]], toy["audit"])


def test_missing_prediction_id_is_rejected(toy):
    with pytest.raises(ValueError, match="evaluation IDs"):
        study.validate_prediction_cohort(toy["splits"], toy["contracts"],
            toy["controls"][study.SCENARIOS[0]].iloc[:-1], toy["audit"])


def test_duplicate_prediction_cells_are_rejected(toy):
    p = toy["controls"][study.SCENARIOS[0]]
    with pytest.raises(ValueError, match="duplicate"):
        study.validate_prediction_cohort(toy["splits"], toy["contracts"],
            pd.concat([p, p.iloc[:1]], ignore_index=True), toy["audit"])


@pytest.mark.parametrize("column", ["true_PD", "expected_profit_true", "epsilon", "PN"])
def test_group_criterion_rejects_evaluator_or_target_columns(toy, column):
    decisions = pd.DataFrame({"applicant_id": toy["audit"].applicant_id, "approved": [True] * 8})
    with pytest.raises(ValueError):
        study.group_criterion(decisions.assign(**{column: 0.}), toy["audit"])


def test_successor_config_is_frozen_and_does_not_refit_or_resample():
    from fair_lending.professor_model import five_percent_penalty_io as io
    config = io.load_config()
    assert tuple(config["penalty_candidates"]) == study.CANDIDATES
    assert config["common_coefficient"] is True
    assert config["profit_used_for_penalty_selection"] is False
    assert config["minimum_B_rate_numerator"] == 1 and config["minimum_B_rate_denominator"] == 20
    assert config["refit"] is False and config["new_noise"] is False and config["new_population"] is False
    assert config["training_rows"] == 80000 and config["prediction_rows"] == 20000


@pytest.mark.parametrize("name,value", [("budget", 1.), ("common_coefficient", False),
    ("profit_used_for_penalty_selection", True), ("penalty_candidates", [.005, .0001]),
    ("minimum_B_rate_denominator", 10), ("refit", True)])
def test_config_rejects_unapproved_design_changes(tmp_path, name, value):
    import yaml
    from fair_lending.professor_model import five_percent_penalty_io as io
    config = io.load_config(); config[name] = value
    p = tmp_path / "changed.yaml"; p.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="predeclared"):
        io.load_config(p)


def test_missing_summary_is_strictly_read_only(tmp_path, monkeypatch):
    from fair_lending.professor_model import five_percent_penalty_io as io
    def forbidden(*args, **kwargs):
        raise AssertionError("Summary-only must not compute, fit or write")
    monkeypatch.setattr(io, "run_experiment", forbidden)
    monkeypatch.setattr(study, "select_common_penalty", forbidden)
    monkeypatch.setattr(parent, "fit_scenario_models", forbidden)
    monkeypatch.setattr(parent, "optimize_policy", forbidden)
    monkeypatch.setattr(Path, "mkdir", forbidden)
    with pytest.raises(FileNotFoundError, match="starts no work"):
        io.summarize_only(root=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_existing_output_refuses_scientific_execution_before_parent_access(tmp_path, monkeypatch):
    from fair_lending.professor_model import five_percent_penalty_io as io
    out = io.paths(tmp_path)["metrics"]; out.mkdir(parents=True)
    existing = out / "kept.json"; existing.write_text("preserved")
    def forbidden(*args, **kwargs):
        raise AssertionError("Existing output should be rejected before parent reads or selection")
    monkeypatch.setattr(io, "_parent_bindings", forbidden)
    monkeypatch.setattr(study, "select_common_penalty", forbidden)
    with pytest.raises(FileExistsError, match="outputs exist"):
        io.run_experiment(root=tmp_path)
    assert existing.read_text() == "preserved"


def test_public_experiment_context_preserves_frozen_values(toy):
    from fair_lending.professor_model import five_percent_penalty_io as io
    original = pd.concat([toy["predictions"](s, method=m)
        for s in study.SCENARIOS for m in study.METHODS], ignore_index=True)
    before = original.copy(deep=True)
    exported = io._augment(original, .0025)
    assert exported.experiment_id.eq(study.EXPERIMENT_ID).all()
    assert exported.loc[exported.method_id.eq("ols_penalty"), "penalty_delta"].eq(.0025).all()
    assert exported.loc[~exported.method_id.eq("ols_penalty"), "penalty_delta"].eq(0).all()
    pd.testing.assert_frame_equal(io._plain(exported), before, check_exact=True)
    pd.testing.assert_frame_equal(original, before, check_exact=True)


def test_csv_parquet_roundtrip_and_immutable_writes(tmp_path):
    from fair_lending.professor_model import five_percent_penalty_io as io
    data = pd.DataFrame({"applicant_id": ["a", "b"], "raw_PD": [-.01, .005], "approved": [False, True]})
    stem = tmp_path / "frozen"
    io.write_pair(stem, data)
    expected = stem.with_suffix(".csv").read_bytes()
    pd.testing.assert_frame_equal(pd.read_parquet(stem.with_suffix(".parquet")), data, check_exact=True)
    pd.testing.assert_frame_equal(pd.read_csv(stem.with_suffix(".csv"), float_precision="round_trip"), data, check_exact=True)
    with pytest.raises(FileExistsError): io.write_pair(stem, data)
    assert stem.with_suffix(".csv").read_bytes() == expected


def test_hash_validation_rejects_changed_parent_copy(tmp_path):
    from fair_lending.professor_model import five_percent_penalty_io as io
    p = tmp_path / "parent.csv"; p.write_text("original")
    hashes = {p.name: io.file_hash(p)}
    io.verify_hashes(hashes, root=tmp_path)
    p.write_text("tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        io.verify_hashes(hashes, root=tmp_path)


@pytest.mark.parametrize("path", ["../outside.json", "/outside.json"])
def test_hash_validation_rejects_outside_paths(tmp_path, path):
    from fair_lending.professor_model import five_percent_penalty_io as io
    with pytest.raises(ValueError, match="repository"):
        io.verify_hashes({path: "unused"}, root=tmp_path)


def test_saved_parent_split_and_contracts_already_satisfy_professor_requirement():
    from fair_lending.professor_model import nov2_followup_io as io
    p = io.paths()["processed"]
    if not (p / "applicant_splits.csv").is_file():
        pytest.skip("Parent artifacts are local evidence, not bundled full datasets")
    splits = pd.read_csv(p / "applicant_splits.csv")
    contracts = pd.read_parquet(p / "loan_contracts.parquet")
    predictions = pd.read_parquet(p / "predictions.parquet")
    assert splits.split.value_counts().to_dict() == {"train": 80000, "evaluation": 20000}
    training = set(splits.query("split == 'train'").applicant_id)
    prediction = set(splits.query("split == 'evaluation'").applicant_id)
    assert not training.intersection(prediction)
    assert set(contracts.applicant_id) == prediction
    for _, frame in predictions.groupby(["scenario_id", "method_id"]):
        assert set(frame.applicant_id) == prediction


def test_group_profit_aggregation_remains_prediction_only(toy):
    from fair_lending.professor_model.profit import assess_true_profit
    selected = select(toy)
    scenario = study.SCENARIOS[0]
    control = toy["controls"][scenario]
    ml = toy["predictions"](scenario, method="ml_histgb")
    policies = {"ols_control": parent.optimize_policy(toy["contracts"], control, toy["budget"]),
                "ols_penalty": selected["selected_policies"][scenario],
                "ml_histgb": parent.optimize_policy(toy["contracts"], ml, toy["budget"])}
    true_pd = pd.DataFrame({"applicant_id": toy["contracts"].applicant_id, "true_PD": .001})
    truth = assess_true_profit(toy["contracts"], true_pd)
    result = parent.evaluate_policies(policies, toy["contracts"], toy["audit"], truth)
    assert len(result["portfolio"]) == 3 and len(result["groups"]) == 6
    for row in result["portfolio"].itertuples():
        group = result["groups"].query("method_id == @row.method_id")
        assert group.n_applicants.sum() == 8
        assert group.number_approved.sum() == row.number_approved
        assert group.true_expected_portfolio_profit_USD.sum() == pytest.approx(
            row.true_expected_portfolio_profit_USD, rel=0, abs=1e-8)
        for g in group.itertuples():
            assert g.true_profit_per_eligible_applicant_USD == g.true_expected_portfolio_profit_USD / g.n_applicants


def test_positive_tiny_approval_and_profit_labels_do_not_round_to_zero():
    from fair_lending.professor_model.five_percent_penalty_figures import approval_label, profit_label
    label = approval_label(1 / 3030, 1)
    assert "0.00%" not in label and "0.0%" not in label
    assert "1" in label
    assert profit_label(1.) not in {"$0.00M", "$0.0M", "$0M", "$0.000B"}


@pytest.mark.parametrize("b,expected", [(0, False), (151, False), (152, True), (153, True)])
def test_exact_five_percent_boundary_with_3030_group_b_applicants(b, expected):
    audit = pd.DataFrame({"applicant_id": [f"p{i:05d}" for i in range(7030)],
                          "B": [0] * 4000 + [1] * 3030})
    decisions = pd.DataFrame({"applicant_id": audit.applicant_id,
        "approved": [True] * 1000 + [False] * 3000 + [True] * b + [False] * (3030-b)})
    result = study.group_criterion(decisions, audit)
    assert result["criterion_met"] is expected
    assert result["required_B_approvals"] == 152
    assert result["minimum_rate_met"] is expected


@pytest.mark.parametrize("a,b,expected", [(2, 1, True), (1, 1, False), (20, 20, False)])
def test_exact_five_percent_is_inclusive_and_group_equality_is_excluded(a, b, expected):
    audit = pd.DataFrame({"applicant_id": [f"p{i:02d}" for i in range(40)], "B": [0]*20+[1]*20})
    decisions = pd.DataFrame({"applicant_id": audit.applicant_id,
        "approved": [True]*a+[False]*(20-a)+[True]*b+[False]*(20-b)})
    result = study.group_criterion(decisions, audit)
    assert result["criterion_met"] is expected
    assert result["required_B_approvals"] == 1


def test_five_percent_stopping_waits_for_both_worlds_without_monotonicity_assumption():
    ids = [f"p{i:02d}" for i in range(60)]
    audit = pd.DataFrame({"applicant_id": ids, "B": [0]*40+[1]*20})
    contracts = make_loan_contracts(pd.DataFrame({"applicant_id":ids,"Y":1000.}),T=30,theta=.1,r=.05)
    controls = {s: pd.DataFrame({"applicant_id":ids,"scenario_id":s,"method_id":"ols_control",
                                "raw_PD":0.,"bounded_PD":0.}).loc[:, parent.PREDICTION_COLUMNS] for s in study.SCENARIOS}
    calls = []
    def optimizer(c, predictions, budget):
        s = predictions.scenario_id.iloc[0]; delta = predictions.raw_PD.iloc[-1]
        calls.append((s, delta))
        b = 0 if delta == .005 or (delta == .0025 and s == study.SCENARIOS[1]) else 1
        return {"decisions":pd.DataFrame({"applicant_id":ids,"approved":[True]*10+[False]*30+[True]*b+[False]*(20-b)}),
                "selection":{"available_bank_capital":budget,"total_funded_principal":10.,"optimal_tau":.001},"checks":{}}
    result = study.select_common_penalty(contracts, controls, {s:audit for s in study.SCENARIOS}, 100., optimizer=optimizer)
    assert result["selected_delta"] == .00125
    assert len(calls) == 6
    assert result["attempts"].delta.tolist() == [.005,.005,.0025,.0025,.00125,.00125]
    assert result["attempts"].iloc[-2:].minimum_rate_met.all()


def test_method_comparison_units_signs_and_capital():
    rows=[]
    for s in study.SCENARIOS:
        for m, gap, profit, principal in [("ols_control",-.2,100.,100.),("ols_penalty",-10.,99.,99.),("ml_histgb",.1,72.,72.)]:
            rows.append({"scenario_id":s,"method_id":m,"approval_gap_pp":gap,
                "true_expected_portfolio_profit_USD":profit,"available_capital_USD":100.,"funded_principal_USD":principal})
    frame=study.method_comparisons(pd.DataFrame(rows)); ml=frame.loc[frame.method_id.eq("ml_histgb")]
    assert ml.absolute_approval_gap_pp.eq(.1).all()
    assert ml.true_profit_difference_vs_control_USD.eq(-28.).all()
    assert ml.true_profit_difference_vs_control_pct.eq(-28.).all()
    assert ml.capital_utilization.eq(.72).all()
    assert ml.unused_capital_USD.eq(28.).all()
    with pytest.raises(ValueError): study.method_comparisons(pd.DataFrame(rows[:-1]))


def test_historical_penalty_comparison_signed_profit_and_group_accounting(toy):
    audit=toy["audit"]
    before=pd.DataFrame({"applicant_id":audit.applicant_id,"approved":[True,False,False,False,False,False,False,True]})
    after=pd.DataFrame({"applicant_id":audit.applicant_id,"approved":[False,True,False,False,False,False,True,True]})
    truth=pd.DataFrame({"applicant_id":audit.applicant_id,"expected_profit_true":[1.,2.,3.,4.,5.,6.,7.,8.]})
    result=study.compare_previous_penalty(study.SCENARIOS[0],after,before,audit,truth).set_index("scope")
    assert result.loc["overall","true_profit_difference_USD"] == 8.
    assert result.loc["B=0","true_profit_difference_USD"] == 1.
    assert result.loc["B=1","true_profit_difference_USD"] == 7.
    assert result.loc["overall","gained_approvals"] == 2
    assert result.loc["overall","lost_approvals"] == 1
    with pytest.raises(ValueError): study.compare_previous_penalty(study.SCENARIOS[0],after,before,audit,truth.iloc[:-1])
