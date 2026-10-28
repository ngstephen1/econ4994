"""Step 9B deterministic small fixtures. Never refit a research model."""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from fair_lending.professor_model import revised_lending as rl
from fair_lending.professor_model import revised_lending_io as io
from fair_lending.professor_model import revised_scenarios as rs
from fair_lending.professor_model import revised_scenarios_io as rio
from fair_lending.professor_model import bank_regression, group_blind
from fair_lending.professor_model.bank_regression import MODEL_ID, ids_fingerprint
from fair_lending.professor_model.config import configuration_fingerprint
from fair_lending.professor_model.contracts import make_loan_contracts
from fair_lending.professor_model.lending import BANK_INPUT_COLUMNS, choose_cutoff
from fair_lending.professor_model.oracle import optimize_oracle
from fair_lending.professor_model.risk_io import file_hash


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


@pytest.fixture
def toy():
    ids = [f"toy_{i}" for i in range(8)]
    applicants = pd.DataFrame({"applicant_id": ids, "Y": np.arange(20, 100, 10) * 1000.,
                               "D": .3, "B": np.arange(8) % 2, "C": np.arange(8) / 10})
    aware_raw = np.array([.02, .04, .01, .05, .03, .06, .07, .08])
    blind_raw = aware_raw - .002 * applicants.B.to_numpy() + .001 * (1 - applicants.B.to_numpy())
    aware = applicants[["applicant_id"]].assign(split="evaluation", model_id=MODEL_ID,
                                                bank_PD_raw=aware_raw, bank_PD_bounded=aware_raw,
                                                prediction_out_of_bounds=False)
    blind = applicants[["applicant_id"]].assign(split="evaluation", model_id=group_blind.MODEL_ID,
                                                bank_PD_group_blind_raw=blind_raw, bank_PD_group_blind_bounded=blind_raw,
                                                prediction_out_of_bounds=False)
    contracts = make_loan_contracts(applicants[["applicant_id", "Y"]], T=30, theta=.1, r=.05)
    profit = np.arange(1, 9) * 100.
    truth = pd.DataFrame({"applicant_id": ids, "true_PD": aware_raw,
                          "expected_receipts_true": contracts.L_i.to_numpy() + profit, "expected_profit_true": profit})
    return {"applicants": applicants, "aware": aware, "blind": blind, "contracts": contracts,
            "risks": rs.build_scenario_risk_estimates(applicants, aware, blind), "truth": truth,
            "audit": applicants[["applicant_id", "B"]].copy(), "budget": .4 * math.fsum(contracts.L_i)}


def banks_for(toy, budget=None):
    return {s: rl.optimize_scenario(toy["contracts"], toy["risks"].loc[toy["risks"].scenario_id == s, rl.RISK_VIEW],
                                    toy["budget"] if budget is None else budget) for s in rl.SCENARIO_IDS}


def policies_for(toy, banks, budget=None):
    budget = toy["budget"] if budget is None else budget
    policies = {s: rl.evaluation_policy(banks[s]["decisions"], banks[s]["selection"],
                                        score_column="estimated_PD_bounded", role="step9b_scenario") for s in rl.SCENARIO_IDS}
    for scenario, reference in ((rl.SCENARIO_IDS[0], rl.REFERENCE_IDS[0]), (rl.SCENARIO_IDS[2], rl.REFERENCE_IDS[1])):
        policies[reference] = deepcopy(policies[scenario]); policies[reference]["role"] = "frozen_reference"
    oracle, _, selection = optimize_oracle(toy["contracts"], toy["truth"], budget)
    policies[rl.REFERENCE_IDS[2]] = rl.evaluation_policy(oracle, selection, score_column="true_PD", role="frozen_reference", oracle=True)
    return policies


@pytest.fixture
def saved(toy, tmp_path, monkeypatch):
    root = tmp_path
    config = io.load_config(); config["evaluation_rows"] = 8
    config_path = root / "configs/professor_model/revised_lending.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump(config))
    old_config = rio.load_config(); old_config.update(population_rows=10, training_rows=2, evaluation_rows=8)
    old_path = root / "configs/professor_model/revised_scenarios.yaml"
    old_path.write_text(yaml.safe_dump(old_config))
    old = {"applicants": toy["applicants"], "aware": toy["aware"], "blind": toy["blind"],
           "contracts": toy["contracts"], "budget": rs.EXACT_BUDGET}
    # Stub only Step 9A's already-tested loader, not the new freeze/gating code.
    monkeypatch.setattr(rio, "load_frozen_inputs", lambda *a, **kw: (old, {}))
    frozen_banks = banks_for(toy, rs.EXACT_BUDGET)
    oracle, _, oracle_selection = optimize_oracle(toy["contracts"], toy["truth"], rs.EXACT_BUDGET)
    files = {
        Path("data/processed/professor_model/true_expected_profit.parquet"): toy["truth"],
        Path("data/processed/professor_model/loan_contracts.parquet"): toy["contracts"],
        Path("data/processed/professor_model/lending_decisions.parquet"): frozen_banks[rl.SCENARIO_IDS[0]]["decisions"].rename(columns={"estimated_PD_bounded": "bank_PD_bounded"}),
        Path("data/processed/professor_model/lending_decisions_group_blind.parquet"): frozen_banks[rl.SCENARIO_IDS[2]]["decisions"].rename(columns={"estimated_PD_bounded": "bank_PD_group_blind_bounded"}),
        Path("data/processed/professor_model/oracle_lending_decisions.parquet"): oracle,
    }
    for relative, frame in files.items():
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(root / relative, index=False)
    for name, selection in (("lending_selection.json", frozen_banks[rl.SCENARIO_IDS[0]]["selection"]),
                            ("group_blind_lending_selection.json", frozen_banks[rl.SCENARIO_IDS[2]]["selection"]),
                            ("oracle_comparison_validation.json", oracle_selection)):
        dump(root / "results/metrics/professor_model" / name, {"oracle_selection" if "oracle" in name else "selection": selection})
    risk_paths = rio.output_paths(root)
    risk_paths["risk_csv"].parent.mkdir(parents=True, exist_ok=True)
    toy["risks"].to_csv(risk_paths["risk_csv"], index=False)
    toy["risks"].to_parquet(risk_paths["risk_parquet"], index=False)
    prior_ties = root / "results/tables/professor_model/revised_scenarios_zero_tie_diagnostics.csv"
    prior_ties.parent.mkdir(parents=True, exist_ok=True)
    rs.zero_tie_diagnostics(toy["risks"], toy["contracts"][["applicant_id", "L_i"]], rs.EXACT_BUDGET).to_csv(prior_ties, index=False)
    source = Path("src/fair_lending/professor_model/frozen_toy.py")
    (root / source).parent.mkdir(parents=True)
    (root / source).write_text("# Frozen fixture source.\n")
    originals = {str(p.relative_to(root)): file_hash(p) for p in root.rglob("*") if p.is_file()}
    metadata = {"status": "Step_9A_validated_risk_only", "schema_version": rs.SCHEMA_VERSION,
                "resolved_configuration": old_config, "configuration_fingerprint": configuration_fingerprint(old_config),
                "configuration_file_sha256": file_hash(old_path), "source_code_hashes": {str(source): file_hash(root / source)},
                "source_code_fingerprint": configuration_fingerprint({str(source): file_hash(root / source)}),
                "original_checksums": originals, "input_checksums": {},
                "artifacts": {k: {"path": str(risk_paths[k].relative_to(root)), "sha256": file_hash(risk_paths[k])} for k in ("risk_csv", "risk_parquet")}}
    dump(root / config["scenario_metadata"], metadata)
    originals[str(Path(config["scenario_metadata"]))] = file_hash(root / config["scenario_metadata"])
    dump(io.paths(root)["preservation"], {"schema_version": "professor-revised-lending-preservation-v1", "checksums": originals})
    inputs, hashes = io.load_risk_inputs(config, originals, root)
    return root, config, config_path, inputs, hashes, originals


def test_fixed_config_and_no_budget_retuning():
    config = io.load_config()
    assert tuple(config["scenarios"]) == rl.SCENARIO_IDS
    assert config["expected_capital_USD"] == rs.EXACT_BUDGET
    assert config["policy"]["comparison"] == "less_than_or_equal"


def test_delegate_exact_scores_ids_and_contracts(toy):
    before = toy["contracts"].copy(deep=True)
    banks = banks_for(toy)
    for scenario, result in banks.items():
        expected = toy["risks"].loc[toy["risks"].scenario_id == scenario].sort_values("applicant_id")
        np.testing.assert_array_equal(expected.estimated_PD_bounded, result["decisions"].estimated_PD_bounded)
        assert result["checks"]["independent_exact_enumeration_passed"]
        assert set(result["decisions"].applicant_id) == set(toy["contracts"].applicant_id)
        assert not {"B", "PN", "true_PD", "expected_profit_true"}.intersection(result["decisions"].columns)
    pd.testing.assert_frame_equal(before, toy["contracts"], check_exact=True)


@pytest.mark.parametrize("column", ["B", "true_PD", "expected_profit_true", "oracle_approved", "PN"])
def test_evaluator_or_group_columns_rejected_by_optimizer(toy, column):
    risk = toy["risks"].loc[toy["risks"].scenario_id == rl.SCENARIO_IDS[0], rl.RISK_VIEW].assign(**{column: 0})
    with pytest.raises(ValueError): rl.optimize_scenario(toy["contracts"], risk, toy["budget"])


def test_decisions_independent_of_truth_and_group_audit(toy):
    before = banks_for(toy)
    toy["truth"]["expected_profit_true"] *= -100
    toy["truth"]["true_PD"] = .99
    toy["audit"]["B"] = 1 - toy["audit"].B
    after = banks_for(toy)
    for s in rl.SCENARIO_IDS:
        pd.testing.assert_frame_equal(before[s]["decisions"], after[s]["decisions"], check_exact=True)


def test_oversized_zero_block_rejects_everyone(toy):
    direct = banks_for(toy)[rl.SCENARIO_IDS[1]]
    assert direct["checks"]["bounded_zero_count"] == 4
    assert not direct["checks"]["zero_block_feasible"]
    assert direct["selection"]["optimal_tau"] is None
    assert not direct["decisions"].approved.any()
    assert direct["candidates"].budget_feasible.sum() == 1


def test_feasible_zero_block_approved_whole(toy):
    direct = banks_for(toy, rs.EXACT_BUDGET)[rl.SCENARIO_IDS[1]]
    zero = direct["decisions"].estimated_PD_bounded.eq(0)
    assert direct["decisions"].loc[zero, "approved"].all()
    assert direct["checks"]["zero_block_feasible"]


def test_exact_budget_boundary_and_next_float(toy):
    contracts = make_loan_contracts(pd.DataFrame({"applicant_id": ["x", "y"], "Y": [1e9 / 3, 1e-9 / 3]}), T=30, theta=.1, r=.05)
    risk = contracts[["applicant_id"]].assign(scenario_id=rl.SCENARIO_IDS[0], estimated_PD_bounded=0.)
    exact = sum((Fraction(float(v)) for v in contracts.L_i), Fraction())
    rounded = float(exact)
    result = rl.optimize_scenario(contracts, risk, rounded)
    assert result["selection"]["number_approved"] == (2 if exact <= Fraction(rounded) else 0)
    result = rl.optimize_scenario(contracts, risk, np.nextafter(rounded, np.inf))
    assert result["selection"]["number_approved"] == 2


def test_equal_objective_prefers_reject_all_and_lowest_cutoff():
    bank = pd.DataFrame({"applicant_id": ["a", "b"], "model_id": MODEL_ID,
                         "bank_PD_bounded": [0., .1], "requested_principal": [1., 1.], "expected_profit_bank": [0., 0.]}).loc[:, BANK_INPUT_COLUMNS]
    assert choose_cutoff(bank, 2.)[2]["optimal_tau"] is None
    bank["expected_profit_bank"] = [1., 0.]
    assert choose_cutoff(bank, 2.)[2]["optimal_tau"] == 0.


def test_repeatability_and_order_independence(toy):
    before = banks_for(toy)
    toy["risks"] = toy["risks"].iloc[::-1]
    toy["contracts"] = toy["contracts"].iloc[::-1]
    after = banks_for(toy)
    for s in rl.SCENARIO_IDS:
        pd.testing.assert_frame_equal(before[s]["decisions"], after[s]["decisions"], check_exact=True)
        pd.testing.assert_frame_equal(before[s]["candidates"], after[s]["candidates"], check_exact=True)
        assert before[s]["selection"] == after[s]["selection"]


def test_group_gap_profit_signed_comparisons_and_denominators(toy):
    banks = banks_for(toy); policies = policies_for(toy, banks)
    portfolio, groups, overlap = rl.evaluate_policies(policies, toy["contracts"], toy["audit"], toy["truth"], toy["budget"])
    assert len(portfolio) == 6 and len(groups) == 12 and len(overlap) == 60
    aware = portfolio.set_index("policy_id").loc[rl.SCENARIO_IDS[0]]
    direct = portfolio.set_index("policy_id").loc[rl.SCENARIO_IDS[1]]
    assert aware.number_approved == 4 and aware.approval_gap_pp == -50
    assert aware.true_expected_portfolio_profit_USD == 1100
    assert direct.true_profit_minus_frozen_bank_USD == -1100
    assert direct.approval_gap_minus_frozen_bank_pp == 50
    g = groups.loc[(groups.policy_id == rl.SCENARIO_IDS[0]) & groups.B.eq(0)].iloc[0]
    assert g.n_applicants == 4 and g.number_approved == 3 and g.approval_rate == .75
    assert g.funded_principal_per_applicant_USD == (60000 + 120000 + 180000) / 4
    assert portfolio.information_scope.eq(rl.EVALUATOR_SCOPE).all()
    assert groups.information_scope.eq(rl.EVALUATOR_SCOPE).all()
    assert overlap.information_scope.eq(rl.EVALUATOR_SCOPE).all()


def test_overlap_counts_jaccard_and_unfunded_profit_labels(toy):
    policies = policies_for(toy, banks_for(toy))
    _, _, overlaps = rl.evaluate_policies(policies, toy["contracts"], toy["audit"], toy["truth"], toy["budget"])
    pair = overlaps.loc[(overlaps.policy_first == rl.SCENARIO_IDS[0]) & (overlaps.policy_second == rl.SCENARIO_IDS[1])].set_index("allocation_category")
    assert pair.n_applicants.to_dict() == {"approved_by_both": 0, "first_only": 4, "second_only": 0, "rejected_by_both": 4}
    assert pair.jaccard_overlap.eq(0).all()
    assert pair.loc["first_only", "first_policy_expected_profit_contribution_USD"] == 1100
    assert pair.loc["rejected_by_both", "potential_true_expected_profit_USD"] == 2500
    assert pair.loc["rejected_by_both", "first_policy_expected_profit_contribution_USD"] == 0
    assert pair.loc["rejected_by_both", "profit_scope"] == "hypothetical_unfunded_by_both"


def test_reject_all_means_undefined_not_zero_risk(toy):
    p, groups, _ = rl.evaluate_policies(policies_for(toy, banks_for(toy)), toy["contracts"], toy["audit"], toy["truth"], toy["budget"])
    row = p.loc[p.policy_id == rl.SCENARIO_IDS[1]].iloc[0]
    assert pd.isna(row.mean_true_PD_approved) and pd.isna(row.mean_true_profit_per_approved_USD)
    assert row.true_expected_portfolio_profit_USD == 0


def test_postfreeze_tie_counts_and_actions(toy):
    banks = banks_for(toy)
    decisions = pd.concat([v["decisions"] for v in banks.values()])
    ties = rl.zero_tie_outcomes(toy["risks"], decisions, toy["contracts"], toy["budget"])
    direct = ties.loc[ties.scenario_id == rl.SCENARIO_IDS[1]].iloc[0]
    assert direct.B1_zero_share == 1 and direct.B0_zero_share == 0
    assert direct.zero_score_count == 4 and direct.zero_block_action == "rejected_all"
    assert direct.reject_all_selected and not direct.whole_zero_block_approved


def test_tampered_funding_or_oracle_risk_rejected(toy):
    policies = policies_for(toy, banks_for(toy))
    policies[rl.SCENARIO_IDS[0]]["decisions"].loc[0, "funded_principal"] += 1
    with pytest.raises(AssertionError): rl.evaluate_policies(policies, toy["contracts"], toy["audit"], toy["truth"], toy["budget"])
    policies = policies_for(toy, banks_for(toy))
    policies[rl.REFERENCE_IDS[2]]["decisions"]["score"] += .00001
    with pytest.raises(AssertionError): rl.evaluate_policies(policies, toy["contracts"], toy["audit"], toy["truth"], toy["budget"])


def test_truth_cannot_load_before_freeze(saved, monkeypatch):
    root, config, cp, inputs, _, _ = saved
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **k: pytest.fail("truth should not load"))
    with pytest.raises(ValueError, match="saved and frozen"):
        io.load_evaluator_inputs(inputs, config, root=root, config_path=cp)


def test_risk_loader_does_not_read_true_profit_or_reference_masks(saved, monkeypatch):
    root, config, _, _, _, originals = saved
    read = pd.read_parquet
    def guarded(path, *a, **kw):
        assert Path(path).name == "scenario_risk_estimates.parquet"
        return read(path, *a, **kw)
    monkeypatch.setattr(pd, "read_parquet", guarded)
    monkeypatch.setattr(bank_regression, "fit_bank_ols", lambda *a, **k: pytest.fail("refit forbidden"))
    monkeypatch.setattr(group_blind, "fit_group_blind", lambda *a, **k: pytest.fail("refit forbidden"))
    inputs, _ = io.load_risk_inputs(config, originals, root)
    assert len(inputs["risks"]) == 24


def test_immutable_lifecycle_roundtrips_and_references(saved, monkeypatch):
    root, config, cp, inputs, hashes, originals = saved
    toy = {**inputs, "applicants": inputs["audit"]}
    banks = banks_for(toy)
    before = {p: file_hash(root / p) for p in originals}
    freeze = io.freeze_banks(inputs, banks, config, hashes, originals, root=root, config_path=cp)
    assert freeze["status"] == "all_three_policies_frozen_before_evaluation"
    # Once frozen, evaluator loading and reporting must never search/refit.
    monkeypatch.setattr(rl, "choose_cutoff", lambda *a, **k: pytest.fail("post-freeze optimizer call"))
    policies, truth, _ = io.load_evaluator_inputs(inputs, config, root=root, config_path=cp)
    report = io.save_evaluation(inputs, policies, truth, config, originals, root=root, config_path=cp)
    assert report["originals_byte_identical"] and report["pair_count"] == 15
    assert report["model_fits"] == report["oracle_optimizations"] == 0
    assert before == {p: file_hash(root / p) for p in originals}
    assert all(p.is_file() for p in io.paths(root).values())
    for p in (io.paths(root)["profit_figure"], io.paths(root)["approval_figure"]):
        from PIL import Image
        with Image.open(p) as image: image.verify()
    with pytest.raises(FileExistsError): io.freeze_banks(inputs, banks, config, hashes, originals, root=root, config_path=cp)


def test_tampered_bank_freeze_blocks_truth(saved, monkeypatch):
    root, config, cp, inputs, hashes, originals = saved
    io.freeze_banks(inputs, banks_for(inputs), config, hashes, originals, root=root, config_path=cp)
    with io.paths(root)["decisions_csv"].open("a") as stream: stream.write("tampered\n")
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **k: pytest.fail("truth should not load"))
    with pytest.raises(ValueError, match="checksum mismatch"):
        io.load_evaluator_inputs(inputs, config, root=root, config_path=cp)


def test_preservation_manifest_and_path_safety(saved):
    root, _, _, _, _, originals = saved
    assert io.preservation_manifest(root) == originals
    with pytest.raises(ValueError, match="escapes repository"): io.verify_hashes({"../outside": "x"}, root)
    path = root / next(p for p in originals if p.endswith("scenario_risk_estimates.csv"))
    path.write_text("changed")
    with pytest.raises(ValueError, match="checksum mismatch"): io.preservation_manifest(root)


def test_mutated_in_memory_decisions_cannot_be_frozen(saved):
    root, config, cp, inputs, hashes, originals = saved
    banks = banks_for(inputs)
    banks[rl.SCENARIO_IDS[0]]["decisions"].loc[0, "approved"] = False
    with pytest.raises((ValueError, AssertionError)):
        io.freeze_banks(inputs, banks, config, hashes, originals, root=root, config_path=cp)
    assert not io.paths(root)["freeze"].exists()


def test_mutated_perceived_profits_cannot_enter_freeze(saved):
    root, config, cp, inputs, hashes, originals = saved
    banks = banks_for(inputs)
    banks[rl.SCENARIO_IDS[0]]["assessments"].loc[0, "expected_profit_bank"] += 100
    with pytest.raises(AssertionError):
        io.freeze_banks(inputs, banks, config, hashes, originals, root=root, config_path=cp)
    assert not io.paths(root)["freeze"].exists()
