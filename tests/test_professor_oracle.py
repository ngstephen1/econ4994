"""True-risk oracle, immutable bank, exact search reuse, and matched audits."""

from __future__ import annotations

from fractions import Fraction
import json
import math
import shutil

import numpy as np
import pandas as pd
import pytest
import yaml

from fair_lending.professor_model.contracts import make_loan_contracts
from fair_lending.professor_model.config import PROJECT_ROOT
from fair_lending.professor_model.oracle import ORACLE_COLUMNS, ORACLE_ID, SCOPE, compare_policies, optimize_oracle
from fair_lending.professor_model.oracle_io import (
    load_oracle_config, load_oracle_inputs, output_paths, save_oracle_results,
)
from fair_lending.professor_model.profit import assess_true_profit
from fair_lending.professor_model.risk_io import file_hash, verify_input_hashes

# Reuse the saved-data test fixture and bank freeze pipeline, not research data.
from test_professor_lending_threshold import files as saved_files, _frozen
from fair_lending.professor_model.lending_io import load_evaluator_inputs, save_evaluator_results


def _toy(risks=(.001, .002, .003), principals=(10., 20., 30.)):
    applicants = pd.DataFrame({"applicant_id": [f"test_{i}" for i in range(len(risks))], "Y": np.asarray(principals) / 3})
    contracts = make_loan_contracts(applicants, T=30, theta=.1, r=.05)
    truth = assess_true_profit(contracts, pd.DataFrame({"applicant_id": applicants.applicant_id, "true_PD": risks}))
    return contracts, truth


def _enumerate(contracts, truth, budget):
    joined = contracts.merge(truth, on="applicant_id", validate="one_to_one")
    best, selected_tau, ids = Fraction(0), None, set()
    for tau in [None, *sorted(joined.true_PD.unique())]:
        selected = joined.iloc[:0] if tau is None else joined.loc[joined.true_PD <= tau]
        cost = sum((Fraction(float(x)) for x in selected.L_i), Fraction(0))
        profit = sum((Fraction(float(x)) for x in selected.expected_profit_true), Fraction(0))
        if cost <= Fraction(float(budget)) and profit > best:
            best, selected_tau, ids = profit, tau, set(selected.applicant_id)
    return selected_tau, float(best), ids


@pytest.fixture
def artifacts(saved_files):
    config6b, config_path6b, root = saved_files
    bank, hashes, decisions, candidates, selection, checkpoint = _frozen(saved_files)
    audit, truth = load_evaluator_inputs(bank, checkpoint, config6b, hashes, root)
    save_evaluator_results(bank, decisions, candidates, selection, audit, truth, checkpoint, config6b, hashes, root=root, config_path=config_path6b)
    # Mirror the hashed real source into the isolated repository fixture;
    # do not weaken input-provenance checks to accommodate the fixture.
    metadata = json.loads((root / config6b["input_directory"] / "lending_metadata.json").read_text())
    for relative in metadata["source_code_hashes"]:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT_ROOT / relative, destination)
    config = load_oracle_config()
    config["evaluation_rows"] = 4
    path = root / "configs/professor_model/oracle_comparison.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config, path, root


def test_oracle_configuration_is_evaluator_only():
    config = load_oracle_config()
    assert config["oracle_score"] == "true_PD"
    assert config["oracle_objective"] == "expected_profit_true"
    assert config["budget_field"] == "selection.available_bank_capital"
    assert config["information_scope"] == "EVALUATOR_ONLY"
    assert "budget_fraction" not in config


@pytest.mark.parametrize("key,value", [("oracle_score", "bank_PD_bounded"), ("oracle_objective", "expected_profit_bank"), ("information_scope", "BANK"), ("budget_field", "recalculate_40pct")])
def test_changed_oracle_semantics_rejected(tmp_path, key, value):
    config = load_oracle_config()
    config[key] = value
    path = tmp_path / "wrong.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError):
        load_oracle_config(path)


@pytest.mark.parametrize("budget", [0., 9., 10., 30., 60., 100.])
def test_oracle_matches_exhaustive_true_cutoff_enumeration(budget):
    contracts, truth = _toy()
    decisions, candidates, selection = optimize_oracle(contracts, truth, budget)
    tau, profit, ids = _enumerate(contracts, truth, budget)
    assert selection["optimal_tau"] == tau and selection["true_expected_portfolio_profit"] == profit
    assert set(decisions.loc[decisions.approved, "applicant_id"]) == ids
    assert math.fsum(decisions.funded_principal) <= budget
    assert candidates.iloc[0].candidate_type == "no_lending"
    assert tuple(decisions.columns) == ORACLE_COLUMNS
    assert decisions.information_scope.eq(SCOPE).all() and decisions.policy_id.eq(ORACLE_ID).all()
    assert not any("bank" in name for name in decisions.columns)


def test_true_risk_order_and_ties_not_bank_predictions_drive_oracle():
    contracts, truth = _toy((.003, .001, .001), (10., 20., 30.))
    decisions, candidates, selection = optimize_oracle(contracts, truth, 40.)
    assert selection["optimal_tau"] is None  # the 50-dollar lowest-risk tie cannot be split
    assert not decisions.approved.any()
    assert candidates.number_approved.tolist() == [0, 2, 3]
    decisions, _, selection = optimize_oracle(contracts, truth, 55.)
    assert selection["optimal_tau"] == .001
    assert decisions.approved.tolist() == [False, True, True]


def test_exact_search_is_reused_once_with_explicit_true_adapter(monkeypatch):
    import fair_lending.professor_model.oracle as module
    original = module.choose_cutoff
    seen = []
    def observed(frame, budget):
        seen.append(frame.copy())
        return original(frame, budget)
    monkeypatch.setattr(module, "choose_cutoff", observed)
    contracts, truth = _toy()
    optimize_oracle(contracts, truth, 30.)
    assert len(seen) == 1
    np.testing.assert_array_equal(seen[0].bank_PD_bounded, truth.true_PD)
    np.testing.assert_array_equal(seen[0].expected_profit_bank, truth.expected_profit_true)


def test_nonfinite_or_mismatched_oracle_truth_rejected():
    contracts, truth = _toy()
    with pytest.raises(ValueError):
        optimize_oracle(contracts, truth.iloc[:2], 30.)
    truth.loc[0, "true_PD"] = np.nan
    with pytest.raises(ValueError):
        optimize_oracle(contracts, truth, 30.)


def test_oracle_input_order_does_not_change_result():
    contracts, truth = _toy()
    first = optimize_oracle(contracts, truth, 30.)
    second = optimize_oracle(contracts.iloc[::-1], truth.iloc[::-1], 30.)
    for a, b in zip(first[:2], second[:2]):
        pd.testing.assert_frame_equal(a, b, check_exact=True)
    assert first[2] == second[2]


def test_bank_not_reoptimized_and_absolute_budget_loaded(artifacts, monkeypatch):
    config, _, root = artifacts
    import fair_lending.professor_model.lending as lending
    import fair_lending.professor_model.bank_regression as regression
    def forbidden(*args, **kwargs):
        raise AssertionError("bank must never be refitted or reoptimized")
    monkeypatch.setattr(lending, "choose_cutoff", forbidden)
    monkeypatch.setattr(regression, "fit_bank_ols", forbidden)
    inputs, hashes = load_oracle_inputs(config, root)
    saved = json.loads((root / config["budget_source"]).read_text())["selection"]
    assert inputs["bank_selection"] == saved
    assert len(inputs["contracts"]) == 4 and verify_input_hashes(hashes, root)
    # The oracle imported the original function before monkeypatching; bank search is never invoked here.
    oracle, _, chosen = optimize_oracle(inputs["contracts"], inputs["truth"], saved["available_bank_capital"])
    assert chosen["available_bank_capital"] == saved["available_bank_capital"]
    original = pd.read_parquet(root / config["input_directory"] / "lending_decisions.parquet")
    pd.testing.assert_frame_equal(inputs["bank_decisions"], original, check_exact=True)


def test_group_denominators_disagreements_and_signed_regret(artifacts):
    config, _, root = artifacts
    inputs, _ = load_oracle_inputs(config, root)
    oracle, _, selection = optimize_oracle(inputs["contracts"], inputs["truth"], inputs["bank_selection"]["available_bank_capital"])
    bank_copy = inputs["bank_decisions"].copy(deep=True)
    summary, groups, disagreements, comparison = compare_policies(inputs["contracts"], inputs["truth"], inputs["audit"], bank_copy, inputs["bank_selection"], oracle, selection)
    assert disagreements.n_applicants.sum() == 4 and len(disagreements) == 4
    for policy in ("bank", "oracle"):
        records = groups.loc[groups.policy == policy]
        assert records.n_applicants.sum() == 4
        for row in records.itertuples():
            assert row.approval_rate == row.number_approved / row.n_applicants
            assert row.funded_principal_per_applicant == row.total_funded_principal / row.n_applicants
    indexed = summary.set_index("policy")
    expected = indexed.loc["oracle", "true_expected_portfolio_profit"] - indexed.loc["bank", "true_expected_portfolio_profit"]
    assert comparison["economic_regret"] == expected
    # All true risks tied: oracle rejects all, bank has a positive feasible portfolio.
    # This concrete counterexample disproves automatic nonnegative regret across rankings.
    assert expected < 0 and comparison["regret_nonnegative_within_tolerance"] is False
    assert comparison["economic_regret_pct_of_oracle"] is None
    pd.testing.assert_frame_equal(bank_copy, inputs["bank_decisions"], check_exact=True)


def test_different_absolute_budget_rejected(artifacts):
    config, _, root = artifacts
    inputs, _ = load_oracle_inputs(config, root)
    oracle, _, selection = optimize_oracle(inputs["contracts"], inputs["truth"], inputs["bank_selection"]["available_bank_capital"] + 1)
    with pytest.raises(ValueError, match="exact saved bank budget"):
        compare_policies(inputs["contracts"], inputs["truth"], inputs["audit"], inputs["bank_decisions"], inputs["bank_selection"], oracle, selection)


def test_changed_saved_bank_decisions_are_rejected(artifacts):
    config, _, root = artifacts
    path = root / config["input_directory"] / "lending_decisions.parquet"
    path.write_bytes(b"changed bank file")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_oracle_inputs(config, root)


def test_positive_regret_percentage_overlap_and_gap_arithmetic():
    from fair_lending.professor_model.lending import choose_cutoff
    from fair_lending.professor_model.bank_regression import MODEL_ID
    contracts, truth = _toy((.001, .002, .003), (10., 10., 10.))
    bank = pd.DataFrame({"applicant_id": contracts.applicant_id, "model_id": MODEL_ID, "bank_PD_bounded": [.3, .2, .1], "requested_principal": contracts.L_i, "expected_profit_bank": [1., 1., 1.]})
    bank_decisions, _, bank_selection = choose_cutoff(bank, 15.)
    oracle, _, selection = optimize_oracle(contracts, truth, 15.)
    audit = pd.DataFrame({"applicant_id": contracts.applicant_id, "B": [0, 0, 1]})
    _, _, disagreements, comparison = compare_policies(contracts, truth, audit, bank_decisions, bank_selection, oracle, selection)
    profit = truth.set_index("applicant_id").expected_profit_true
    expected = profit.loc["test_0"] - profit.loc["test_2"]
    assert comparison["economic_regret"] == pytest.approx(expected)
    assert comparison["economic_regret_pct_of_oracle"] == pytest.approx(100 * expected / profit.loc["test_0"])
    assert comparison["jaccard_overlap"] == 0
    assert disagreements.n_applicants.tolist() == [0, 1, 1, 1]
    assert comparison["bank_group_approval_gap_pp"] == 100.
    assert comparison["oracle_group_approval_gap_pp"] == -50.
    assert comparison["bank_minus_oracle_group_gap_pp"] == 150.


def test_export_round_trips_preserves_originals_and_separates_truth(artifacts):
    config, config_path, root = artifacts
    inputs, hashes = load_oracle_inputs(config, root)
    frozen_bank = inputs["bank_decisions"].copy(deep=True)
    decisions, candidates, selection = optimize_oracle(inputs["contracts"], inputs["truth"], inputs["bank_selection"]["available_bank_capital"])
    report = save_oracle_results(inputs, decisions, candidates, selection, config, hashes, root=root, config_path=config_path)
    assert report["valid"] and verify_input_hashes(hashes, root)
    paths = output_paths(config, root)
    pd.testing.assert_frame_equal(pd.read_parquet(paths["oracle_parquet"]), decisions, check_exact=True)
    pd.testing.assert_frame_equal(pd.read_csv(paths["oracle_csv"], float_precision="round_trip"), decisions, check_exact=True)
    pd.testing.assert_frame_equal(inputs["bank_decisions"], frozen_bank, check_exact=True)
    assert "true_PD" not in inputs["bank"].columns and "true_PD" not in frozen_bank
    metadata = json.loads(paths["metadata"].read_text())
    assert metadata["bank_reoptimized"] is False and metadata["models_refitted"] is False and metadata["random_draws"] == 0
    assert all(a["information_role"] == SCOPE and file_hash(root / a["path"]) == a["sha256"] for a in metadata["artifacts"].values())
    assert all(p.exists() for p in paths.values())
    with pytest.raises(FileExistsError):
        save_oracle_results(inputs, decisions, candidates, selection, config, hashes, root=root, config_path=config_path)
