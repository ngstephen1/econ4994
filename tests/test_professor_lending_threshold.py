"""Exact-prefix policy tests; independent enumeration is never a knapsack test."""

from __future__ import annotations

from fractions import Fraction
import json
import math

import numpy as np
import pandas as pd
import pytest
import yaml

from fair_lending.professor_model.bank_regression import MODEL_ID, ids_fingerprint
from fair_lending.professor_model.config import configuration_fingerprint
from fair_lending.professor_model.contracts import make_loan_contracts
from fair_lending.professor_model.lending import (
    BANK_INPUT_COLUMNS, DECISION_COLUMNS, capital_budget, choose_cutoff,
    evaluate_portfolio, validate_bank_inputs, validate_solution,
)
from fair_lending.professor_model.lending_io import (
    freeze_lending_decisions, load_evaluator_inputs, load_lending_config,
    load_lending_inputs, output_paths, save_evaluator_results,
)
from fair_lending.professor_model.profit import assess_bank_profit, assess_true_profit
from fair_lending.professor_model.risk_io import file_hash, verify_input_hashes


def _bank(risks=(0., .1, .1, .2), principals=(40., 30., 30., 10.), profits=(10., 20., 20., 100.)):
    return pd.DataFrame({"applicant_id": [f"toy_{i:03d}" for i in range(len(risks))], "model_id": MODEL_ID, "bank_PD_bounded": np.array(risks, dtype=float), "requested_principal": np.array(principals, dtype=float), "expected_profit_bank": np.array(profits, dtype=float)}).loc[:, BANK_INPUT_COLUMNS]


def _exhaustive_thresholds(bank, budget):
    best_tau, best_value, best_ids = None, Fraction(0), set()
    for tau in [None, *sorted(bank.bank_PD_bounded.unique())]:
        mask = np.zeros(len(bank), dtype=bool) if tau is None else bank.bank_PD_bounded <= tau
        principal = sum((Fraction(float(v)) for v in bank.loc[mask, "requested_principal"]), Fraction(0))
        value = sum((Fraction(float(v)) for v in bank.loc[mask, "expected_profit_bank"]), Fraction(0))
        if principal <= Fraction(float(budget)) and value > best_value:
            best_tau, best_value, best_ids = tau, value, set(bank.loc[mask, "applicant_id"])
    return best_tau, float(best_value), best_ids


def test_canonical_policy_is_bounded_40_percent_common_cutoff():
    config = load_lending_config()
    assert config["evaluation_rows"] == 20000
    assert config["policy"]["budget_fraction"] == .4
    assert config["policy"]["risk_column"] == "bank_PD_bounded"
    assert config["policy"]["tied_risks"] == "fund_all_or_none"


@pytest.mark.parametrize("key,value", [("risk_column", "bank_PD_raw"), ("budget_fraction", .5), ("tied_risks", "split_to_fill_budget"), ("profit_tie_break", "highest_true_profit")])
def test_changed_policy_rejected(tmp_path, key, value):
    config = load_lending_config()
    config["policy"][key] = value
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError):
        load_lending_config(path)


def test_ties_not_split_and_no_skipping_oversized_lower_risk_prefix():
    bank = _bank()
    decisions, candidates, summary = choose_cutoff(bank, 80.)
    assert summary["optimal_tau"] == 0 and decisions.approved.tolist() == [True, False, False, False]
    assert candidates.number_approved.tolist() == [0, 1, 3, 4]
    assert decisions.funded_principal.tolist() == [40., 0., 0., 0.]
    # Even though the final $10 request fits, it cannot bypass the tied prefix.
    assert summary["unused_capital"] == 40 and summary["capital_limits_next_cutoff"]


def test_profitable_largest_feasible_cutoff_is_verified_not_assumed():
    bank = _bank((.1, .2, .3), (20., 20., 20.), (10., -20., 100.))
    decisions, _, summary = choose_cutoff(bank, 40.)
    assert summary["optimal_tau"] == .1
    assert summary["largest_feasible_cutoff_selected"] is False
    assert decisions.approved.sum() == 1


@pytest.mark.parametrize("profits", [(-1., -2.), (0., 0.)])
def test_no_lending_is_explicit_and_wins_zero_objective_tie(profits):
    decisions, candidates, summary = choose_cutoff(_bank((0., 0.), (1., 1.), profits), 10.)
    assert summary["optimal_tau"] is None and summary["bank_expected_portfolio_profit"] == 0
    assert decisions.optimal_tau.isna().all() and not decisions.approved.any()
    assert candidates.iloc[0].candidate_type == "no_lending" and candidates.iloc[0].selected


def test_zero_risk_group_cannot_be_partially_funded():
    decisions, _, summary = choose_cutoff(_bank((0., 0.), (10., 10.), (10., 20.)), 15.)
    assert summary["optimal_tau"] is None and summary["zero_risk_approved"] == 0
    assert not decisions.approved.any()


def test_lowest_cutoff_tie_break():
    _, _, summary = choose_cutoff(_bank((.1, .2), (10., 10.), (5., 0.)), 20.)
    assert summary["optimal_tau"] == .1


def test_exact_objective_comparison_not_rounded_tie():
    _, candidates, summary = choose_cutoff(_bank((.1, .2), (1., 1.), (1., 1e-20)), 2.)
    assert candidates.iloc[1].bank_expected_portfolio_profit == candidates.iloc[2].bank_expected_portfolio_profit
    assert summary["optimal_tau"] == .2  # exact underlying sum is strictly larger


def test_no_budget_slack_even_at_float_boundary():
    decisions, candidates, summary = choose_cutoff(_bank((.1, .2), (.1, .2), (1., 1.)), .3)
    assert summary["number_approved"] == 1
    assert candidates.iloc[-1].budget_feasible == False
    assert math.fsum(decisions.funded_principal) <= .3


def test_exact_large_dollar_budget_boundary_remains_feasible():
    bank = _bank((0., .1), (300000000.125, 299999999.875), (1e8, 2e8))
    _, _, summary = choose_cutoff(bank, 600000000.)
    assert summary["number_approved"] == 2 and summary["exact_budget_exhausted"]
    assert summary["unused_capital"] == 0


@pytest.mark.parametrize("budget", [0., 40., 80., 100., 110., 200.])
def test_solution_matches_exhaustive_cutoff_enumeration(budget):
    bank = _bank()
    decisions, candidates, summary = choose_cutoff(bank, budget)
    tau, value, ids = _exhaustive_thresholds(bank, budget)
    assert summary["optimal_tau"] == tau and summary["bank_expected_portfolio_profit"] == value
    assert set(decisions.loc[decisions.approved, "applicant_id"]) == ids
    assert validate_solution(bank, budget, decisions, candidates, summary)["maximum_perceived_profit_among_feasible_cutoffs"]


def test_all_approved_and_rejected_obey_same_cutoff_and_full_principal():
    bank = _bank()
    decisions, _, summary = choose_cutoff(bank, 100.)
    np.testing.assert_array_equal(decisions.approved, decisions.bank_PD_bounded <= summary["optimal_tau"])
    np.testing.assert_array_equal(decisions.funded_principal, decisions.approved * decisions.requested_principal)
    assert (decisions.loc[~decisions.approved, "bank_PD_bounded"] > summary["optimal_tau"]).all()
    assert tuple(decisions.columns) == DECISION_COLUMNS


def test_reruns_and_input_order_are_deterministic():
    original = choose_cutoff(_bank(), 100.)
    repeated = choose_cutoff(_bank().sample(frac=1, random_state=83), 100.)
    for first, second in zip(original[:2], repeated[:2]):
        pd.testing.assert_frame_equal(first, second, check_exact=True)
    assert original[2] == repeated[2]


@pytest.mark.parametrize("column", ["true_PD", "expected_profit_true", "B", "bank_PD_raw", "PN"])
def test_non_bank_information_excluded_from_optimizer(column):
    bank = _bank()
    bank[column] = 1e9
    with pytest.raises(ValueError, match="observable-only"):
        choose_cutoff(bank, 100.)


@pytest.mark.parametrize("column,value", [("bank_PD_bounded", -.01), ("bank_PD_bounded", 1.1), ("requested_principal", 0), ("expected_profit_bank", np.nan)])
def test_invalid_bank_inputs_rejected(column, value):
    bank = _bank()
    bank.loc[0, column] = value
    with pytest.raises(ValueError):
        validate_bank_inputs(bank)


@pytest.fixture
def files(tmp_path):
    config = load_lending_config()
    config.update(expected_population_rows=8, evaluation_rows=4)
    synthetic = tmp_path / config["synthetic_directory"]
    processed = tmp_path / config["input_directory"]
    conf_dir = tmp_path / "configs/professor_model"
    for directory in (synthetic, processed, conf_dir):
        directory.mkdir(parents=True)
    config_path = conf_dir / "lending_threshold.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    old_config = conf_dir / "loan_economics.yaml"
    old_config.write_text("fixture: true\n", encoding="utf-8")
    ids = [f"saved_{i}" for i in range(8)]
    applicants = pd.DataFrame({"applicant_id": ids, "Y": np.arange(1, 9) * 1000., "D": .3, "B": [0, 0, 1, 0, 0, 1, 1, 0], "C": 0.})
    splits = pd.DataFrame({"applicant_id": ids, "split": ["evaluation", "train"] * 4})
    raw = np.array([-.01, .001, -.02, .002, .001, .003, .002, .004])
    estimates = pd.DataFrame({"applicant_id": ids, "split": splits.split, "model_id": MODEL_ID, "bank_PD_raw": raw, "bank_PD_bounded": np.clip(raw, 0, 1), "prediction_out_of_bounds": (raw < 0) | (raw > 1)})
    selected = applicants.loc[splits.split == "evaluation"]
    contracts = make_loan_contracts(selected, T=30, theta=.1, r=.05)
    bank = assess_bank_profit(contracts, estimates.loc[splits.split == "evaluation", ["applicant_id", "model_id", "bank_PD_bounded"]])
    truth = assess_true_profit(contracts, pd.DataFrame({"applicant_id": selected.applicant_id, "true_PD": .001}))
    originals = {}
    for path, frame in ((synthetic / "applicants_bank_view.parquet", applicants), (synthetic / "applicant_splits.csv", splits), (synthetic / "bank_risk_estimates.parquet", estimates), (processed / "loan_contracts.parquet", contracts), (processed / "bank_expected_profit.parquet", bank), (processed / "true_expected_profit.parquet", truth)):
        frame.to_csv(path, index=False) if path.suffix == ".csv" else frame.to_parquet(path, index=False)
        originals[path.name] = {"path": str(path.relative_to(tmp_path)), "sha256": file_hash(path)}
    metadata = {"status": "Step_6A_validated", "resolved_configuration": {"fixture": True}, "configuration_fingerprint": configuration_fingerprint({"fixture": True}), "configuration_file_sha256": file_hash(old_config), "source_code_hashes": {}, "input_checksums": {}, "artifacts": originals, "evaluation_ID_fingerprint": ids_fingerprint(selected.applicant_id)}
    (processed / "loan_economics_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return config, config_path, tmp_path


def _frozen(files):
    config, config_path, root = files
    bank, hashes, _ = load_lending_inputs(config, root)
    _, budget = capital_budget(bank, .4)
    decisions, candidates, summary = choose_cutoff(bank, budget)
    checkpoint = freeze_lending_decisions(bank, decisions, candidates, summary, config, hashes, root=root, config_path=config_path)
    return bank, hashes, decisions, candidates, summary, checkpoint


def test_bank_loader_does_not_read_true_profit(files, monkeypatch):
    config, _, root = files
    original = pd.read_parquet
    seen = []
    def recording(path, *args, **kwargs):
        seen.append(path.name)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(pd, "read_parquet", recording)
    bank, _, _ = load_lending_inputs(config, root)
    assert seen == ["loan_contracts.parquet", "bank_expected_profit.parquet"]
    assert len(bank) == 4 and tuple(bank.columns) == BANK_INPUT_COLUMNS


def test_truth_loader_requires_saved_decisions(files):
    config, _, root = files
    bank, hashes, _ = load_lending_inputs(config, root)
    with pytest.raises(ValueError, match="saved and frozen"):
        load_evaluator_inputs(bank, {}, config, hashes, root)


def test_tampered_frozen_decision_prevents_truth_loading(files):
    config, _, root = files
    bank, hashes, _, _, _, checkpoint = _frozen(files)
    output_paths(config, root)["decisions_csv"].write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="frozen bank decision artifact changed"):
        load_evaluator_inputs(bank, checkpoint, config, hashes, root)


def test_original_input_tampering_is_not_silently_accepted(files):
    config, _, root = files
    (root / config["input_directory"] / "bank_expected_profit.parquet").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_lending_inputs(config, root)


def test_saved_artifacts_audit_and_truth_invariance(files):
    config, config_path, root = files
    bank, hashes, decisions, candidates, summary, checkpoint = _frozen(files)
    audit, truth = load_evaluator_inputs(bank, checkpoint, config, hashes, root)
    groups, evaluated = evaluate_portfolio(bank, decisions, audit, truth)
    assert groups.number_approved.sum() == summary["number_approved"]
    for group in groups.to_dict("records"):
        assert group["average_funded_principal_per_applicant"] == group["total_funded_principal"] / group["n_applicants"]
    assert evaluated["B1_minus_B0_approval_gap_pp"] == 0
    assert evaluated["profit_estimation_error"] == evaluated["bank_expected_portfolio_profit"] - evaluated["true_expected_portfolio_profit"]
    changed = truth.copy()
    changed["expected_profit_true"] = -1e9
    evaluate_portfolio(bank, decisions, audit, changed)
    rebuilt, _, repeated = choose_cutoff(bank, summary["available_bank_capital"])
    pd.testing.assert_frame_equal(rebuilt, decisions, check_exact=True)
    assert repeated == summary
    report = save_evaluator_results(bank, decisions, candidates, summary, audit, truth, checkpoint, config, hashes, root=root, config_path=config_path)
    assert report["valid"] and verify_input_hashes(hashes, root)
    paths = output_paths(config, root)
    pd.testing.assert_frame_equal(pd.read_parquet(paths["decisions_parquet"]), pd.read_csv(paths["decisions_csv"], float_precision="round_trip"), check_exact=True)
    assert tuple(pd.read_csv(paths["decisions_csv"]).columns) == DECISION_COLUMNS
    metadata = json.loads(paths["metadata"].read_text())
    assert metadata["risk_models_refitted"] is False and metadata["random_draws"] == 0
    assert metadata["source_code_fingerprint"] == configuration_fingerprint(metadata["source_code_hashes"])
    assert all(file_hash(root / a["path"]) == a["sha256"] for a in metadata["artifacts"].values())
    assert all(path.exists() for path in paths.values())
    with pytest.raises(FileExistsError):
        freeze_lending_decisions(bank, decisions, candidates, summary, config, hashes, root=root, config_path=config_path)


def test_freeze_rejects_wrong_budget_fraction(files):
    config, config_path, root = files
    bank, hashes, _ = load_lending_inputs(config, root)
    decisions, candidates, summary = choose_cutoff(bank, math.fsum(bank.requested_principal))
    with pytest.raises(ValueError, match="40%"):
        freeze_lending_decisions(bank, decisions, candidates, summary, config, hashes, root=root, config_path=config_path)
