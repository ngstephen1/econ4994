"""Hand arithmetic, evaluation-only accounting, and preserved information boundaries."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import yaml

from fair_lending.professor_model.bank_regression import MODEL_ID, ids_fingerprint
from fair_lending.professor_model.config import configuration_fingerprint
from fair_lending.professor_model.contracts import (
    CONTRACT_COLUMNS, PAYMENT_RULE_ID, make_loan_contracts, scheduled_payments, validate_contracts,
)
from fair_lending.professor_model.economics_io import (
    load_economic_inputs, load_economics_config, load_true_risk_for_economics,
    output_paths, save_economic_artifacts, validate_economic_outputs, validate_economics_config,
)
from fair_lending.professor_model.profit import (
    BANK_PROFIT_COLUMNS, BANK_RISK_COLUMNS, PREVIEW_COLUMNS, PREVIEW_SCOPE,
    TRUE_PROFIT_COLUMNS, assess_bank_profit, assess_true_profit, break_even_probability,
    economic_diagnostics, expected_cashflows, payment_survival,
)
from fair_lending.professor_model.risk_io import file_hash, verify_input_hashes


def _contracts(Y=(1000.,), T=30, r=.05, theta=.1):
    applicants = pd.DataFrame({"applicant_id": [f"test_{i}" for i in range(len(Y))], "Y": Y})
    return make_loan_contracts(applicants, T=T, r=r, theta=theta)


@pytest.fixture
def files(tmp_path):
    """Saved-data fixture only; no population/noise/risk/model generation calls."""
    config = load_economics_config()
    config.update(expected_input_rows=8, evaluation_rows=4, preview_rows=4)
    directory = tmp_path / config["input_directory"]
    directory.mkdir(parents=True)
    config_dir = tmp_path / "configs/professor_model"
    config_dir.mkdir(parents=True)
    step5_config = config_dir / "bank_regression.yaml"
    step5_config.write_text("fixture: true\n", encoding="utf-8")
    config_path = config_dir / "loan_economics.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    ids = [f"fixture_{i}" for i in range(8)]
    bank = pd.DataFrame({"applicant_id": ids, "Y": np.arange(1, 9) * 1000., "D": [.3] * 8, "B": [0, 0, 1, 0, 0, 1, 1, 0], "C": [0.] * 8})
    splits = pd.DataFrame({"applicant_id": ids, "split": ["evaluation", "train"] * 4})
    raw = np.array([-.03, .01, .003, .02, .06, .001, .05, 0.])
    estimates = pd.DataFrame({"applicant_id": ids, "split": splits.split, "model_id": MODEL_ID, "bank_PD_raw": raw, "bank_PD_bounded": np.clip(raw, 0, 1), "prediction_out_of_bounds": (raw < 0) | (raw > 1)})
    truth = pd.DataFrame({"applicant_id": ids, "P_income": .5, "P_other": .5, "true_PD": [.002, .003, .01, .001, .1, .2, .02, .005]})
    artifacts = {}
    for filename, frame in (("applicants_bank_view.parquet", bank), ("applicant_splits.csv", splits), ("bank_risk_estimates.parquet", estimates), ("simulation_truth.parquet", truth)):
        path = directory / filename
        frame.to_csv(path, index=False) if path.suffix == ".csv" else frame.to_parquet(path, index=False)
        artifacts[filename] = {"path": str(path.relative_to(tmp_path)), "sha256": file_hash(path)}
    # Verify that no loader opens target/noise/private tables as data.
    private = directory / "applicants_simulator_private.parquet"
    private.write_bytes(b"opaque private file, cannot be parsed as Parquet")
    signal = directory / "training_signals.parquet"
    signal.write_bytes(b"opaque raw PN file, not needed for Step 6A")
    metadata = {
        "status": "Step_5_validated", "model_id": MODEL_ID,
        "resolved_configuration": {"fixture": True}, "configuration_fingerprint": configuration_fingerprint({"fixture": True}),
        "configuration_file_sha256": file_hash(step5_config), "source_code_hashes": {},
        "input_checksums": {str(p.relative_to(tmp_path)): file_hash(p) for p in (private, signal)},
        "artifacts": artifacts,
        "evaluation_ID_fingerprint": ids_fingerprint(splits.loc[splits.split == "evaluation", "applicant_id"]),
    }
    (directory / "bank_regression_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return config, config_path, tmp_path


def _pipeline(files):
    config, _, root = files
    evaluation, hashes, _ = load_economic_inputs(config, root)
    c = config["contract"]
    contracts = make_loan_contracts(evaluation, T=c["T"], theta=c["theta"], r=c["r"])
    bank = assess_bank_profit(contracts, evaluation.loc[:, BANK_RISK_COLUMNS])
    hidden = load_true_risk_for_economics(evaluation, config, hashes, root)
    truth = assess_true_profit(contracts, hidden)
    return evaluation, hashes, contracts, bank, truth


def test_canonical_working_terms_and_no_extra_accounting():
    config = load_economics_config()
    assert config["contract"] == {"T": 30, "theta": .1, "r": .05, "rate_basis": "per_period", "payment_rule_id": PAYMENT_RULE_ID, "schedule": "ell_times_one_plus_r_power_t", "principal": "T_times_ell"}
    assert config["expected_input_rows"] == 100000 and config["evaluation_rows"] == 20000
    assert config["accounting"] == {"transaction_cost": 0, "discounting": False, "prepayment": False, "recovery": 0}


@pytest.mark.parametrize("section,key,value", [("contract", "rate_basis", "annual"), ("contract", "schedule", "mortgage_amortization"), ("contract", "principal", "ell"), ("risk", "bank_probability", "bank_PD_raw"), ("risk", "interpretation", "whole_loan_default"), ("accounting", "transaction_cost", 5), ("accounting", "discounting", True), ("accounting", "prepayment", True), ("accounting", "recovery", .5)])
def test_unsupported_interpretations_rejected(section, key, value):
    config = load_economics_config()
    config[section][key] = value
    with pytest.raises(ValueError):
        validate_economics_config(config)


def test_principal_identity_and_exact_contract_schema():
    contracts = _contracts((1000., 77777.77))
    assert tuple(contracts.columns) == CONTRACT_COLUMNS
    np.testing.assert_array_equal(contracts.ell_i, .1 * contracts.Y)
    np.testing.assert_array_equal(contracts.L_i, 30 * contracts.ell_i)
    np.testing.assert_allclose(contracts.L_i, 3 * contracts.Y, rtol=1e-15)
    assert contracts.payment_rule_id.eq(PAYMENT_RULE_ID).all()


def test_correct_30_period_schedule_starts_at_one():
    contracts = _contracts()
    schedule = scheduled_payments(contracts)
    assert schedule.shape == (1, 30)
    assert schedule[0, 0] == pytest.approx(105.)
    assert schedule[0, -1] == pytest.approx(100 * 1.05 ** 30)
    np.testing.assert_allclose(schedule[0], [100 * 1.05 ** t for t in range(1, 31)], rtol=1e-14)


def test_two_period_hand_example():
    contracts = _contracts(T=2)
    np.testing.assert_allclose(scheduled_payments(contracts)[0], [105., 110.25], atol=1e-12)
    receipts, profit = expected_cashflows(contracts, np.array([.1]))
    assert receipts[0] == pytest.approx(105 * .9 + 110.25 * .81)
    assert receipts[0] == pytest.approx(183.8025)
    assert profit[0] == pytest.approx(-16.1975)


def test_survival_includes_current_period_and_is_nonincreasing():
    survival = payment_survival(np.array([.1, .2]), 3)
    np.testing.assert_allclose(survival, [[.9, .81, .729], [.8, .64, .512]])
    assert (np.diff(survival, axis=1) <= 0).all()


def test_no_default_receives_all_scheduled_payments():
    contracts = _contracts((1000., 2000.))
    receipts, _ = expected_cashflows(contracts, np.zeros(2))
    np.testing.assert_array_equal(receipts, scheduled_payments(contracts).sum(axis=1))


def test_certain_immediate_default_gives_zero_receipts():
    contracts = _contracts((1000., 2000.))
    receipts, profit = expected_cashflows(contracts, np.ones(2))
    np.testing.assert_array_equal(receipts, [0., 0.])
    np.testing.assert_array_equal(profit, -contracts.L_i)


def test_zero_rate_zero_default_has_zero_profit():
    receipts, profit = expected_cashflows(_contracts(r=0), np.array([0.]))
    assert receipts[0] == pytest.approx(3000., abs=1e-10)
    assert profit[0] == pytest.approx(0., abs=1e-10)


@pytest.mark.parametrize("r", [0., .01, .05, .2])
def test_break_even_formula_and_neighboring_signs(r):
    p = break_even_probability(r)
    assert p == pytest.approx(r / (1 + r), abs=1e-15)
    _, profit = expected_cashflows(_contracts(r=r), np.array([p]))
    assert profit[0] == pytest.approx(0., abs=1e-9)
    if r > 0:
        assert expected_cashflows(_contracts(r=r), np.array([p - .001]))[1][0] > 0
        assert expected_cashflows(_contracts(r=r), np.array([p + .001]))[1][0] < 0


def test_higher_default_risk_never_increases_profit():
    probabilities = np.linspace(0, 1, 101)
    _, profit = expected_cashflows(_contracts((1000.,) * len(probabilities)), probabilities)
    assert (np.diff(profit) < 0).all()


@pytest.mark.parametrize("probability", [np.array([-.01]), np.array([1.01]), np.array([np.nan]), np.array([np.inf]), np.array([[.1]]), np.array([])])
def test_invalid_probabilities_rejected_without_clipping(probability):
    with pytest.raises(ValueError):
        payment_survival(probability, 30)


@pytest.mark.parametrize("terms", [{"T": 0}, {"T": True}, {"T": 30.5}, {"theta": 0}, {"r": -.05}, {"r": float("nan")}])
def test_invalid_contract_terms_rejected(terms):
    with pytest.raises(ValueError):
        _contracts(**terms)


def test_tampered_principal_rejected():
    contracts = _contracts()
    contracts.loc[0, "L_i"] += 1
    with pytest.raises(ValueError, match="identity"):
        validate_contracts(contracts)


def test_bank_and_truth_use_same_formula_separate_inputs():
    contracts = _contracts(T=2)
    bank_risk = pd.DataFrame({"applicant_id": ["test_0"], "model_id": MODEL_ID, "bank_PD_bounded": [.1]})
    bank = assess_bank_profit(contracts, bank_risk)
    truth = assess_true_profit(contracts, pd.DataFrame({"applicant_id": ["test_0"], "true_PD": [.2]}))
    assert bank.expected_receipts_bank.iloc[0] == pytest.approx(183.8025)
    assert truth.expected_receipts_true.iloc[0] == pytest.approx(105 * .8 + 110.25 * .64)
    np.testing.assert_array_equal(bank.expected_profit_bank, bank.expected_receipts_bank - contracts.L_i)
    np.testing.assert_array_equal(truth.expected_profit_true, truth.expected_receipts_true - contracts.L_i)
    assert tuple(bank.columns) == BANK_PROFIT_COLUMNS
    assert tuple(truth.columns) == TRUE_PROFIT_COLUMNS
    assert not set(bank.columns) & {"true_PD", "expected_profit_true", "bank_PD_raw", "PN"}


def test_bank_entry_point_rejects_hidden_or_raw_predictor_columns():
    contracts = _contracts()
    bank_risk = pd.DataFrame({"applicant_id": ["test_0"], "model_id": MODEL_ID, "bank_PD_bounded": [.1], "true_PD": [.01]})
    with pytest.raises(ValueError, match="exact schema"):
        assess_bank_profit(contracts, bank_risk)
    with pytest.raises(ValueError, match="exact schema"):
        assess_bank_profit(contracts, bank_risk.rename(columns={"true_PD": "bank_PD_raw"}))


def test_risk_rows_are_aligned_by_id_not_position():
    contracts = _contracts((1000., 2000.))
    risk = pd.DataFrame({"applicant_id": ["test_1", "test_0"], "true_PD": [1., 0.]})
    result = assess_true_profit(contracts, risk)
    assert result.expected_receipts_true.iloc[0] > 0 and result.expected_receipts_true.iloc[1] == 0
    with pytest.raises(ValueError):
        assess_true_profit(contracts, risk.iloc[:1])


def test_bank_loader_reads_only_saved_observed_tables_and_evaluation(files, monkeypatch):
    config, _, root = files
    original = pd.read_parquet
    seen = []
    def recording(path, *args, **kwargs):
        seen.append(path.name)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(pd, "read_parquet", recording)
    evaluation, _, schemas = load_economic_inputs(config, root)
    assert seen == ["applicants_bank_view.parquet", "bank_risk_estimates.parquet"]
    assert len(schemas) == 3 and len(evaluation) == 4
    assert evaluation.split.eq("evaluation").all()
    assert "true_PD" not in evaluation and "PN" not in evaluation


def test_no_generation_or_regression_fit_called(files, monkeypatch):
    from fair_lending.professor_model import bank_regression, noisy_signal, population, risk
    def forbidden(*args, **kwargs):
        raise AssertionError("existing data/models must not be regenerated/refitted")
    monkeypatch.setattr(bank_regression, "fit_bank_ols", forbidden)
    monkeypatch.setattr(population, "generate_applicants", forbidden)
    monkeypatch.setattr(noisy_signal, "generate_noisy_signal", forbidden)
    monkeypatch.setattr(risk, "compute_true_risk", forbidden)
    _pipeline(files)


def test_original_checksum_tampering_rejected(files):
    config, _, root = files
    (root / config["input_directory"] / "bank_risk_estimates.parquet").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_economic_inputs(config, root)


def test_raw_scores_not_used_in_economics(files):
    evaluation, _, contracts, bank, _ = _pipeline(files)
    assert (evaluation.bank_PD_raw < 0).any()
    evaluation["bank_PD_raw"] = -1000.
    repeated = assess_bank_profit(contracts, evaluation.loc[:, BANK_RISK_COLUMNS])
    pd.testing.assert_frame_equal(repeated, bank, check_exact=True)


def test_truth_changes_do_not_affect_bank_assessments(files):
    evaluation, _, contracts, bank, _ = _pipeline(files)
    truth = assess_true_profit(contracts, pd.DataFrame({"applicant_id": contracts.applicant_id, "true_PD": 1.}))
    assert truth.expected_receipts_true.eq(0).all()
    pd.testing.assert_frame_equal(assess_bank_profit(contracts, evaluation.loc[:, BANK_RISK_COLUMNS]), bank, check_exact=True)


def test_group_summary_and_profit_sign_diagnostics(files):
    evaluation, _, contracts, bank, truth = _pipeline(files)
    summary, distributions, preview = economic_diagnostics(evaluation, contracts, bank, truth)
    overall = summary.loc[summary.scope == "overall"].iloc[0]
    assert overall.n_applicants == 4
    assert overall.fraction_profit_bank_positive == .5
    assert overall.fraction_profit_true_positive == .75
    assert overall.fraction_profit_sign_disagreement == .25
    error = bank.expected_profit_bank.to_numpy() - truth.expected_profit_true.to_numpy()
    assert overall.mean_profit_estimation_error == pytest.approx(error.mean())
    assert overall.profit_MAE == pytest.approx(np.abs(error).mean())
    assert overall.profit_RMSE == pytest.approx(np.sqrt((error ** 2).mean()))
    assert summary.loc[summary.scope != "overall", "n_applicants"].sum() == 4
    for group in (0, 1):
        row = summary.loc[summary.scope == f"B={group}"].iloc[0]
        mask = evaluation.B == group
        assert row.n_applicants == mask.sum()
        assert row.total_requested_principal == pytest.approx(contracts.loc[mask, "L_i"].sum())
        assert row.profit_MAE == pytest.approx(np.abs(error[mask]).mean())
    assert len(distributions) == 24 and not distributions.duplicated(["scope", "variable"]).any()
    assert tuple(preview.columns) == PREVIEW_COLUMNS and preview.information_scope.eq(PREVIEW_SCOPE).all()


def test_reproducibility_and_export_round_trip(files):
    config, config_path, root = files
    evaluation, hashes, contracts, bank, truth = _pipeline(files)
    for original, repeated in zip((evaluation, hashes, contracts, bank, truth), _pipeline(files)):
        pd.testing.assert_frame_equal(original, repeated, check_exact=True) if isinstance(original, pd.DataFrame) else None
    assert validate_economic_outputs(contracts, bank, truth)["valid"]
    paths, report, preview = save_economic_artifacts(evaluation, contracts, bank, truth, config, hashes, root=root, config_path=config_path)
    assert report["valid"] and report["checks"]["original_inputs_unchanged"]
    for name, expected in (("loan_contracts", contracts), ("bank_expected_profit", bank), ("true_expected_profit", truth)):
        pd.testing.assert_frame_equal(pd.read_csv(paths[f"{name}_csv"], float_precision="round_trip"), expected, check_exact=True)
        pd.testing.assert_frame_equal(pd.read_parquet(paths[f"{name}_parquet"]), expected, check_exact=True)
    assert len(preview) == 4 and verify_input_hashes(hashes, root)
    metadata = json.loads(paths["metadata"].read_text())
    assert metadata["random_draws"] == 0 and metadata["regression_refitted"] is False
    assert metadata["source_code_fingerprint"] == configuration_fingerprint(metadata["source_code_hashes"])
    assert metadata["artifacts"]["bank_expected_profit_csv"]["information_role"] == "BANK_OBSERVABLE"
    assert metadata["artifacts"]["true_expected_profit_csv"]["information_role"] == "EVALUATOR_ONLY"
    assert all(file_hash(root / a["path"]) == a["sha256"] for a in metadata["artifacts"].values())
    assert all(path.exists() for path in paths.values())
    with pytest.raises(FileExistsError):
        save_economic_artifacts(evaluation, contracts, bank, truth, config, hashes, root=root, config_path=config_path)


def test_training_applicants_cannot_be_exported_as_evaluation(files):
    config, config_path, root = files
    evaluation, hashes, contracts, bank, truth = _pipeline(files)
    evaluation.loc[0, "split"] = "train"
    with pytest.raises(ValueError, match="evaluation cohort"):
        save_economic_artifacts(evaluation, contracts, bank, truth, config, hashes, root=root, config_path=config_path)
    assert not any(path.exists() for path in output_paths(config, root).values())
