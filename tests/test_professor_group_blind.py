"""Step 7B deterministic toy tests; no research population is regenerated."""
from __future__ import annotations

from fractions import Fraction
import json
import math

import numpy as np
import pandas as pd
import pytest
import yaml

from fair_lending.professor_model import group_blind as gb
from fair_lending.professor_model import group_blind_io as io
from fair_lending.professor_model.bank_regression import MODEL_ID as AWARE_ID, ids_fingerprint
from fair_lending.professor_model.config import configuration_fingerprint
from fair_lending.professor_model.contracts import make_loan_contracts
from fair_lending.professor_model.lending import BANK_INPUT_COLUMNS, choose_cutoff
from fair_lending.professor_model.oracle import optimize_oracle
from fair_lending.professor_model.profit import assess_true_profit, expected_cashflows
from fair_lending.professor_model.risk_io import file_hash, verify_input_hashes


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


@pytest.fixture
def toy():
    i = np.arange(200)
    observed = pd.DataFrame({"applicant_id": [f"toy_{j:03}" for j in i], "Y": 30000 + i**2 * 3., "D": .2 + (i % 7) / 30, "B": (i % 4 == 0).astype(int), "C": np.sin(i), "PN": .004 - 1e-8 * (30000 + i**2 * 3.) + .002 * (.2 + (i % 7) / 30) - .0002 * np.sin(i) + .0005 * np.cos(i), "split": np.where(i < 150, "train", "evaluation")})
    evaluation = observed.loc[observed.split == "evaluation"]
    contracts = make_loan_contracts(evaluation[["applicant_id", "Y"]], T=30, theta=.1, r=.05)
    aware = observed[["applicant_id", "split"]].assign(model_id=AWARE_ID, bank_PD_raw=.004 - 1e-8 * observed.Y + .002 * observed.D - .0002 * observed.C + .001 * observed.B)
    aware["bank_PD_bounded"] = np.clip(aware.bank_PD_raw, 0, 1)
    aware["prediction_out_of_bounds"] = (aware.bank_PD_raw < 0) | (aware.bank_PD_raw > 1)
    aware_eval = aware.loc[aware.split == "evaluation"].reset_index(drop=True)
    receipts, profits = expected_cashflows(contracts, aware_eval.bank_PD_bounded.to_numpy())
    bank_profit = aware_eval[["applicant_id", "model_id", "bank_PD_bounded"]].assign(expected_receipts_bank=receipts, expected_profit_bank=profits)
    bank = contracts[["applicant_id", "L_i"]].merge(bank_profit, on="applicant_id").rename(columns={"L_i": "requested_principal"}).loc[:, BANK_INPUT_COLUMNS]
    budget = .4 * math.fsum(contracts.L_i)
    decisions, _, selection = choose_cutoff(bank, budget)
    truth = assess_true_profit(contracts, evaluation[["applicant_id"]].assign(true_PD=.002 + .0003 * np.cos(np.arange(len(evaluation)))))
    oracle, _, oracle_selection = optimize_oracle(contracts, truth, budget)
    return {"observed": observed, "aware": aware, "contracts": contracts, "bank_profit": bank_profit, "bank_decisions": decisions, "bank_selection": selection, "original_model": {"training_ID_fingerprint": ids_fingerprint(observed.loc[observed.split == "train", "applicant_id"]), "training_count": 150, "training_mean_PN": float(observed.loc[observed.split == "train", "PN"].mean())}, "truth": truth, "oracle": oracle, "oracle_selection": oracle_selection}


def _new(toy):
    model = gb.fit_group_blind(toy["observed"].loc[:, gb.FIT_COLUMNS])
    predictions = gb.risk_estimates(toy["observed"], model)
    profit = gb.assess_profit(toy["contracts"], predictions.loc[predictions.split == "evaluation", ["applicant_id", "model_id", gb.BOUNDED]])
    decisions, candidates, selection = gb.optimize_cutoff(toy["contracts"], profit, toy["bank_selection"]["available_bank_capital"])
    return model, predictions, profit, decisions, candidates, selection


@pytest.fixture
def saved(toy, tmp_path, monkeypatch):
    """Mirror frozen toy inputs. Stub only the previously tested Step 5 loader."""
    root = tmp_path
    config = io.load_config()
    config.update(expected_population_rows=200, training_rows=150, evaluation_rows=50)
    config_path = root / "configs/professor_model/group_blind_comparison.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump(config))
    original = {
        io.SYNTHETIC / "applicant_splits.csv": toy["observed"][["applicant_id", "split"]],
        io.SYNTHETIC / "bank_risk_estimates.parquet": toy["aware"],
        io.PROCESSED / "loan_contracts.parquet": toy["contracts"],
        io.PROCESSED / "bank_expected_profit.parquet": toy["bank_profit"],
        io.PROCESSED / "lending_decisions.parquet": toy["bank_decisions"],
        io.PROCESSED / "true_expected_profit.parquet": toy["truth"],
        io.PROCESSED / "oracle_lending_decisions.parquet": toy["oracle"],
    }
    for relative, frame in original.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".csv":
            frame.to_csv(path, index=False)
        else:
            frame.to_parquet(path, index=False)
    _write_json(root / io.METRICS / "bank_regression_model.json", toy["original_model"])
    _write_json(root / io.METRICS / "lending_selection.json", {"selection": toy["bank_selection"]})
    _write_json(root / io.METRICS / "oracle_comparison_validation.json", {"oracle_selection": toy["oracle_selection"]})
    old_config = root / "configs/professor_model/oracle_comparison.yaml"
    old_config.write_text("fixture: true\n")
    hashes = {str(p.relative_to(root)): file_hash(p) for p in root.rglob("*") if p.is_file() and p != config_path}
    metadata = {"status": "Step_7A_validated", "resolved_configuration": {"fixture": True}, "configuration_fingerprint": configuration_fingerprint({"fixture": True}), "source_code_hashes": {}, "source_code_fingerprint": configuration_fingerprint({}), "input_checksums": hashes, "artifacts": {}, "configuration_file_sha256": file_hash(old_config), "exact_saved_budget": toy["bank_selection"]["available_bank_capital"]}
    _write_json(root / io.PROCESSED / "oracle_comparison_metadata.json", metadata)
    monkeypatch.setattr(io, "load_bank_config", lambda path: {})
    monkeypatch.setattr(io, "load_bank_inputs", lambda config, root: (toy["observed"].drop(columns="split"), {}, []))
    return root, config, config_path


def test_exact_feature_order_and_units(toy):
    matrix = gb.design_matrix(toy["observed"])
    assert tuple(matrix.columns) == ("const", "Y", "D", "C")
    np.testing.assert_array_equal(matrix.Y, toy["observed"].Y)
    np.testing.assert_array_equal(matrix.D, toy["observed"].D)
    np.testing.assert_array_equal(matrix.C, toy["observed"].C)
    assert matrix.const.eq(1).all()


@pytest.mark.parametrize("forbidden", ["B", "epsilon", "omega", "eta", "true_PD", "log_Y", "expected_profit_true", "bank_PD_raw"])
def test_forbidden_fit_columns_rejected(toy, forbidden):
    fit = toy["observed"].loc[:, gb.FIT_COLUMNS].assign(**{forbidden: 0.})
    with pytest.raises(ValueError, match="exact truth-free"):
        gb.fit_group_blind(fit)


def test_training_target_is_raw_and_same_ids(toy):
    frame = toy["observed"].loc[:, gb.FIT_COLUMNS].copy()
    frame.loc[0, "PN"] = -.5
    model = gb.fit_group_blind(frame)
    train = frame.loc[frame.split == "train"].sort_values("applicant_id")
    expected = np.linalg.lstsq(gb.design_matrix(train), train.PN, rcond=1e-15)[0]
    np.testing.assert_allclose([model["coefficients"][t] for t in gb.DESIGN], expected, rtol=1e-10, atol=1e-14)
    assert model["training_ID_fingerprint"] == toy["original_model"]["training_ID_fingerprint"]
    assert model["training_mean_PN"] == train.PN.mean()
    assert model["target_transform"] == model["prediction_rescaling"] == "none"


def test_evaluation_targets_do_not_enter_fit(toy):
    frame = toy["observed"].loc[:, gb.FIT_COLUMNS].copy()
    model = gb.fit_group_blind(frame)
    frame.loc[frame.split == "evaluation", "PN"] = np.nan
    assert gb.fit_group_blind(frame) == model


def test_predictions_invariant_to_B_hidden_values_and_target(toy):
    model = _new(toy)[0]
    frame = toy["observed"].copy()
    raw = gb.predict_group_blind(frame, model)
    frame["B"] = 1 - frame.B
    for column in ("PN", "epsilon", "true_PD", "eta"):
        frame[column] = 999.
    np.testing.assert_array_equal(gb.predict_group_blind(frame, model), raw)


def test_raw_bounded_flags_remain_separate(toy):
    model = _new(toy)[0]
    model["coefficients"] = {"const": -.2, "Y": 1e-5, "D": 0., "C": 0.}
    prediction = gb.risk_estimates(toy["observed"], model)
    np.testing.assert_array_equal(prediction[gb.BOUNDED], np.clip(prediction[gb.RAW], 0, 1))
    np.testing.assert_array_equal(prediction.prediction_out_of_bounds, (prediction[gb.RAW] < 0) | (prediction[gb.RAW] > 1))
    assert tuple(prediction.columns) == gb.PREDICTION_COLUMNS
    assert prediction[gb.RAW].max() > 1


def test_reproducibility_and_no_resplit(toy):
    first, second = _new(toy), _new(toy)
    assert first[0] == second[0] and first[-1] == second[-1]
    for a, b in zip(first[1:-1], second[1:-1]):
        pd.testing.assert_frame_equal(a, b, check_exact=True)
    pd.testing.assert_frame_equal(first[1][["applicant_id", "split"]], toy["observed"][["applicant_id", "split"]])


def test_profit_reuses_contracts_and_exact_accounting(toy):
    _, pred, profit, *_ = _new(toy)
    p = pred.loc[pred.split == "evaluation", gb.BOUNDED].to_numpy()
    contracts = toy["contracts"]
    expected = [math.fsum((1 - x)**t * row.ell_i * (1 + row.r)**t for t in range(1, 31)) for x, row in zip(p, contracts.itertuples())]
    np.testing.assert_allclose(profit.expected_receipts_bank, expected, rtol=1e-14)
    np.testing.assert_array_equal(profit.expected_profit_bank, profit.expected_receipts_bank - contracts.L_i.to_numpy())


@pytest.mark.parametrize("column", ["true_PD", "expected_profit_true", gb.RAW])
def test_no_truth_or_raw_scores_in_accounting_or_decisions(toy, column):
    _, pred, profit, *_ = _new(toy)
    risk = pred.loc[pred.split == "evaluation", ["applicant_id", "model_id", gb.BOUNDED]].assign(**{column: 0.})
    with pytest.raises(ValueError):
        gb.assess_profit(toy["contracts"], risk)
    with pytest.raises(ValueError):
        gb.optimize_cutoff(toy["contracts"], profit.assign(**{column: 0.}), toy["bank_selection"]["available_bank_capital"])


@pytest.mark.parametrize("budget_fraction", [0., .4, 1.])
def test_exact_cutoff_matches_independent_fraction_enumeration(toy, budget_fraction):
    _, _, profit, *_ = _new(toy)
    budget = budget_fraction * math.fsum(toy["contracts"].L_i)
    decisions, candidates, selection = gb.optimize_cutoff(toy["contracts"], profit, budget)
    joined = toy["contracts"][["applicant_id", "L_i"]].merge(profit, on="applicant_id")
    best, best_tau = Fraction(0), None
    for tau in sorted(joined[gb.BOUNDED].unique()):
        selected = joined.loc[joined[gb.BOUNDED] <= tau]
        cost = sum((Fraction(float(x)) for x in selected.L_i), Fraction(0))
        value = sum((Fraction(float(x)) for x in selected.expected_profit_bank), Fraction(0))
        if cost <= Fraction(budget) and value > best:
            best, best_tau = value, tau
    assert selection["optimal_tau"] == best_tau
    assert selection["bank_expected_portfolio_profit"] == float(best)
    assert selection["available_bank_capital"] == budget
    np.testing.assert_array_equal(decisions.funded_principal, np.where(decisions.approved, decisions.requested_principal, 0))
    assert candidates.selected.sum() == 1


def test_ties_indivisible_and_zero_cutoff_not_reject_all(toy):
    profit = _new(toy)[2]
    profit[gb.BOUNDED] = 0.
    decisions, _, summary = gb.optimize_cutoff(toy["contracts"], profit, 1.)
    assert summary["optimal_tau"] is None and not decisions.approved.any()
    # A rounded float sum can lie below the exact binary-float sum. Ensure
    # this test's budget truly covers the whole tie, without changing the solver.
    budget = math.nextafter(math.fsum(toy["contracts"].L_i), math.inf)
    decisions, _, summary = gb.optimize_cutoff(toy["contracts"], profit, budget)
    assert summary["optimal_tau"] == 0 and decisions.approved.all()


def test_risk_and_group_metrics_on_toy(toy):
    pred = _new(toy)[1]
    metrics, groups, changes = gb.compare_risk(toy["observed"], toy["aware"], pred, toy["truth"])
    assert len(metrics) == 8 and len(groups) == 8 and len(changes) == 6
    evaluation = toy["observed"].query("split == 'evaluation'").merge(pred, on=["applicant_id", "split"]).merge(toy["truth"], on="applicant_id")
    row = metrics.query("policy == 'group_blind' and prediction == 'raw' and reference == 'hidden_true_PD'").iloc[0]
    assert row.MAE == pytest.approx(np.abs(evaluation[gb.RAW] - evaluation.true_PD).mean())
    row = groups.query("policy == 'group_blind' and prediction == 'bounded' and B == 1").iloc[0]
    expected = evaluation.loc[evaluation.B == 1]
    assert row.n == len(expected)
    assert row.mean_predicted_PD == pytest.approx(expected[gb.BOUNDED].mean())


def _comparison(toy):
    _, pred, profit, decisions, _, selection = _new(toy)
    policies = {"group_aware": (toy["bank_decisions"], toy["bank_selection"], "bank_PD_bounded", toy["bank_profit"][["applicant_id", "expected_profit_bank"]]), "group_blind": (decisions, selection, gb.BOUNDED, profit[["applicant_id", "expected_profit_bank"]]), "oracle": (toy["oracle"], toy["oracle_selection"], "true_PD", None)}
    audit = toy["observed"].query("split == 'evaluation'")[["applicant_id", "B"]]
    return gb.compare_allocations(toy["contracts"], toy["truth"], audit, policies, toy["bank_selection"]["available_bank_capital"])


def test_policy_comparison_overlap_groups_and_signed_profits(toy):
    summary, groups, overlap, comparison = _comparison(toy)
    assert len(summary) == 3 and len(groups) == 6 and len(overlap) == 12
    assert overlap.query("scope == 'overall'").n_applicants.sum() == 50
    assert overlap.query("scope != 'overall'").n_applicants.sum() == 50
    for _, row in groups.iterrows():
        assert row.approval_rate == row.number_approved / row.n_applicants
        assert row.funded_principal_per_applicant == row.funded_principal / row.n_applicants
    indexed = summary.set_index("policy")
    assert comparison["group_blind_minus_aware_true_profit_USD"] == indexed.loc["group_blind", "true_expected_portfolio_profit"] - indexed.loc["group_aware", "true_expected_portfolio_profit"]
    expected = indexed.loc["oracle", "true_expected_portfolio_profit"] - indexed.loc["group_blind", "true_expected_portfolio_profit"]
    assert indexed.loc["group_blind", "signed_oracle_minus_policy_true_profit"] == expected
    shared = overlap.query("scope == 'overall' and allocation_group == 'approved_by_both'").n_applicants.iloc[0]
    union = overlap.query("scope == 'overall' and allocation_group != 'rejected_by_both'").n_applicants.sum()
    assert comparison["jaccard_overlap"] == shared / union


def test_signed_oracle_difference_is_not_floored_at_zero(toy):
    # A restricted oracle family need not dominate a differently ranked policy.
    # Use a frozen reject-all oracle to check reporting, not to reselect a policy.
    toy["oracle"]["approved"] = False
    toy["oracle"]["funded_principal"] = 0.
    toy["oracle"]["optimal_tau"] = np.nan
    toy["oracle_selection"].update(optimal_tau=None, number_approved=0, total_funded_principal=0.)
    summary = _comparison(toy)[0].set_index("policy")
    assert summary.loc["group_aware", "signed_oracle_minus_policy_true_profit"] < 0
    assert pd.isna(summary.loc["group_aware", "signed_difference_pct_of_oracle"])


def test_invalid_group_blind_feature_artifact_rejected(toy):
    model = _new(toy)[0]
    model["feature_order"].append("B")
    with pytest.raises(ValueError, match="model contract"):
        gb.predict_group_blind(toy["observed"], model)


def test_configuration_rejects_B_and_changed_budget_source(tmp_path):
    config = io.load_config()
    config["model"]["predictors"].append("B")
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        io.load_config(path)
    config = io.load_config()
    config["budget_source"] = "recomputed_budget"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        io.load_config(path)


def test_bank_phase_does_not_read_truth_or_oracle(saved, monkeypatch):
    root, config, _ = saved
    original = pd.read_parquet
    seen = []
    def observed(path, *args, **kwargs):
        seen.append(str(path))
        assert not any(term in str(path) for term in ("true_expected_profit", "simulation_truth", "oracle_lending_decisions"))
        return original(path, *args, **kwargs)
    monkeypatch.setattr(pd, "read_parquet", observed)
    inputs, _ = io.load_inputs(config, root)
    assert len(inputs["observed"]) == 200 and len(seen) == 4


def test_checksum_tamper_is_rejected(saved):
    root, config, _ = saved
    path = root / io.PROCESSED / "lending_decisions.parquet"
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="checksum mismatch"):
        io.load_inputs(config, root)


def test_truth_cannot_load_before_freeze(saved):
    root, config, config_path = saved
    _, hashes = io.load_inputs(config, root)
    with pytest.raises(FileNotFoundError):
        io.load_truth(config, hashes, root=root, config_path=config_path)


def test_end_to_end_freeze_then_truth_and_unchanged_originals(saved):
    root, config, config_path = saved
    inputs, hashes = io.load_inputs(config, root)
    model, pred, profit, decisions, candidates, selection = _new(inputs)
    io.freeze(inputs, model, pred, profit, decisions, candidates, selection, config, hashes, root=root, config_path=config_path)
    report = io.save_evaluation(inputs, pred, profit, decisions, selection, config, hashes, root=root, config_path=config_path)
    assert report["valid"] and all(report["checks"].values())
    assert verify_input_hashes(hashes, root)
    for stem in ("predictions", "profit", "decisions"):
        outputs = io.paths(root)
        pd.testing.assert_frame_equal(pd.read_csv(outputs[f"{stem}_csv"], float_precision="round_trip"), pd.read_parquet(outputs[f"{stem}_parquet"]), check_exact=True)
    assert not any("true" in col or col == "B" for col in decisions.columns)
    with pytest.raises(FileExistsError):
        io.freeze(inputs, model, pred, profit, decisions, candidates, selection, config, hashes, root=root, config_path=config_path)


@pytest.mark.parametrize("key", ["model", "decisions_parquet", "selection"])
def test_frozen_artifact_tampering_blocks_truth(saved, key):
    root, config, config_path = saved
    inputs, hashes = io.load_inputs(config, root)
    io.freeze(inputs, *_new(inputs), config, hashes, root=root, config_path=config_path)
    path = io.paths(root)[key]
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="frozen group-blind artifact changed"):
        io.load_truth(config, hashes, root=root, config_path=config_path)


def test_saved_original_model_and_split_identity_enforced(saved):
    root, config, _ = saved
    inputs, _ = io.load_inputs(config, root)
    assert inputs["original_model"]["training_ID_fingerprint"] == gb.fit_group_blind(inputs["observed"].loc[:, gb.FIT_COLUMNS])["training_ID_fingerprint"]
    config["training_rows"] = 149
    config["evaluation_rows"] = 51
    with pytest.raises(ValueError, match="cohorts"):
        io.load_inputs(config, root)
