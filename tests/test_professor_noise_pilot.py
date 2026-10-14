"""Noise-only pilot tests on isolated toy artifacts, not new research populations."""
from __future__ import annotations

from copy import deepcopy
import json
import math

import numpy as np
import pandas as pd
import pytest
import yaml

from fair_lending.professor_model import bank_regression as aware
from fair_lending.professor_model import group_blind as blind
from fair_lending.professor_model import noise_pilot as pilot
from fair_lending.professor_model import noise_pilot_io as io
from fair_lending.professor_model.config import configuration_fingerprint
from fair_lending.professor_model.lending import choose_cutoff
from fair_lending.professor_model.risk_io import file_hash

from test_professor_group_blind import toy as saved_toy


@pytest.fixture
def fixed(saved_toy):
    t = saved_toy
    observed = t["observed"].loc[:, pilot.OBSERVED_COLUMNS].copy()
    full_truth = observed[["applicant_id"]].assign(true_PD=.002)
    full_truth.loc[full_truth.applicant_id.isin(t["truth"].applicant_id), "true_PD"] = t["truth"].true_PD.to_numpy()
    return observed, full_truth, t


def _banks(fixed, replication=0):
    observed, full_truth, t = fixed
    targets, stream = pilot.draw_targets(full_truth, replication)
    banks = pilot.fit_bank_policies(observed, targets, t["contracts"], t["bank_selection"]["available_bank_capital"])
    return targets, stream, banks


def _record(fixed, replication):
    observed, _, t = fixed
    targets, stream, banks = _banks(fixed, replication)
    frozen = pilot.bank_freeze_payload(banks)
    metrics, groups, comparison = pilot.evaluate_replication(banks, observed, targets, t["contracts"], t["truth"], t["oracle"], t["oracle_selection"], t["bank_selection"]["available_bank_capital"], frozen)
    rows = lambda f: f.astype(object).where(f.notna(), None).to_dict("records")
    return {"status": "success", "replication": replication, "experiment_fingerprint": "toy", "stream": stream,
            "banks": frozen, "policy_metrics": rows(metrics), "group_approval": rows(groups), "comparison": comparison,
            "runtime_seconds": 1., "oracle_reference": {"true_expected_portfolio_profit": math.fsum(t["truth"].set_index("applicant_id").loc[t["oracle"].loc[t["oracle"].approved, "applicant_id"], "expected_profit_true"])}}


@pytest.fixture
def evidence(tmp_path, fixed):
    _, _, t = fixed
    marker = tmp_path / "original.txt"
    marker.write_text("immutable original PN\n")
    manifest = {"experiment_fingerprint": "toy", "identity": {"original_checksums": {"original.txt": file_hash(marker)}}}
    directory = tmp_path / io.legacy.PROCESSED
    directory.mkdir(parents=True)
    t["truth"].to_parquet(directory / "true_expected_profit.parquet", index=False)
    t["oracle"].to_parquet(directory / "oracle_lending_decisions.parquet", index=False)
    metric_directory = tmp_path / io.legacy.METRICS
    metric_directory.mkdir(parents=True)
    (metric_directory / "oracle_comparison_validation.json").write_text(json.dumps({"oracle_selection": t["oracle_selection"]}))
    table_directory = tmp_path / io.legacy.TABLES
    table_directory.mkdir(parents=True)
    oracle_profit = math.fsum(t["truth"].set_index("applicant_id").loc[t["oracle"].loc[t["oracle"].approved, "applicant_id"], "expected_profit_true"])
    pd.DataFrame([{"policy": "oracle", "true_expected_portfolio_profit": oracle_profit}]).to_csv(table_directory / "bank_vs_oracle_summary.csv", index=False)
    return tmp_path, manifest


def test_five_exact_seedsequence_children():
    children = pilot.streams()
    assert len(children) == 5
    assert [c.spawn_key for c in children] == [(i,) for i in range(5)]
    assert all(c.entropy == 499408 for c in children)


def test_exact_noise_formula_and_no_clipping(fixed):
    _, truth, _ = fixed
    targets, stream = pilot.draw_targets(truth, 2)
    eta = np.random.Generator(np.random.PCG64(np.random.SeedSequence(499408).spawn(5)[2])).normal(0, .1, len(truth))
    np.testing.assert_array_equal(targets.PN, .8 * truth.sort_values("applicant_id").true_PD.to_numpy() + eta)
    assert (targets.PN < 0).any()
    assert stream["spawn_key"] == [2]


def test_independent_reproducible_noise_and_fixed_inputs(fixed):
    observed, truth, t = fixed
    snapshots = [pilot.frame_fingerprint(f) for f in (observed, truth, t["contracts"], t["truth"], t["oracle"], t["observed"])]
    draws = [pilot.draw_targets(truth, i) for i in range(5)]
    assert len({stream["eta_sha256"] for _, stream in draws}) == 5
    assert len({stream["target_fingerprint"] for _, stream in draws}) == 5
    for i, (target, stream) in enumerate(draws):
        again, again_stream = pilot.draw_targets(truth.iloc[::-1], i)
        pd.testing.assert_frame_equal(again, target, check_exact=True)
        assert stream == again_stream
        np.testing.assert_array_equal(target.applicant_id, observed.applicant_id)
    assert snapshots == [pilot.frame_fingerprint(f) for f in (observed, truth, t["contracts"], t["truth"], t["oracle"], t["observed"])]


@pytest.mark.parametrize("index", [-1, 5, 50, True, 1.5])
def test_sixth_or_invalid_replication_rejected(fixed, index):
    with pytest.raises(ValueError):
        pilot.draw_targets(fixed[1], index)


@pytest.mark.parametrize("field", ["true_PD", "epsilon", "omega", "eta", "PN", "P_income"])
def test_private_columns_never_reach_bank_fits(fixed, field):
    observed, _, t = fixed
    targets = pilot.draw_targets(fixed[1], 0)[0]
    with pytest.raises(ValueError, match="observable-only"):
        pilot.fit_bank_policies(observed.assign(**{field: 0.}), targets, t["contracts"], t["bank_selection"]["available_bank_capital"])


def test_both_feature_allowlists_and_identical_training_ids(fixed):
    _, _, banks = _banks(fixed)
    a, b = (banks[p]["model"] for p in pilot.POLICIES)
    assert a["feature_order"] == ["Y", "D", "C", "B"]
    assert b["feature_order"] == ["Y", "D", "C"]
    assert a["design_order"] == ["const", "Y", "D", "C", "B"]
    assert b["design_order"] == ["const", "Y", "D", "C"]
    assert a["training_ID_fingerprint"] == b["training_ID_fingerprint"] == fixed[2]["original_model"]["training_ID_fingerprint"]
    assert a["training_mean_PN"] == b["training_mean_PN"]


def test_evaluation_PN_does_not_enter_fit_or_decisions(fixed):
    observed, _, t = fixed
    targets, _, banks = _banks(fixed)
    changed = targets.copy()
    changed.loc[changed.applicant_id.isin(observed.loc[observed.split == "evaluation", "applicant_id"]), "PN"] = 123.
    rerun = pilot.fit_bank_policies(observed, changed, t["contracts"], t["bank_selection"]["available_bank_capital"])
    assert pilot.bank_freeze_payload(rerun) == pilot.bank_freeze_payload(banks)


def test_raw_bounded_predictions_separated(fixed):
    _, _, banks = _banks(fixed)
    for policy in pilot.POLICIES:
        pred = banks[policy]["predictions"]
        raw, bounded = ("bank_PD_raw", "bank_PD_bounded") if policy == "group_aware" else (blind.RAW, blind.BOUNDED)
        np.testing.assert_array_equal(pred[bounded], np.clip(pred[raw], 0, 1))
        np.testing.assert_array_equal(pred.prediction_out_of_bounds, (pred[raw] < 0) | (pred[raw] > 1))
        assert not any(x in pred for x in ("true_PD", "PN", "eta"))


def test_deterministic_replication_and_model_replay(fixed):
    targets, stream, first = _banks(fixed)
    _, stream2, second = _banks(fixed)
    assert stream == stream2 and pilot.bank_freeze_payload(first) == pilot.bank_freeze_payload(second)
    evaluation = fixed[0].query("split == 'evaluation'")
    np.testing.assert_array_equal(aware.predict_bank_ols(evaluation, first["group_aware"]["model"]), first["group_aware"]["predictions"].bank_PD_raw)
    np.testing.assert_array_equal(blind.predict_group_blind(evaluation, first["group_blind"]["model"]), first["group_blind"]["predictions"][blind.RAW])


def _toy_bank(risk, principal, profit):
    return pd.DataFrame({"applicant_id": [f"small_{i}" for i in range(len(risk))], "model_id": aware.MODEL_ID, "bank_PD_bounded": risk, "requested_principal": principal, "expected_profit_bank": profit})


@pytest.mark.parametrize("budget", [0., 10., 30., 60.])
def test_exact_candidate_audit_full_funding_and_ties(budget):
    bank = _toy_bank([0., 0., .1, .2], [10., 20., 10., 20.], [4., 7., 3., -1.])
    decisions, candidates, selection = choose_cutoff(bank, budget)
    report = pilot.exact_cutoff_diagnostics(bank, decisions, candidates, selection, budget)
    assert report["maximum_budget_violation_USD"] == 0
    assert report["selected_budget_residual_USD"] <= 0
    assert report["maximum_feasible_candidate_budget_residual_USD"] <= 0
    assert report["zero_block_principal_USD"] == 30.
    assert report["zero_block_feasible"] == (budget >= 30)
    assert decisions.approved.iloc[0] == decisions.approved.iloc[1]
    if budget < 30:
        assert selection["optimal_tau"] is None and selection["number_approved"] == 0
        assert report["feasible_cutoff_candidates_including_reject_all"] == 1
    else:
        assert selection["optimal_tau"] in (0., .1)


def test_independent_audit_catches_incorrect_selection():
    bank = _toy_bank([.1, .2], [10., 20.], [1., 2.])
    decisions, candidates, selection = choose_cutoff(bank, 30.)
    selection["optimal_tau"] = .1
    with pytest.raises(ValueError, match="exact optimum"):
        pilot.exact_cutoff_diagnostics(bank, decisions, candidates, selection, 30.)


def test_accounting_and_budget_are_fixed(fixed):
    _, _, banks = _banks(fixed)
    contracts = fixed[2]["contracts"].set_index("applicant_id")
    for policy in pilot.POLICIES:
        rows = banks[policy]["profit"]
        np.testing.assert_array_equal(rows.expected_profit_bank, rows.expected_receipts_bank - contracts.loc[rows.applicant_id, "L_i"].to_numpy())
        assert banks[policy]["selection"]["available_bank_capital"] == fixed[2]["bank_selection"]["available_bank_capital"]
        assert banks[policy]["diagnostics"]["independent_exact_enumeration_passed"]


def test_truth_cannot_be_evaluated_before_freeze(evidence, fixed, monkeypatch):
    root, manifest = evidence
    targets, stream, banks = _banks(fixed)
    def forbidden(*a, **kw):
        raise AssertionError("truth read before freeze")
    monkeypatch.setattr(pd, "read_parquet", forbidden)
    with pytest.raises(FileNotFoundError):
        io.evaluate_frozen(banks, 0, stream, fixed[0], targets, fixed[2]["contracts"], fixed[2]["bank_selection"]["available_bank_capital"], manifest, root)


def test_frozen_allocation_then_evaluator_and_unchanged_oracle(evidence, fixed):
    root, manifest = evidence
    targets, stream, banks = _banks(fixed)
    oracle_path = root / io.legacy.PROCESSED / "oracle_lending_decisions.parquet"
    before = file_hash(oracle_path)
    io.freeze_bank(banks, 0, stream, manifest, root)
    metrics, groups, comparison, oracle = io.evaluate_frozen(banks, 0, stream, fixed[0], targets, fixed[2]["contracts"], fixed[2]["bank_selection"]["available_bank_capital"], manifest, root)
    assert before == file_hash(oracle_path)
    assert len(metrics) == 2 and len(groups) == 4
    assert oracle["number_approved"] == fixed[2]["oracle"].approved.sum()
    for _, row in metrics.iterrows():
        by_group = groups.loc[groups.policy == row.policy].set_index("B")
        assert row.approval_gap_pp == pytest.approx(100 * (by_group.loc[1, "approval_rate"] - by_group.loc[0, "approval_rate"]))
        assert row.signed_oracle_minus_policy_true_profit == oracle["true_expected_portfolio_profit"] - row.true_expected_portfolio_profit
    indexed = metrics.set_index("policy")
    assert comparison["group_blind_minus_aware_true_profit_USD"] == indexed.loc["group_blind", "true_expected_portfolio_profit"] - indexed.loc["group_aware", "true_expected_portfolio_profit"]
    banks["group_aware"]["decisions"].loc[0, "approved"] = not banks["group_aware"]["decisions"].loc[0, "approved"]
    with pytest.raises(ValueError, match="frozen"):
        io.evaluate_frozen(banks, 0, stream, fixed[0], targets, fixed[2]["contracts"], fixed[2]["bank_selection"]["available_bank_capital"], manifest, root)


def test_immutable_record_roundtrip_and_tampering(evidence, fixed):
    root, manifest = evidence
    record = _record(fixed, 0)
    targets, stream, banks = _banks(fixed)
    io.freeze_bank(banks, 0, stream, manifest, root)
    io.save_record(record, manifest, root)
    loaded = io.load_record(0, manifest, root)
    assert loaded["comparison"] == record["comparison"]
    with pytest.raises(FileExistsError):
        io.save_record(record, manifest, root)
    path = io.record_path(0, root)
    content = json.loads(path.read_text())
    content["replication"] = 1
    path.write_text(json.dumps(content))
    with pytest.raises(ValueError):
        io.load_record(0, manifest, root)


def test_original_PN_change_blocks_freeze(evidence, fixed):
    root, manifest = evidence
    (root / "original.txt").write_text("modified original PN")
    _, stream, banks = _banks(fixed)
    with pytest.raises(ValueError, match="original scientific evidence"):
        io.freeze_bank(banks, 0, stream, manifest, root)


def test_five_summaries_reproduce_and_arithmetic_is_correct(fixed):
    records = [_record(fixed, i) for i in range(5)]
    first = pilot.result_tables(records)
    second = pilot.result_tables(list(reversed(records)))
    for a, b in zip(first, second):
        pd.testing.assert_frame_equal(a, b, check_exact=True)
    assert [len(frame) for frame in first[:3]] == [5, 45, 20]
    summary = first[3].set_index("metric")
    assert summary.loc["beta_B_hat", "mean"] == first[0].beta_B_hat.mean()
    assert summary.loc["beta_B_hat", "sample_SD"] == first[0].beta_B_hat.std(ddof=1)
    assert summary.loc["gap_change_pp", "minimum"] == first[0].gap_change_pp.min()
    for records_bad in (records[:4], [*records[:4], records[0]]):
        with pytest.raises(ValueError):
            pilot.result_tables(records_bad)


def test_summary_only_never_draws_fits_or_optimizes(evidence, fixed, monkeypatch):
    root, manifest = evidence
    for i in range(5):
        record = _record(fixed, i)
        _, stream, banks = _banks(fixed, i)
        io.freeze_bank(banks, i, stream, manifest, root)
        io.save_record(record, manifest, root)
    def forbidden(*args, **kwargs):
        raise AssertionError("summary must never compute new experiments")
    for module, name in ((pilot, "draw_targets"), (pilot, "fit_bank_policies"), (aware, "fit_bank_ols"), (blind, "fit_group_blind")):
        monkeypatch.setattr(module, name, forbidden)
    result = io.summarize(manifest, root)
    assert result["n_replications"] == 5 and not result["full_Monte_Carlo_started"]
    assert io.summarize(manifest, root) == result


@pytest.mark.parametrize("key,value", [("replications", 6), ("master_seed", 4994), ("signal_scale", 1), ("noise", {"mean": 0, "sd": .01})])
def test_pilot_design_cannot_silently_change(tmp_path, key, value):
    config = io.load_config()
    config[key] = value
    path = tmp_path / "wrong.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        io.load_config(path)


def test_manifest_source_input_identity_and_no_overwrite(tmp_path):
    config = io.load_config()
    manifest = io.prepare_manifest(config, root=tmp_path)
    assert manifest["spawn_keys"] == [[0], [1], [2], [3], [4]]
    assert io.prepare_manifest(config, root=tmp_path, resume=True) == manifest
    with pytest.raises(FileExistsError):
        io.prepare_manifest(config, root=tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/original.yaml").write_text("changed: true")
    with pytest.raises(ValueError, match="provenance differs"):
        io.prepare_manifest(config, root=tmp_path, resume=True)
