"""Full-study controls, first-five equivalence, decomposition and strict resume."""
from __future__ import annotations

from copy import deepcopy
import json
import math

import numpy as np
import pandas as pd
import pytest

from fair_lending.professor_model import noise_full as full
from fair_lending.professor_model import noise_full_io as io
from fair_lending.professor_model import noise_pilot as pilot
from fair_lending.professor_model import noise_pilot_io as pio
from fair_lending.professor_model import bank_regression as aware
from fair_lending.professor_model.config import configuration_fingerprint as fp
from fair_lending.professor_model.risk_io import file_hash
from test_professor_group_blind import toy as saved_toy
from test_professor_noise_pilot import fixed, evidence


@pytest.fixture
def context(evidence, fixed):
    root, pilot_manifest = evidence
    observed, truth, toy = fixed
    manifest = {"experiment_fingerprint": "full_toy", "identity": {"configuration": {"training_rows": 150},
                "scientific_config_fingerprint": "science", "source_fingerprint": "source", "solver_settings_fingerprint": fp(io.SETTINGS),
                "dependencies": {}, "original_checksums": pilot_manifest["identity"]["original_checksums"], "source_code_hashes": {},
                "fixed_budget": toy["bank_selection"]["available_bank_capital"], "training_ID_fingerprint": toy["original_model"]["training_ID_fingerprint"]}}
    reference, mask, row = io.prepare_reference(observed, truth, manifest, root)
    return root, manifest, reference, mask, row


@pytest.mark.parametrize("index", list(range(5)))
def test_first_five_streams_targets_and_model_outputs_exactly_match_pilot(fixed, index):
    observed, truth, toy = fixed
    expected_targets, expected_stream = pilot.draw_targets(truth, index)
    targets, eta, stream = full.draw_targets(truth.iloc[::-1], index)
    pd.testing.assert_frame_equal(targets, expected_targets, check_exact=True)
    assert stream == expected_stream
    before = [pilot.frame_fingerprint(f) for f in (observed, truth, toy["contracts"], toy["oracle"])]
    first = pilot.fit_bank_policies(observed, targets, toy["contracts"], toy["bank_selection"]["available_bank_capital"])
    second = pilot.fit_bank_policies(observed, expected_targets, toy["contracts"], toy["bank_selection"]["available_bank_capital"])
    assert pilot.bank_freeze_payload(first) == pilot.bank_freeze_payload(second)
    assert before == [pilot.frame_fingerprint(f) for f in (observed, truth, toy["contracts"], toy["oracle"])]


def test_200_unique_streams_and_invalid_indices(fixed):
    truth = fixed[1]
    streams = [full.draw_targets(truth, i)[2] for i in range(200)]
    assert len({s["eta_sha256"] for s in streams}) == 200
    assert [s["spawn_key"] for s in streams] == [[i] for i in range(200)]
    assert full.draw_targets(truth, 199)[2] == streams[-1]
    for i in (-1, 200, True):
        with pytest.raises(ValueError): full.draw_targets(truth, i)


def test_evaluator_only_OLS_decomposition_matches_direct_eta_regression(fixed):
    observed, truth, _ = fixed
    X, signal, mask = full.reference_design(observed, truth)
    reference = full.fit_signal_reference(X, signal)
    for i in (0, 5, 199):
        targets, eta, _ = full.draw_targets(truth, i)
        pn_coef = np.linalg.pinv(X.to_numpy(), rcond=1e-15) @ targets.loc[mask, "PN"].to_numpy()
        eta_coef = np.linalg.pinv(X.to_numpy(), rcond=1e-15) @ eta[mask]
        result = full.decomposition(float(pn_coef[-1]), eta, mask, full.B_row(X), reference)
        assert result["beta_B_noise"] == pytest.approx(eta_coef[-1], abs=1e-14)
        assert abs(result["residual"]) < 1e-12
    with pytest.raises(ValueError): full.decomposition(99., eta, mask, full.B_row(X), reference)


def test_reference_is_fitted_once_and_not_an_operational_bank(context, fixed, monkeypatch):
    root, manifest, reference, *_ = context
    assert reference["information_scope"] == "EVALUATOR_ONLY_NOT_AN_OPERATIONAL_BANK"
    def forbidden(*a, **kw): raise AssertionError("signal reference must be loaded, not refitted")
    monkeypatch.setattr(full, "fit_signal_reference", forbidden)
    again, _, _ = io.prepare_reference(fixed[0], fixed[1], manifest, root)
    assert again == reference


def test_one_run_preserves_inputs_and_validates_freeze(context, fixed):
    root, manifest, reference, mask, row = context
    before = [pilot.frame_fingerprint(f) for f in (fixed[0], fixed[1], fixed[2]["contracts"], fixed[2]["oracle"])]
    record = io.run_one(5, fixed[0], fixed[1], fixed[2]["contracts"], manifest["identity"]["fixed_budget"], reference, mask, row, manifest, root)
    assert record is not None
    loaded = io.load_record(5, manifest, root)
    assert loaded == record
    assert [m["negative_raw_share"] for m in record["policy_metrics"]] == [m["negative_raw_count"] / 50 for m in record["policy_metrics"]]
    assert io.resume_record(5, manifest, root) == record
    assert before == [pilot.frame_fingerprint(f) for f in (fixed[0], fixed[1], fixed[2]["contracts"], fixed[2]["oracle"])]
    assert io.execution_counts(manifest, root)["unique_replications_attempted"] == 1


@pytest.mark.parametrize("corruption", ["source", "missing_group", "duplicate_group", "profit", "gap", "reference", "budget", "allowlist", "nonfinite"])
def test_malformed_or_incomplete_success_never_passes_resume(context, fixed, corruption):
    root, manifest, reference, mask, row = context
    record = io.run_one(6, fixed[0], fixed[1], fixed[2]["contracts"], manifest["identity"]["fixed_budget"], reference, mask, row, manifest, root)
    assert record
    bad = deepcopy(record)
    if corruption == "source": bad["source_fingerprint"] = "wrong"
    if corruption == "missing_group": bad["group_approval"].pop()
    if corruption == "duplicate_group": bad["group_approval"][-1] = bad["group_approval"][0]
    if corruption == "profit": bad["comparison"]["group_blind_minus_aware_true_profit_USD"] += 1
    if corruption == "gap": bad["policy_metrics"][0]["approval_gap_pp"] += 1
    if corruption == "reference": bad["signal_reference_sha256"] = "changed"
    if corruption == "budget": bad["policy_metrics"][0]["available_bank_capital"] += 1
    if corruption == "allowlist": bad["banks"]["group_blind"]["model"]["design_order"].append("B")
    if corruption == "nonfinite": bad["policy_metrics"][0]["raw_PN_RMSE"] = None
    # Even a self-consistent outer checksum must not bypass semantic validation.
    path = io.result_path(6, root)
    path.write_text(json.dumps({**bad, "content_fingerprint": fp(bad)}))
    assert io.resume_record(6, manifest, root) is None
    assert not path.exists()
    assert len(list((root / io.RUN_DIRECTORY / "invalid_records").glob("*.json"))) == 1
    assert io.execution_counts(manifest, root)["invalid_saved_records"] == 1


def test_failed_attempt_is_retained_then_retried_without_counting_failure_as_success(context, fixed, monkeypatch):
    root, manifest, reference, mask, row = context
    original = pilot.fit_bank_policies
    def failed(*a, **kw): raise RuntimeError("intentional toy fitting failure")
    monkeypatch.setattr(pilot, "fit_bank_policies", failed)
    args = (7, fixed[0], fixed[1], fixed[2]["contracts"], manifest["identity"]["fixed_budget"], reference, mask, row, manifest, root)
    assert io.run_one(*args) is None and not io.result_path(7, root).exists()
    monkeypatch.setattr(pilot, "fit_bank_policies", original)
    assert io.run_one(*args) is not None
    counts = io.execution_counts(manifest, root)
    assert counts["attempt_records"] == 2 and counts["unique_replications_attempted"] == 1 and counts["failed_attempts"] == 1


def test_distribution_MCSE_quantiles_and_undefined_values():
    values = np.array([1., 2., 4., 8.])
    result = full.distribution(values)
    assert result["sample_SD"] == values.std(ddof=1)
    assert result["MCSE_mean"] == values.std(ddof=1) / 2
    assert result["p2.5"] == np.quantile(values, .025, method="linear")
    assert result["p97.5"] == np.quantile(values, .975, method="linear")
    assert full.distribution([None])["n_defined"] == 0


def test_tidy_aggregation_frequencies_and_complete_denominators(context, fixed):
    root, manifest, reference, mask, row = context
    records = [io.run_one(i, fixed[0], fixed[1], fixed[2]["contracts"], manifest["identity"]["fixed_budget"], reference, mask, row, manifest, root) for i in (0, 1)]
    assert all(records)
    tables = full.summarize_records(records, expected=2)
    assert len(tables["results"]) == len(tables["ties"]) == 4
    assert len(tables["coefficients"]) == 9 and len(tables["groups"]) == 16
    frequencies = {(f["event"], f["scope"]): f for f in tables["frequencies"]}
    assert frequencies[("reject_all", "pooled_policy_replications")]["denominator"] == 4
    assert frequencies[("reject_all", "group_aware")]["denominator"] == 2
    pair = tables["paired"]
    assert frequencies[("blind_smaller_absolute_gap", "replications")]["fraction"] == (pair.blind_abs_gap_pp < pair.aware_abs_gap_pp).mean()
    rerun = full.summarize_records(list(reversed(records)), expected=2)
    for key in ("results", "coefficients", "groups", "ties", "summary"): pd.testing.assert_frame_equal(tables[key], rerun[key], check_exact=True)
    with pytest.raises(ValueError): full.summarize_records(records)
    with pytest.raises(ValueError): full.summarize_records([records[0], records[0]], expected=2)


def test_unique_run_ids_and_study_source_invalidation(context):
    _, manifest, *_ = context
    assert len({io.run_id(i, manifest) for i in range(200)}) == 200
    other = deepcopy(manifest)
    other["experiment_fingerprint"] = "changed"
    assert io.run_id(0, manifest) != io.run_id(0, other)


def test_truth_gate_precedes_any_evaluator_read(context, fixed, monkeypatch):
    root, manifest, *_ = context
    def forbidden(*a, **kw): raise AssertionError("truth cannot load before bank freeze")
    monkeypatch.setattr(pd, "read_parquet", forbidden)
    with pytest.raises(FileNotFoundError):
        io.evaluate({}, root / "missing_freeze.json", fixed[0], None, fixed[2]["contracts"], 1., manifest, root)


def test_summarize_only_cannot_launch_computation(context, monkeypatch):
    root, manifest, *_ = context
    def forbidden(*a, **kw): raise AssertionError("summarization launched computation")
    monkeypatch.setattr(full, "draw_targets", forbidden)
    monkeypatch.setattr(pilot, "fit_bank_policies", forbidden)
    monkeypatch.setattr(full, "fit_signal_reference", forbidden)
    with pytest.raises(ValueError, match="incomplete"):
        io.summarize(manifest, root)
