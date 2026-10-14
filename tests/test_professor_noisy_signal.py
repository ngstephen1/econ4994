"""Reserved-stream, raw-unit, information-boundary, and read-only-input tests."""

from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from fair_lending.professor_model.config import configuration_fingerprint, load_population_config, resolve_config
from fair_lending.professor_model.noisy_signal import (
    NOISE_COLUMNS, SIGNAL_COLUMNS, SIGNAL_PREVIEW_COLUMNS, SIGNAL_PREVIEW_SCOPE,
    generate_noisy_signal, load_signal_config, noisy_audit_preview,
    raw_noisy_measurement, validate_noisy_signal, validate_signal_config,
)
from fair_lending.professor_model.risk import load_true_risk_config
from fair_lending.professor_model.risk_io import file_hash, verify_input_hashes
from fair_lending.professor_model.signal_io import join_signal_inputs, load_signal_inputs, save_signal_artifacts


@pytest.fixture
def applicants():
    epsilon = np.array([0., 1., -1., .2, -.2, .5])
    omega = np.array([.1, -.1, .02, -.05, .05, .04])
    probability = np.array([.001, .002, .003, .004, .005, .006])
    return pd.DataFrame({
        "applicant_id": [f"toy_{index:03d}" for index in range(6)],
        "Y": np.exp(np.array([11., 10., 12., 11.5, 10.5, 10.8])),
        "log_Y": [11., 10., 12., 11.5, 10.5, 10.8], "D": [.35, .2, .4, .3, .5, .1],
        "B": [0, 1, 0, 1, 0, 0], "C": .4 * epsilon + omega,
        "epsilon": epsilon, "omega": omega,
        # Hand-specified saved truth fixture, not a true-risk generator call.
        "P_income": 2 * probability, "P_other": [.5] * 6, "true_PD": probability,
    })


@pytest.fixture
def config():
    config = load_signal_config()
    config["expected_input_rows"] = 6
    return config


def _fixture_files(applicants, config, root):
    directory = root / config["input_directory"]
    directory.mkdir(parents=True)
    configs = root / "configs/professor_model"
    configs.mkdir(parents=True)
    population_config = resolve_config(load_population_config(), n_samples=6)
    risk_config = load_true_risk_config()
    risk_config["expected_input_rows"] = 6
    pop_config_path = configs / "baseline.yaml"
    risk_config_path = configs / "true_risk.yaml"
    pop_config_path.write_text(yaml.safe_dump(population_config), encoding="utf-8")
    risk_config_path.write_text(yaml.safe_dump(risk_config), encoding="utf-8")
    frames = {
        "applicants_bank_view": applicants[["applicant_id", "Y", "D", "B", "C"]],
        "applicants_simulator_private": applicants[["applicant_id", "log_Y", "epsilon", "omega"]],
        "simulation_truth": applicants[["applicant_id", "P_income", "P_other", "true_PD"]],
    }
    artifacts = {}
    for name, frame in frames.items():
        for suffix in ("csv", "parquet"):
            path = directory / f"{name}.{suffix}"
            if suffix == "csv":
                frame.to_csv(path, index=False)
            else:
                frame.to_parquet(path, index=False)
            artifacts[path.name] = {"path": str(path.relative_to(root)), "sha256": file_hash(path)}
    # Deliberately non-sorted preview order verifies the preserved ID selection.
    preview = directory / "simulation_truth_audit_preview_EVALUATOR_ONLY.csv"
    applicants.iloc[[2, 0, 1, 3, 5, 4]].to_csv(preview, index=False)
    artifacts[preview.name] = {"path": str(preview.relative_to(root)), "sha256": file_hash(preview)}
    pop_metadata = {
        "resolved_configuration": population_config,
        "configuration_sha256": configuration_fingerprint(population_config),
        "source_sha256": {str(pop_config_path.relative_to(root)): file_hash(pop_config_path)},
        "artifacts": {key: value for key, value in artifacts.items() if key.startswith("applicants_")},
    }
    risk_metadata = {
        "resolved_configuration": risk_config,
        "configuration_fingerprint": configuration_fingerprint(risk_config),
        "configuration_file_sha256": file_hash(risk_config_path),
        "artifacts": {key: value for key, value in artifacts.items() if key.startswith("simulation_")},
    }
    (directory / "population_metadata.json").write_text(json.dumps(pop_metadata), encoding="utf-8")
    (directory / "true_risk_metadata.json").write_text(json.dumps(risk_metadata), encoding="utf-8")
    return directory


def test_canonical_configuration_uses_requested_units_and_reserved_stream():
    config = load_signal_config()
    assert config["noise"] == {"distribution": "normal", "mean": 0., "sd": .1}
    assert config["signal_scale"] == .8
    assert config["expected_input_rows"] == 100000
    assert config["randomness"] == {"master_seed": 4994, "spawn_key": [5], "bit_generator": "PCG64", "applicant_order": "ascending_lexicographic_applicant_id"}


@pytest.mark.parametrize("section,key,value", [
    ("noise", "sd", .001), ("noise", "mean", .01), ("noise", "distribution", "truncated_normal"),
    ("randomness", "spawn_key", [0]), ("randomness", "bit_generator", "MT19937"),
    ("randomness", "master_seed", -1), ("randomness", "master_seed", True),
])
def test_invalid_or_retuned_noise_settings_rejected(config, section, key, value):
    config[section][key] = value
    with pytest.raises(ValueError):
        validate_signal_config(config)


def test_removing_point_eight_scale_rejected(config):
    config["signal_scale"] = 1
    with pytest.raises(ValueError):
        validate_signal_config(config)


def test_hand_checked_negative_measurement_is_not_clipped():
    result = raw_noisy_measurement(np.array([.05]), np.array([-.1]))
    assert result[0] == pytest.approx(-.06, abs=1e-15)
    assert result[0] < 0


def test_zero_noise_toy_preserves_point_eight_scale():
    probability = np.array([0., .0014, .05, 1.])
    np.testing.assert_array_equal(raw_noisy_measurement(probability, np.zeros(4)), .8 * probability)
    assert load_signal_config()["noise"]["sd"] == .1


def test_measurement_above_one_is_not_clipped_or_sigmoid_transformed():
    assert raw_noisy_measurement(np.array([1.]), np.array([.5]))[0] == pytest.approx(1.3)


def test_direct_child_is_equivalent_to_sixth_spawn_and_exact_raw_draws(applicants, config):
    targets, noise, stream = generate_noisy_signal(applicants, config)
    child = np.random.SeedSequence(4994).spawn(6)[5]
    expected = np.random.Generator(np.random.PCG64(child)).normal(0, .1, len(applicants))
    np.testing.assert_array_equal(noise.eta, expected)
    np.testing.assert_array_equal(targets.PN, .8 * applicants.true_PD.to_numpy() + expected)
    assert stream["spawn_key"] == [5]
    assert tuple(noise.columns) == NOISE_COLUMNS
    assert tuple(targets.columns) == SIGNAL_COLUMNS
    assert not {"eta", "true_PD", "epsilon", "P_income", "P_other"}.intersection(targets.columns)


def test_eta_is_not_sample_centered_or_reused_from_population_stream(applicants, config):
    _, noise, _ = generate_noisy_signal(applicants, config)
    assert abs(noise.eta.mean()) > 1e-12
    for index in range(5):
        child = np.random.SeedSequence(4994, spawn_key=(index,))
        other = np.random.Generator(np.random.PCG64(child)).normal(0, .1, len(applicants))
        assert not np.array_equal(noise.eta, other)


def test_same_input_configuration_and_seed_reproduce_exactly(applicants, config):
    first = generate_noisy_signal(applicants, config)
    second = generate_noisy_signal(applicants, config)
    for left, right in zip(first[:2], second[:2], strict=True):
        pd.testing.assert_frame_equal(left, right, check_exact=True)
    assert first[2] == second[2]


def test_changed_noise_seed_changes_only_eta_and_PN(applicants, config):
    original = applicants.copy(deep=True)
    changed = copy.deepcopy(config)
    changed["randomness"]["master_seed"] = 4995
    first = generate_noisy_signal(applicants, config)
    second = generate_noisy_signal(applicants, changed)
    assert not np.array_equal(first[0].PN, second[0].PN)
    assert not np.array_equal(first[1].eta, second[1].eta)
    pd.testing.assert_frame_equal(applicants, original, check_exact=True)


def test_changing_only_B_leaves_draws_unchanged(applicants, config):
    changed = applicants.copy(deep=True)
    changed["B"] = 1 - changed.B
    first = generate_noisy_signal(applicants, config)
    second = generate_noisy_signal(changed, config)
    for left, right in zip(first[:2], second[:2], strict=True):
        pd.testing.assert_frame_equal(left, right, check_exact=True)


def test_draws_attached_by_fixed_id_order_not_input_order(applicants, config):
    first = generate_noisy_signal(applicants, config)
    second = generate_noisy_signal(applicants.iloc[::-1], config)
    for left, right in zip(first[:2], second[:2], strict=True):
        pd.testing.assert_frame_equal(left, right, check_exact=True)
    assert first[0].applicant_id.tolist() == sorted(applicants.applicant_id)


def test_nonzero_sample_mean_identity_and_diagnostics(applicants, config):
    targets, noise, _ = generate_noisy_signal(applicants, config)
    report = validate_noisy_signal(applicants, targets, noise, config)
    assert report["valid"] and all(report["checks"].values())
    comparison = report["signal_noise_comparison"]
    assert comparison["mean_PN"] == pytest.approx(.8 * applicants.true_PD.mean() + noise.eta.mean(), abs=1e-15)
    assert comparison["noise_sd_over_scaled_signal_sd"] == pytest.approx(noise.eta.std(ddof=1) / (.8 * applicants.true_PD).std(ddof=1))
    assert report["distributions"]["eta"]["sd"] == pytest.approx(noise.eta.std(ddof=1))
    assert sum(row["count"] for row in report["PN_range_counts"]) == 6
    joined = applicants.merge(targets, on="applicant_id").merge(noise, on="applicant_id")
    for row in report["group_comparison"]:
        subset = joined.loc[joined.B == row["B"]]
        assert row["n"] == len(subset)
        assert row["mean_eta"] == pytest.approx(subset.eta.mean())
        assert row["mean_PN"] == pytest.approx(subset.PN.mean())


@pytest.mark.parametrize("mutation", ["missing", "infinite", "duplicate", "mismatched_ids", "hidden_column", "clipped", "recentered", "wrong_count"])
def test_validation_rejects_modified_or_leaking_outputs(applicants, config, mutation):
    targets, noise, _ = generate_noisy_signal(applicants, config)
    if mutation == "missing":
        noise.loc[0, "eta"] = np.nan
    elif mutation == "infinite":
        targets.loc[0, "PN"] = np.inf
    elif mutation == "duplicate":
        targets.loc[0, "applicant_id"] = targets.loc[1, "applicant_id"]
    elif mutation == "mismatched_ids":
        noise.loc[0, "applicant_id"] = "wrong"
    elif mutation == "hidden_column":
        targets["eta"] = noise.eta
    elif mutation == "clipped":
        targets["PN"] = targets.PN.clip(0, 1)
        assert not (targets.PN == generate_noisy_signal(applicants, config)[0].PN).all()
    elif mutation == "recentered":
        noise["eta"] -= noise.eta.mean()
        targets["PN"] = .8 * applicants.true_PD + noise.eta
    else:
        targets = targets.iloc[:-1]
    report = validate_noisy_signal(applicants, targets, noise, config)
    assert not report["valid"] and report["errors"]


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "mismatched", "extra_truth"])
def test_join_rejects_invalid_original_tables(applicants, mutation):
    bank = applicants[["applicant_id", "Y", "D", "B", "C"]].copy()
    private = applicants[["applicant_id", "log_Y", "epsilon", "omega"]].copy()
    truth = applicants[["applicant_id", "P_income", "P_other", "true_PD"]].copy()
    if mutation == "missing":
        private.loc[0, "applicant_id"] = None
    elif mutation == "duplicate":
        truth.loc[0, "applicant_id"] = truth.loc[1, "applicant_id"]
    elif mutation == "mismatched":
        truth.loc[0, "applicant_id"] = "unknown"
    else:
        truth["PN"] = 0
    with pytest.raises(ValueError):
        join_signal_inputs(bank, private, truth, 6)


def test_loader_and_export_preserve_inputs_and_shared_preview(applicants, config, tmp_path, monkeypatch):
    directory = _fixture_files(applicants, config, tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError("Step 4 must not regenerate applicants or true risk")
    monkeypatch.setattr("fair_lending.professor_model.population.generate_applicants", forbidden)
    monkeypatch.setattr("fair_lending.professor_model.risk.compute_true_risk", forbidden)
    joined, hashes, preview_ids, schemas = load_signal_inputs(config, tmp_path)
    assert len(schemas) == 3
    targets, noise, _ = generate_noisy_signal(joined, config)
    paths, report = save_signal_artifacts(joined, targets, noise, config, hashes, preview_ids, root=tmp_path)
    assert report["valid"] and report["checks"]["original_inputs_unchanged"]
    assert verify_input_hashes(hashes, tmp_path)
    pd.testing.assert_frame_equal(pd.read_parquet(directory / "applicants_bank_view.parquet"), applicants[["applicant_id", "Y", "D", "B", "C"]], check_exact=True)
    for name, frame in (("targets", targets), ("noise", noise)):
        pd.testing.assert_frame_equal(pd.read_parquet(paths[f"{name}_parquet"]), frame, check_exact=True)
        pd.testing.assert_frame_equal(pd.read_csv(paths[f"{name}_csv"], float_precision="round_trip"), frame, check_exact=True)
    preview = pd.read_csv(paths["preview"])
    assert tuple(preview.columns) == SIGNAL_PREVIEW_COLUMNS
    assert preview.information_scope.eq(SIGNAL_PREVIEW_SCOPE).all()
    assert preview.applicant_id.tolist() == preview_ids
    metadata = json.loads(paths["metadata"].read_text())
    assert metadata["input_checksums"] == hashes
    assert metadata["source_code_fingerprint"] == configuration_fingerprint(metadata["source_code_hashes"])
    assert metadata["artifacts"]["targets_csv"]["information_role"] == "BANK_FUTURE_REGRESSION_TARGET"
    assert metadata["transformations"]["noise_recentered"] is False
    for name, record in metadata["artifacts"].items():
        assert record["sha256"] == file_hash(paths[name])
    with pytest.raises(FileExistsError):
        save_signal_artifacts(joined, targets, noise, config, hashes, preview_ids, root=tmp_path)


def test_original_checksum_failure_blocks_generation_loader(applicants, config, tmp_path):
    directory = _fixture_files(applicants, config, tmp_path)
    (directory / "simulation_truth.csv").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_signal_inputs(config, tmp_path)


def test_changed_input_blocks_export(applicants, config, tmp_path):
    directory = _fixture_files(applicants, config, tmp_path)
    joined, hashes, preview_ids, _ = load_signal_inputs(config, tmp_path)
    targets, noise, _ = generate_noisy_signal(joined, config)
    (directory / "simulation_truth.csv").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="inputs changed"):
        save_signal_artifacts(joined, targets, noise, config, hashes, preview_ids, root=tmp_path)
    assert not (directory / "training_signals.csv").exists()


def test_preview_has_only_requested_columns_and_retains_raw_PN(applicants, config):
    targets, noise, _ = generate_noisy_signal(applicants, config)
    ids = applicants.applicant_id.tolist()
    preview = noisy_audit_preview(applicants, targets, noise, ids)
    assert tuple(preview.columns) == SIGNAL_PREVIEW_COLUMNS
    np.testing.assert_array_equal(preview.PN, targets.PN)
    assert preview.information_scope.eq(SIGNAL_PREVIEW_SCOPE).all()
