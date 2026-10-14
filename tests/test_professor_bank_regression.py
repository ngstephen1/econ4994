"""Step 5 information boundaries, exact specification, and artifact replay.

Toy samples below are test fixtures, not regenerated professor populations.
No assertion requires favorable signs, significance, R-squared, or accuracy.
"""

from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest

from fair_lending.professor_model.bank_io import (
    freeze_bank_artifacts, load_bank_config, load_bank_inputs, load_evaluator_truth,
    output_paths, save_bank_evaluation, validate_bank_config,
)
from fair_lending.professor_model.bank_regression import (
    DESIGN_COLUMNS, ESTIMATE_COLUMNS, FEATURES, PREVIEW_COLUMNS, attach_splits,
    design_matrix, error_metrics, evaluate_bank_risk, fit_bank_ols, join_bank_inputs,
    make_splits, predict_bank_ols, prediction_summary, risk_estimates,
)
from fair_lending.professor_model.config import configuration_fingerprint
from fair_lending.professor_model.risk_io import file_hash, verify_input_hashes


@pytest.fixture
def sample():
    rng = np.random.default_rng(103)
    n = 800
    bank = pd.DataFrame({
        "applicant_id": [f"toy_{i:04d}" for i in range(n)],
        "Y": rng.uniform(10000, 180000, n), "D": rng.uniform(.05, .8, n),
        "B": rng.binomial(1, .3, n), "C": rng.normal(0, .5, n),
    })
    targets = pd.DataFrame({"applicant_id": bank.applicant_id, "PN": -.02 + 1e-7 * bank.Y + .02 * bank.D + .03 * bank.C - .01 * bank.B + rng.normal(0, .1, n)})
    truth = pd.DataFrame({"applicant_id": bank.applicant_id, "P_income": .01, "P_other": .5, "true_PD": .005})
    config = load_bank_config()
    config["expected_input_rows"] = n
    config["split"].update(train_rows=640, evaluation_rows=160)
    return bank, targets, truth, config


def _fit(sample):
    bank, targets, _, config = sample
    joined = join_bank_inputs(bank, targets, len(bank))
    splits = make_splits(joined.applicant_id, config["split"])
    applicants = attach_splits(joined, splits)
    model = fit_bank_ols(applicants)
    return applicants, splits, model, risk_estimates(applicants, model)


def _input_files(sample, root):
    bank, targets, truth, config = sample
    directory = root / config["input_directory"]
    directory.mkdir(parents=True)
    conf_dir = root / "configs/professor_model"
    conf_dir.mkdir(parents=True)
    paths = {}
    for name, frame in (("applicants_bank_view", bank), ("training_signals", targets), ("simulation_truth", truth)):
        path = directory / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        paths[name] = {"path": str(path.relative_to(root)), "sha256": file_hash(path)}
    # An opaque private file must remain unread, even though its hash is checked.
    private = directory / "applicants_simulator_private.parquet"
    private.write_bytes(b"opaque private input; not a readable parquet file")
    paths["private"] = {"path": str(private.relative_to(root)), "sha256": file_hash(private)}
    for name, key, yaml_name in (
        ("population_metadata.json", "configuration_sha256", "baseline.yaml"),
        ("true_risk_metadata.json", "configuration_fingerprint", "true_risk.yaml"),
        ("noisy_signal_metadata.json", "configuration_fingerprint", "noisy_signal.yaml"),
    ):
        config_file = conf_dir / yaml_name
        config_file.write_text("untouched: true\n", encoding="utf-8")
        metadata = {"resolved_configuration": {"fixture": True}, key: configuration_fingerprint({"fixture": True}), "configuration_file_sha256": file_hash(config_file), "artifacts": paths}
        (directory / name).write_text(json.dumps(metadata), encoding="utf-8")
    config_path = conf_dir / "bank_regression.yaml"
    import yaml
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def test_canonical_design_and_split_configuration():
    config = load_bank_config()
    assert FEATURES == ("Y", "D", "C", "B")
    assert DESIGN_COLUMNS == ("const", "Y", "D", "C", "B")
    assert config["split"] == {"train_rows": 80000, "evaluation_rows": 20000, "master_seed": 4994, "spawn_key": [6], "bit_generator": "PCG64", "applicant_order": "ascending_lexicographic_applicant_id"}


@pytest.mark.parametrize("field,value", [("predictors", ["log_Y", "D", "C", "B"]), ("target_transform", "clip"), ("prediction_rescaling", "divide_by_0.8"), ("intercept", False), ("regularization", .1), ("target", "true_PD")])
def test_reject_changed_scientific_specification(field, value):
    config = load_bank_config()
    config["model"][field] = value
    with pytest.raises(ValueError):
        validate_bank_config(config)


def test_exact_design_units_and_intercept(sample):
    bank = sample[0]
    matrix = design_matrix(bank)
    assert tuple(matrix.columns) == DESIGN_COLUMNS
    np.testing.assert_array_equal(matrix.const, np.ones(len(bank)))
    for column in FEATURES:
        np.testing.assert_array_equal(matrix[column], bank[column])


def test_private_and_administrative_columns_never_enter_fit(sample, monkeypatch):
    import fair_lending.professor_model.bank_regression as module
    applicants, _, _, _ = _fit(sample)
    for column in ("log_Y", "epsilon", "omega", "eta", "true_PD", "P_income", "P_other", "period_index"):
        applicants[column] = np.arange(len(applicants))
    original = module.sm.OLS
    seen = {}

    def observed_ols(endog, exog, **kwargs):
        seen["columns"] = tuple(exog.columns)
        seen["target"] = endog.copy()
        return original(endog, exog, **kwargs)

    monkeypatch.setattr(module.sm, "OLS", observed_ols)
    model = fit_bank_ols(applicants)
    assert seen["columns"] == DESIGN_COLUMNS
    train_target = applicants.loc[applicants.split == "train"].sort_values("applicant_id").PN
    np.testing.assert_array_equal(seen["target"], train_target)
    assert (seen["target"] < 0).any()  # raw PN is not clipped
    assert model["training_mean_PN"] == train_target.mean()


@pytest.mark.parametrize("defect", ["duplicate", "missing", "mismatched", "dropped", "infinite", "wrong_columns"])
def test_bad_inputs_rejected_without_dropping(sample, defect):
    bank, targets, _, _ = sample
    targets = targets.copy()
    if defect == "duplicate":
        targets.loc[0, "applicant_id"] = targets.loc[1, "applicant_id"]
    elif defect == "missing":
        targets.loc[0, "applicant_id"] = None
    elif defect == "mismatched":
        targets.loc[0, "applicant_id"] = "not_an_applicant"
    elif defect == "dropped":
        targets = targets.iloc[:-1]
    elif defect == "infinite":
        targets.loc[0, "PN"] = np.inf
    else:
        targets["true_PD"] = 0
    with pytest.raises(ValueError):
        join_bank_inputs(bank, targets, len(bank))


def test_complete_canonical_split_and_exact_reserved_child():
    config = load_bank_config()["split"]
    ids = pd.Series([f"prof_{i:06d}" for i in range(1, 100001)])
    split = make_splits(ids, config)
    assert split.split.value_counts().to_dict() == {"train": 80000, "evaluation": 20000}
    train_ids = set(split.loc[split.split == "train", "applicant_id"])
    eval_ids = set(split.loc[split.split == "evaluation", "applicant_id"])
    assert not train_ids & eval_ids and len(train_ids | eval_ids) == 100000
    child = np.random.SeedSequence(4994).spawn(7)[6]
    expected = np.random.Generator(np.random.PCG64(child)).permutation(100000)[:80000]
    assert train_ids == set(ids.iloc[expected])
    pd.testing.assert_frame_equal(split, make_splits(ids.sample(frac=1, random_state=4), config), check_exact=True)


def test_reproducible_fit_and_predictions(sample):
    applicants, _, model, predictions = _fit(sample)
    repeated = fit_bank_ols(applicants.sample(frac=1, random_state=6))
    assert model == repeated
    pd.testing.assert_frame_equal(predictions, risk_estimates(applicants, repeated), check_exact=True)
    assert np.isfinite(list(model["coefficients"].values())).all()
    assert np.isfinite(predictions[["bank_PD_raw", "bank_PD_bounded"]]).all().all()


def test_manual_prediction_and_raw_preservation(sample):
    applicants, _, model, predictions = _fit(sample)
    coefficients = model["coefficients"]
    manual = sum(coefficients[key] * (1 if key == "const" else applicants[key].to_numpy()) for key in DESIGN_COLUMNS)
    np.testing.assert_allclose(predictions.bank_PD_raw, manual, atol=1e-15, rtol=1e-13)
    np.testing.assert_array_equal(predictions.bank_PD_bounded, np.clip(predictions.bank_PD_raw, 0, 1))
    np.testing.assert_array_equal(predictions.prediction_out_of_bounds, (predictions.bank_PD_raw < 0) | (predictions.bank_PD_raw > 1))
    assert tuple(predictions.columns) == ESTIMATE_COLUMNS


def test_evaluation_pn_cannot_change_fitted_coefficients(sample):
    applicants, _, model, _ = _fit(sample)
    applicants.loc[applicants.split == "evaluation", "PN"] = 1e9
    assert fit_bank_ols(applicants) == model


def test_truth_cannot_change_predictions(sample):
    applicants, _, model, predictions = _fit(sample)
    applicants["true_PD"] = np.linspace(0, 1, len(applicants))
    first = predict_bank_ols(applicants, model)
    applicants["true_PD"] = -1e9
    np.testing.assert_array_equal(first, predict_bank_ols(applicants, model))
    np.testing.assert_array_equal(first, predictions.bank_PD_raw)


def test_pn_not_needed_for_frozen_prediction(sample):
    applicants, _, model, predictions = _fit(sample)
    applicants["PN"] = np.nan
    np.testing.assert_array_equal(predict_bank_ols(applicants, model), predictions.bank_PD_raw)
    np.testing.assert_array_equal(predict_bank_ols(applicants.drop(columns="PN"), model), predictions.bank_PD_raw)


def test_rank_deficiency_rejected(sample):
    applicants, _, _, _ = _fit(sample)
    applicants["C"] = applicants.D
    with pytest.raises(ValueError, match="full column rank"):
        fit_bank_ols(applicants)


def test_nonfinite_features_rejected(sample):
    bank = sample[0].copy()
    bank.loc[0, "C"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        design_matrix(bank)


def test_invalid_split_join_rejected(sample):
    applicants, splits, _, _ = _fit(sample)
    splits.loc[0, "split"] = "validation"
    with pytest.raises(ValueError):
        attach_splits(applicants.drop(columns="split"), splits)


def test_error_metrics_hand_calculation():
    metrics = error_metrics(np.array([.1, .2, .3]), np.array([.0, .3, .4]))
    assert metrics["MAE"] == pytest.approx(.1)
    assert metrics["RMSE"] == pytest.approx(.1)
    assert metrics["mean_signed_error"] == pytest.approx(.1 / 3)
    assert metrics["R_squared"] == pytest.approx(-.5)
    assert metrics["Pearson_correlation"] == pytest.approx(np.corrcoef([.1, .2, .3], [.0, .3, .4])[0, 1])


def test_undefined_correlation_is_explicit():
    metrics = error_metrics(np.array([.1, .2]), np.array([.15, .15]))
    assert metrics["Pearson_correlation"] is None
    assert metrics["correlation_status"] == "undefined_constant_vector"


def test_distribution_counts_keep_out_of_range_and_ties():
    raw = np.array([-.2, 0, .3, 1, 1.2])
    summary = prediction_summary(raw, np.clip(raw, 0, 1))
    assert summary["below_zero_count"] == summary["above_one_count"] == 1
    assert summary["below_zero_share"] == .2
    assert summary["bounded_zero_count"] == summary["bounded_one_count"] == 2
    assert summary["sample_SD"] == pytest.approx(raw.std(ddof=1))


def test_evaluation_groups_and_training_constant(sample):
    applicants, _, model, predictions = _fit(sample)
    truth = sample[2]
    metrics, groups, preview, summary = evaluate_bank_risk(applicants, predictions, truth, model)
    assert metrics.n.unique().tolist() == [160]
    assert tuple(preview.columns) == PREVIEW_COLUMNS and len(preview) == 150
    assert set(preview.applicant_id) <= set(applicants.loc[applicants.split == "evaluation", "applicant_id"])
    assert preview.information_scope.unique().tolist() == ["EVALUATOR_ONLY_NOT_FOR_LENDER_TRAINING"]
    pn = applicants.loc[applicants.split == "evaluation", "PN"].to_numpy()
    const = metrics.loc[metrics.estimate == "constant_training_mean_PN"].iloc[0]
    assert const.MAE == pytest.approx(np.abs(pn - model["training_mean_PN"]).mean())
    assert const.RMSE == pytest.approx(np.sqrt(np.mean((pn - model["training_mean_PN"]) ** 2)))
    assert groups.n_applicants.sum() == summary["n"] == 160
    evaluation = applicants.loc[applicants.split == "evaluation"].merge(predictions, on=["applicant_id", "split"])
    for record in groups.to_dict("records"):
        frame = evaluation.loc[evaluation.B == record["B"]]
        assert record["n_applicants"] == len(frame)
        assert record["raw_MAE"] == pytest.approx(np.abs(frame.bank_PD_raw - .005).mean())
        assert record["raw_mean_signed_error"] == pytest.approx((frame.bank_PD_raw - .005).mean())


def test_evaluator_truth_mismatch_rejected(sample):
    applicants, _, model, predictions = _fit(sample)
    truth = sample[2].copy()
    truth.loc[0, "applicant_id"] = "unknown"
    with pytest.raises(ValueError, match="mismatched IDs"):
        evaluate_bank_risk(applicants, predictions, truth, model)


def test_bank_loader_reads_only_two_observed_tables(sample, tmp_path, monkeypatch):
    _input_files(sample, tmp_path)
    original = pd.read_parquet
    seen = []
    def observed_read(path, *args, **kwargs):
        seen.append(path.name)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(pd, "read_parquet", observed_read)
    joined, hashes, schemas = load_bank_inputs(sample[3], tmp_path)
    assert seen == ["applicants_bank_view.parquet", "training_signals.parquet"]
    assert len(joined) == 800 and len(schemas) == 2
    assert verify_input_hashes(hashes, tmp_path)


def test_original_checksum_mismatch_rejected(sample, tmp_path):
    _input_files(sample, tmp_path)
    path = tmp_path / sample[3]["input_directory"] / "applicants_simulator_private.parquet"
    path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_bank_inputs(sample[3], tmp_path)


def test_truth_loader_requires_frozen_saved_model(sample, tmp_path):
    _input_files(sample, tmp_path)
    _, hashes, _ = load_bank_inputs(sample[3], tmp_path)
    with pytest.raises(ValueError, match="frozen"):
        load_evaluator_truth(sample[3], {}, hashes, tmp_path)


def test_export_replay_boundary_and_original_preservation(sample, tmp_path):
    config_path = _input_files(sample, tmp_path)
    config = sample[3]
    joined, hashes, _ = load_bank_inputs(config, tmp_path)
    original_config_hash = file_hash(config_path)
    applicants, splits, model, predictions = _fit(sample)
    before = copy.deepcopy(model)
    checkpoint = freeze_bank_artifacts(applicants, splits, predictions, model, config, hashes, root=tmp_path, config_path=config_path)
    paths = output_paths(config, tmp_path)
    saved = json.loads(paths["model"].read_text())
    assert saved["frozen_before_evaluator_truth"] is True
    np.testing.assert_array_equal(predict_bank_ols(applicants, saved), predictions.bank_PD_raw)
    pd.testing.assert_frame_equal(pd.read_csv(paths["estimates_csv"], float_precision="round_trip"), pd.read_parquet(paths["estimates_parquet"]), check_exact=True)
    truth = load_evaluator_truth(config, checkpoint, hashes, tmp_path)
    report = save_bank_evaluation(applicants, predictions, truth, model, config, checkpoint, hashes, root=tmp_path, config_path=config_path)
    assert model == before and report["valid"]
    assert verify_input_hashes(hashes, tmp_path) and file_hash(config_path) == original_config_hash
    metadata = json.loads(paths["metadata"].read_text())
    assert metadata["source_code_fingerprint"] == configuration_fingerprint(metadata["source_code_hashes"])
    assert all(file_hash(tmp_path / a["path"]) == a["sha256"] for a in metadata["artifacts"].values())
    assert report["checks"]["frozen_bank_artifacts_unchanged"]
    assert tuple(pd.read_csv(paths["estimates_csv"]).columns) == ESTIMATE_COLUMNS
    assert all(path.exists() for path in paths.values())
    with pytest.raises(FileExistsError):
        freeze_bank_artifacts(applicants, splits, predictions, model, config, hashes, root=tmp_path, config_path=config_path)


def test_modified_frozen_predictions_prevent_truth_loading(sample, tmp_path):
    config_path = _input_files(sample, tmp_path)
    _, hashes, _ = load_bank_inputs(sample[3], tmp_path)
    applicants, splits, model, predictions = _fit(sample)
    checkpoint = freeze_bank_artifacts(applicants, splits, predictions, model, sample[3], hashes, root=tmp_path, config_path=config_path)
    output_paths(sample[3], tmp_path)["estimates_csv"].write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="changed before truth"):
        load_evaluator_truth(sample[3], checkpoint, hashes, tmp_path)
