"""Deterministic safeguards for the Prompt 9 sensitivity framework."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from fair_lending.analysis.logit import fit_logit_model
from fair_lending.models.features import RACE_AWARE, RACE_BLIND, get_feature_spec
from fair_lending.sensitivity.design import (
    ANALYSIS_VERSION,
    RunSpec,
    build_experiment_design,
    replication_seed,
    resolve_sensitivity_config,
    run_id,
    upstream_treatment,
)
from fair_lending.sensitivity.metrics import (
    detection_table,
    infer_detection_thresholds,
    mechanism_signature,
    relative_recovery,
    signature_table,
    summarize_estimate,
    summarize_runs,
)
from fair_lending.sensitivity.persistence import (
    load_matching_record,
    record_path,
    write_record_atomic,
)
from fair_lending.sensitivity.runner import execute_design, records_frame
from fair_lending.simulation.approval import approval_probabilities
from fair_lending.simulation.config import stable_fingerprint
from fair_lending.simulation.generator import (
    OUTPUT_COLUMNS,
    SYNTHETIC_SCHEMA_VERSION,
    generate_from_resolved_config,
)
from fair_lending.simulation.treatments import apply_treatment_overrides


INTERCEPT = 2.0362202310934663
MASTER_SEED = 4_994


def _spec(
    *,
    n_rows: int = 2_500,
    replication: int = 0,
    direct: float = 0.0,
    upstream: float = 0.0,
    families: tuple[str, ...] = ("direct",),
) -> RunSpec:
    child_seed, spawn_key = replication_seed(MASTER_SEED, replication)
    return RunSpec(
        n_rows=n_rows,
        replication_index=replication,
        master_seed=MASTER_SEED,
        child_seed=child_seed,
        spawn_key=spawn_key,
        direct_black_log_odds=direct,
        upstream_strength=upstream,
        families=families,
    )


@lru_cache(maxsize=None)
def _data(direct: float, upstream: float, n_rows: int = 20_000):
    config = resolve_sensitivity_config(
        _spec(n_rows=n_rows, direct=direct, upstream=upstream)
    )
    data, _ = generate_from_resolved_config(config, INTERCEPT)
    return data, config


def _toy_run(**overrides) -> dict:
    row = {
        "status": "completed",
        "run_id": "toy",
        "analysis_version": ANALYSIS_VERSION,
        "schema_version": SYNTHETIC_SCHEMA_VERSION,
        "config_fingerprint": "config",
        "families": ["direct"],
        "n_rows": 100_000,
        "replication_index": 0,
        "direct_black_log_odds": -0.25,
        "upstream_strength": 0.0,
        "scenario_family": "direct_discrimination",
        "raw_approval_gap": -0.03,
        "raw_ci_low": -0.04,
        "raw_ci_high": -0.02,
        "model_0_black_coefficient": -0.24,
        "model_1_black_coefficient": -0.25,
        "model_2_black_coefficient": -0.25,
        "model_2_black_p_value": 0.01,
        "model_2_black_ci_low": -0.35,
        "model_2_black_ci_high": -0.15,
        "model_2_adjusted_probability_gap": -0.03,
        "model_3_black_coefficient": -0.26,
        "true_direct_log_odds": -0.25,
        "true_direct_probability_gap": -0.03,
        "test_true_probability_gap": -0.03,
        "race_blind_predicted_probability_gap": 0.0,
        "race_aware_predicted_probability_gap": -0.03,
        "race_blind_true_probability_mae": 0.02,
        "race_aware_true_probability_mae": 0.01,
        "mechanism_signature": "Pattern B",
    }
    row.update(overrides)
    return row


def test_full_design_has_declared_3140_unique_runs() -> None:
    design = build_experiment_design("all")
    assert len(design) == 3_140
    scientific_cells = {
        (item.n_rows, item.replication_index, item.direct_black_log_odds, item.upstream_strength)
        for item in design
    }
    assert len(scientific_cells) == len(design)


def test_quick_design_has_declared_small_grid() -> None:
    design = build_experiment_design("all", quick=True)
    assert len(design) == 16
    assert {item.n_rows for item in design} == {1_000, 2_500}
    assert {item.replication_index for item in design} == {0, 1}


def test_replication_seed_is_deterministic() -> None:
    assert replication_seed(MASTER_SEED, 7) == replication_seed(MASTER_SEED, 7)


def test_common_random_number_seed_is_shared_across_settings_and_sizes() -> None:
    design = build_experiment_design("all", quick=True)
    for replication in (0, 1):
        seeds = {item.child_seed for item in design if item.replication_index == replication}
        assert len(seeds) == 1


def test_distinct_replications_receive_distinct_child_seeds() -> None:
    seeds = {replication_seed(MASTER_SEED, index)[0] for index in range(20)}
    assert len(seeds) == 20


def test_upstream_strength_uses_declared_scales() -> None:
    treatment = upstream_treatment(1.5)
    assert treatment["black_income_log_shift"] == pytest.approx(1.5 * np.log(0.90))
    assert treatment["black_credit_score_shift"] == -37.5
    assert treatment["black_liquid_assets_log_shift"] == pytest.approx(1.5 * np.log(0.75))
    assert treatment["black_income_multiplier"] == pytest.approx(0.90**1.5)
    assert treatment["black_liquid_assets_multiplier"] == pytest.approx(0.75**1.5)


def test_treatment_override_does_not_mutate_source_config() -> None:
    source = resolve_sensitivity_config(_spec())
    fingerprint = stable_fingerprint(source)
    apply_treatment_overrides(
        source,
        direct_black_log_odds=-0.5,
        black_income_log_shift=np.log(0.8),
        black_credit_score_shift=-40.0,
        black_liquid_assets_log_shift=np.log(0.6),
        source_label="test",
    )
    assert stable_fingerprint(source) == fingerprint


def test_direct_sweep_changes_only_direct_mechanism() -> None:
    config = resolve_sensitivity_config(_spec(direct=-0.25))
    assert config["scenario_effects"]["direct"]["log_odds_by_race"]["Black"] == -0.25
    assert not config["scenario_effects"]["upstream"]["enabled"]
    for treatment in config["scenario_effects"]["upstream"]["treatments"].values():
        assert set(treatment["additive_shift_by_race"].values()) == {0.0}


def test_upstream_sweep_changes_only_three_black_upstream_parameters() -> None:
    config = resolve_sensitivity_config(_spec(upstream=1.0))
    direct = config["scenario_effects"]["direct"]
    assert set(direct["log_odds_by_race"].values()) == {0.0}
    treatments = config["scenario_effects"]["upstream"]["treatments"]
    assert treatments["annual_income"]["additive_shift_by_race"]["Black"] == pytest.approx(np.log(0.90))
    assert treatments["credit_score"]["additive_shift_by_race"]["Black"] == -25.0
    assert treatments["liquid_assets"]["additive_shift_by_race"]["Black"] == pytest.approx(np.log(0.75))
    for treatment in treatments.values():
        shifts = treatment["additive_shift_by_race"]
        assert all(value == 0.0 for race, value in shifts.items() if race != "Black")


@pytest.mark.parametrize(
    ("direct", "upstream", "expected"),
    [
        (0.0, 0.0, "fair_baseline"),
        (-0.25, 0.0, "direct_discrimination"),
        (0.0, 1.0, "upstream_inequality"),
        (-0.25, 1.0, "mixed_mechanism"),
    ],
)
def test_mechanism_switches_map_to_four_canonical_scenarios(direct, upstream, expected) -> None:
    assert _spec(direct=direct, upstream=upstream).scenario_family == expected


def test_resolved_generation_preserves_schema_seed_and_frozen_intercept() -> None:
    spec = _spec(n_rows=1_000, direct=-0.25, upstream=1.0)
    config = resolve_sensitivity_config(spec)
    data, _ = generate_from_resolved_config(config, INTERCEPT)
    assert data.shape == (1_000, 24)
    assert list(data) == OUTPUT_COLUMNS
    assert config["simulation"]["random_seed"] == spec.child_seed
    np.testing.assert_allclose(
        data["approval_probability_true"],
        approval_probabilities(data, config, INTERCEPT),
        rtol=0.0,
        atol=0.0,
    )


def test_resolved_generation_is_exactly_reproducible() -> None:
    config = resolve_sensitivity_config(_spec(n_rows=1_000, direct=-0.25))
    first, _ = generate_from_resolved_config(config, INTERCEPT)
    second, _ = generate_from_resolved_config(config, INTERCEPT)
    assert_frame_equal(first, second, check_exact=True)


def test_direct_adjusted_coefficient_recovers_configured_effect() -> None:
    data, config = _data(-0.25, 0.0)
    coefficient = fit_logit_model(data, config, "model_2")["summary"]["black_coefficient"]
    assert coefficient == pytest.approx(-0.25, abs=0.15)


def test_upstream_adjustment_is_materially_closer_to_zero() -> None:
    data, config = _data(0.0, 1.0)
    unadjusted = fit_logit_model(data, config, "model_0")["summary"]["black_coefficient"]
    adjusted = fit_logit_model(data, config, "model_2")["summary"]["black_coefficient"]
    assert abs(adjusted) < 0.5 * abs(unadjusted)
    assert adjusted == pytest.approx(0.0, abs=0.15)


def test_feature_regimes_preserve_leakage_policy() -> None:
    blind = set(get_feature_spec(RACE_BLIND).columns)
    aware = set(get_feature_spec(RACE_AWARE).columns)
    forbidden = {
        "approval_probability_true", "approved", "denial_reason", "application_id",
        "neighborhood_minority_share", "ethnicity", "sex", "age_group",
    }
    assert "race" not in blind
    assert aware == blind | {"race"}
    assert not forbidden.intersection(aware)


def test_run_id_is_stable_and_sensitive_to_scientific_inputs() -> None:
    first = _spec(direct=-0.25)
    config = resolve_sensitivity_config(first)
    fingerprint = stable_fingerprint(config)
    assert run_id(first, fingerprint) == run_id(first, fingerprint)
    changed = _spec(direct=-0.35)
    changed_fingerprint = stable_fingerprint(resolve_sensitivity_config(changed))
    assert run_id(first, fingerprint) != run_id(changed, changed_fingerprint)


def test_atomic_record_round_trip_and_resume_integrity(tmp_path) -> None:
    path = tmp_path / "record.json"
    record = _toy_run()
    write_record_atomic(record, path)
    assert load_matching_record(
        path, expected_run_id="toy", expected_config_fingerprint="config"
    ) == record
    assert load_matching_record(
        path, expected_run_id="different", expected_config_fingerprint="config"
    ) is None


def test_resume_rejects_schema_version_mismatch(tmp_path) -> None:
    path = tmp_path / "record.json"
    write_record_atomic(_toy_run(schema_version="obsolete"), path)
    assert load_matching_record(
        path, expected_run_id="toy", expected_config_fingerprint="config"
    ) is None


def test_failed_or_corrupt_records_are_not_resumed(tmp_path) -> None:
    path = tmp_path / "record.json"
    write_record_atomic(_toy_run(status="failed"), path)
    assert load_matching_record(path, expected_run_id="toy", expected_config_fingerprint="config") is None
    path.write_text("{broken", encoding="utf-8")
    assert load_matching_record(path, expected_run_id="toy", expected_config_fingerprint="config") is None


def test_toy_estimate_summary_units_bias_and_rmse() -> None:
    estimate = pd.Series([-0.02, -0.04])
    truth = pd.Series([-0.03, -0.03])
    result = summarize_estimate(estimate, truth)
    assert result["mean"] == pytest.approx(-0.03)
    assert result["bias"] == pytest.approx(0.0)
    assert result["rmse"] == pytest.approx(0.01)
    assert result["mc_sd"] == pytest.approx(np.sqrt(0.0002))


def test_relative_recovery_handles_nonzero_and_zero_truth() -> None:
    result = relative_recovery(np.array([-0.03, -0.015, 0.0]), np.array([-0.03, -0.03, 0.0]))
    np.testing.assert_allclose(result[:2], [1.0, 0.5])
    assert np.isnan(result[2])


def test_detection_coverage_and_false_positive_calculations() -> None:
    rows = [
        _toy_run(run_id="a"),
        _toy_run(run_id="b", replication_index=1, model_2_black_p_value=0.20),
        _toy_run(
            run_id="c", replication_index=0, direct_black_log_odds=0.0,
            scenario_family="fair_baseline", true_direct_log_odds=0.0,
            model_2_black_coefficient=0.01, model_2_black_p_value=0.01,
            model_2_black_ci_low=-0.02, model_2_black_ci_high=0.04,
        ),
    ]
    table = detection_table(pd.DataFrame(rows))
    direct = table.loc[table["direct_black_log_odds"].eq(-0.25)].iloc[0]
    fair = table.loc[table["direct_black_log_odds"].eq(0.0)].iloc[0]
    assert direct["model_2_detection_probability"] == 0.5
    assert direct["model_2_coverage_probability"] == 1.0
    assert fair["model_2_false_positive_rate"] == 1.0


@pytest.mark.parametrize(
    ("updates", "expected"),
    [
        ({"raw_ci_low": -0.01, "raw_ci_high": 0.01, "model_2_black_p_value": 0.5,
          "race_blind_predicted_probability_gap": 0.0}, "Pattern A"),
        ({}, "Pattern B"),
        ({"model_2_black_p_value": 0.5, "race_blind_predicted_probability_gap": -0.03,
          "race_aware_predicted_probability_gap": -0.03}, "Pattern C"),
        ({"race_blind_predicted_probability_gap": -0.02,
          "race_aware_predicted_probability_gap": -0.04}, "Pattern D"),
    ],
)
def test_predeclared_mechanism_signatures(updates, expected) -> None:
    assert mechanism_signature(_toy_run(**updates)) == expected


def test_signature_summary_marks_modal_pattern() -> None:
    frame = pd.DataFrame([
        _toy_run(run_id="a"),
        _toy_run(run_id="b", replication_index=1),
        _toy_run(run_id="c", replication_index=2, mechanism_signature="Other / ambiguous"),
    ])
    result = signature_table(frame)
    modal = result.loc[result["is_modal_signature"]]
    assert modal.iloc[0]["mechanism_signature"] == "Pattern B"
    assert modal.iloc[0]["signature_share"] == pytest.approx(2 / 3)


def test_threshold_inference_uses_declared_80_percent_rules() -> None:
    detection = pd.DataFrame([
        {"n_rows": 100_000, "direct_black_log_odds": -0.10, "upstream_strength": 0.0,
         "model_2_negative_detection_probability": 0.79, "raw_gap_below_minus_2pp_probability": 0.0,
         "race_aware_75pct_recovery_probability": 0.79, "race_blind_below_50pct_recovery_probability": 0.81},
        {"n_rows": 100_000, "direct_black_log_odds": -0.25, "upstream_strength": 0.0,
         "model_2_negative_detection_probability": 0.80, "raw_gap_below_minus_2pp_probability": 0.0,
         "race_aware_75pct_recovery_probability": 0.80, "race_blind_below_50pct_recovery_probability": 0.90},
        {"n_rows": 100_000, "direct_black_log_odds": 0.0, "upstream_strength": 0.5,
         "model_2_negative_detection_probability": 0.0, "raw_gap_below_minus_2pp_probability": 0.80,
         "race_aware_75pct_recovery_probability": 0.0, "race_blind_below_50pct_recovery_probability": 0.0},
    ])
    result = infer_detection_thresholds(detection).set_index("threshold")
    assert result.loc["model_2_negative_detection_80pct", "value"] == 0.25
    assert result.loc["raw_gap_below_minus_2pp_80pct", "value"] == 0.5
    assert result.loc["race_aware_75pct_recovery_in_80pct_runs", "value"] == 0.25
    assert result.loc["race_blind_below_50pct_recovery_in_80pct_runs", "value"] == 0.10


def test_records_frame_expands_only_requested_memberships_and_sorts() -> None:
    rows = [
        _toy_run(run_id="b", replication_index=1, families=["direct", "detection"]),
        _toy_run(run_id="a", replication_index=0, families=["direct", "detection"]),
    ]
    frame = records_frame(rows, {"direct"})
    assert frame["experiment_family"].unique().tolist() == ["direct"]
    assert frame["run_id"].tolist() == ["a", "b"]


def test_summary_has_one_row_per_setting_and_estimand() -> None:
    rows = [
        _toy_run(run_id="a", replication_index=0),
        _toy_run(run_id="b", replication_index=1),
    ]
    result = summarize_runs(records_frame(rows, {"direct"}), "direct")
    keys = [
        "experiment_family", "n_rows", "direct_black_log_odds", "upstream_strength",
        "scenario_family", "estimand",
    ]
    assert not result.duplicated(keys).any()
    assert set(result["n_completed"]) == {2}


def test_execute_design_persists_then_resumes_completed_run(tmp_path) -> None:
    spec = _spec(n_rows=1_000, direct=-0.25)
    first, failures, resumed = execute_design(
        [spec], intercept=INTERCEPT, revision=None, dirty=None, resume=False,
        workers=1, project_root=tmp_path, progress_every=1,
    )
    assert len(first) == 1 and not failures and resumed == 0
    assert record_path(first[0]["run_id"], tmp_path).exists()
    second, failures, resumed = execute_design(
        [spec], intercept=INTERCEPT, revision=None, dirty=None, resume=True,
        workers=1, project_root=tmp_path, progress_every=1,
    )
    assert len(second) == 1 and not failures and resumed == 1
    assert second[0]["run_id"] == first[0]["run_id"]
