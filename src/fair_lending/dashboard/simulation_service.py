"""In-memory simulation and export services for interactive research use."""

from __future__ import annotations

import io
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from fair_lending.analysis.descriptive import black_white_raw_gap, descriptive_outcomes
from fair_lending.analysis.estimands import true_direct_effect
from fair_lending.analysis.logit import prepare_regression_sample
from fair_lending.simulation.calibration import load_calibration_artifact
from fair_lending.simulation.config import (
    git_revision,
    package_version,
    resolve_simulation_config,
    stable_fingerprint,
)
from fair_lending.simulation.generator import generate_from_resolved_config
from fair_lending.simulation.treatments import (
    apply_treatment_overrides,
    configured_treatments,
)


@dataclass(frozen=True)
class CustomTreatments:
    """Small, intentionally bounded set of temporary treatment overrides."""

    direct_black_log_odds: float
    black_income_multiplier: float
    black_credit_score_shift: float
    black_liquid_assets_multiplier: float


@dataclass(frozen=True)
class SimulationRequest:
    """Hashable dashboard request suitable for deterministic Streamlit caching."""

    scenario: str = "fair_baseline"
    effect_level: str = "moderate"
    n_samples: int = 10_000
    random_seed: int = 4_994
    custom_treatments: CustomTreatments | None = None


def resolve_dashboard_config(request: SimulationRequest) -> dict[str, Any]:
    """Resolve canonical YAML and apply custom parameters only to a deep copy."""
    config = resolve_simulation_config(
        request.scenario,
        request.effect_level,
        request.n_samples,
        request.random_seed,
    )
    custom = request.custom_treatments
    if custom is None:
        return config

    config = apply_treatment_overrides(
        config,
        direct_black_log_odds=custom.direct_black_log_odds,
        black_income_log_shift=float(np.log(custom.black_income_multiplier)),
        black_credit_score_shift=custom.black_credit_score_shift,
        black_liquid_assets_log_shift=float(
            np.log(custom.black_liquid_assets_multiplier)
        ),
        source_label="dashboard_custom",
    )
    config["simulation"]["dashboard_custom_treatments"] = True
    return config


def generate_dashboard_simulation(
    request: SimulationRequest, *, intercept: float | None = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Generate one in-memory sample without calibrating or persisting it."""
    config = resolve_dashboard_config(request)
    if intercept is None:
        calibration = load_calibration_artifact()
        frozen_intercept = float(calibration["intercept"])
        calibration_fingerprint = calibration["config_fingerprint"]
    else:
        frozen_intercept = float(intercept)
        calibration_fingerprint = None
    data, generation = generate_from_resolved_config(config, frozen_intercept)
    revision, dirty = git_revision()
    metadata = {
        "mode": "dashboard_in_memory",
        "scenario": request.scenario,
        "effect_level": request.effect_level,
        "n_rows": request.n_samples,
        "seed": request.random_seed,
        "custom_treatments": (
            asdict(request.custom_treatments)
            if request.custom_treatments is not None
            else None
        ),
        "configured_treatments": configured_treatments(config),
        "intercept": frozen_intercept,
        "intercept_calibration_source": "frozen_fair_baseline_artifact",
        "intercept_calibration_config_fingerprint": calibration_fingerprint,
        "config_fingerprint": stable_fingerprint(config),
        "resolved_configuration": config,
        "random_stream_strategy": "NumPy SeedSequence child streams",
        "random_stream_spawn_keys": generation["random_stream_spawn_keys"],
        "population_diagnostics": generation["population_diagnostics"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "package_version": package_version(),
        "git_revision": revision,
        "git_worktree_dirty": dirty,
        "persisted": False,
    }
    return data, metadata


def summarize_simulation(
    data: pd.DataFrame, metadata: dict[str, Any]
) -> dict[str, Any]:
    """Calculate the dashboard's declared sample and mechanism summaries."""
    gap = black_white_raw_gap(data, metadata["scenario"])
    config = metadata["resolved_configuration"]
    regression_sample = prepare_regression_sample(data, config)
    truth = true_direct_effect(regression_sample, config, float(metadata["intercept"]))
    race = data["race"].astype(object)
    white = data.loc[race.eq("White")]
    black = data.loc[race.eq("Black")]
    return {
        "n_applications": int(len(data)),
        "overall_approval_rate": float(data["approved"].mean()),
        "white_approval_rate": float(white["approved"].mean()),
        "black_approval_rate": float(black["approved"].mean()),
        "black_white_approval_gap": float(gap["raw_approval_gap"]),
        "gap_ci_low": float(gap["ci_low"]),
        "gap_ci_high": float(gap["ci_high"]),
        "median_income": float(data["annual_income"].median()),
        "median_credit_score": float(data["credit_score"].median()),
        "median_dti": float(data["debt_to_income_ratio"].median()),
        "median_ltv": float(data["loan_to_value_ratio"].median()),
        **truth,
        **metadata["configured_treatments"],
    }


def race_outcome_table(data: pd.DataFrame, metadata: dict[str, Any]) -> pd.DataFrame:
    """Return all configured race summaries for the approval explorer."""
    config = metadata["resolved_configuration"]
    return descriptive_outcomes(data, config, metadata["scenario"]).loc[
        lambda frame: frame["row_type"].eq("race")
    ].reset_index(drop=True)


def export_dataset_csv(data: pd.DataFrame) -> bytes:
    """Serialize the active sample to CSV without writing to the repository."""
    return data.to_csv(index=False).encode("utf-8")


def export_dataset_parquet(data: pd.DataFrame) -> bytes:
    """Serialize the active sample to Parquet without writing to the repository."""
    buffer = io.BytesIO()
    data.to_parquet(buffer, index=False, engine="pyarrow")
    return buffer.getvalue()


def export_summary_csv(summary: dict[str, Any]) -> bytes:
    """Serialize scalar summary values as a one-row CSV."""
    scalar = {
        key: value
        for key, value in summary.items()
        if isinstance(value, (str, int, float, bool)) or value is None
    }
    return pd.DataFrame([scalar]).to_csv(index=False).encode("utf-8")


def export_reproducibility_json(metadata: dict[str, Any]) -> bytes:
    """Serialize the exact resolved in-memory experiment configuration."""
    payload = {
        key: metadata[key]
        for key in (
            "mode",
            "scenario",
            "effect_level",
            "n_rows",
            "seed",
            "custom_treatments",
            "configured_treatments",
            "intercept",
            "intercept_calibration_source",
            "intercept_calibration_config_fingerprint",
            "config_fingerprint",
            "resolved_configuration",
            "random_stream_strategy",
            "random_stream_spawn_keys",
        )
    }
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
