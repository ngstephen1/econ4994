"""One-time fair-baseline approval-intercept calibration."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import expit

from fair_lending.simulation.approval import baseline_linear_predictor
from fair_lending.simulation.config import (
    PROJECT_ROOT,
    calibration_resolved_config,
    git_revision,
    package_version,
    stable_fingerprint,
)
from fair_lending.simulation.population import create_random_streams, generate_population


DEFAULT_ARTIFACT_PATH = PROJECT_ROOT / "results" / "metrics" / "calibrated_intercept.json"


def solve_intercept(
    linear_predictor: np.ndarray,
    target: float,
    minimum: float,
    maximum: float,
    tolerance: float,
) -> tuple[float, float]:
    """Solve mean(sigmoid(alpha + x)) = target by deterministic bisection."""
    lower = float(minimum)
    upper = float(maximum)
    lower_value = float(np.mean(expit(lower + linear_predictor)) - target)
    upper_value = float(np.mean(expit(upper + linear_predictor)) - target)
    if lower_value > 0.0 or upper_value < 0.0:
        raise ValueError("Calibration target is not bracketed by search bounds")
    while upper - lower > tolerance:
        midpoint = (lower + upper) / 2.0
        value = float(np.mean(expit(midpoint + linear_predictor)) - target)
        if value < 0.0:
            lower = midpoint
        else:
            upper = midpoint
    intercept = (lower + upper) / 2.0
    achieved = float(np.mean(expit(intercept + linear_predictor)))
    return intercept, achieved


def expected_calibration_fingerprint() -> str:
    """Fingerprint the canonical population used to solve the shared intercept."""
    return stable_fingerprint(calibration_resolved_config())


def load_calibration_artifact(
    path: Path | str = DEFAULT_ARTIFACT_PATH,
    *,
    verify_fingerprint: bool = True,
) -> dict[str, Any]:
    artifact_path = Path(path)
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    required = {
        "intercept",
        "target_mean_probability",
        "achieved_mean_probability",
        "calibration_population_size",
        "calibration_seed",
        "config_fingerprint",
        "generated_at_utc",
    }
    missing = required - set(artifact)
    if missing:
        raise ValueError(f"Calibration artifact is missing fields: {sorted(missing)}")
    if verify_fingerprint and artifact["config_fingerprint"] != expected_calibration_fingerprint():
        raise ValueError(
            "Calibration artifact config fingerprint does not match the current calibration"
        )
    return artifact


def calibrate_intercept(
    *,
    force: bool = False,
    artifact_path: Path | str = DEFAULT_ARTIFACT_PATH,
    reason: str | None = None,
) -> dict[str, Any]:
    """Create or reuse the single frozen fair-baseline intercept artifact."""
    artifact_path = Path(artifact_path)
    if artifact_path.exists() and not force:
        return load_calibration_artifact(artifact_path)
    previous_artifact = None
    if artifact_path.exists():
        previous_artifact = load_calibration_artifact(
            artifact_path, verify_fingerprint=False
        )

    config = calibration_resolved_config()
    streams, spawn_keys = create_random_streams(config["simulation"]["random_seed"])
    population, diagnostics = generate_population(
        config, streams, include_application_id=False
    )
    predictor = baseline_linear_predictor(population, config)
    intercept_spec = config["approval_model"]["intercept"]
    search_bounds = intercept_spec["search_bounds"]
    intercept, achieved = solve_intercept(
        predictor,
        float(intercept_spec["target_mean_probability"]),
        float(search_bounds["minimum"]),
        float(search_bounds["maximum"]),
        float(intercept_spec["solver_tolerance"]),
    )
    if abs(achieved - intercept_spec["target_mean_probability"]) > intercept_spec[
        "target_tolerance"
    ]:
        raise RuntimeError("Solved intercept did not reach the configured target tolerance")

    revision, dirty = git_revision()
    artifact = {
        "intercept": intercept,
        "target_mean_probability": float(intercept_spec["target_mean_probability"]),
        "achieved_mean_probability": achieved,
        "calibration_population_size": int(config["simulation"]["n_samples"]),
        "calibration_seed": int(config["simulation"]["random_seed"]),
        "config_fingerprint": stable_fingerprint(config),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "package_version": package_version(),
        "git_revision": revision,
        "git_worktree_dirty": dirty,
        "random_stream_spawn_keys": spawn_keys,
        "calibration_diagnostics": diagnostics,
        "recalibration_reason": reason,
        "previous_calibration": (
            {
                "intercept": previous_artifact["intercept"],
                "target_mean_probability": previous_artifact[
                    "target_mean_probability"
                ],
                "achieved_mean_probability": previous_artifact[
                    "achieved_mean_probability"
                ],
                "config_fingerprint": previous_artifact["config_fingerprint"],
                "generated_at_utc": previous_artifact["generated_at_utc"],
            }
            if previous_artifact is not None
            else None
        ),
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return artifact
