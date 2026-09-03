"""Public generation API and separate Parquet/metadata persistence."""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fair_lending.simulation.approval import approval_probabilities
from fair_lending.simulation.calibration import (
    calibrate_intercept,
    expected_calibration_fingerprint,
)
from fair_lending.simulation.config import (
    PROJECT_ROOT,
    git_revision,
    package_version,
    resolve_simulation_config,
    stable_fingerprint,
)
from fair_lending.simulation.population import create_random_streams, generate_population


OUTPUT_COLUMNS = [
    "application_id",
    "race",
    "ethnicity",
    "sex",
    "age_group",
    "annual_income",
    "credit_score",
    "employment_years",
    "liquid_assets",
    "existing_monthly_debt",
    "debt_to_income_ratio",
    "loan_amount",
    "property_value",
    "loan_to_value_ratio",
    "loan_purpose",
    "loan_type",
    "occupancy_type",
    "neighborhood_income_index",
    "neighborhood_minority_share",
    "local_unemployment_rate",
    "income_to_loan_ratio",
    "approval_probability_true",
    "approved",
    "denial_reason",
]
SYNTHETIC_SCHEMA_VERSION = "1.0-24-fields"


def dataset_filename(scenario: str, effect_level: str, n_rows: int, seed: int) -> str:
    return f"synthetic_{scenario}_{effect_level}_n{n_rows}_seed{seed}.parquet"


def generate_from_resolved_config(
    config: dict[str, Any], intercept: float
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Generate one dataset from an already-resolved in-memory configuration.

    This low-level entry point supports declared sensitivity experiments without
    modifying YAML. It performs no calibration, persistence, or Git inspection.
    """
    n_rows = int(config["simulation"]["n_samples"])
    seed = int(config["simulation"]["random_seed"])
    streams, spawn_keys = create_random_streams(seed)
    applications, population_diagnostics = generate_population(config, streams)
    probabilities = approval_probabilities(applications, config, float(intercept))
    applications["approval_probability_true"] = probabilities
    applications["approved"] = (
        streams["approval"].random(n_rows) < probabilities
    ).astype(np.int8)
    applications["denial_reason"] = pd.Series(
        np.full(n_rows, None, dtype=object), dtype="object"
    )
    return applications.loc[:, OUTPUT_COLUMNS], {
        "random_stream_spawn_keys": spawn_keys,
        "population_diagnostics": population_diagnostics,
    }


def generate_synthetic_data(
    scenario: str = "fair_baseline",
    effect_level: str = "moderate",
    n_rows: int = 100_000,
    seed: int = 4_994,
    *,
    intercept: float | None = None,
    recalibrate_intercept: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Generate a synthetic mortgage-application dataset and run metadata."""
    config = resolve_simulation_config(scenario, effect_level, n_rows, seed)
    if intercept is None:
        calibration = calibrate_intercept(force=recalibrate_intercept)
        frozen_intercept = float(calibration["intercept"])
        calibration_source = "artifact"
    else:
        frozen_intercept = float(intercept)
        intercept_spec = config["approval_model"]["intercept"]
        calibration = {
            "intercept": frozen_intercept,
            "target_mean_probability": float(
                intercept_spec["target_mean_probability"]
            ),
            "config_fingerprint": expected_calibration_fingerprint(),
        }
        calibration_source = "explicit_argument"

    applications, generation = generate_from_resolved_config(config, frozen_intercept)

    revision, dirty = git_revision()
    metadata = {
        "scenario": scenario,
        "effect_level": effect_level,
        "n_rows": n_rows,
        "seed": seed,
        "resolved_configuration": config,
        "intercept": frozen_intercept,
        "target_baseline_approval_rate": float(
            config["approval_model"]["intercept"]["target_mean_probability"]
        ),
        "intercept_calibration_source": calibration_source,
        "intercept_calibration_config_fingerprint": calibration["config_fingerprint"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "package_version": package_version(),
        "git_revision": revision,
        "git_worktree_dirty": dirty,
        "config_fingerprint": stable_fingerprint(config),
        "random_stream_strategy": "NumPy SeedSequence child streams",
        "random_stream_spawn_keys": generation["random_stream_spawn_keys"],
        "population_diagnostics": generation["population_diagnostics"],
        "output_file_path": None,
        "row_count": int(len(applications)),
        "schema_column_count": int(len(applications.columns)),
    }
    return applications, metadata


def save_synthetic_dataset(
    data: pd.DataFrame,
    metadata: dict[str, Any],
    output_path: Path | str | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    """Persist one dataset as Parquet and adjacent reproducibility metadata."""
    if output_path is None:
        output_path = PROJECT_ROOT / "data" / "synthetic" / dataset_filename(
            metadata["scenario"],
            metadata["effect_level"],
            metadata["n_rows"],
            metadata["seed"],
        )
    path = Path(output_path)
    if path.suffix == "":
        path = path / dataset_filename(
            metadata["scenario"],
            metadata["effect_level"],
            metadata["n_rows"],
            metadata["seed"],
        )
    if path.suffix.lower() != ".parquet":
        raise ValueError("Synthetic research datasets must be saved as .parquet")
    if list(data.columns) != OUTPUT_COLUMNS:
        raise ValueError("Refusing to persist a dataset that does not match the 24-field schema")

    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(path, index=False, engine="pyarrow")
    saved_metadata = copy.deepcopy(metadata)
    saved_metadata.update(
        {
            "output_file_path": str(path.resolve()),
            "row_count": int(len(data)),
            "schema_column_count": int(len(data.columns)),
        }
    )
    metadata_path = path.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(saved_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path, metadata_path, saved_metadata
