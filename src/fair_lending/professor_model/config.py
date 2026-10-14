"""Small YAML configuration contract for the professor's applicant population."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs/professor_model/baseline.yaml"
SCHEMA_VERSION = "professor-population-v1"
# Fixed ordering is part of the random-stream contract, not a YAML mapping order.
STREAM_NAMES = ("log_Y", "D", "B", "epsilon", "omega")
BANK_COLUMNS = ("applicant_id", "Y", "D", "B", "C")
PRIVATE_COLUMNS = ("applicant_id", "log_Y", "epsilon", "omega")
PREVIEW_COLUMNS = (
    "information_scope", "applicant_id", "Y", "log_Y", "D", "B", "epsilon", "omega", "C"
)
PREVIEW_SCOPE = "EVALUATOR_ONLY_NOT_FOR_LENDER_TRAINING"


class PopulationConfigurationError(ValueError):
    """An unsupported or invalid Step 2 configuration."""


def configuration_fingerprint(config: dict[str, Any]) -> str:
    """Hash the resolved configuration independently of YAML key ordering."""
    serialized = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise PopulationConfigurationError(f"{name} must be a finite number")
    return float(value)


def validate_config(config: dict[str, Any]) -> None:
    """Reject unsupported distributions/dependencies rather than guessing."""
    try:
        if config["schema_version"] != SCHEMA_VERSION:
            raise PopulationConfigurationError("unsupported population schema_version")
        simulation = config["simulation"]
        for name, minimum in (("n_samples", 1), ("random_seed", 0), ("audit_preview_rows", 1)):
            value = simulation[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise PopulationConfigurationError(f"simulation.{name} must be an integer >= {minimum}")
        if simulation["audit_preview_rows"] > 200:
            raise PopulationConfigurationError("audit_preview_rows must not exceed 200")
        population = config["population"]
        if population["independent_primitives"] is not True:
            raise PopulationConfigurationError("Step 2 requires independent primitive draws")
        for variable in ("log_Y", "D", "epsilon", "omega"):
            spec = population[variable]
            if spec["distribution"] != "normal":
                raise PopulationConfigurationError(f"{variable} must use a normal distribution")
            _finite_number(spec["mean"], f"{variable}.mean")
            if _finite_number(spec["sd"], f"{variable}.sd") <= 0:
                raise PopulationConfigurationError(f"{variable}.sd must be positive")
        group = population["B"]
        if group["distribution"] != "bernoulli":
            raise PopulationConfigurationError("B must use a Bernoulli distribution")
        if not 0 <= _finite_number(group["probability"], "B.probability") <= 1:
            raise PopulationConfigurationError("B.probability must be in [0,1]")
        _finite_number(config["credit_signal"]["epsilon_coefficient"], "epsilon_coefficient")
        # Restrict writes to the new family's namespaces, preserving older data.
        for key, prefix in (
            ("data_directory", "data/synthetic/professor_model"),
            ("validation_report", "results/metrics/professor_model"),
            ("group_comparison", "results/tables/professor_model"),
        ):
            path = Path(config["outputs"][key])
            if path.is_absolute() or ".." in path.parts or not path.is_relative_to(prefix):
                raise PopulationConfigurationError(f"outputs.{key} must be within {prefix}")
    except (KeyError, TypeError) as exc:
        raise PopulationConfigurationError(f"missing or malformed population configuration: {exc}") from exc


def load_population_config(path: Path | str = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load the separate professor-model configuration with explicit SD units."""
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise PopulationConfigurationError("configuration must be a YAML mapping")
    validate_config(config)
    return config


def resolve_config(
    config: dict[str, Any], *, n_samples: int | None = None, seed: int | None = None
) -> dict[str, Any]:
    """Copy and validate caller overrides without mutating the canonical config."""
    resolved = copy.deepcopy(config)
    if n_samples is not None:
        resolved["simulation"]["n_samples"] = n_samples
    if seed is not None:
        resolved["simulation"]["random_seed"] = seed
    validate_config(resolved)
    return resolved
