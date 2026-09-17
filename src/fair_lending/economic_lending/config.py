"""Configuration loading for the Version 2 economic population baseline."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "economic_lending" / "baseline.yaml"


class EconomicConfigurationError(ValueError):
    """Raised when the Version 2 population configuration is invalid."""


def config_fingerprint(config: dict[str, Any]) -> str:
    """Return a stable SHA-256 fingerprint for a resolved configuration."""

    serialized = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def load_economic_config(path: Path | str = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load and validate the standalone Version 2 population YAML."""

    config_path = Path(path)
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise EconomicConfigurationError("economic lending config must be a YAML mapping")
    config = copy.deepcopy(loaded)
    _validate_config(config)
    config["metadata"] = {
        "config_path": str(config_path),
        "config_fingerprint": config_fingerprint(loaded),
    }
    return config


def _validate_shares(shares: dict[str, float], expected: set[str], name: str) -> None:
    if set(shares) != expected:
        raise EconomicConfigurationError(f"{name} categories must be {sorted(expected)}")
    if any(float(value) < 0 for value in shares.values()):
        raise EconomicConfigurationError(f"{name} shares must be nonnegative")
    if abs(sum(float(value) for value in shares.values()) - 1.0) > 1e-12:
        raise EconomicConfigurationError(f"{name} shares must sum to one")


def _validate_config(config: dict[str, Any]) -> None:
    required = {"population", "contract", "true_risk", "cohorts", "randomness", "artifacts"}
    missing = required - set(config)
    if missing:
        raise EconomicConfigurationError(f"missing configuration sections: {sorted(missing)}")
    _validate_shares(config["population"]["group_shares"], {"A", "B"}, "group")
    _validate_shares(
        config["cohorts"]["shares"],
        {"historical_train", "historical_validation", "evaluation"},
        "cohort",
    )
    contract = config["contract"]
    if int(contract["term_periods"]) <= 0:
        raise EconomicConfigurationError("contract term_periods must be positive")
    if float(contract["transaction_cost"]) < 0:
        raise EconomicConfigurationError("transaction_cost must be nonnegative")
    if config["true_risk"]["link"] != "logistic":
        raise EconomicConfigurationError("the baseline true-risk link must be logistic")


def create_random_streams(
    seed: int,
    stream_names: list[str] | tuple[str, ...],
) -> tuple[dict[str, Any], dict[str, list[int]]]:
    """Create stable named NumPy child streams from one master seed."""

    import numpy as np

    if seed < 0:
        raise EconomicConfigurationError("seed must be nonnegative")
    if len(stream_names) != len(set(stream_names)):
        raise EconomicConfigurationError("random stream names must be unique")
    children = np.random.SeedSequence(seed).spawn(len(stream_names))
    streams = {
        name: np.random.default_rng(child)
        for name, child in zip(stream_names, children, strict=True)
    }
    spawn_keys = {
        name: list(child.spawn_key)
        for name, child in zip(stream_names, children, strict=True)
    }
    return streams, spawn_keys
