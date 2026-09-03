"""Load, resolve, and validate simulation configuration files."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import subprocess
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIRECTORY = PROJECT_ROOT / "configs" / "simulation"
ALLOWED_SCENARIOS = (
    "fair_baseline",
    "direct_discrimination",
    "upstream_inequality",
    "mixed_mechanism",
)
EFFECT_LEVELS = ("mild", "moderate", "strong")
EXPECTED_SCENARIO_SWITCHES = {
    "fair_baseline": (False, False),
    "direct_discrimination": (False, True),
    "upstream_inequality": (True, False),
    "mixed_mechanism": (True, True),
}


class ConfigurationError(ValueError):
    """Raised when a simulation configuration is incomplete or inconsistent."""


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ConfigurationError(f"Expected a YAML mapping in {path}")
    return value


def stable_fingerprint(value: Any) -> str:
    """Return a SHA-256 hash from stable JSON serialization."""
    serialized = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def approval_transform_parameters(
    config: dict[str, Any], field: str
) -> dict[str, float | str]:
    """Parse the canonical documented transform into numeric parameters.

    The YAML keeps human-readable equations. Parsing their numeric values here
    ensures that a calibrated center, scale, or log offset is not duplicated as
    a hidden source-code constant.
    """
    expression = config["approval_model"]["continuous_terms"][field][
        "transform"
    ].replace(" ", "")
    number = r"([-+]?\d+(?:\.\d+)?)"
    patterns = (
        (
            rf"\(ln\(value\)-ln\({number}\)\)/{number}",
            ("center", "scale"),
            "log_centered",
        ),
        (
            rf"\(ln\(value\+{number}\)-ln\({number}\)\)/{number}",
            ("offset", "center", "scale"),
            "log_offset_centered",
        ),
        (
            rf"\(value-{number}\)/{number}",
            ("center", "scale"),
            "linear_centered",
        ),
    )
    for pattern, names, kind in patterns:
        match = re.fullmatch(pattern, expression)
        if match:
            parsed: dict[str, float | str] = {"kind": kind}
            parsed.update(
                {name: float(value) for name, value in zip(names, match.groups(), strict=True)}
            )
            if float(parsed["scale"]) <= 0.0:
                raise ConfigurationError(f"Approval transform scale must be positive for {field}")
            return parsed
    raise ConfigurationError(
        f"Unsupported approval transform for {field}: {expression!r}"
    )


def _zero_map(categories: list[str]) -> dict[str, float]:
    return {category: 0.0 for category in categories}


def resolve_simulation_config(
    scenario: str = "fair_baseline",
    effect_level: str = "moderate",
    n_rows: int = 100_000,
    seed: int = 4_994,
) -> dict[str, Any]:
    """Resolve baseline, treatment-library, scenario, and run overrides.

    YAML mappings are deep-copied, never modified on disk or in a shared cache.
    Disabled mechanisms are materialized as exact identity/zero treatments so
    generator code does not need to infer scenario behavior in multiple places.
    """
    if scenario not in ALLOWED_SCENARIOS:
        raise ConfigurationError(
            f"Unknown scenario {scenario!r}; expected one of {ALLOWED_SCENARIOS}"
        )
    if effect_level not in EFFECT_LEVELS:
        raise ConfigurationError(
            f"Unknown effect level {effect_level!r}; expected one of {EFFECT_LEVELS}"
        )
    if not isinstance(n_rows, int) or isinstance(n_rows, bool) or n_rows <= 0:
        raise ConfigurationError("n_rows must be a positive integer")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ConfigurationError("seed must be a non-negative integer")

    baseline = copy.deepcopy(_load_yaml(CONFIG_DIRECTORY / "baseline.yaml"))
    effects = _load_yaml(CONFIG_DIRECTORY / "effect_sizes.yaml")
    scenario_spec = _load_yaml(
        CONFIG_DIRECTORY / "scenarios" / f"{scenario}.yaml"
    )["scenario"]

    expected_upstream, expected_direct = EXPECTED_SCENARIO_SWITCHES[scenario]
    actual_switches = (
        scenario_spec["upstream"]["enabled"],
        scenario_spec["direct"]["enabled"],
    )
    if actual_switches != (expected_upstream, expected_direct):
        raise ConfigurationError(
            f"Scenario switch drift for {scenario}: got {actual_switches}, "
            f"expected {(expected_upstream, expected_direct)}"
        )

    race_categories = list(
        baseline["population"]["demographics"]["race"]["shares"]
    )
    affected_variables = list(effects["upstream_effects"]["affected_variables"])

    if expected_upstream:
        upstream_treatments = copy.deepcopy(
            effects["upstream_effects"]["levels"][effect_level]
        )
        upstream_level: str | None = effect_level
    else:
        upstream_treatments = {}
        template = effects["upstream_effects"]["levels"][effect_level]
        for variable in affected_variables:
            upstream_treatments[variable] = {
                "shifted_parameter": template[variable]["shifted_parameter"],
                "additive_shift_by_race": _zero_map(race_categories),
            }
        upstream_level = None

    if expected_direct:
        direct_coefficients = copy.deepcopy(
            effects["direct_effects"]["levels"][effect_level][
                "log_odds_by_race"
            ]
        )
        direct_level: str | None = effect_level
    else:
        direct_coefficients = _zero_map(race_categories)
        direct_level = None

    baseline["simulation"].update(
        {
            "scenario": scenario,
            "effect_level": effect_level,
            "n_samples": n_rows,
            "random_seed": seed,
        }
    )
    baseline["scenario"] = copy.deepcopy(scenario_spec)
    baseline["scenario_effects"] = {
        "focal_experiment": copy.deepcopy(effects["focal_experiment"]),
        "upstream": {
            "enabled": expected_upstream,
            "effect_level": upstream_level,
            "affected_variables": affected_variables,
            "treatments": upstream_treatments,
        },
        "direct": {
            "enabled": expected_direct,
            "effect_level": direct_level,
            "scale": effects["direct_effects"]["scale"],
            "log_odds_by_race": direct_coefficients,
        },
    }
    validate_resolved_config(baseline)
    return baseline


def validate_resolved_config(config: dict[str, Any]) -> None:
    """Fail early when a resolved config violates core DGP invariants."""
    simulation = config["simulation"]
    scenario = simulation["scenario"]
    if scenario not in ALLOWED_SCENARIOS:
        raise ConfigurationError(f"Invalid resolved scenario: {scenario!r}")
    if simulation["n_samples"] <= 0 or simulation["random_seed"] < 0:
        raise ConfigurationError("Invalid sample size or seed")

    share_maps = [
        value["shares"]
        for value in config["population"]["demographics"].values()
    ]
    share_maps.extend(
        config["loan_property_variables"][name]["shares"]
        for name in ("loan_purpose", "loan_type", "occupancy_type")
    )
    for shares in share_maps:
        if any(probability < 0.0 for probability in shares.values()):
            raise ConfigurationError("Category probabilities must be non-negative")
        if abs(sum(shares.values()) - 1.0) > 1e-12:
            raise ConfigurationError("Category probabilities must sum to one")

    upstream, direct = EXPECTED_SCENARIO_SWITCHES[scenario]
    resolved_effects = config["scenario_effects"]
    if resolved_effects["upstream"]["enabled"] is not upstream:
        raise ConfigurationError("Resolved upstream switch does not match scenario")
    if resolved_effects["direct"]["enabled"] is not direct:
        raise ConfigurationError("Resolved direct switch does not match scenario")

    races = list(config["population"]["demographics"]["race"]["shares"])
    direct_map = resolved_effects["direct"]["log_odds_by_race"]
    if list(direct_map) != races or direct_map["White"] != 0.0:
        raise ConfigurationError("Direct-effect race map or reference is invalid")
    if not direct and any(value != 0.0 for value in direct_map.values()):
        raise ConfigurationError("Disabled direct mechanism must be exactly zero")

    allowed_upstream = {"annual_income", "credit_score", "liquid_assets"}
    affected = set(resolved_effects["upstream"]["affected_variables"])
    if affected != allowed_upstream:
        raise ConfigurationError("Version 1 upstream variables have drifted")
    for treatment in resolved_effects["upstream"]["treatments"].values():
        shifts = treatment["additive_shift_by_race"]
        if list(shifts) != races or shifts["White"] != 0.0:
            raise ConfigurationError("Upstream race map or reference is invalid")
        if not upstream and any(value != 0.0 for value in shifts.values()):
            raise ConfigurationError("Disabled upstream mechanism must be identity")

    if config["context_variables"]["neighborhood_minority_share"][
        "baseline_race_dependency"
    ]:
        raise ConfigurationError("Version 1 context must remain race-independent")
    if config["outcomes"]["denial_reason"]["enabled"]:
        raise ConfigurationError("Version 1 denial reasons must remain disabled")
    for field in config["approval_model"]["continuous_terms"]:
        approval_transform_parameters(config, field)


def calibration_resolved_config() -> dict[str, Any]:
    """Return the canonical one-million-row fair calibration configuration."""
    baseline = _load_yaml(CONFIG_DIRECTORY / "baseline.yaml")
    calibration = baseline["approval_model"]["intercept"]
    return resolve_simulation_config(
        scenario="fair_baseline",
        effect_level="moderate",
        n_rows=int(calibration["calibration_population_size"]),
        seed=int(calibration["calibration_seed"]),
    )


def package_version() -> str:
    """Return installed package version, with the source-tree version fallback."""
    try:
        return importlib_metadata.version("fair-lending-capstone")
    except importlib_metadata.PackageNotFoundError:
        return "0.1.0"


def git_revision() -> tuple[str | None, bool | None]:
    """Return the current Git revision and whether the worktree is dirty."""
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return revision, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None
