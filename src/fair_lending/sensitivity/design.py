"""Declared grids, common-random-number seeds, and run identifiers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from fair_lending.simulation.config import resolve_simulation_config, stable_fingerprint
from fair_lending.simulation.treatments import apply_treatment_overrides
from fair_lending.simulation.generator import SYNTHETIC_SCHEMA_VERSION


ANALYSIS_VERSION = 4
MODEL_POLICY = "unpenalized_logit_drop_separated_nuisance_dummies_newton_bfgs_fallback_and_validation_tuned_ml_c_0.1_1_10"
DIRECT_GRID = (0.0, -0.05, -0.10, -0.15, -0.25, -0.35, -0.50)
UPSTREAM_GRID = (0.0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50)
MIXED_DIRECT_GRID = (0.0, -0.10, -0.25, -0.50)
MIXED_UPSTREAM_GRID = (0.0, 0.50, 1.00, 1.50)
SAMPLE_SIZE_GRID = (1_000, 2_500, 5_000, 10_000, 25_000, 50_000, 100_000)
DETECTION_SIZE_GRID = (5_000, 10_000, 25_000, 50_000, 100_000)


@dataclass(frozen=True)
class RunSpec:
    """One unique scientific condition and Monte Carlo replication."""

    n_rows: int
    replication_index: int
    master_seed: int
    child_seed: int
    spawn_key: tuple[int, ...]
    direct_black_log_odds: float
    upstream_strength: float
    families: tuple[str, ...]

    @property
    def scenario_family(self) -> str:
        direct = self.direct_black_log_odds != 0.0
        upstream = self.upstream_strength != 0.0
        if direct and upstream:
            return "mixed_mechanism"
        if direct:
            return "direct_discrimination"
        if upstream:
            return "upstream_inequality"
        return "fair_baseline"


def replication_seed(master_seed: int, replication_index: int) -> tuple[int, tuple[int, ...]]:
    """Return a stable SeedSequence child shared across mechanism settings."""
    if master_seed < 0 or replication_index < 0:
        raise ValueError("Master seed and replication index must be non-negative")
    child = np.random.SeedSequence(master_seed).spawn(replication_index + 1)[
        replication_index
    ]
    return int(child.generate_state(1, dtype=np.uint32)[0]), tuple(child.spawn_key)


def upstream_treatment(strength: float) -> dict[str, float]:
    """Scale the moderate upstream treatment linearly on DGP parameter scales."""
    if not np.isfinite(strength) or strength < 0.0:
        raise ValueError("Upstream strength must be finite and non-negative")
    return {
        "black_income_log_shift": float(strength * np.log(0.90)),
        "black_credit_score_shift": float(strength * -25.0),
        "black_liquid_assets_log_shift": float(strength * np.log(0.75)),
        "black_income_multiplier": float(0.90**strength),
        "black_liquid_assets_multiplier": float(0.75**strength),
    }


def resolve_sensitivity_config(spec: RunSpec) -> dict[str, Any]:
    """Create a temporary resolved config without changing canonical YAML."""
    config = resolve_simulation_config(
        "fair_baseline", "moderate", spec.n_rows, spec.child_seed
    )
    upstream = upstream_treatment(spec.upstream_strength)
    config = apply_treatment_overrides(
        config,
        direct_black_log_odds=spec.direct_black_log_odds,
        black_income_log_shift=upstream["black_income_log_shift"],
        black_credit_score_shift=upstream["black_credit_score_shift"],
        black_liquid_assets_log_shift=upstream[
            "black_liquid_assets_log_shift"
        ],
        source_label="sensitivity_custom",
    )
    config["simulation"]["scenario"] = spec.scenario_family
    config["simulation"]["sensitivity_analysis_version"] = ANALYSIS_VERSION
    config["simulation"]["direct_black_log_odds"] = spec.direct_black_log_odds
    config["simulation"]["upstream_strength"] = spec.upstream_strength
    return config


def run_identity_payload(spec: RunSpec, config_fingerprint: str) -> dict[str, Any]:
    """Return only stable scientific fields used to identify a completed run."""
    return {
        "analysis_version": ANALYSIS_VERSION,
        "schema_version": SYNTHETIC_SCHEMA_VERSION,
        "model_policy": MODEL_POLICY,
        "n_rows": spec.n_rows,
        "replication_index": spec.replication_index,
        "master_seed": spec.master_seed,
        "child_seed": spec.child_seed,
        "spawn_key": list(spec.spawn_key),
        "direct_black_log_odds": spec.direct_black_log_odds,
        "upstream_strength": spec.upstream_strength,
        "config_fingerprint": config_fingerprint,
        "feature_regimes": ["race_blind", "race_aware_sensitivity"],
        "regression_models": ["model_0", "model_1", "model_2", "model_3"],
    }


def run_id(spec: RunSpec, config_fingerprint: str) -> str:
    return stable_fingerprint(run_identity_payload(spec, config_fingerprint))


def _spec(
    n_rows: int,
    replication: int,
    master_seed: int,
    direct: float,
    upstream: float,
    family: str,
) -> RunSpec:
    child_seed, spawn_key = replication_seed(master_seed, replication)
    return RunSpec(
        n_rows=n_rows,
        replication_index=replication,
        master_seed=master_seed,
        child_seed=child_seed,
        spawn_key=spawn_key,
        direct_black_log_odds=float(direct),
        upstream_strength=float(upstream),
        families=(family,),
    )


def _deduplicate(specs: list[RunSpec]) -> list[RunSpec]:
    combined: dict[tuple[Any, ...], RunSpec] = {}
    for spec in specs:
        key = (
            spec.n_rows,
            spec.replication_index,
            spec.master_seed,
            spec.child_seed,
            spec.direct_black_log_odds,
            spec.upstream_strength,
        )
        if key in combined:
            families = tuple(sorted(set(combined[key].families) | set(spec.families)))
            combined[key] = replace(combined[key], families=families)
        else:
            combined[key] = spec
    return sorted(
        combined.values(),
        key=lambda item: (
            item.n_rows,
            item.direct_black_log_odds,
            item.upstream_strength,
            item.replication_index,
        ),
    )


def build_experiment_design(
    experiment: str = "all",
    *,
    replications: int | None = None,
    rows: int | None = None,
    master_seed: int = 4_994,
    quick: bool = False,
) -> list[RunSpec]:
    """Build the declared family design and deduplicate overlapping settings."""
    allowed = {"direct", "upstream", "mixed", "sample_size", "all"}
    if experiment not in allowed:
        raise ValueError(f"Unknown experiment {experiment!r}; expected {sorted(allowed)}")
    if replications is not None and replications <= 0:
        raise ValueError("replications must be positive")
    if rows is not None and rows < 1_000:
        raise ValueError("rows must be at least 1,000")

    requested = (
        {"direct", "upstream", "mixed", "sample_size"}
        if experiment == "all"
        else {experiment}
    )
    if quick:
        direct_grid = (0.0, -0.25)
        upstream_grid = (0.0, 1.0)
        mixed_direct = direct_grid
        mixed_upstream = upstream_grid
        size_grid = (1_000, 2_500)
        detection_sizes = size_grid
        default_rows = 2_500
        standard_reps = mixed_reps = 2
    else:
        direct_grid = DIRECT_GRID
        upstream_grid = UPSTREAM_GRID
        mixed_direct = MIXED_DIRECT_GRID
        mixed_upstream = MIXED_UPSTREAM_GRID
        size_grid = SAMPLE_SIZE_GRID
        detection_sizes = DETECTION_SIZE_GRID
        default_rows = 100_000
        standard_reps = 50
        mixed_reps = 30

    fixed_rows = rows if rows is not None else default_rows
    specs: list[RunSpec] = []
    if "direct" in requested:
        count = replications if replications is not None else standard_reps
        for replication in range(count):
            for direct in direct_grid:
                specs.append(
                    _spec(fixed_rows, replication, master_seed, direct, 0.0, "direct")
                )
    if "upstream" in requested:
        count = replications if replications is not None else standard_reps
        for replication in range(count):
            for upstream in upstream_grid:
                specs.append(
                    _spec(fixed_rows, replication, master_seed, 0.0, upstream, "upstream")
                )
    if "mixed" in requested:
        count = replications if replications is not None else mixed_reps
        for replication in range(count):
            for direct in mixed_direct:
                for upstream in mixed_upstream:
                    specs.append(
                        _spec(fixed_rows, replication, master_seed, direct, upstream, "mixed")
                    )
    if "sample_size" in requested:
        count = replications if replications is not None else standard_reps
        scenarios = (
            (0.0, 0.0),
            (-0.25, 0.0),
            (0.0, 1.0),
            (-0.25, 1.0),
        )
        for replication in range(count):
            for n_rows in size_grid:
                for direct, upstream in scenarios:
                    specs.append(
                        _spec(
                            n_rows,
                            replication,
                            master_seed,
                            direct,
                            upstream,
                            "sample_size",
                        )
                    )
            for n_rows in detection_sizes:
                for direct in direct_grid:
                    specs.append(
                        _spec(
                            n_rows,
                            replication,
                            master_seed,
                            direct,
                            0.0,
                            "detection",
                        )
                    )
    return _deduplicate(specs)
