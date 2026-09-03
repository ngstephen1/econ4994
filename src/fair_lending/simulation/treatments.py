"""Shared in-memory treatment overrides for controlled synthetic experiments."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np


def apply_treatment_overrides(
    config: dict[str, Any],
    *,
    direct_black_log_odds: float,
    black_income_log_shift: float,
    black_credit_score_shift: float,
    black_liquid_assets_log_shift: float,
    source_label: str,
) -> dict[str, Any]:
    """Apply focal Black treatments to a deep copy of a resolved configuration.

    This function deliberately does not call the canonical scenario validator:
    custom experiments may combine mechanisms outside the four named YAML
    scenarios. All non-Black race effects remain unchanged, including the White
    reference value of zero.
    """
    values = np.asarray(
        [
            direct_black_log_odds,
            black_income_log_shift,
            black_credit_score_shift,
            black_liquid_assets_log_shift,
        ],
        dtype=float,
    )
    if not np.isfinite(values).all():
        raise ValueError("Synthetic treatment overrides must be finite")

    resolved = copy.deepcopy(config)
    direct = resolved["scenario_effects"]["direct"]
    direct["log_odds_by_race"]["Black"] = float(direct_black_log_odds)
    direct["enabled"] = direct_black_log_odds != 0.0
    direct["effect_level"] = source_label if direct["enabled"] else None

    treatments = resolved["scenario_effects"]["upstream"]["treatments"]
    shifts = {
        "annual_income": float(black_income_log_shift),
        "credit_score": float(black_credit_score_shift),
        "liquid_assets": float(black_liquid_assets_log_shift),
    }
    for field, shift in shifts.items():
        treatments[field]["additive_shift_by_race"]["Black"] = shift
    treatments["annual_income"]["conditional_black_multiplier"] = float(
        np.exp(black_income_log_shift)
    )
    treatments["liquid_assets"]["conditional_black_multiplier"] = float(
        np.exp(black_liquid_assets_log_shift)
    )
    upstream_active = any(shift != 0.0 for shift in shifts.values())
    upstream = resolved["scenario_effects"]["upstream"]
    upstream["enabled"] = upstream_active
    upstream["effect_level"] = source_label if upstream_active else None
    resolved["simulation"]["treatment_override_source"] = source_label
    return resolved


def configured_treatments(config: dict[str, Any]) -> dict[str, Any]:
    """Extract the focal Black treatments in presentation-friendly units."""
    upstream = config["scenario_effects"]["upstream"]["treatments"]
    direct = config["scenario_effects"]["direct"]
    return {
        "direct_enabled": bool(direct["enabled"]),
        "direct_black_log_odds": float(direct["log_odds_by_race"]["Black"]),
        "upstream_enabled": bool(config["scenario_effects"]["upstream"]["enabled"]),
        "black_income_log_shift": float(
            upstream["annual_income"]["additive_shift_by_race"]["Black"]
        ),
        "black_income_multiplier": float(
            np.exp(upstream["annual_income"]["additive_shift_by_race"]["Black"])
        ),
        "black_credit_score_shift": float(
            upstream["credit_score"]["additive_shift_by_race"]["Black"]
        ),
        "black_liquid_assets_log_shift": float(
            upstream["liquid_assets"]["additive_shift_by_race"]["Black"]
        ),
        "black_liquid_assets_multiplier": float(
            np.exp(upstream["liquid_assets"]["additive_shift_by_race"]["Black"])
        ),
    }
