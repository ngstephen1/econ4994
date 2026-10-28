"""Step 9A: frozen-score adjustments and risk-only audits, never lending.

The diagram's direct adjustment lowers default risk for B=1 by 0.1 in raw
probability units. No link function, refitting, or truth/PN lookup is involved.
"""
from __future__ import annotations

from fractions import Fraction
import math

import numpy as np
import pandas as pd

from .bank_regression import ESTIMATE_COLUMNS, MODEL_ID as AWARE_ID, validate_ids
from .group_blind import MODEL_ID as BLIND_ID, PREDICTION_COLUMNS as BLIND_COLUMNS

SCHEMA_VERSION = "professor-revised-scenarios-v1"
DIRECT_ADJUSTMENT = -0.1
EXACT_BUDGET = 1826087220.7466326
OBSERVED_COLUMNS = ("applicant_id", "Y", "D", "B", "C")
SCENARIOS = (
    {"scenario_id": "professor_no_direct_group_aware", "base_model_id": AWARE_ID,
     "manual_adjustment": False, "role": "professor_branch"},
    {"scenario_id": "professor_direct_minus_0p1_B", "base_model_id": BLIND_ID,
     "manual_adjustment": True, "role": "professor_branch"},
    {"scenario_id": "group_blind_no_adjustment_reference", "base_model_id": BLIND_ID,
     "manual_adjustment": False, "role": "analytical_control"},
)
RISK_COLUMNS = (
    "applicant_id", "scenario_id", "base_model_id", "B", "estimated_PD_base_raw",
    "group_adjustment", "estimated_PD_adjusted_raw", "estimated_PD_bounded",
    "clipped_below_zero", "clipped_above_one",
)


def validate_config(config: dict) -> None:
    """Freeze the scientific branches/sign; cohort counts can be small in tests."""
    expected = {
        "schema_version": SCHEMA_VERSION, "population_rows": config.get("population_rows"),
        "training_rows": config.get("training_rows"), "evaluation_rows": config.get("evaluation_rows"),
        "preview_applicants_per_group": config.get("preview_applicants_per_group"),
        "direct_adjustment_per_B": DIRECT_ADJUSTMENT,
        "adjustment_scale": "additive_raw_probability_before_clipping",
        "bounding": "clip_0_1_separate_column",
        "budget_source": "results/metrics/professor_model/lending_selection.json",
        "exact_saved_budget_USD": EXACT_BUDGET, "scenarios": list(SCENARIOS),
    }
    if config != expected:
        raise ValueError("Step 9A requires the exact branches, -0.1 raw adjustment and saved budget")
    for key in ("population_rows", "training_rows", "evaluation_rows"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError("positive integer cohort counts required")
    if config["population_rows"] != config["training_rows"] + config["evaluation_rows"]:
        raise ValueError("cohorts must cover the original population")
    if type(config["preview_applicants_per_group"]) is not int or not 5 <= config["preview_applicants_per_group"] <= 10:
        raise ValueError("preview must request 5–10 unique applicants per B group")


def _observed(applicants: pd.DataFrame) -> pd.DataFrame:
    validate_ids(applicants, "evaluation bank view")
    if tuple(applicants.columns) != OBSERVED_COLUMNS or applicants.empty:
        raise ValueError("exact observable-only schema and nonempty evaluation cohort required")
    values = applicants.loc[:, OBSERVED_COLUMNS[1:]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or not applicants.Y.gt(0).all() or not applicants.B.isin([0, 1]).all():
        raise ValueError("finite observables, positive income and binary B required")
    # CSV has no integer-width metadata. Preserve B values, use a portable
    # int64 audit column in new outputs; the original int8 source is untouched.
    return applicants.sort_values("applicant_id").reset_index(drop=True).astype({"B": "int64"})


def _reference(frame: pd.DataFrame, applicants: pd.DataFrame, *, blind: bool) -> np.ndarray:
    validate_ids(frame, "frozen evaluation predictions")
    columns = BLIND_COLUMNS if blind else ESTIMATE_COLUMNS
    model_id = BLIND_ID if blind else AWARE_ID
    raw_column = "bank_PD_group_blind_raw" if blind else "bank_PD_raw"
    bounded_column = "bank_PD_group_blind_bounded" if blind else "bank_PD_bounded"
    if tuple(frame.columns) != columns or not frame.model_id.eq(model_id).all() or not frame.split.eq("evaluation").all():
        raise ValueError("exact frozen evaluation prediction contract required")
    frame = frame.sort_values("applicant_id").reset_index(drop=True)
    if frame.applicant_id.tolist() != applicants.applicant_id.tolist():
        raise ValueError("frozen predictions must cover exactly the evaluation IDs")
    raw = frame[raw_column].to_numpy(dtype=float)
    if not np.isfinite(raw).all() or not np.array_equal(frame[bounded_column], np.clip(raw, 0, 1)):
        raise ValueError("invalid frozen raw/bounded predictions")
    if not pd.api.types.is_bool_dtype(frame.prediction_out_of_bounds) or not np.array_equal(frame.prediction_out_of_bounds, (raw < 0) | (raw > 1)):
        raise ValueError("invalid frozen prediction flags")
    return raw


def build_scenario_risk_estimates(applicants, aware, blind) -> pd.DataFrame:
    """Use saved evaluation raw predictions as authoritative, preserving bits."""
    observed = _observed(applicants)
    references = {
        AWARE_ID: _reference(aware, observed, blind=False),
        BLIND_ID: _reference(blind, observed, blind=True),
    }
    rows = []
    for scenario in SCENARIOS:
        base = references[scenario["base_model_id"]]
        adjustment = (np.where(observed.B.to_numpy() == 1, DIRECT_ADJUSTMENT, 0.)
                      if scenario["manual_adjustment"] else np.zeros(len(observed)))
        raw = base + adjustment
        rows.append(pd.DataFrame({
            "applicant_id": observed.applicant_id, "scenario_id": scenario["scenario_id"],
            "base_model_id": scenario["base_model_id"], "B": observed.B,
            "estimated_PD_base_raw": base, "group_adjustment": adjustment,
            "estimated_PD_adjusted_raw": raw, "estimated_PD_bounded": np.clip(raw, 0, 1),
            "clipped_below_zero": raw < 0, "clipped_above_one": raw > 1,
        }))
    return pd.concat(rows, ignore_index=True).loc[:, RISK_COLUMNS].sort_values(
        ["scenario_id", "applicant_id"]).reset_index(drop=True)


def validate_scenario_estimates(estimates, applicants, aware, blind) -> dict:
    """Require exact cell coverage, original bases, adjustment and clipping."""
    if tuple(estimates.columns) != RISK_COLUMNS or estimates.duplicated(["scenario_id", "applicant_id"]).any():
        raise ValueError("invalid or duplicated scenario rows")
    expected = build_scenario_risk_estimates(applicants, aware, blind)
    pd.testing.assert_frame_equal(estimates.sort_values(["scenario_id", "applicant_id"]).reset_index(drop=True), expected, check_exact=True)
    return {
        "valid": True, "scenarios": len(SCENARIOS), "rows": len(estimates),
        "applicants_per_scenario": len(applicants), "exact_frozen_reference_reproduction": True,
        "direct_B0_equals_blind_reference": True, "direct_B1_equals_blind_minus_0p1": True,
        "raw_base_plus_adjustment_identity": True, "exact_clip_identity": True,
        "complete_unique_scenario_applicant_cells": True,
    }


def _summary(frame: pd.DataFrame) -> dict:
    """Data stay in raw units; risk statistics explicitly display pp (×100)."""
    raw, bounded = frame.estimated_PD_adjusted_raw, frame.estimated_PD_bounded
    return {
        "n_applicants": len(frame),
        "mean_base_raw_PD_pp": float(100 * frame.estimated_PD_base_raw.mean()),
        "mean_raw_PD_pp": float(100 * raw.mean()), "median_raw_PD_pp": float(100 * raw.median()),
        "minimum_raw_PD_pp": float(100 * raw.min()), "maximum_raw_PD_pp": float(100 * raw.max()),
        "mean_bounded_PD_pp": float(100 * bounded.mean()), "median_bounded_PD_pp": float(100 * bounded.median()),
        "negative_raw_count": int((raw < 0).sum()), "negative_raw_share": float((raw < 0).mean()),
        "above_one_raw_count": int((raw > 1).sum()),
        "clipped_to_zero_count": int(frame.clipped_below_zero.sum()),
        "clipped_to_one_count": int(frame.clipped_above_one.sum()),
        "bounded_zero_count": int((bounded == 0).sum()), "bounded_zero_share": float((bounded == 0).mean()),
        "bounded_one_count": int((bounded == 1).sum()), "bounded_one_share": float((bounded == 1).mean()),
        "adjusted_applicants": int(frame.group_adjustment.ne(0).sum()),
        "mean_adjustment_pp": float(100 * frame.group_adjustment.mean()),
    }


def summarize_risks(estimates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall, groups = [], []
    for scenario, frame in estimates.groupby("scenario_id", sort=True):
        overall.append({"scenario_id": scenario, **_summary(frame)})
        for group, subset in frame.groupby("B", sort=True):
            groups.append({"scenario_id": scenario, "B": int(group), **_summary(subset)})
    return pd.DataFrame(overall), pd.DataFrame(groups)


def zero_tie_diagnostics(estimates, contracts, budget: float) -> pd.DataFrame:
    """Capital diagnostic only. No cutoff, profit, or lending selection occurs.

    Feasibility compares exact rational values of saved binary floats, avoiding
    a rounded sum masking a tiny budget violation. Displayed totals use fsum.
    """
    validate_ids(contracts, "diagnostic principal")
    if tuple(contracts.columns) != ("applicant_id", "L_i") or not np.isfinite(contracts.L_i).all() or not contracts.L_i.gt(0).all():
        raise ValueError("exact ID/principal view with finite positive requests required")
    if isinstance(budget, bool) or not np.isfinite(budget) or budget <= 0:
        raise ValueError("positive finite diagnostic budget required")
    rows = []
    for scenario, frame in estimates.groupby("scenario_id", sort=True):
        validate_ids(frame, "scenario principal join")
        if set(frame.applicant_id) != set(contracts.applicant_id):
            raise ValueError("principal and scenario evaluation IDs differ")
        joined = frame.merge(contracts, on="applicant_id", validate="one_to_one")
        zero = joined.loc[joined.estimated_PD_bounded == 0]
        principal = math.fsum(zero.L_i)
        feasible = sum((Fraction.from_float(float(v)) for v in zero.L_i), Fraction()) <= Fraction.from_float(float(budget))
        rows.append({
            "scenario_id": scenario, "diagnostic_only": True, "n_applicants": len(frame),
            "zero_score_count": len(zero), "zero_score_share": len(zero) / len(frame),
            "B0_zero_score_count": int(zero.B.eq(0).sum()), "B1_zero_score_count": int(zero.B.eq(1).sum()),
            "zero_score_requested_principal_USD": principal, "available_bank_capital_USD": budget,
            "zero_score_principal_budget_ratio": principal / budget,
            "entire_zero_score_group_feasible": bool(feasible),
        })
    return pd.DataFrame(rows)


def evaluator_preview(applicants, estimates, per_group=8) -> pd.DataFrame:
    """First saved IDs in each B group; no selection on risk or hidden data."""
    observed = _observed(applicants)
    selected = observed.groupby("B", sort=True).head(per_group)
    result = selected.merge(estimates, on=["applicant_id", "B"], validate="one_to_many")
    result.insert(0, "information_scope", "EVALUATOR_ONLY_NOT_FOR_LENDER_TRAINING")
    return result.sort_values(["B", "applicant_id", "scenario_id"]).reset_index(drop=True)
