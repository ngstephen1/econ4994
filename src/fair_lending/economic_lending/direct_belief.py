"""Synthetic direct discrimination through post-model repayment-belief distortion."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from fair_lending.economic_lending.config import PROJECT_ROOT, config_fingerprint
from fair_lending.economic_lending.profit import expected_economics_from_probability
from fair_lending.economic_lending.schema import POLICY_ASSESSMENTS_COLUMNS


DIRECT_CONFIG_PATH = PROJECT_ROOT / "configs/economic_lending/direct_belief_distortion.yaml"
DIRECT_MECHANISM_ID = "direct_belief_distortion_v1"


def load_direct_belief_config(path: Path | str = DIRECT_CONFIG_PATH) -> dict[str, Any]:
    loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(loaded, dict) or loaded.get("mechanism_id") != DIRECT_MECHANISM_ID:
        raise ValueError("invalid direct-belief-distortion configuration")
    if tuple(float(x) for x in loaded["deltas"]) != (0.0, 0.05, 0.10, 0.20):
        raise ValueError("direct distortion deltas must match the predeclared grid")
    config = copy.deepcopy(loaded)
    config["metadata"] = {"config_path": str(path), "config_fingerprint": config_fingerprint(loaded)}
    return config


def logit_probability(probability: np.ndarray | pd.Series) -> np.ndarray:
    probability = np.asarray(probability, dtype=float)
    if np.isnan(probability).any() or ((probability <= 0) | (probability >= 1)).any():
        raise ValueError("base probabilities must lie strictly inside (0, 1)")
    return np.log(probability) - np.log1p(-probability)


def distort_repayment_beliefs(
    applicants: pd.DataFrame,
    base_probabilities: pd.DataFrame,
    probability_column: str,
    delta: float,
    *,
    treated_group: str = "B",
) -> pd.DataFrame:
    """Apply an exact treated-group log-odds shift after risk estimation."""

    if delta < 0:
        raise ValueError("delta must be nonnegative")
    source = applicants.loc[:, ["applicant_id", "group"]].merge(
        base_probabilities.loc[:, ["applicant_id", probability_column]],
        on="applicant_id", validate="one_to_one"
    )
    base = source[probability_column].to_numpy(dtype=float)
    shift = float(delta) * source["group"].eq(treated_group).to_numpy(dtype=float)
    used = 1.0 / (1.0 + np.exp(-(logit_probability(base) - shift)))
    return pd.DataFrame({"applicant_id":source.applicant_id.to_numpy(),
                         "repayment_probability_base":base,
                         "repayment_probability_used":used})


def direct_policy_id(family: str, world_id: str, delta: float) -> str:
    world = "additive" if world_id == "additive_logistic_baseline" else "nonlinear"
    family_name = {"traditional":"traditional", "ml":"ml", "true_risk_reference":"true_risk_reference"}[family]
    return f"{family_name}_{world}_direct_d{int(round(delta*100)):03d}"


def build_direct_policy_assessments(
    loan_options: pd.DataFrame,
    distorted_probabilities: pd.DataFrame,
    policy_id: str,
) -> pd.DataFrame:
    """Recompute perceived economics from used—not base—probability."""

    economics = expected_economics_from_probability(
        loan_options, distorted_probabilities, "repayment_probability_used",
        "expected_receipts_perceived", "expected_profit_perceived"
    )
    result = (loan_options.loc[:, ["applicant_id","option_id"]]
              .merge(distorted_probabilities,on="applicant_id",validate="one_to_one")
              .merge(economics,on="applicant_id",validate="one_to_one"))
    result["policy_id"] = policy_id
    result["assessment_status"] = "direct_belief_distortion"
    return result.loc[:, POLICY_ASSESSMENTS_COLUMNS]
