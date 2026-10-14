"""Raw noisy risk measurements and diagnostics, without lender estimation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from fair_lending.professor_model.config import PROJECT_ROOT

SIGNAL_SCHEMA_VERSION = "professor-noisy-signal-v1"
SIGNAL_CONFIG_PATH = PROJECT_ROOT / "configs/professor_model/noisy_signal.yaml"
SIGNAL_COLUMNS = ("applicant_id", "PN")
NOISE_COLUMNS = ("applicant_id", "eta")
SIGNAL_PREVIEW_COLUMNS = (
    "information_scope", "applicant_id", "Y", "D", "B", "C", "true_PD", "eta", "PN"
)
SIGNAL_PREVIEW_SCOPE = "EVALUATOR_ONLY_NOT_FOR_LENDER_TRAINING"
SIGNAL_SCALE = 0.8


def validate_signal_config(config: dict[str, Any]) -> None:
    """Preserve the supplied equation and reserve child [5]; only seed may vary."""
    try:
        if config["schema_version"] != SIGNAL_SCHEMA_VERSION:
            raise ValueError("unsupported noisy-signal schema")
        if config["signal_scale"] != SIGNAL_SCALE or config["noise"] != {"distribution": "normal", "mean": 0.0, "sd": 0.1}:
            raise ValueError("signal scale and Normal mean/SD must match the supplied equation")
        random = config["randomness"]
        if random["spawn_key"] != [5] or random["bit_generator"] != "PCG64" or random["applicant_order"] != "ascending_lexicographic_applicant_id":
            raise ValueError("noisy signal must use PCG64 child [5] and fixed ID order")
        seed = random["master_seed"]
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("master_seed must be a nonnegative integer")
        for key in ("expected_input_rows", "preview_rows"):
            if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] < 1:
                raise ValueError(f"{key} must be a positive integer")
        if config["preview_rows"] > 200:
            raise ValueError("preview_rows must not exceed 200")
        for value, prefix in (
            (config["input_directory"], "data/synthetic/professor_model"),
            (config["outputs"]["validation_report"], "results/metrics/professor_model"),
            (config["outputs"]["summary"], "results/tables/professor_model"),
            (config["outputs"]["group_comparison"], "results/tables/professor_model"),
            (config["outputs"]["figures_directory"], "results/figures/professor_model"),
        ):
            path = Path(value)
            if path.is_absolute() or ".." in path.parts or not path.is_relative_to(prefix):
                raise ValueError(f"path must remain inside {prefix}")
    except (KeyError, TypeError) as exc:
        raise ValueError(f"malformed noisy-signal config: {exc}") from exc


def load_signal_config(path: Path | str = SIGNAL_CONFIG_PATH) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("noisy-signal config must be a YAML mapping")
    validate_signal_config(config)
    return config


def raw_noisy_measurement(true_probability: np.ndarray, eta: np.ndarray) -> np.ndarray:
    """Compute raw 0.8 * probability + eta, including negative measurements."""
    probability = np.asarray(true_probability, dtype=float)
    noise = np.asarray(eta, dtype=float)
    if probability.ndim != 1 or noise.shape != probability.shape:
        raise ValueError("probability and eta require matching one-dimensional arrays")
    if not np.isfinite(probability).all() or not np.isfinite(noise).all():
        raise ValueError("probability and eta must be finite")
    if ((probability < 0) | (probability > 1)).any():
        raise ValueError("true probability must be in [0,1]")
    return SIGNAL_SCALE * probability + noise


def generate_noisy_signal(
    applicants: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Draw one eta per fixed ID from a fresh directly constructed PCG64 child.

    Population RNGs and previously generated truth are never called. B and all
    other applicant covariates are absent from the calculation allowlist.
    """
    validate_signal_config(config)
    if not {"applicant_id", "true_PD"}.issubset(applicants.columns):
        raise ValueError("signal generation requires applicant_id and true_PD")
    source = applicants.loc[:, ["applicant_id", "true_PD"]].copy()
    if len(source) != config["expected_input_rows"] or source.isna().any().any() or not source.applicant_id.is_unique:
        raise ValueError("signal inputs need exact row count, unique IDs, and no missing values")
    if not source.applicant_id.map(lambda value: isinstance(value, str)).all():
        raise ValueError("applicant IDs must be strings for documented lexicographic ordering")
    source = source.sort_values("applicant_id", kind="stable").reset_index(drop=True)
    probability = source.true_PD.to_numpy(dtype=float)
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("saved true_PD must be finite and in [0,1]")
    random = config["randomness"]
    child = np.random.SeedSequence(entropy=random["master_seed"], spawn_key=tuple(random["spawn_key"]))
    rng = np.random.Generator(np.random.PCG64(child))
    eta = rng.normal(loc=config["noise"]["mean"], scale=config["noise"]["sd"], size=len(source))
    ids = source.applicant_id.to_numpy(copy=True)
    targets = pd.DataFrame({"applicant_id": ids, "PN": raw_noisy_measurement(probability, eta)})
    private = pd.DataFrame({"applicant_id": ids, "eta": eta})
    stream = {
        "master_seed": random["master_seed"], "spawn_key": list(child.spawn_key), "bit_generator": "PCG64",
        "applicant_order": random["applicant_order"],
        "ordered_id_sha256": hashlib.sha256(json.dumps(ids.tolist(), separators=(",", ":")).encode()).hexdigest(),
    }
    return targets, private, stream


def noisy_audit_preview(
    applicants: pd.DataFrame, targets: pd.DataFrame, noise: pd.DataFrame, preview_ids: list[str]
) -> pd.DataFrame:
    """Preserve the exact ID selection/order from the existing Step 3 preview."""
    if not preview_ids or len(preview_ids) != len(set(preview_ids)):
        raise ValueError("preview IDs must be nonempty and unique")
    combined = applicants.merge(targets, on="applicant_id", validate="one_to_one").merge(noise, on="applicant_id", validate="one_to_one")
    preview = combined.set_index("applicant_id").loc[preview_ids].reset_index()
    preview.insert(0, "information_scope", SIGNAL_PREVIEW_SCOPE)
    return preview.loc[:, SIGNAL_PREVIEW_COLUMNS]


def _summary(values: pd.Series) -> dict[str, Any]:
    return {
        "mean": float(values.mean()), "median": float(values.median()),
        "sd": float(values.std(ddof=1)) if len(values) > 1 else None,
        "minimum": float(values.min()), "maximum": float(values.max()),
        "quantiles": {f"p{int(q * 100):02d}": float(values.quantile(q)) for q in (.01, .05, .25, .50, .75, .95, .99)},
    }


def validate_noisy_signal(
    applicants: pd.DataFrame, targets: pd.DataFrame, noise: pd.DataFrame, config: dict[str, Any]
) -> dict[str, Any]:
    """Validate exact stream/formula identities without requiring zero sample mean."""
    validate_signal_config(config)
    report: dict[str, Any] = {"schema_version": SIGNAL_SCHEMA_VERSION, "valid": False, "errors": [], "warnings": [], "checks": {}}

    def check(name: str, value: bool) -> None:
        report["checks"][name] = bool(value)
        if not value:
            report["errors"].append(name)

    check("exact_target_schema", tuple(targets.columns) == SIGNAL_COLUMNS)
    check("exact_private_schema", tuple(noise.columns) == NOISE_COLUMNS)
    check("exact_count", len(targets) == len(noise) == len(applicants) == config["expected_input_rows"])
    check("no_missing_values", not targets.isna().any().any() and not noise.isna().any().any())
    if report["errors"]:
        return report
    for name, frame in (("target", targets), ("noise", noise), ("input", applicants)):
        check(f"unique_{name}_ids", frame.applicant_id.is_unique and not frame.applicant_id.isna().any())
        check(f"matching_{name}_ids", set(frame.applicant_id) == set(applicants.applicant_id))
    try:
        finite = np.isfinite(targets.PN.to_numpy(dtype=float)).all() and np.isfinite(noise.eta.to_numpy(dtype=float)).all()
    except (TypeError, ValueError):
        finite = False
    check("finite_signal_and_noise", bool(finite))
    if report["errors"]:
        return report
    combined = applicants.merge(targets, on="applicant_id", validate="one_to_one").merge(noise, on="applicant_id", validate="one_to_one")
    combined["scaled_true_PD"] = SIGNAL_SCALE * combined.true_PD
    check("raw_signal_equation", bool(np.allclose(combined.PN, raw_noisy_measurement(combined.true_PD.to_numpy(), combined.eta.to_numpy()), rtol=1e-13, atol=1e-15)))
    expected_targets, expected_noise, stream = generate_noisy_signal(applicants, config)
    ordered_ids = expected_targets.applicant_id
    observed_targets = targets.set_index("applicant_id").loc[ordered_ids].reset_index()
    observed_noise = noise.set_index("applicant_id").loc[ordered_ids].reset_index()
    check("exact_configured_noise_draws", observed_noise.equals(expected_noise))
    check("deterministic_raw_targets", observed_targets.equals(expected_targets))
    signal_mean = float(combined.scaled_true_PD.mean())
    noise_mean = float(combined.eta.mean())
    target_mean = float(combined.PN.mean())
    check("sample_mean_identity", bool(np.isclose(target_mean, signal_mean + noise_mean, rtol=1e-12, atol=1e-15)))
    report["n_applicants"] = len(combined)
    report["random_stream"] = stream
    report["distributions"] = {name: _summary(combined[name]) for name in ("eta", "PN", "scaled_true_PD")}
    report["PN_range_counts"] = []
    for label, mask in (("below_zero", combined.PN < 0), ("within_zero_one", combined.PN.between(0, 1)), ("above_one", combined.PN > 1)):
        report["PN_range_counts"].append({"range": label, "count": int(mask.sum()), "percent": float(100 * mask.mean())})
    def correlation(variable: str) -> float | None:
        if len(combined) < 2 or combined[variable].nunique() < 2 or combined.true_PD.nunique() < 2:
            return None
        return float(combined[variable].corr(combined.true_PD))
    signal_sd = combined.scaled_true_PD.std(ddof=1)
    report["signal_noise_comparison"] = {
        "correlation_PN_true_PD": correlation("PN"), "correlation_eta_true_PD": correlation("eta"),
        "noise_sd_over_scaled_signal_sd": float(combined.eta.std(ddof=1) / signal_sd) if pd.notna(signal_sd) and signal_sd > 0 else None,
        "mean_PN": target_mean, "scaled_mean_true_PD": signal_mean, "mean_eta": noise_mean,
        "sample_mean_identity_residual": target_mean - signal_mean - noise_mean,
    }
    report["group_comparison"] = []
    for group in (0, 1):
        subset = combined.loc[combined.B == group]
        report["group_comparison"].append({"B": group, "n": len(subset), **{f"mean_{name}": float(subset[name].mean()) if len(subset) else None for name in ("eta", "PN", "scaled_true_PD")}})
    report["sd_convention"] = "sample standard deviation, ddof=1"
    report["units"] = "raw probability units; SD=0.1 equals 10 percentage points"
    report["warnings"] = [
        "PN is an unbounded noisy measurement, not a true probability; negative values are retained.",
        "Individual measurement noise is large relative to the varying signal; this does not by itself establish what a future regression can learn.",
    ]
    report["valid"] = not report["errors"]
    return report
