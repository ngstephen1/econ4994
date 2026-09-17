"""Repeated-seed uncertainty analysis for the frozen systemic-opportunity DGP."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import resource
import subprocess
import traceback
import itertools
import uuid
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml
from threadpoolctl import threadpool_limits

from fair_lending.economic_lending.config import (
    PROJECT_ROOT,
    config_fingerprint,
    load_economic_config,
)
from fair_lending.economic_lending.history import build_at_risk_payment_history
from fair_lending.economic_lending.ml import (
    load_ml_config,
    predict_ml_applicant_risk,
    tune_hist_gradient_boosting,
)
from fair_lending.economic_lending.nonlinear import (
    load_nonlinear_config,
    nonlinear_repayment_probabilities,
)
from fair_lending.economic_lending.outcomes import simulate_repayment_outcomes_from_uniforms
from fair_lending.economic_lending.portfolio import (
    OPTIMALITY_ROUNDOFF,
    allocation_overlap,
    evaluate_portfolio,
    portfolio_group_audit,
    solve_fixed_request_portfolio,
)
from fair_lending.economic_lending.profit import expected_profit_from_options
from fair_lending.economic_lending.recovery import scientific_provenance, digest, save_attempt
from fair_lending.economic_lending.repayment import true_repayment_probabilities
from fair_lending.economic_lending.systemic import (
    controlled_funding_audit,
    draw_systemic_population_inputs,
    generate_systemic_world,
    identify_opportunity_switchers,
    load_systemic_config,
    validate_matched_worlds,
)
from fair_lending.economic_lending.traditional import (
    build_policy_assessments,
    fit_traditional_logit,
    predict_applicant_risk,
)


MC_CONFIG_PATH = PROJECT_ROOT / "configs" / "economic_lending" / "systemic_monte_carlo.yaml"
MC_ANALYSIS_VERSION = "systemic-opportunity-mc-v2-recovery"
PRIMARY_ESTIMANDS = (
    "delta_a_funding_rate",
    "delta_b_funding_rate",
    "delta_funding_gap",
    "delta_principal_gap",
    "delta_funded_requested_gap",
    "delta_true_expected_portfolio_profit",
    "oracle_regret",
    "delta_oracle_regret",
    "delta_realized_portfolio_profit",
)


@dataclass(frozen=True)
class ReplicationSeeds:
    replication_index: int
    replication_seed: int
    replication_spawn_key: tuple[int, ...]
    population_seed: int
    repayment_seed: int
    ml_seed: int


def load_systemic_mc_config(path: Path | str = MC_CONFIG_PATH) -> dict[str, Any]:
    """Load the Monte Carlo design and verify it points to the frozen mechanism."""

    path = Path(path)
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict) or loaded.get("experiment_id") != "systemic_opportunity_mc_v1":
        raise ValueError("Monte Carlo config must declare systemic_opportunity_mc_v1")
    systemic = load_systemic_config()
    if loaded["systemic_config_fingerprint"] != systemic["metadata"]["config_fingerprint"]:
        raise ValueError("Monte Carlo config does not match the frozen systemic config")
    if [float(x) for x in loaded["design"]["systemic_strengths"]] != [0.0, 0.1, 0.2, 0.4]:
        raise ValueError("Monte Carlo strengths must match Prompt 16")
    config = copy.deepcopy(loaded)
    config["metadata"] = {
        "config_path": str(path),
        "config_fingerprint": config_fingerprint(loaded),
    }
    return config


def code_revision() -> str:
    """Return the repository revision used in stable replication identities."""

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    head = result.stdout.strip() if result.returncode == 0 else "unknown"
    return f"{head}:{scientific_provenance()['fingerprint']}"


def replication_seeds(master_seed: int, replication_index: int) -> ReplicationSeeds:
    """Spawn independent top-level replication and named child seeds."""

    if master_seed < 0 or replication_index < 0:
        raise ValueError("master seed and replication index must be nonnegative")
    replication_sequence = np.random.SeedSequence(master_seed).spawn(replication_index + 1)[
        replication_index
    ]
    population, repayment, ml = replication_sequence.spawn(3)
    return ReplicationSeeds(
        replication_index=replication_index,
        replication_seed=int(replication_sequence.generate_state(1, dtype=np.uint64)[0]),
        replication_spawn_key=tuple(replication_sequence.spawn_key),
        population_seed=int(population.generate_state(1, dtype=np.uint64)[0]),
        repayment_seed=int(repayment.generate_state(1, dtype=np.uint64)[0]),
        ml_seed=int(ml.generate_state(1, dtype=np.uint32)[0]),
    )


def replication_id(
    config_fingerprint_value: str,
    seeds: ReplicationSeeds,
    revision: str,
) -> str:
    """Build the stable scientific identity for one independent replication."""

    identity = {
        "analysis_version": MC_ANALYSIS_VERSION,
        "config_fingerprint": config_fingerprint_value,
        "replication_index": seeds.replication_index,
        "replication_seed": seeds.replication_seed,
        "spawn_key": list(seeds.replication_spawn_key),
        "code_revision": revision,
    }
    serialized = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def replication_path(config: dict[str, Any], run_id: str) -> Path:
    return PROJECT_ROOT / config["storage"]["replication_directory"] / f"{run_id}.json"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_replication_atomic(record: dict[str, Any], path: Path) -> None:
    """Persist immutable attempt evidence and an atomic latest-status pointer."""
    value = _json_safe(record)
    value.pop("record_checksum", None)
    value["record_checksum"] = digest(value)
    save_attempt(path, value)


def expected_record_counts(config: dict[str, Any]) -> dict[str, int]:
    design = config["design"]
    worlds = len(design["risk_worlds"]) * len(design["systemic_strengths"])
    portfolios = worlds * len(design["policies"]) * len(design["budget_fractions"])
    return {
        "runs": portfolios,
        "model_performance": worlds * len(design["policies"]),
        "ml_selections": worlds,
        "controlled_audits": portfolios * 2,
    }


def load_valid_replication(
    path: Path,
    *,
    expected_run_id: str,
    expected_config_fingerprint: str,
    expected_revision: str,
    expected_counts: dict[str, int],
    expected_design: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Return a completed, matching, structurally intact replication only."""

    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(record, dict):
        return None
    checksum = record.get("record_checksum")
    try:
        if checksum != digest({k: v for k, v in record.items() if k != "record_checksum"}):
            return None
    except (TypeError, ValueError):
        return None
    identity_ok = (
        record.get("status") == "success"
        and record.get("run_id") == expected_run_id
        and record.get("analysis_version") == MC_ANALYSIS_VERSION
        and record.get("config_fingerprint") == expected_config_fingerprint
        and record.get("code_revision") == expected_revision
    )
    if not identity_ok:
        return None
    if any(not isinstance(record.get(key), list) or len(record[key]) != count for key, count in expected_counts.items()):
        return None
    design = expected_design or load_systemic_mc_config()["design"]
    try:
        if record["n_applicants"] != design["n_applicants"] or sum(record["cohort_counts"].values()) != record["n_applicants"]:
            return None
        worlds = set(itertools.product(design["risk_worlds"], design["systemic_strengths"]))
        models = {(*w, p) for w in worlds for p in design["policies"]}
        cells = {(*m, b) for m in models for b in design["budget_fractions"]}
        keys = ["risk_world", "systemic_strength", "family", "budget_id"]
        for table, names, expected in [
            ("runs", keys, cells), ("switcher_lending", keys, {c for c in cells if c[1] != 0}),
            ("model_performance", keys[:3], models), ("ml_selections", keys[:2], worlds),
            ("controlled_audits", keys + ["model"], {(*c, m) for c in cells for m in ["U0_race_only", "U1_downstream_controls"]}),
        ]:
            actual = [tuple(row[k] for k in names) for row in record[table]]
            if len(actual) != len(set(actual)) or set(actual) != expected:
                return None
        for row in record["model_performance"]:
            if not all(isinstance(row[k], (int, float)) and math.isfinite(row[k]) and row[k] >= 0
                       for k in ["risk_mae", "risk_rmse", "profit_mae", "profit_rmse"]):
                return None
        for row in record["runs"]:
            for name in (*PRIMARY_ESTIMANDS, "bank_budget", "total_principal_funded", "a_funding_rate", "b_funding_rate"):
                if not isinstance(row[name], (float, int)) or not math.isfinite(row[name]):
                    return None
            if row["replication_index"] != record["replication_index"]:
                return None
            if row["bank_budget"] != record["budgets"][row["budget_id"]]:
                return None
            if not all(0 <= row[k] <= 1 for k in ["a_funding_rate", "b_funding_rate"]):
                return None
            metadata = row["solver_metadata"]
            if not row["solver_success"] or not 0 <= row["solver_mip_gap"] <= OPTIMALITY_ROUNDOFF or not metadata["budget_feasible"]:
                return None
            if metadata["budget_residual_dollars"] > 0 or metadata["budget_constraint_numerical_guard_dollars"] != 0:
                return None
            if row["total_principal_funded"] > row["bank_budget"] + 1e-6:
                return None
    except (KeyError, TypeError, ValueError):
        return None
    return record


def neutral_budgets(
    neutral_evaluation_loans: pd.DataFrame,
    budget_fractions: dict[str, float],
) -> dict[str, float]:
    """Construct per-replication budgets from neutral evaluation requests only."""

    total = float(neutral_evaluation_loans["requested_principal"].sum())
    return {name: total * float(fraction) for name, fraction in budget_fractions.items()}


def _policy_id(family: str, risk_world: str, strength: float, replication_index: int) -> str:
    strength_token = f"{strength:.2f}".replace(".", "p")
    return f"mc_r{replication_index:03d}_{family}_{risk_world}_s{strength_token}"


def _oracle_policy(loans: pd.DataFrame, truth: pd.DataFrame, policy_id: str) -> pd.DataFrame:
    probability = truth[["applicant_id", "repayment_probability_per_period_true"]].rename(
        columns={"repayment_probability_per_period_true": "repayment_probability_oracle"}
    )
    return build_policy_assessments(
        loans,
        probability,
        policy_id=policy_id,
        probability_column="repayment_probability_oracle",
    )


def _model_performance(
    risk_world: str,
    strength: float,
    family: str,
    policy: pd.DataFrame,
    truth: pd.DataFrame,
    true_economics: pd.DataFrame,
) -> dict[str, Any]:
    joined = (
        policy.merge(truth, on="applicant_id", validate="one_to_one")
        .merge(true_economics, on="applicant_id", validate="one_to_one")
    )
    risk_error = joined["repayment_probability_used"] - joined["repayment_probability_per_period_true"]
    profit_error = joined["expected_profit_perceived"] - joined["expected_profit_true"]
    return {
        "risk_world": risk_world,
        "systemic_strength": strength,
        "family": family,
        "risk_mae": float(risk_error.abs().mean()),
        "risk_rmse": float(np.sqrt(np.mean(np.square(risk_error)))),
        "profit_mae": float(profit_error.abs().mean()),
        "profit_rmse": float(np.sqrt(np.mean(np.square(profit_error)))),
    }


def _pathway_record(
    risk_world: str,
    strength: float,
    neutral: dict[str, pd.DataFrame],
    current: dict[str, pd.DataFrame],
    switcher_ids: set[str],
) -> list[dict[str, Any]]:
    applicant_fields = ["employment_years", "annual_income", "liquid_assets"]
    neutral_frame = (
        neutral["applicants"][["applicant_id", "group", *applicant_fields]]
        .merge(neutral["loan_options"][["applicant_id", "first_period_dti", "requested_ltv"]], on="applicant_id")
        .merge(neutral["truth"], on="applicant_id")
        .merge(neutral["economics"], on="applicant_id")
    )
    current_frame = (
        current["applicants"][["applicant_id", *applicant_fields]]
        .merge(current["loan_options"][["applicant_id", "first_period_dti", "requested_ltv"]], on="applicant_id")
        .merge(current["truth"], on="applicant_id")
        .merge(current["economics"], on="applicant_id")
    )
    paired = neutral_frame.merge(current_frame, on="applicant_id", suffixes=("_neutral", "_systemic"))
    variables = [
        *applicant_fields,
        "first_period_dti",
        "requested_ltv",
        "repayment_probability_per_period_true",
        "full_repayment_probability_true",
        "expected_profit_true",
    ]
    rows = []
    for population, sample in (
        ("group_b_switchers", paired[paired["applicant_id"].isin(switcher_ids)]),
        ("all_group_b", paired[paired["group"].astype(str).eq("B")]),
    ):
        row: dict[str, Any] = {
            "risk_world": risk_world,
            "systemic_strength": strength,
            "population": population,
            "n": int(len(sample)),
            "status": "available" if len(sample) else "no_switchers",
        }
        for variable in variables:
            difference = sample[f"{variable}_systemic"] - sample[f"{variable}_neutral"]
            row[f"delta_{variable}"] = float(difference.mean()) if len(sample) else None
        rows.append(row)
    return rows


def _transition_counts(neutral: pd.DataFrame, treated: pd.DataFrame, ids: set[str]) -> dict[str, int]:
    first = neutral.set_index("applicant_id")["approved"].astype(bool)
    second = treated.set_index("applicant_id")["approved"].astype(bool)
    index = pd.Index(sorted(ids)).intersection(first.index).intersection(second.index)
    a = first.loc[index]
    b = second.loc[index]
    return {
        "switchers_in_evaluation": int(len(index)),
        "funded_both": int((a & b).sum()),
        "neutral_only": int((a & ~b).sum()),
        "systemic_only": int((~a & b).sum()),
        "unfunded_both": int((~a & ~b).sum()),
    }


def _failure_status(stage: str) -> str:
    if stage == "model":
        return "failed_model"
    if stage == "solver":
        return "failed_solver"
    if stage == "validation":
        return "failed_validation"
    return "other"


def run_systemic_mc_replication(
    replication_index: int,
    *,
    mc_config_path: Path | str = MC_CONFIG_PATH,
    diagnostic_directory: Path | None = None,
) -> dict[str, Any]:
    """Use one native math thread per process; never retune model settings."""
    with threadpool_limits(limits=1):
        return _run_systemic_mc_replication(replication_index, mc_config_path=mc_config_path,
                                           diagnostic_directory=diagnostic_directory)


def _run_systemic_mc_replication(
    replication_index: int,
    *,
    mc_config_path: Path | str = MC_CONFIG_PATH,
    diagnostic_directory: Path | None = None,
) -> dict[str, Any]:
    """Execute one complete independent replication and return compact results."""

    started = perf_counter()
    stage = "initialization"
    mc = load_systemic_mc_config(mc_config_path)
    systemic = load_systemic_config()
    baseline = load_economic_config()
    nonlinear = load_nonlinear_config()
    ml_config = load_ml_config()
    revision = code_revision()
    seeds = replication_seeds(int(mc["design"]["master_seed"]), replication_index)
    run_id = replication_id(mc["metadata"]["config_fingerprint"], seeds, revision)
    if diagnostic_directory is not None:
        diagnostic_directory = Path(diagnostic_directory) / f"r{replication_index}_{uuid.uuid4().hex}"
        print(f"replication {replication_index}: diagnostic inputs -> {diagnostic_directory}", flush=True)
    base_record: dict[str, Any] = {
        "analysis_version": MC_ANALYSIS_VERSION,
        "experiment_id": mc["experiment_id"],
        "config_fingerprint": mc["metadata"]["config_fingerprint"],
        "systemic_config_fingerprint": systemic["metadata"]["config_fingerprint"],
        "code_revision": revision,
        "provenance": scientific_provenance(),
        "execution": {"native_threads_per_worker": 1, "solver_time_limit_seconds": 60},
        "run_id": run_id,
        "replication_index": replication_index,
        "seeds": {
            "master_seed": int(mc["design"]["master_seed"]),
            "replication_seed": seeds.replication_seed,
            "replication_spawn_key": list(seeds.replication_spawn_key),
            "population_seed": seeds.population_seed,
            "repayment_seed": seeds.repayment_seed,
            "ml_seed": seeds.ml_seed,
        },
    }
    try:
        n = int(mc["design"]["n_applicants"])
        strengths = [float(x) for x in mc["design"]["systemic_strengths"]]
        risk_worlds = list(mc["design"]["risk_worlds"])
        policies_declared = list(mc["design"]["policies"])

        stage = "validation"
        draws = draw_systemic_population_inputs(
            n, baseline, systemic, seed=seeds.population_seed
        )
        base_worlds = {
            strength: generate_systemic_world(draws, baseline, systemic, strength)
            for strength in strengths
        }
        for strength in strengths:
            validate_matched_worlds(draws, base_worlds[0.0], base_worlds[strength])
        b_access = []
        switcher_ids: dict[float, set[str]] = {}
        for strength in strengths:
            mechanism = base_worlds[strength]["mechanism"].merge(
                base_worlds[strength]["applicants"][["applicant_id", "group"]],
                on="applicant_id",
            )
            b_frame = mechanism[mechanism["group"].astype(str).eq("B")]
            switchers = identify_opportunity_switchers(
                base_worlds[0.0]["mechanism"],
                base_worlds[strength]["mechanism"],
                base_worlds[0.0]["applicants"],
            )
            ids = set(switchers.loc[switchers["upstream_displaced"], "applicant_id"])
            switcher_ids[strength] = ids
            b_access.append(
                {
                    "systemic_strength": strength,
                    "group_b_n": int(len(b_frame)),
                    "group_b_opportunity_access_rate": float(b_frame["opportunity_access"].mean()),
                    "switcher_count": int(len(ids)),
                    "switcher_share": float(len(ids) / len(b_frame)),
                }
            )
        access_lookup = {row["systemic_strength"]: row for row in b_access}
        access_rates = [access_lookup[strength]["group_b_opportunity_access_rate"] for strength in strengths]
        if any(later > earlier for earlier, later in zip(access_rates, access_rates[1:])):
            raise AssertionError("matched Group B opportunity access must be nonincreasing")

        neutral_eval_ids = base_worlds[0.0]["applicants"].loc[
            base_worlds[0.0]["applicants"]["cohort"].astype(str).eq("evaluation"),
            "applicant_id",
        ]
        neutral_eval_loans = base_worlds[0.0]["loan_options"].loc[
            base_worlds[0.0]["loan_options"]["applicant_id"].isin(neutral_eval_ids)
        ]
        budgets = neutral_budgets(neutral_eval_loans, mc["design"]["budget_fractions"])
        repayment_uniforms = np.random.default_rng(seeds.repayment_seed).random(
            (n, int(baseline["contract"]["term_periods"]))
        )
        replication_ml_config = copy.deepcopy(ml_config)
        replication_ml_config["random_state"] = seeds.ml_seed

        stage = "model"
        worlds: dict[tuple[str, float], dict[str, pd.DataFrame]] = {}
        policies: dict[tuple[str, float, str], pd.DataFrame] = {}
        model_performance: list[dict[str, Any]] = []
        ml_selections: list[dict[str, Any]] = []
        pathway: list[dict[str, Any]] = []
        model_fit_seconds = 0.0
        for risk_world in risk_worlds:
            for strength in strengths:
                print(f"replication {replication_index}: fitting {risk_world}, s={strength}", flush=True)
                common = base_worlds[strength]
                applicants = common["applicants"]
                loans = common["loan_options"]
                if risk_world == "additive_logistic_baseline":
                    truth = true_repayment_probabilities(applicants, loans, baseline["true_risk"])
                elif risk_world == "nonlinear_v1":
                    truth = nonlinear_repayment_probabilities(
                        applicants, loans, baseline["true_risk"], nonlinear
                    )
                else:
                    raise ValueError(f"unknown risk world: {risk_world}")
                outcomes = simulate_repayment_outcomes_from_uniforms(
                    loans, truth, repayment_uniforms
                )
                economics = expected_profit_from_options(loans, truth)
                history = build_at_risk_payment_history(applicants, loans, outcomes)
                train = history[history["cohort"].eq("historical_train")].reset_index(drop=True)
                validation = history[history["cohort"].eq("historical_validation")].reset_index(drop=True)
                eval_ids = applicants.loc[
                    applicants["cohort"].astype(str).eq("evaluation"), "applicant_id"
                ]
                world = {
                    "applicants": applicants[applicants["applicant_id"].isin(eval_ids)].reset_index(drop=True),
                    "loan_options": loans[loans["applicant_id"].isin(eval_ids)].reset_index(drop=True),
                    "truth": truth[truth["applicant_id"].isin(eval_ids)].reset_index(drop=True),
                    "outcomes": outcomes[outcomes["applicant_id"].isin(eval_ids)].reset_index(drop=True),
                    "economics": economics[economics["applicant_id"].isin(eval_ids)].reset_index(drop=True),
                }
                worlds[(risk_world, strength)] = world

                fit_started = perf_counter()
                traditional = fit_traditional_logit(train, baseline["true_risk"])
                traditional_prediction = predict_applicant_risk(
                    traditional,
                    world["applicants"],
                    world["loan_options"],
                    baseline["true_risk"],
                )
                traditional_policy = build_policy_assessments(
                    world["loan_options"],
                    traditional_prediction,
                    policy_id=_policy_id("traditional", risk_world, strength, replication_index),
                )
                selection = tune_hist_gradient_boosting(train, validation, replication_ml_config)
                ml_prediction = predict_ml_applicant_risk(
                    selection.estimator, world["applicants"], world["loan_options"]
                )
                ml_policy = build_policy_assessments(
                    world["loan_options"],
                    ml_prediction,
                    policy_id=_policy_id("ml", risk_world, strength, replication_index),
                    probability_column="repayment_probability_ml",
                )
                model_fit_seconds += perf_counter() - fit_started
                oracle_policy = _oracle_policy(
                    world["loan_options"],
                    world["truth"],
                    _policy_id("true_risk_reference", risk_world, strength, replication_index),
                )
                policies[(risk_world, strength, "traditional")] = traditional_policy
                policies[(risk_world, strength, "ml")] = ml_policy
                policies[(risk_world, strength, "true_risk_reference")] = oracle_policy
                for family, policy in (
                    ("traditional", traditional_policy),
                    ("ml", ml_policy),
                    ("true_risk_reference", oracle_policy),
                ):
                    if not np.array_equal(
                        policy["repayment_probability_base"].to_numpy(),
                        policy["repayment_probability_used"].to_numpy(),
                    ):
                        raise AssertionError("systemic policy contains a direct belief distortion")
                    model_performance.append(
                        _model_performance(
                            risk_world, strength, family, policy, world["truth"], world["economics"]
                        )
                    )
                ml_selections.append(
                    {
                        "risk_world": risk_world,
                        "systemic_strength": strength,
                        **{key: value.item() if isinstance(value, np.generic) else value for key, value in selection.selected_parameters.items()},
                    }
                )
                del history, train, validation

            neutral_full = {
                **base_worlds[0.0],
                "truth": true_repayment_probabilities(base_worlds[0.0]["applicants"], base_worlds[0.0]["loan_options"], baseline["true_risk"])
                if risk_world == "additive_logistic_baseline"
                else nonlinear_repayment_probabilities(base_worlds[0.0]["applicants"], base_worlds[0.0]["loan_options"], baseline["true_risk"], nonlinear),
            }
            neutral_full["economics"] = expected_profit_from_options(neutral_full["loan_options"], neutral_full["truth"])
            for strength in strengths:
                current_truth = true_repayment_probabilities(base_worlds[strength]["applicants"], base_worlds[strength]["loan_options"], baseline["true_risk"]) if risk_world == "additive_logistic_baseline" else nonlinear_repayment_probabilities(base_worlds[strength]["applicants"], base_worlds[strength]["loan_options"], baseline["true_risk"], nonlinear)
                current_full = {**base_worlds[strength], "truth": current_truth}
                current_full["economics"] = expected_profit_from_options(current_full["loan_options"], current_truth)
                pathway.extend(_pathway_record(risk_world, strength, neutral_full, current_full, switcher_ids[strength]))

        stage = "solver"
        decisions: dict[tuple[str, float, str, str], pd.DataFrame] = {}
        runs: list[dict[str, Any]] = []
        audits: list[dict[str, Any]] = []
        solver_seconds = 0.0
        for risk_world in risk_worlds:
            for strength in strengths:
                world = worlds[(risk_world, strength)]
                access = access_lookup[strength]
                for family in policies_declared:
                    policy = policies[(risk_world, strength, family)]
                    performance = next(
                        row for row in model_performance
                        if row["risk_world"] == risk_world
                        and row["systemic_strength"] == strength
                        and row["family"] == family
                    )
                    for budget_id, budget in budgets.items():
                        stage = "solver"
                        print(f"replication {replication_index}: solving {risk_world}, s={strength}, {family}, {budget_id}", flush=True)
                        solve_started = perf_counter()
                        solution = solve_fixed_request_portfolio(
                            world["loan_options"], policy, budget,
                            world_id=f"mc_r{replication_index}_{risk_world}_s{strength:.2f}",
                            budget_id=budget_id,
                            policy_id=str(policy["policy_id"].iloc[0]),
                            diagnostic_directory=diagnostic_directory,
                        )
                        solver_seconds += perf_counter() - solve_started
                        if not solution.solver_metadata["success"] or not solution.solver_metadata["budget_feasible"]:
                            raise RuntimeError("portfolio solver returned an unsuccessful result")
                        key = (risk_world, strength, family, budget_id)
                        decisions[key] = solution.decisions
                        summary, _ = evaluate_portfolio(
                            solution.decisions,
                            world["loan_options"],
                            world["truth"],
                            world["economics"],
                            world["outcomes"],
                            policy[["applicant_id", "repayment_probability_used"]],
                            budget,
                        )
                        group = portfolio_group_audit(
                            solution.decisions,
                            world["applicants"],
                            world["loan_options"],
                            world["truth"],
                            world["economics"],
                        ).set_index("group")
                        runs.append(
                            {
                                "replication_index": replication_index,
                                "risk_world": risk_world,
                                "systemic_strength": strength,
                                "family": family,
                                "budget_id": budget_id,
                                "bank_budget": budget,
                                **access,
                                "a_funding_rate": float(group.loc["A", "funding_rate"]),
                                "b_funding_rate": float(group.loc["B", "funding_rate"]),
                                "funding_gap_b_minus_a": float(group.loc["B_minus_A", "funding_rate"]),
                                "a_principal_per_applicant": float(group.loc["A", "mean_funded_principal_per_applicant"]),
                                "b_principal_per_applicant": float(group.loc["B", "mean_funded_principal_per_applicant"]),
                                "principal_gap_b_minus_a": float(group.loc["B_minus_A", "mean_funded_principal_per_applicant"]),
                                "a_funded_requested_ratio": float(group.loc["A", "funded_requested_principal_ratio"]),
                                "b_funded_requested_ratio": float(group.loc["B", "funded_requested_principal_ratio"]),
                                "funded_requested_gap_b_minus_a": float(group.loc["B_minus_A", "funded_requested_principal_ratio"]),
                                "true_expected_portfolio_profit": summary["true_expected_portfolio_profit"],
                                "realized_portfolio_profit": summary["realized_portfolio_profit"],
                                "total_principal_funded": summary["total_principal_funded"],
                                "budget_utilization": summary["budget_utilization"],
                                "risk_mae": performance["risk_mae"],
                                "risk_rmse": performance["risk_rmse"],
                                "profit_mae": performance["profit_mae"],
                                "profit_rmse": performance["profit_rmse"],
                                "solver_success": bool(solution.solver_metadata["success"]),
                                "solver_status": solution.solver_metadata["solver_status"],
                                "solver_mip_gap": solution.solver_metadata["optimality_gap"],
                                "solver_runtime_seconds": solution.solver_metadata["runtime_seconds"],
                                "solver_metadata": solution.solver_metadata,
                            }
                        )
                        stage = "secondary_audit"
                        try:
                            audit = controlled_funding_audit(
                                solution.decisions, world["applicants"],
                                world["loan_options"], baseline["true_risk"],
                            )
                        except Exception as error:
                            audit = pd.DataFrame([{
                                "model": model, "identified": False, "converged": False,
                                "group_b_coefficient": None, "audit_status": "failed_secondary_audit",
                                "failure_message": str(error),
                            } for model in ["U0_race_only", "U1_downstream_controls"]])
                        for row in audit.to_dict("records"):
                            audits.append(
                                {
                                    "replication_index": replication_index,
                                    "risk_world": risk_world,
                                    "systemic_strength": strength,
                                    "family": family,
                                    "budget_id": budget_id,
                                    **row,
                                }
                            )

        stage = "validation"
        run_frame = pd.DataFrame(runs)
        oracle = run_frame[run_frame["family"].eq("true_risk_reference")][
            ["risk_world", "systemic_strength", "budget_id", "true_expected_portfolio_profit"]
        ].rename(columns={"true_expected_portfolio_profit": "oracle_true_expected_portfolio_profit"})
        run_frame = run_frame.merge(
            oracle,
            on=["risk_world", "systemic_strength", "budget_id"],
            validate="many_to_one",
        )
        run_frame["oracle_regret"] = (
            run_frame["oracle_true_expected_portfolio_profit"]
            - run_frame["true_expected_portfolio_profit"]
        )
        overlap_values = []
        for row in run_frame.itertuples(index=False):
            policy_decision = decisions[(row.risk_world, row.systemic_strength, row.family, row.budget_id)]
            oracle_decision = decisions[(row.risk_world, row.systemic_strength, "true_risk_reference", row.budget_id)]
            overlap_values.append(
                allocation_overlap(policy_decision, oracle_decision, worlds[(row.risk_world, row.systemic_strength)]["loan_options"])["jaccard_overlap"]
            )
        run_frame["oracle_allocation_overlap"] = overlap_values

        level_columns = {
            "a_funding_rate": "delta_a_funding_rate",
            "b_funding_rate": "delta_b_funding_rate",
            "funding_gap_b_minus_a": "delta_funding_gap",
            "principal_gap_b_minus_a": "delta_principal_gap",
            "funded_requested_gap_b_minus_a": "delta_funded_requested_gap",
            "true_expected_portfolio_profit": "delta_true_expected_portfolio_profit",
            "realized_portfolio_profit": "delta_realized_portfolio_profit",
            "oracle_regret": "delta_oracle_regret",
        }
        neutral = run_frame[run_frame["systemic_strength"].eq(0.0)][
            ["risk_world", "family", "budget_id", *level_columns]
        ].rename(columns={column: f"neutral_{column}" for column in level_columns})
        run_frame = run_frame.merge(
            neutral, on=["risk_world", "family", "budget_id"], validate="many_to_one"
        )
        for level, delta in level_columns.items():
            run_frame[delta] = run_frame[level] - run_frame[f"neutral_{level}"]

        switcher_lending: list[dict[str, Any]] = []
        for risk_world in risk_worlds:
            for strength in strengths:
                if strength == 0:
                    continue
                for family in policies_declared:
                    for budget_id in budgets:
                        counts = _transition_counts(
                            decisions[(risk_world, 0.0, family, budget_id)],
                            decisions[(risk_world, strength, family, budget_id)],
                            switcher_ids[strength],
                        )
                        switcher_lending.append(
                            {
                                "replication_index": replication_index,
                                "risk_world": risk_world,
                                "systemic_strength": strength,
                                "family": family,
                                "budget_id": budget_id,
                                **counts,
                            }
                        )

        runtime = perf_counter() - started
        if code_revision() != revision:
            raise RuntimeError("source/config/dependencies changed during replication; refusing success")
        return {
            **base_record,
            "status": "success",
            "n_applicants": n,
            "cohort_counts": base_worlds[0.0]["applicants"]["cohort"].astype(str).value_counts().to_dict(),
            "budgets": budgets,
            "opportunity": b_access,
            "runs": run_frame.to_dict("records"),
            "pathway": pathway,
            "switcher_lending": switcher_lending,
            "model_performance": model_performance,
            "ml_selections": ml_selections,
            "controlled_audits": audits,
            "timing": {
                "total_seconds": runtime,
                "model_fit_seconds": model_fit_seconds,
                "solver_seconds": solver_seconds,
                "portfolios_solved": int(len(run_frame)),
            },
            "peak_rss_mb": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0**2)),
            "raw_rows_persisted": False,
        }
    except Exception as error:
        return {
            **base_record,
            "status": _failure_status(stage),
            "failure_stage": stage,
            "failure_type": type(error).__name__,
            "failure_message": str(error),
            "failure_traceback": traceback.format_exc(),
            "timing": {"total_seconds": perf_counter() - started},
            "raw_rows_persisted": False,
        }


def summarize_scalar(values: Iterable[float]) -> dict[str, Any]:
    """Return Monte Carlo dispersion, precision, intervals, and sign stability."""

    series = pd.Series(list(values), dtype=float).dropna()
    n = int(len(series))
    if not n:
        return {key: None for key in ["mean", "sd", "mcse", "relative_mcse", "p2_5", "p5", "p25", "p50", "p75", "p95", "p97_5", "fraction_positive", "fraction_negative", "sign_stability"]} | {"n_successful": 0}
    mean = float(series.mean())
    sd = float(series.std(ddof=1)) if n > 1 else None
    mcse = sd / math.sqrt(n) if sd is not None else None
    sign_stability = float((series > 0).mean()) if mean > 0 else float((series < 0).mean()) if mean < 0 else float((series == 0).mean())
    quantiles = series.quantile([.025, .05, .25, .5, .75, .95, .975])
    return {
        "n_successful": n,
        "mean": mean,
        "sd": sd,
        "mcse": mcse,
        "relative_mcse": mcse / abs(mean) if mcse is not None and abs(mean) > 1e-12 else None,
        "p2_5": float(quantiles.loc[.025]),
        "p5": float(quantiles.loc[.05]),
        "p25": float(quantiles.loc[.25]),
        "p50": float(quantiles.loc[.5]),
        "p75": float(quantiles.loc[.75]),
        "p95": float(quantiles.loc[.95]),
        "p97_5": float(quantiles.loc[.975]),
        "fraction_positive": float((series > 0).mean()),
        "fraction_negative": float((series < 0).mean()),
        "sign_stability": sign_stability,
    }


def records_to_frames(records: list[dict[str, Any]]) -> dict[str, pd.DataFrame]:
    """Reconstruct every summary input solely from compact replication records."""

    successful = [record for record in records if record.get("status") == "success"]
    frames: dict[str, pd.DataFrame] = {}
    for key in ["runs", "pathway", "switcher_lending", "model_performance", "ml_selections", "controlled_audits"]:
        rows = []
        for record in successful:
            for row in record.get(key, []):
                rows.append({"run_id": record["run_id"], **row})
        frames[key] = pd.DataFrame(rows)
    frames["replications"] = pd.DataFrame(
        [
            {
                "run_id": record["run_id"],
                "replication_index": record["replication_index"],
                "status": record["status"],
                "total_seconds": record.get("timing", {}).get("total_seconds"),
                "model_fit_seconds": record.get("timing", {}).get("model_fit_seconds"),
                "solver_seconds": record.get("timing", {}).get("solver_seconds"),
                "peak_rss_mb": record.get("peak_rss_mb"),
            }
            for record in records
        ]
    )
    return frames


def monte_carlo_summary(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["risk_world", "systemic_strength", "family", "budget_id"]
    for key, frame in runs.groupby(keys, sort=True, dropna=False):
        for estimand in PRIMARY_ESTIMANDS:
            rows.append(dict(zip(keys, key, strict=True), estimand=estimand, **summarize_scalar(frame[estimand])))
    return pd.DataFrame(rows)


def switcher_summary(pathway: pd.DataFrame, lending: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    opportunity = runs.drop_duplicates(["replication_index", "systemic_strength"])
    for strength, frame in opportunity.groupby("systemic_strength"):
        for estimand in ["switcher_count", "switcher_share", "group_b_opportunity_access_rate"]:
            rows.append({"record_type": "opportunity", "risk_world": "all", "systemic_strength": strength, "population": "group_b", "family": "all", "budget_id": "all", "estimand": estimand, **summarize_scalar(frame[estimand])})
    pathway_metrics = [column for column in pathway if column.startswith("delta_")]
    for key, frame in pathway.groupby(["risk_world", "systemic_strength", "population"], sort=True):
        for estimand in pathway_metrics:
            rows.append({"record_type": "pathway", "risk_world": key[0], "systemic_strength": key[1], "population": key[2], "family": "all", "budget_id": "all", "estimand": estimand, **summarize_scalar(frame[estimand])})
    for key, frame in lending.groupby(["risk_world", "systemic_strength", "family", "budget_id"], sort=True):
        for estimand in ["switchers_in_evaluation", "funded_both", "neutral_only", "systemic_only", "unfunded_both"]:
            rows.append({"record_type": "lending_transition", "risk_world": key[0], "systemic_strength": key[1], "population": "group_b_switchers", "family": key[2], "budget_id": key[3], "estimand": estimand, **summarize_scalar(frame[estimand])})
    return pd.DataFrame(rows)


def model_performance_summary(
    performance: pd.DataFrame,
    runs: pd.DataFrame,
    selections: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for key, frame in performance.groupby(["risk_world", "systemic_strength", "family"], sort=True):
        for estimand in ["risk_mae", "risk_rmse", "profit_mae", "profit_rmse"]:
            rows.append({"record_type": "performance", "risk_world": key[0], "systemic_strength": key[1], "family": key[2], "budget_id": "all", "estimand": estimand, **summarize_scalar(frame[estimand])})
    fitted = runs[runs["family"].isin(["traditional", "ml"])]
    for key, frame in fitted.groupby(["risk_world", "systemic_strength", "family", "budget_id"], sort=True):
        for estimand in ["oracle_regret", "oracle_allocation_overlap"]:
            rows.append({"record_type": "portfolio", "risk_world": key[0], "systemic_strength": key[1], "family": key[2], "budget_id": key[3], "estimand": estimand, **summarize_scalar(frame[estimand])})
    if not selections.empty:
        grouped = selections.groupby(["risk_world", "systemic_strength", "learning_rate", "max_iter", "max_leaf_nodes"], dropna=False).size().rename("selection_count").reset_index()
        totals = grouped.groupby(["risk_world", "systemic_strength"])["selection_count"].transform("sum")
        for row, total in zip(grouped.to_dict("records"), totals, strict=True):
            rows.append({"record_type": "hyperparameter_frequency", "risk_world": row["risk_world"], "systemic_strength": row["systemic_strength"], "family": "ml", "budget_id": "all", "estimand": f"lr={row['learning_rate']};iter={int(row['max_iter'])};leaves={int(row['max_leaf_nodes'])}", "n_successful": int(total), "mean": float(row["selection_count"] / total), "selection_count": int(row["selection_count"])})
    return pd.DataFrame(rows)


def monotonicity_summary(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, frame in runs.groupby(["replication_index", "risk_world", "family", "budget_id"]):
        lookup = frame.set_index("systemic_strength")
        if 0.2 not in lookup.index or 0.4 not in lookup.index:
            continue
        for estimand in ["delta_b_funding_rate", "delta_funding_gap", "delta_true_expected_portfolio_profit"]:
            rows.append({"replication_index": key[0], "risk_world": key[1], "family": key[2], "budget_id": key[3], "estimand": estimand, "abs_s40_ge_abs_s20": bool(abs(lookup.loc[.4, estimand]) >= abs(lookup.loc[.2, estimand]))})
    raw = pd.DataFrame(rows)
    if raw.empty:
        return raw
    return raw.groupby(["risk_world", "family", "budget_id", "estimand"], as_index=False).agg(n_replications=("abs_s40_ge_abs_s20", "size"), fraction_abs_s40_ge_abs_s20=("abs_s40_ge_abs_s20", "mean"))
