"""Recovery safeguards, independent of favorable research results."""

import copy
import importlib.util
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml
from concurrent.futures import ProcessPoolExecutor

from fair_lending.economic_lending import portfolio
from fair_lending.economic_lending.config import PROJECT_ROOT
from fair_lending.economic_lending.recovery import scientific_provenance
from fair_lending.economic_lending.systemic_mc import (
    expected_record_counts, load_systemic_mc_config, load_valid_replication,
    summarize_scalar, write_replication_atomic,
    run_systemic_mc_replication, _json_safe,
)
from test_systemic_monte_carlo import _complete_record


def runner_module():
    spec = importlib.util.spec_from_file_location("mc_runner", PROJECT_ROOT / "experiments/run_systemic_monte_carlo.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def small_problem(costs=(60., 50., 40.), profits=(12., 10., 7.)):
    keys = {"applicant_id": list(range(len(costs))), "option_id": [str(i) for i in range(len(costs))]}
    return pd.DataFrame({**keys, "requested_principal": costs}), pd.DataFrame({**keys, "expected_profit_perceived": profits})


def solve(loans, policy, budget, **kwargs):
    return portfolio.solve_fixed_request_portfolio(loans, policy, budget, world_id="w", budget_id="b", policy_id="p", **kwargs)


@pytest.mark.parametrize("costs,budget", [((60., 40.), 100.), ((.01, .02, .03), .03), ((.01, .02, .03), .039), ((290_000_000., 10_000_000.), 300_000_000.)])
def test_original_budget_feasible_set_matches_exhaustive(costs, budget):
    from decimal import Decimal
    profits = tuple(2. + i for i in range(len(costs)))
    loans, policy = small_problem(costs, profits)
    result = solve(loans, policy, budget)
    best = max(sum(p*x for p,x in zip(profits, bits))
               for bits in itertools.product([0,1], repeat=len(costs))
               if sum((Decimal(str(c))*x for c,x in zip(costs,bits)), Decimal(0)) <= Decimal(str(budget)))
    assert result.solver_metadata["perceived_objective_value"] == best
    assert result.solver_metadata["budget_constraint_numerical_guard_dollars"] == 0


def test_timeout_is_failure_with_replayable_diagnostic(tmp_path, monkeypatch):
    def timeout(**kwargs):
        assert kwargs["options"]["time_limit"] == 2.
        return SimpleNamespace(success=False, status=1, message="Time limit", x=np.array([1., 0., 0.]))
    monkeypatch.setattr(portfolio, "milp", timeout)
    loans, policy = small_problem()
    with pytest.raises(RuntimeError, match="status=1"):
        solve(loans, policy, 100., time_limit=2., diagnostic_directory=tmp_path)
    case = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert case["solver_status"] == 1
    assert case["bank_budget"] == 100.
    assert case["candidate"] == [1., 0., 0.]
    assert len(case["loans"]) == len(case["assessments"]) == 3


def test_infeasible_candidate_is_not_repaired_by_shrinking_budget(monkeypatch, tmp_path):
    calls = []
    def bad(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(success=True, status=0, message="optimal", x=np.ones(3), fun=-29., mip_gap=0.)
    monkeypatch.setattr(portfolio, "milp", bad)
    with pytest.raises(RuntimeError, match="infeasible"):
        solve(*small_problem(), 100., diagnostic_directory=tmp_path)
    assert len(calls) == 1
    case = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert case["budget_residual_dollars"] == 50.


def test_diagnostic_replay_matches_original(tmp_path):
    first = solve(*small_problem(), 100., diagnostic_directory=tmp_path)
    case = json.loads(next(tmp_path.glob("*.json")).read_text())
    second = solve(pd.DataFrame(case["loans"]), pd.DataFrame(case["assessments"]), case["bank_budget"])
    pd.testing.assert_frame_equal(first.decisions, second.decisions)


def test_dirty_source_invalidates_provenance(tmp_path):
    source = tmp_path / "src/fair_lending/economic_lending"
    source.mkdir(parents=True)
    module = source / "test.py"
    module.write_text("x = 1\n")
    before = scientific_provenance(tmp_path)
    module.write_text("x = 2\n")
    assert before["fingerprint"] != scientific_provenance(tmp_path)["fingerprint"]


@pytest.mark.parametrize("mutation", ["duplicate", "missing_metric", "infeasible", "guard", "wrong_cell", "nan", "model_nan"])
def test_integrity_rejects_rehashed_but_invalid_records(tmp_path, mutation):
    config = load_systemic_mc_config()
    counts = expected_record_counts(config)
    record = _complete_record("id", config["metadata"]["config_fingerprint"], "rev", counts)
    row = record["runs"][0]
    if mutation == "duplicate": record["runs"][1] = copy.deepcopy(row)
    elif mutation == "missing_metric": del row["delta_funding_gap"]
    elif mutation == "infeasible": row["total_principal_funded"] = 101.
    elif mutation == "guard": row["solver_metadata"]["budget_constraint_numerical_guard_dollars"] = .1
    elif mutation == "wrong_cell": row["budget_id"] = "unexpected"
    elif mutation == "nan": row["delta_funding_gap"] = float("nan")
    elif mutation == "model_nan": record["model_performance"][0]["risk_mae"] = None
    path = tmp_path / "record.json"
    write_replication_atomic(record, path)
    assert load_valid_replication(path, expected_run_id="id", expected_config_fingerprint=config["metadata"]["config_fingerprint"],
                                  expected_revision="rev", expected_counts=counts) is None


def test_attempt_history_retains_failure_after_success(tmp_path):
    path = tmp_path / "record.json"
    for status in ["failed_solver", "success"]:
        write_replication_atomic({"status": status}, path)
    attempts = [json.loads(p.read_text()) for p in (tmp_path / "attempts/record").glob("*.json")]
    assert {r["status"] for r in attempts} == {"failed_solver", "success"}
    assert json.loads(path.read_text())["status"] == "success"


def test_summary_only_missing_record_never_invokes_worker(tmp_path, monkeypatch):
    runner = runner_module()
    config = load_systemic_mc_config()
    config["storage"]["replication_directory"] = str(tmp_path)
    def forbidden(*args, **kwargs):
        pytest.fail("summary-only launched work")
    monkeypatch.setattr(runner, "_worker", forbidden)
    with pytest.raises(RuntimeError, match="no work launched"):
        runner._load_or_run([13], config, resume=True, workers=1, summarize_only=True)
    assert list(tmp_path.iterdir()) == []


def test_one_replication_does_not_claim_zero_uncertainty():
    result = summarize_scalar([5.])
    assert result["sd"] is None and result["mcse"] is None


def test_replication13_actual_failed_portfolio_regression():
    case = json.loads((Path(__file__).parent / "fixtures/systemic_r13_knapsack.json").read_text())
    result = solve(*small_problem(case["principal"], case["profit"]), case["bank_budget"])
    assert result.solver_metadata["budget_residual_dollars"] <= 0
    assert result.solver_metadata["integrality_error"] <= 1e-9
    assert result.solver_metadata["optimality_gap"] <= portfolio.OPTIMALITY_ROUNDOFF
    assert result.solver_metadata["budget_constraint_numerical_guard_dollars"] == 0


@pytest.mark.parametrize("gap,accepted", [(1.5e-16, True), (1e-5, False)])
def test_optimality_roundoff_is_not_a_substantive_gap(monkeypatch, gap, accepted):
    monkeypatch.setattr(portfolio, "milp", lambda **kwargs: SimpleNamespace(
        success=True, status=0, message="optimal", x=np.array([1., 0., 1.]), fun=-19.,
        mip_gap=gap, mip_dual_bound=-19.))
    if accepted:
        result = solve(*small_problem(), 100.)
        assert result.solver_metadata["optimality_gap"] == gap  # Never conceal the raw gap.
    else:
        with pytest.raises(RuntimeError, match="roundoff"):
            solve(*small_problem(), 100.)


def _deterministic_small_replication(arguments):
    """Real pipeline, small population, unchanged scientific coefficients/grid."""
    from threadpoolctl import threadpool_limits
    index, config_path = arguments
    with threadpool_limits(limits=1):
        result = run_systemic_mc_replication(index, mc_config_path=config_path)
    assert result["status"] == "success", result.get("failure_message")
    def strip(value):
        if isinstance(value, dict):
            return {k: strip(v) for k, v in value.items()
                    if k not in {"timing", "peak_rss_mb", "runtime_seconds", "solver_runtime_seconds"}}
        if isinstance(value, list):
            return [strip(v) for v in value]
        return value
    return _json_safe(strip(result))


def test_one_and_two_worker_outputs_match_exactly(tmp_path):
    config = yaml.safe_load((PROJECT_ROOT / "configs/economic_lending/systemic_monte_carlo.yaml").read_text())
    config["design"]["n_applicants"] = 300
    path = tmp_path / "small_mc.yaml"
    path.write_text(yaml.safe_dump(config))
    tasks = [(0, str(path)), (1, str(path))]
    with ProcessPoolExecutor(max_workers=1) as executor:
        serial = list(executor.map(_deterministic_small_replication, tasks))
    with ProcessPoolExecutor(max_workers=2) as executor:
        parallel = list(executor.map(_deterministic_small_replication, tasks))
    assert serial == parallel
