"""Full-study immutable records, attempt history, strict resume and reporting."""
from __future__ import annotations

from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import subprocess
import time
import traceback
import uuid

import numpy as np
import pandas as pd
import yaml

from . import noise_full as full
from . import noise_pilot as pilot
from . import noise_pilot_io as pio
from . import bank_regression as aware
from . import group_blind as blind
from .config import PROJECT_ROOT, configuration_fingerprint as fp
from .lending import BANK_INPUT_COLUMNS, choose_cutoff
from .profit import assess_bank_profit
from .risk_io import file_hash, verify_input_hashes

CONFIG_PATH = PROJECT_ROOT / "configs/professor_model/noise_robustness_full.yaml"
RUN_DIRECTORY = Path("results/metrics/professor_model/noise_robustness_full")
TABLE_DIRECTORY = Path("results/tables/professor_model/noise_robustness_full")
FIGURE_DIRECTORY = Path("results/figures/professor_model/noise_robustness_full")
NEW_FILES = ("configs/professor_model/noise_robustness_full.yaml", "src/fair_lending/professor_model/noise_full.py",
             "src/fair_lending/professor_model/noise_full_io.py", "experiments/run_professor_noise_full.py", "tests/test_professor_noise_full.py")
SETTINGS = {"OLS": "statsmodels_OLS_pinv_unregularized", "aware_features": ["const", "Y", "D", "C", "B"],
            "blind_features": ["const", "Y", "D", "C"], "cutoff": "exact_binary_float_integer_prefixes",
            "ties": "indivisible", "reject_all": "null_cutoff", "probability": "raw_and_separate_clip_0_1",
            "decomposition_atol": full.DECOMPOSITION_ATOL, "decomposition_rtol": full.DECOMPOSITION_RTOL,
            "original_abs_gap_pp": full.ORIGINAL_ABS_GAP_PP, "quantiles": "numpy_linear", "SD_ddof": 1}


def load_config():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    expected = {**pio.load_config(), "schema_version": "professor-noise-full-v1", "replications": 200}
    if config != expected:
        raise ValueError("Step 8B requires exactly the frozen 200-noise-replication design")
    return config


def snapshot(root=PROJECT_ROOT):
    # The pilot snapshot utility deliberately excludes pilot outputs. Restore
    # those files here: the full study must preserve them as original evidence.
    folders = ("data/synthetic/professor_model", "data/processed/professor_model", "results/tables/professor_model",
               "results/metrics/professor_model", "results/figures/professor_model", "configs", "src", "experiments", "tests")
    result = {}
    for folder in folders:
        for path in (root / folder).rglob("*"):
            relative = path.relative_to(root)
            if not path.is_file() or "__pycache__" in path.parts or path.suffix in (".pyc", ".DS_Store") or str(relative) in NEW_FILES or any(relative.is_relative_to(p) for p in (RUN_DIRECTORY, TABLE_DIRECTORY, FIGURE_DIRECTORY)):
                continue
            result[str(relative)] = file_hash(path)
    return dict(sorted(result.items()))


def prepare_manifest(config, root=PROJECT_ROOT, resume=False):
    sources = {str(p.relative_to(PROJECT_ROOT)): file_hash(p) for p in (PROJECT_ROOT / "src/fair_lending/professor_model").glob("*.py")}
    sources["experiments/run_professor_noise_full.py"] = file_hash(PROJECT_ROOT / "experiments/run_professor_noise_full.py")
    scientific = {k: v for k, v in config.items() if k not in ("replications", "schema_version")}
    identity = {"configuration": config, "config_file_sha256": file_hash(CONFIG_PATH), "original_checksums": snapshot(root),
                "source_code_hashes": sources, "source_fingerprint": fp(sources), "scientific_config_fingerprint": fp(scientific),
                "solver_settings_fingerprint": fp(SETTINGS), "settings": SETTINGS, "python": platform.python_version(),
                "dependencies": {name: version(name) for name in ("numpy", "pandas", "scipy", "statsmodels", "pyarrow", "PyYAML", "matplotlib")}}
    identity["fixed_budget"] = json.loads((root / config["budget_source"]).read_text())["selection"]["available_bank_capital"]
    splits = pd.read_csv(root / pio.legacy.SYNTHETIC / "applicant_splits.csv")
    identity["training_ID_fingerprint"] = aware.ids_fingerprint(splits.loc[splits.split == "train", "applicant_id"])
    path = root / RUN_DIRECTORY / "manifest.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if not resume:
            raise FileExistsError("Full-study manifest exists; use --resume or --summarize-only")
        if saved["experiment_fingerprint"] != fp(identity) or saved["identity"] != identity:
            raise ValueError("full-study source/config/input/dependency provenance changed")
        return saved
    if resume:
        raise FileNotFoundError("no full-study manifest exists")
    git = lambda *args: subprocess.run(["git", *args], cwd=PROJECT_ROOT, text=True, capture_output=True, check=True).stdout.strip()
    manifest = {"schema_version": config["schema_version"], "experiment_fingerprint": fp(identity), "identity": identity,
                "git_revision": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
                "created_at_utc": datetime.now(timezone.utc).isoformat(), "replications": 200}
    pio.immutable_json(path, manifest)
    return manifest


def verify_originals(manifest, root=PROJECT_ROOT):
    if not verify_input_hashes(manifest["identity"]["original_checksums"], root) or not verify_input_hashes(manifest["identity"]["source_code_hashes"], root):
        raise ValueError("original datasets/science/pilot evidence changed")


def run_id(index, manifest):
    if type(index) is not int or not 0 <= index < 200:
        raise ValueError("only replication indices 0..199 permitted")
    return fp({"experiment": manifest["experiment_fingerprint"], "replication": index, "master_seed": 499408, "spawn_key": [index]})


def result_path(index, root=PROJECT_ROOT):
    return root / RUN_DIRECTORY / "replications" / f"replication_{index:03d}.json"


def checked_json(path):
    value = json.loads(path.read_text())
    if value.pop("content_fingerprint", None) != fp(value):
        raise ValueError(f"record checksum invalid: {path}")
    return value


def publish(path, value):
    pio.immutable_json(path, {**value, "content_fingerprint": fp(value)})


def prepare_reference(observed, truth, manifest, root=PROJECT_ROOT):
    matrix, target, mask = full.reference_design(observed, truth)
    path = root / RUN_DIRECTORY / "signal_reference_EVALUATOR_ONLY.json"
    if path.exists():
        saved = checked_json(path)
        if saved["experiment_fingerprint"] != manifest["experiment_fingerprint"] or saved["reference"]["matrix_fingerprint"] != pilot.frame_fingerprint(matrix.reset_index(drop=True)):
            raise ValueError("signal-reference provenance mismatch")
        reference = saved["reference"]
    else:
        reference = full.fit_signal_reference(matrix, target)
        publish(path, {"experiment_fingerprint": manifest["experiment_fingerprint"], "reference": reference})
    return reference, mask, full.B_row(matrix)


def pilot_candidate(index, manifest, root=PROJECT_ROOT):
    """Validate recorded fixed source maps, not a directory glob enlarged by 8B."""
    if index >= 5:
        return None, "outside five-run pilot"
    try:
        old = json.loads((root / pio.RUN_DIRECTORY / "manifest.json").read_text())
        identity = old["identity"]
        if old["schema_version"] != "professor-noise-pilot-v1" or old["experiment_fingerprint"] != fp(identity):
            raise ValueError("pilot schema/identity mismatch")
        scientific = {k: v for k, v in identity["configuration"].items() if k not in ("replications", "schema_version")}
        if fp(scientific) != manifest["identity"]["scientific_config_fingerprint"] or identity["dependencies"] != manifest["identity"]["dependencies"] or identity["python"] != manifest["identity"]["python"]:
            raise ValueError("pilot scientific configuration/dependencies differ")
        if file_hash(root / "configs/professor_model/noise_robustness_pilot.yaml") != identity["configuration_file_sha256"] or not verify_input_hashes(identity["source_code_hashes"], root):
            raise ValueError("pilot configuration/source contents changed")
        record = pio.load_record(index, old, root)
        return record, "validated pilot schema, per-replication configuration, sources, dependencies, seed and integrity"
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return None, f"pilot not reusable: {type(exc).__name__}: {exc}"


def replay_pilot(record, observed, contracts, budget):
    """Reconstruct predictions/decisions from saved coefficients; no OLS refit."""
    evaluation = observed.loc[observed.split == "evaluation"]
    banks = {}
    for policy in pilot.POLICIES:
        model = record["banks"][policy]["model"]
        if policy == "group_aware":
            predictions = aware.risk_estimates(evaluation, model)
            profit = assess_bank_profit(contracts, predictions[["applicant_id", "model_id", "bank_PD_bounded"]])
            adapter = contracts[["applicant_id", "L_i"]].merge(profit, on="applicant_id", validate="one_to_one").rename(columns={"L_i": "requested_principal"}).loc[:, BANK_INPUT_COLUMNS]
            decisions, candidates, selection = choose_cutoff(adapter, budget)
        else:
            predictions = blind.risk_estimates(evaluation, model)
            profit = blind.assess_profit(contracts, predictions[["applicant_id", "model_id", blind.BOUNDED]])
            decisions, candidates, selection = blind.optimize_cutoff(contracts, profit, budget)
            adapter = contracts[["applicant_id", "L_i"]].merge(profit, on="applicant_id", validate="one_to_one").rename(columns={"L_i": "requested_principal", blind.BOUNDED: "bank_PD_bounded"}).loc[:, BANK_INPUT_COLUMNS]
        banks[policy] = {"model": model, "predictions": predictions, "profit": profit, "decisions": decisions, "selection": selection,
                         "diagnostics": pilot.exact_cutoff_diagnostics(adapter, decisions, candidates, selection, budget), "candidate_fingerprint": pilot.frame_fingerprint(candidates)}
    if pilot.bank_freeze_payload(banks) != record["banks"]:
        raise ValueError("pilot scientific outputs did not replay exactly")
    return banks


def freeze(banks, index, stream, attempt_id, manifest, root=PROJECT_ROOT):
    verify_originals(manifest, root)
    value = {"status": "frozen_before_true_profit_evaluation", "run_id": run_id(index, manifest), "replication": index,
             "experiment_fingerprint": manifest["experiment_fingerprint"], "stream": stream, "banks": pilot.bank_freeze_payload(banks)}
    path = root / RUN_DIRECTORY / "freezes" / f"{attempt_id}.json"
    publish(path, value)
    return path


def evaluate(banks, freeze_path, observed, targets, contracts, budget, manifest, root=PROJECT_ROOT):
    frozen = checked_json(freeze_path)
    if frozen["status"] != "frozen_before_true_profit_evaluation" or frozen["experiment_fingerprint"] != manifest["experiment_fingerprint"] or frozen["banks"] != pilot.bank_freeze_payload(banks):
        raise ValueError("bank freeze must match before truth evaluation")
    verify_originals(manifest, root)
    truth = pd.read_parquet(root / pio.legacy.PROCESSED / "true_expected_profit.parquet")
    oracle = pd.read_parquet(root / pio.legacy.PROCESSED / "oracle_lending_decisions.parquet")
    selection = json.loads((root / pio.legacy.METRICS / "oracle_comparison_validation.json").read_text())["oracle_selection"]
    metrics, groups, comparison = pilot.evaluate_replication(banks, observed, targets, contracts, truth, oracle, selection, budget, frozen["banks"])
    oracle_value = {"true_expected_portfolio_profit": math.fsum(truth.set_index("applicant_id").loc[oracle.loc[oracle.approved, "applicant_id"], "expected_profit_true"]),
                    "number_approved": int(oracle.approved.sum()), "optimal_tau": selection["optimal_tau"], "decision_fingerprint": pilot.frame_fingerprint(oracle)}
    return metrics, groups, comparison, oracle_value


def validate_record(record, index, manifest, root=PROJECT_ROOT):
    identity = manifest["identity"]
    if record["schema_version"] != "professor-noise-full-v1" or record["status"] != "success" or record["replication"] != index or record["run_id"] != run_id(index, manifest) or record["experiment_fingerprint"] != manifest["experiment_fingerprint"] or not math.isfinite(record["runtime_seconds"]) or record["runtime_seconds"] < 0:
        raise ValueError("replication identity/status mismatch")
    for key in ("scientific_config_fingerprint", "source_fingerprint", "solver_settings_fingerprint", "dependencies"):
        if record[key] != identity[key]:
            raise ValueError(f"replication provenance mismatch: {key}")
    if record["stream"]["spawn_key"] != [index] or record["stream"]["master_seed"] != 499408:
        raise ValueError("stream identity mismatch")
    freeze_path = root / record["freeze_path"]
    if not freeze_path.resolve().is_relative_to((root / RUN_DIRECTORY / "freezes").resolve()) or file_hash(freeze_path) != record["freeze_sha256"]:
        raise ValueError("freeze path/hash mismatch")
    frozen = checked_json(freeze_path)
    if frozen["run_id"] != record["run_id"] or frozen["banks"] != record["banks"] or frozen["stream"] != record["stream"]:
        raise ValueError("frozen scientific result mismatch")
    metrics, groups = record["policy_metrics"], record["group_approval"]
    if len(metrics) != 2 or {m["policy"] for m in metrics} != set(pilot.POLICIES) or len(groups) != 4 or {(g["policy"], g["B"]) for g in groups} != {(p, b) for p in pilot.POLICIES for b in (0, 1)}:
        raise ValueError("incomplete/duplicate cell coverage")
    indexed = {}
    for m in metrics:
        p = m["policy"]
        model = record["banks"][p]["model"]
        expected_terms = ["const", "Y", "D", "C", "B"] if p == "group_aware" else ["const", "Y", "D", "C"]
        if model["design_order"] != expected_terms or model["feature_order"] != expected_terms[1:] or set(model["coefficients"]) != set(expected_terms) or not np.isfinite(list(model["coefficients"].values())).all() or model["training_count"] != identity["configuration"]["training_rows"] or model["training_ID_fingerprint"] != identity["training_ID_fingerprint"]:
            raise ValueError("model allowlist/cohort differs")
        numeric = [m[k] for k in ("true_expected_portfolio_profit", "raw_PN_RMSE", "raw_true_PD_MAE", "bounded_true_PD_RMSE", "funded_principal", "approval_gap_pp")]
        if not np.isfinite(numeric).all() or m["maximum_budget_violation_USD"] != 0 or not m["independent_exact_enumeration_passed"] or m["funded_principal"] > m["available_bank_capital"]:
            raise ValueError("invalid finite/feasibility metrics")
        if m["optimal_cutoff"] != record["banks"][p]["selection"]["optimal_tau"] or m["available_bank_capital"] != identity["fixed_budget"]:
            raise ValueError("cutoff/budget changed")
        g = {v["B"]: v for v in groups if v["policy"] == p}
        if sum(v["n_applicants"] for v in g.values()) != m["n_applicants"] or sum(v["number_approved"] for v in g.values()) != m["number_approved"]:
            raise ValueError("group/count inconsistency")
        for v in g.values():
            if v["approval_rate"] != v["number_approved"] / v["n_applicants"]:
                raise ValueError("group rate arithmetic mismatch")
        if not np.isclose(m["approval_gap_pp"], 100 * (g[1]["approval_rate"] - g[0]["approval_rate"]), rtol=0, atol=1e-12):
            raise ValueError("gap arithmetic mismatch")
        if m["number_approved"] == 0 and (not m["reject_all_selected"] or m["true_expected_portfolio_profit"] != 0 or m["optimal_cutoff"] is not None):
            raise ValueError("reject-all incorrectly represented")
        if m["signed_oracle_minus_policy_true_profit"] != record["oracle_reference"]["true_expected_portfolio_profit"] - m["true_expected_portfolio_profit"]:
            raise ValueError("signed oracle difference incorrect")
        indexed[p] = m
    a, b = (indexed[p] for p in pilot.POLICIES)
    if record["comparison"]["group_blind_minus_aware_true_profit_USD"] != b["true_expected_portfolio_profit"] - a["true_expected_portfolio_profit"]:
        raise ValueError("paired profit arithmetic incorrect")
    if not np.isclose(record["comparison"]["group_blind_minus_aware_approval_gap_pp"], b["approval_gap_pp"] - a["approval_gap_pp"], rtol=0, atol=1e-12):
        raise ValueError("paired gap arithmetic incorrect")
    d = record["decomposition"]
    if not d["valid"] or not np.isclose(d["beta_B_total"], d["beta_B_signal"] + d["beta_B_noise"], atol=full.DECOMPOSITION_ATOL, rtol=full.DECOMPOSITION_RTOL) or d["beta_B_total"] != record["banks"]["group_aware"]["model"]["coefficients"]["B"]:
        raise ValueError("coefficient decomposition invalid")
    reference_path = root / RUN_DIRECTORY / "signal_reference_EVALUATOR_ONLY.json"
    if file_hash(reference_path) != record["signal_reference_sha256"] or d["beta_B_signal"] != checked_json(reference_path)["reference"]["beta_B_signal"]:
        raise ValueError("signal-reference identity differs")


def load_record(index, manifest, root=PROJECT_ROOT):
    record = checked_json(result_path(index, root))
    validate_record(record, index, manifest, root)
    return record


def save_record(record, manifest, root=PROJECT_ROOT):
    validate_record(record, record["replication"], manifest, root)
    verify_originals(manifest, root)
    publish(result_path(record["replication"], root), record)


def attempt_path(index, attempt_id, root=PROJECT_ROOT):
    return root / RUN_DIRECTORY / "attempts" / f"replication_{index:03d}_{attempt_id}.json"


def run_one(index, observed, truth, contracts, budget, reference, train_mask, B_row, manifest, root=PROJECT_ROOT):
    start, attempt_id, stage = time.perf_counter(), uuid.uuid4().hex, "generation"
    origin, reason = "new_noise_fit", ""
    try:
        targets, eta, stream = full.draw_targets(truth, index)
        old, reason = pilot_candidate(index, manifest, root)
        stage = "fitting_or_pilot_replay"
        if old is not None:
            if old["stream"] != stream:
                raise ValueError("first-five pilot stream/PN mismatch")
            banks = replay_pilot(old, observed, contracts, budget)
            origin = "validated_pilot_reuse"
        else:
            banks = pilot.fit_bank_policies(observed, targets, contracts, budget)
        stage = "freeze"
        freeze_path = freeze(banks, index, stream, attempt_id, manifest, root)
        stage = "evaluation"
        metrics, groups, comparison, oracle = evaluate(banks, freeze_path, observed, targets, contracts, budget, manifest, root)
        rows = lambda f: f.astype(object).where(f.notna(), None).to_dict("records")
        if old is not None and (rows(metrics) != old["policy_metrics"] or rows(groups) != old["group_approval"] or comparison != old["comparison"] or oracle != old["oracle_reference"]):
            raise ValueError("first-five evaluated scientific outputs differ from pilot")
        stage = "decomposition"
        decomp = full.decomposition(banks["group_aware"]["model"]["coefficients"]["B"], eta, train_mask, B_row, reference)
        policy_rows = rows(metrics)
        for m in policy_rows:
            m["negative_raw_share"] = m["negative_raw_count"] / m["n_applicants"]
        record = {"schema_version": "professor-noise-full-v1", "status": "success", "replication": index, "run_id": run_id(index, manifest),
                  "experiment_fingerprint": manifest["experiment_fingerprint"], "stream": stream, "origin": origin, "pilot_reuse_reason": reason,
                  "banks": pilot.bank_freeze_payload(banks), "policy_metrics": policy_rows, "group_approval": rows(groups), "comparison": comparison,
                  "decomposition": decomp, "oracle_reference": oracle, "runtime_seconds": time.perf_counter() - start,
                  "pilot_original_runtime_seconds": old["runtime_seconds"] if old is not None else None,
                  "freeze_path": str(freeze_path.relative_to(root)), "freeze_sha256": file_hash(freeze_path),
                  "signal_reference_sha256": file_hash(root / RUN_DIRECTORY / "signal_reference_EVALUATOR_ONLY.json"),
                  **{k: manifest["identity"][k] for k in ("scientific_config_fingerprint", "source_fingerprint", "solver_settings_fingerprint", "dependencies")}}
        stage = "validation_and_publication"
        save_record(record, manifest, root)
        publish(attempt_path(index, attempt_id, root), {"replication": index, "run_id": record["run_id"], "status": "success", "origin": origin,
                                                       "runtime_seconds": time.perf_counter() - start, "experiment_fingerprint": manifest["experiment_fingerprint"]})
        return record
    except BaseException as exc:
        publish(attempt_path(index, attempt_id, root), {"replication": index, "run_id": run_id(index, manifest), "status": "failed", "stage": stage,
                                                       "runtime_seconds": time.perf_counter() - start, "experiment_fingerprint": manifest["experiment_fingerprint"],
                                                       "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()})
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        return None


def resume_record(index, manifest, root=PROJECT_ROOT):
    path = result_path(index, root)
    if not path.exists():
        return None
    try:
        return load_record(index, manifest, root)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        # Preserve invalid evidence instead of overwriting or counting it.
        target = root / RUN_DIRECTORY / "invalid_records" / f"{path.stem}_{uuid.uuid4().hex}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        path.rename(target)
        publish(attempt_path(index, uuid.uuid4().hex, root), {"replication": index, "run_id": run_id(index, manifest), "status": "invalid_saved_record",
                "runtime_seconds": 0., "reason": str(exc), "preserved_path": str(target.relative_to(root)), "experiment_fingerprint": manifest["experiment_fingerprint"]})
        print(f"Replication {index}: invalid saved record preserved at {target}; rerun required", flush=True)
        return None


def execution_counts(manifest, root=PROJECT_ROOT):
    attempts = [checked_json(p) for p in sorted((root / RUN_DIRECTORY / "attempts").glob("*.json"))]
    return {"attempt_records": len(attempts), "unique_replications_attempted": len({a["replication"] for a in attempts}),
            "failed_attempts": sum(a["status"] == "failed" for a in attempts), "invalid_saved_records": sum(a["status"] == "invalid_saved_record" for a in attempts),
            "cumulative_attempt_runtime_seconds": sum(a["runtime_seconds"] for a in attempts)}


def figures(tables, original_beta, root=PROJECT_ROOT):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory = root / FIGURE_DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    paired, n = tables["paired"], len(tables["paired"])
    def finish(fig, filename, caption):
        fig.text(.5, .015, caption, ha="center", fontsize=8, color=".35")
        fig.savefig(directory / filename, dpi=180, bbox_inches="tight")
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    ax.hist(paired.beta_B_hat, bins=25, color="#243e59", edgecolor="white")
    ax.axvline(0, color=".3", linewidth=1, label="Zero")
    ax.axvline(original_beta, color="#a65e39", linestyle="--", label="Original group-aware coefficient")
    ax.set(xlabel="Fitted B coefficient (raw PN units)", ylabel="Replications", title=f"Noise-only distribution of B coefficients (R={n})")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    finish(fig, "beta_B_distribution.png", "One fixed population; varying Gaussian measurement noise only.")
    fig, axes = plt.subplots(2, 1, figsize=(8, 6.1), sharex=True, layout="constrained")
    bins = np.histogram_bin_edges(np.r_[paired.aware_gap_pp, paired.blind_gap_pp, -full.ORIGINAL_ABS_GAP_PP], bins=35)
    for ax, key, label, color in zip(axes, ("aware_gap_pp", "blind_gap_pp"), ("Group-aware", "Group-blind"), ("#243e59", "#288679")):
        ax.hist(paired[key], bins=bins, color=color, edgecolor="white")
        ax.axvline(0, color=".3", linewidth=.8)
        ax.axvline(-full.ORIGINAL_ABS_GAP_PP, color="#a65e39", linestyle="--", label="Original aware gap: −14.385705 pp")
        ax.set(ylabel="Replications", title=f"{label} (R={n})")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].set_xlabel("B=1 minus B=0 approval gap (percentage points)")
    finish(fig, "approval_gap_distributions.png", "Empirical noise-realization spreads; includes reject-all outcomes.")
    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    ax.hist(paired.abs_gap_change_pp, bins=30, color="#288679", edgecolor="white")
    ax.axvline(0, color=".3", linestyle="--")
    ax.set(xlabel="|Blind gap| − |Aware gap| (percentage points)", ylabel="Replications", title=f"Change in absolute approval disparity (R={n})")
    ax.spines[["top", "right"]].set_visible(False)
    finish(fig, "absolute_gap_changes.png", "Negative: blind has smaller absolute gap. Zero includes equal-gap/reject-all pairs.")
    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    values = paired.true_profit_change_USD.to_numpy() / 1e6
    if np.ptp(values) > 0:
        edges = np.sinh(np.linspace(np.arcsinh(values.min()), np.arcsinh(values.max()), 35))
        edges[0], edges[-1] = np.nextafter(values.min(), -np.inf), np.nextafter(values.max(), np.inf)
        ax.hist(values, bins=edges, color="#243e59", edgecolor="white", linewidth=.5)
        ax.set_xscale("symlog", linthresh=1)
    else:
        ax.hist(values, bins=1, color="#243e59")
    ax.axvline(0, color=".3", linestyle="--")
    ax.set(xlabel="Blind − aware true expected profit (USD millions; symlog scale)", ylabel="Replications", title=f"Economic consequences of excluding B (R={n})")
    ax.spines[["top", "right"]].set_visible(False)
    finish(fig, "true_profit_changes.png", "Full range retained; nonlinear x-axis resolves both central changes and large tail outcomes.")


def summarize(manifest, root=PROJECT_ROOT):
    """Strictly read-only with respect to experiments: no draws, fits, or searches."""
    verify_originals(manifest, root)
    records, missing = [], []
    for index in range(200):
        try:
            records.append(load_record(index, manifest, root))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            missing.append({"replication": index, "reason": str(exc)})
    counts = execution_counts(manifest, root)
    if missing:
        status = {"status": "incomplete", "n_successful": len(records), "missing_or_invalid": missing, **counts}
        publish(root / RUN_DIRECTORY / "status_history" / f"incomplete_{uuid.uuid4().hex}.json", status)
        raise ValueError(f"Full study incomplete: {len(records)}/200 valid successes; failure history retained")
    if len({r["run_id"] for r in records}) != 200 or len({r["stream"]["eta_sha256"] for r in records}) != 200:
        raise ValueError("replication/noise identities are not unique")
    if any(r["oracle_reference"] != records[0]["oracle_reference"] for r in records):
        raise ValueError("oracle changed across replications")
    tables = full.summarize_records(records)
    output_hashes = {}
    for key, filename in (("results", "noise_full_results.csv"), ("coefficients", "noise_full_coefficient_summary.csv"),
                          ("groups", "noise_full_group_outcome_summary.csv"), ("ties", "noise_full_tie_diagnostics.csv"), ("summary", "noise_full_summary.csv")):
        path = root / TABLE_DIRECTORY / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        tables[key].to_csv(path, index=False)
        output_hashes[str(path.relative_to(root))] = file_hash(path)
    original_beta = json.loads((root / pio.legacy.METRICS / "bank_regression_model.json").read_text())["coefficients"]["B"]
    figures(tables, original_beta, root)
    for path in (root / FIGURE_DIRECTORY).glob("*.png"):
        output_hashes[str(path.relative_to(root))] = file_hash(path)
    report = {"status": "complete_200_replications", "experiment_fingerprint": manifest["experiment_fingerprint"],
              "n_successful": 200, "pilot_reused": sum(r["origin"] == "validated_pilot_reuse" for r in records),
              "new_noise_fits": sum(r["origin"] == "new_noise_fit" for r in records), **counts,
              "frequencies": tables["frequencies"], "oracle_reference": records[0]["oracle_reference"],
              "max_abs_decomposition_residual": max(abs(r["decomposition"]["residual"]) for r in records),
              "signal_reference": checked_json(root / RUN_DIRECTORY / "signal_reference_EVALUATOR_ONLY.json")["reference"],
              "original_beta_B": original_beta, "original_abs_gap_threshold_pp": full.ORIGINAL_ABS_GAP_PP,
              "original_files_preserved": len(manifest["identity"]["original_checksums"]), "outputs": output_hashes,
              "replication_result_hashes": {str(i): file_hash(result_path(i, root)) for i in range(200)},
              "limitation": "Conditional measurement-noise sensitivity on one fixed synthetic population, not real-world discrimination inference."}
    # Consolidated reports are regenerable; immutable attempts/results remain
    # the evidence. Preserve every distinct completion report as well.
    report_path = root / RUN_DIRECTORY / "reports" / f"complete_{fp(report)}.json"
    if not report_path.exists():
        publish(report_path, report)
    (root / RUN_DIRECTORY / "latest_report.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    verify_originals(manifest, root)
    return report
