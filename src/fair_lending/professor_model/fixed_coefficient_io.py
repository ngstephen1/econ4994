"""Immutable, source-versioned Step 10B cases; no fitting or noise generation."""
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

from . import fixed_coefficient as science
from . import bank_regression as aware, group_blind as blind
from . import revised_lending_io as prior
from . import noise_full_io as old_noise
from .config import PROJECT_ROOT, configuration_fingerprint as fp
from .noise_pilot import frame_fingerprint
from .noise_pilot_io import immutable_json
from .risk_io import file_hash

CONFIG = PROJECT_ROOT / "configs/professor_model/fixed_coefficient.yaml"
METRICS = Path("results/metrics/professor_model/fixed_coefficient")
TABLES = Path("results/tables/professor_model/fixed_coefficient")
DATA = Path("data/processed/professor_model/fixed_coefficient")
SYNTHETIC = Path("data/synthetic/professor_model/baseline")
PROCESSED = Path("data/processed/professor_model")
ALL_CASES = ("canonical", *(f"noise_{i:03d}" for i in range(200)))


def load_config(path=CONFIG):
    config = yaml.safe_load(Path(path).read_text())
    expected = {"schema_version": science.SCHEMA, "population_rows": 100000, "training_rows": 80000,
                "evaluation_rows": 20000, "saved_noise_replications": 200,
                "intervention": "subtract_saved_aware_B_contribution_before_clipping", "raw_replay_absolute_tolerance": 1e-15,
                "bounding": "clip_0_1_separate_column", "positive_floor": None, "noise_reconstruction": False,
                "canonical_PN_diagnostics": "saved_targets_after_freeze", "noise_PN_diagnostics": "unavailable_no_noise_replay",
                "budget_source": "results/metrics/professor_model/lending_selection.json", "expected_capital_USD": 1826087220.7466326,
                "policy_rule": "exact_common_cutoff_indivisible_ties_reject_all", "original_reference_policies_reoptimized": False,
                "execution": "sequential"}
    if config != expected: raise ValueError("Step 10B requires the frozen canonical plus 200-saved-draw specification")
    return config


def source_hashes():
    paths = set(prior.source_hashes())
    for name in ("fixed_coefficient", "fixed_coefficient_io", "group_blind", "noise_full", "noise_full_io", "noise_pilot_io"):
        paths.add(f"src/fair_lending/professor_model/{name}.py")
    paths.add("experiments/run_professor_fixed_coefficient.py")
    return {p: file_hash(PROJECT_ROOT / p) for p in sorted(paths)}


def read_json(path): return json.loads(Path(path).read_text())


def publish(path, value):
    immutable_json(path, {**value, "content_fingerprint": fp(value)})


def checked(path):
    value = read_json(path); fingerprint = value.pop("content_fingerprint", None)
    if fingerprint != fp(value): raise ValueError(f"content fingerprint invalid: {path}")
    return value


def verify_originals(root=PROJECT_ROOT):
    record = read_json(root / METRICS / "preservation_manifest.json")
    if record["schema_version"] != "professor-fixed-coefficient-preservation-v1": raise ValueError("wrong preservation schema")
    prior.verify_hashes(record["original_checksums"], root)
    index = read_json(root / "results/metrics/professor_model/professor_model_evidence_index.json")
    core = {k: v for k, v in index.items() if k != "extensions"}
    if fp(core) != record["prior_index_core_fingerprint"]: raise ValueError("Step 10A evidence-index core changed")
    return record


def prepare(root=PROJECT_ROOT, config_path=CONFIG, readonly=False):
    """Bank-safe preflight: hash old result bytes; parse only bank freezes."""
    config = load_config(config_path); preservation = verify_originals(root)
    index = read_json(root / "results/metrics/professor_model/professor_model_evidence_index.json")
    for item in index["artifacts"]: prior.verify_hashes({item["path"]: item["sha256"]}, root)
    old_manifest = read_json(root / "results/metrics/professor_model/noise_robustness_full/manifest.json")
    identity = old_manifest["identity"]
    if fp(identity) != old_manifest["experiment_fingerprint"]: raise ValueError("old noise manifest fingerprint differs")
    for key in ("source_code_hashes", "original_checksums"): prior.verify_hashes(identity[key], root)
    dependencies = {k: version(k) for k in identity["dependencies"]}
    if dependencies != identity["dependencies"] or platform.python_version() != identity["python"]:
        raise ValueError("saved-model replay dependency/Python environment differs")
    old_hashes = read_json(root / "results/metrics/professor_model/mechanism_audit/preservation_manifest.json")["checksums"]
    freezes = {}
    for path in sorted((root / old_noise.RUN_DIRECTORY / "freezes").glob("*.json")):
        relative = str(path.relative_to(root))
        prior.verify_hashes({relative: old_hashes[relative]}, root)
        value = checked(path); i = value["replication"]
        if type(i) is not int or not 0 <= i < 200 or i in freezes or value["experiment_fingerprint"] != old_manifest["experiment_fingerprint"] or value["status"] != "frozen_before_true_profit_evaluation":
            raise ValueError("invalid/duplicate old bank freeze")
        if value["run_id"] != old_noise.run_id(i, old_manifest): raise ValueError("old bank-only run identity differs")
        record_path = str(old_noise.result_path(i, root).relative_to(root))
        prior.verify_hashes({record_path: old_hashes[record_path]}, root)
        freezes[i] = {"path": relative, "sha256": file_hash(path), "record_path": record_path,
                      "record_sha256": old_hashes[record_path], "value": value}
    if set(freezes) != set(range(200)): raise ValueError("all 200 unique saved bank freezes/records required")
    observed = aware.attach_splits(pd.read_parquet(root / SYNTHETIC / "applicants_bank_view.parquet"), pd.read_csv(root / SYNTHETIC / "applicant_splits.csv"))
    if len(observed) != 100000 or observed.split.value_counts().to_dict() != {"train": 80000, "evaluation": 20000}: raise ValueError("authoritative cohort differs")
    observed = observed.loc[observed.split == "evaluation", science.OBSERVED].copy()
    contracts = pd.read_parquet(root / PROCESSED / "loan_contracts.parquet")
    science.validate_observed(observed, contracts)
    budget = read_json(root / config["budget_source"])["selection"]["available_bank_capital"]
    if budget != config["expected_capital_USD"] or budget != identity["fixed_budget"]: raise ValueError("original capital changed")
    if not contracts["T"].eq(30).all() or not contracts.theta.eq(.1).all() or not contracts.r.eq(.05).all(): raise ValueError("original contract terms differ")
    sources = source_hashes()
    run_identity = {"configuration": config, "configuration_sha256": file_hash(config_path), "source_hashes": sources,
                    "dependencies": dependencies, "python": platform.python_version(),
                    "preservation_sha256": file_hash(root / METRICS / "preservation_manifest.json"),
                    "original_input_fingerprint": fp(preservation["original_checksums"]),
                    "old_noise_experiment": old_manifest["experiment_fingerprint"], "old_freeze_record_bindings": {str(i): {k: v for k, v in f.items() if k != "value"} for i, f in freezes.items()},
                    "prior_index_core_fingerprint": preservation["prior_index_core_fingerprint"]}
    experiment = fp(run_identity); directory = root / METRICS / experiment[:16]
    manifest_path = directory / "manifest.json"
    manifest = {"schema_version": science.SCHEMA, "experiment_fingerprint": experiment, "identity": run_identity,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "git_revision": subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout.strip(),
                "git_dirty": True, "expected_cases": list(ALL_CASES)}
    if manifest_path.exists():
        manifest = checked(manifest_path)
        if manifest["identity"] != run_identity or manifest["experiment_fingerprint"] != experiment: raise ValueError("existing experiment identity differs")
    elif readonly: raise FileNotFoundError("no matching Step 10B manifest; summary-only never creates one")
    else: publish(manifest_path, manifest)
    return {"root": root, "config": config, "manifest": manifest, "directory": directory,
            "observed": observed, "contracts": contracts, "budget": budget, "old_manifest": old_manifest, "freezes": freezes}


def check_sources(ctx):
    if source_hashes() != ctx["manifest"]["identity"]["source_hashes"]: raise ValueError("implementation changed during run")


def load_banks(ctx, case_id):
    observed, contracts, root = ctx["observed"], ctx["contracts"], ctx["root"]
    if case_id not in ALL_CASES: raise ValueError("unknown case")
    if case_id == "canonical":
        payloads = {}
        for policy, model_file, pred_file, profit_file, dec_file, sel_file in (
            (science.W, "bank_regression_model.json", "bank_risk_estimates.parquet", "bank_expected_profit.parquet", "lending_decisions.parquet", "lending_selection.json"),
            (science.F, "group_blind_regression_model.json", "bank_risk_estimates_group_blind.parquet", "bank_expected_profit_group_blind.parquet", "lending_decisions_group_blind.parquet", "group_blind_lending_selection.json")):
            pred = pd.read_parquet(root / SYNTHETIC / pred_file).loc[lambda f: f.split.eq("evaluation")].reset_index(drop=True)
            payloads[policy] = {"model": read_json(root / "results/metrics/professor_model" / model_file),
                                "selection": read_json(root / "results/metrics/professor_model" / sel_file)["selection"],
                                "predictions_fingerprint": frame_fingerprint(pred),
                                "profit_fingerprint": frame_fingerprint(pd.read_parquet(root / PROCESSED / profit_file)),
                                "decisions_fingerprint": frame_fingerprint(pd.read_parquet(root / PROCESSED / dec_file))}
        provenance = {"kind": "original_canonical_realization", "population_seed": 4994, "noise_spawn_key": [5]}
    else:
        i = int(case_id.split("_")[1]); entry = ctx["freezes"][i]
        if file_hash(root / entry["path"]) != entry["sha256"] or file_hash(root / entry["record_path"]) != entry["record_sha256"]: raise ValueError("saved draw evidence changed")
        frozen = checked(root / entry["path"])
        payloads = {science.W: frozen["banks"]["group_aware"], science.F: frozen["banks"]["group_blind"]}
        provenance = {"kind": "saved_noise_draw", "replication": i, "stream": frozen["stream"],
                      "source_freeze_path": entry["path"], "source_freeze_sha256": entry["sha256"],
                      "source_record_path": entry["record_path"], "source_record_sha256": entry["record_sha256"]}
    banks = {policy: science.replay_reference(observed, contracts, payload, policy) for policy, payload in payloads.items()}
    for b in banks.values():
        if b["selection"]["available_bank_capital"] != ctx["budget"]: raise ValueError("case budget differs")
    pred, intervention = science.counterfactual_scores(observed, banks[science.W]["predictions"], banks[science.W]["model"])
    banks[science.Z] = science.choose_counterfactual(contracts, pred, ctx["budget"])
    banks[science.Z]["intervention"] = intervention
    return banks, provenance


def bank_payload(banks):
    return {p: {"selection": b["selection"], "source_model": b.get("model"), "intervention": b.get("intervention"),
                **{f"{name}_fingerprint": frame_fingerprint(b[name]) for name in ("predictions", "profit", "decisions")},
                "candidate_fingerprint": frame_fingerprint(b["candidates"]) if p == science.Z else None}
            for p, b in banks.items()}


def freeze_case(ctx, case_id, banks, provenance, attempt):
    check_sources(ctx); artifacts = {}
    if case_id == "canonical":
        data = ctx["root"] / DATA / ctx["manifest"]["experiment_fingerprint"][:16] / attempt
        for name in ("predictions", "profit", "decisions", "candidates"):
            frame = banks[science.Z][name]; data.mkdir(parents=True, exist_ok=True)
            for suffix in (["csv", "parquet"] if name != "candidates" else ["csv"]):
                path = data / f"counterfactual_{name}.{suffix}"
                if suffix == "csv": prior.write_csv(path, frame)
                else:
                    with path.open("xb") as stream: frame.to_parquet(stream, index=False)
                    pd.testing.assert_frame_equal(pd.read_parquet(path), frame, check_exact=True)
                artifacts[str(path.relative_to(ctx["root"]))] = file_hash(path)
    value = {"schema_version": science.SCHEMA, "status": "bank_frozen_before_evaluation", "case_id": case_id,
             "experiment_fingerprint": ctx["manifest"]["experiment_fingerprint"], "provenance": provenance,
             "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "banks": bank_payload(banks), "artifacts": artifacts}
    path = ctx["directory"] / "freezes" / f"{case_id}_{attempt}.json"; publish(path, value)
    return path


def load_evaluator(ctx, case_id, banks, freeze_path):
    check_sources(ctx); frozen = checked(freeze_path)
    if frozen["status"] != "bank_frozen_before_evaluation" or frozen["case_id"] != case_id or frozen["experiment_fingerprint"] != ctx["manifest"]["experiment_fingerprint"] or frozen["banks"] != bank_payload(banks): raise ValueError("bank freeze does not match current case")
    prior.verify_hashes(frozen["artifacts"], ctx["root"])
    truth = pd.read_parquet(ctx["root"] / PROCESSED / "true_expected_profit.parquet")
    targets = old = None
    if case_id == "canonical":
        targets = pd.read_parquet(ctx["root"] / SYNTHETIC / "training_signals.parquet")
        targets = targets.loc[targets.applicant_id.isin(ctx["observed"].applicant_id)].copy()
    else:
        old = old_noise.load_record(int(case_id.split("_")[1]), ctx["old_manifest"], ctx["root"])
        entry = ctx["freezes"][old["replication"]]
        if old["freeze_path"] != entry["path"] or old["freeze_sha256"] != entry["sha256"]: raise ValueError("old record/bank-freeze binding differs")
    return truth, targets, old


def rows(frame): return frame.astype(object).where(frame.notna(), None).to_dict("records")


def verify_baseline_results(result, old):
    if old is None: return
    table = result["policies"].set_index("policy")
    for policy, original in ((science.W, "group_aware"), (science.F, "group_blind")):
        m = next(m for m in old["policy_metrics"] if m["policy"] == original); r = table.loc[policy]
        for new, previous in (("number_approved", "number_approved"), ("true_profit_USD", "true_expected_portfolio_profit"),
                              ("funded_principal_USD", "funded_principal"), ("perceived_profit_USD", "perceived_portfolio_profit")):
            if r[new] != m[previous]: raise ValueError(f"frozen baseline metric differs: {policy}/{new}")
        for key in ("approval_gap_pp", "raw_true_PD_MAE", "bounded_true_PD_MAE"):
            if not np.isclose(r[key], m[key], atol=1e-15, rtol=0): raise ValueError(f"baseline metric differs: {key}")


def validate_record(ctx, record):
    if record["schema_version"] != science.SCHEMA or record["status"] != "success" or record["case_id"] not in ALL_CASES or record["experiment_fingerprint"] != ctx["manifest"]["experiment_fingerprint"]: raise ValueError("invalid case result identity")
    path = ctx["root"] / record["freeze_path"]
    if not path.resolve().is_relative_to(ctx["directory"].resolve()) or file_hash(path) != record["freeze_sha256"]: raise ValueError("case freeze path/hash differs")
    frozen = checked(path)
    if frozen["status"] != "bank_frozen_before_evaluation" or frozen["case_id"] != record["case_id"] or frozen["experiment_fingerprint"] != record["experiment_fingerprint"] or frozen["provenance"] != record["provenance"]: raise ValueError("result/freeze binding differs")
    if any(record[k] != 0 for k in ("new_fits", "new_noise_draws", "reference_cutoff_searches")) or not math.isfinite(record["runtime_seconds"]) or record["runtime_seconds"] < 0: raise ValueError("invalid no-fit/no-draw contract or runtime")
    intervention = frozen["banks"][science.Z]["intervention"]
    coefficients = dict(frozen["banks"][science.W]["source_model"]["coefficients"])
    if intervention["source_coefficients"] != coefficients: raise ValueError("wrong source coefficients")
    coefficients["B"] = 0.
    if intervention["coefficients_used"] != coefficients or intervention["positive_floor"] is not None: raise ValueError("intervention changed more than B")
    prior.verify_hashes(frozen["artifacts"], ctx["root"])
    sizes = {"policies": 3, "groups": 6, "paired": 3, "overlap": 12}
    for name, n in sizes.items():
        if len(record[name]) != n or any(r["case_id"] != record["case_id"] for r in record[name]): raise ValueError("incomplete case cells")
    table = pd.DataFrame(record["policies"])
    if set(table.policy) != set(science.POLICIES) or table.policy.duplicated().any(): raise ValueError("duplicate policy cells")
    groups = pd.DataFrame(record["groups"])
    if set(zip(groups.policy, groups.B)) != {(p, b) for p in science.POLICIES for b in (0, 1)}: raise ValueError("group coverage differs")
    pairs = pd.DataFrame(record["paired"])
    if set(zip(pairs.policy_first, pairs.policy_second)) != set(science.PAIRS): raise ValueError("paired coverage differs")
    overlap = pd.DataFrame(record["overlap"])
    if overlap.duplicated(["policy_first", "policy_second", "allocation_category"]).any() or set(zip(overlap.policy_first, overlap.policy_second, overlap.allocation_category)) != {(a,b,c) for a,b in science.PAIRS for c in ("approved_by_both", "first_only", "second_only", "rejected_by_both")}: raise ValueError("overlap coverage differs")
    for r in table.to_dict("records"):
        if not all(math.isfinite(r[k]) for k in ("true_profit_USD", "perceived_profit_USD", "funded_principal_USD", "approval_gap_pp", "raw_true_PD_MAE")) or r["funded_principal_USD"] > ctx["budget"]: raise ValueError("invalid finite/feasibility metric")
        selection = frozen["banks"][r["policy"]]["selection"]
        if r["number_approved"] != selection["number_approved"] or r["funded_principal_USD"] != selection["total_funded_principal"] or r["perceived_profit_USD"] != selection["bank_expected_portfolio_profit"]: raise ValueError("reported outcome differs from frozen bank")
        g = groups.loc[groups.policy == r["policy"]].set_index("B")
        if int(g.n_applicants.sum()) != r["n_applicants"] or int(g.number_approved.sum()) != r["number_approved"]: raise ValueError("group counts differ")
        if any(v.approval_rate != v.number_approved/v.n_applicants or not 0 <= v.number_approved <= v.n_applicants for v in g.itertuples()): raise ValueError("group rate arithmetic differs")
        if not np.isclose(math.fsum(g.true_profit_USD),r["true_profit_USD"],rtol=1e-15,atol=1e-6): raise ValueError("group profit sum differs")
        if not np.isclose(r["approval_gap_pp"], 100*(g.loc[1,"approval_rate"]-g.loc[0,"approval_rate"]),atol=1e-12,rtol=0): raise ValueError("group gap differs")
    indexed = table.set_index("policy")
    for r in record["paired"]:
        if r["true_profit_difference_USD"] != indexed.loc[r["policy_first"],"true_profit_USD"] - indexed.loc[r["policy_second"],"true_profit_USD"]: raise ValueError("paired profit difference differs")
    return record


def find_success(ctx, case_id):
    valid = []
    for path in sorted((ctx["directory"] / "records").glob(f"{case_id}_*.json")):
        try: valid.append(validate_record(ctx, checked(path)))
        except (ValueError, KeyError, TypeError, OSError, AssertionError): continue  # retained invalid evidence; resume retries
    if len(valid) > 1: raise ValueError("duplicate successful records for case")
    return valid[0] if valid else None


def run_case(ctx, case_id, resume=False):
    saved = find_success(ctx, case_id)
    if saved is not None:
        if not resume: raise FileExistsError("case exists; use --resume")
        print(f"{case_id}: resumed verified success", flush=True); return saved
    attempt = uuid.uuid4().hex; stage = "bank_replay"; start = time.perf_counter()
    try:
        print(f"{case_id}: replay frozen W/F; construct and optimize Z", flush=True)
        banks, provenance = load_banks(ctx, case_id)
        stage = "freeze"; freeze = freeze_case(ctx, case_id, banks, provenance, attempt)
        print(f"{case_id}: Z frozen; evaluator stage", flush=True)
        stage = "evaluation"; truth, targets, old = load_evaluator(ctx, case_id, banks, freeze)
        result = science.evaluate_case(banks, ctx["observed"], ctx["contracts"], truth, case_id, targets)
        verify_baseline_results(result, old)
        record = {"schema_version": science.SCHEMA, "status": "success", "case_id": case_id,
                  "experiment_fingerprint": ctx["manifest"]["experiment_fingerprint"],
                  "provenance": provenance, "freeze_path": str(freeze.relative_to(ctx["root"])), "freeze_sha256": file_hash(freeze),
                  "runtime_seconds": time.perf_counter()-start, "new_fits": 0, "new_noise_draws": 0,
                  "reference_cutoff_searches": 0, **{k: rows(v) for k,v in result.items()}}
        stage = "validation"; validate_record(ctx, record)
        publish(ctx["directory"] / "records" / f"{case_id}_{attempt}.json", record)
        publish(ctx["directory"] / "attempts" / f"{case_id}_{attempt}.json", {"status": "success", "case_id": case_id, "runtime_seconds": time.perf_counter()-start})
        print(f"{case_id}: success in {record['runtime_seconds']:.2f}s", flush=True)
        return record
    except Exception as exc:
        publish(ctx["directory"] / "attempts" / f"{case_id}_{attempt}.json", {"status": "failed", "case_id": case_id,
                "stage": stage, "exception": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(), "runtime_seconds": time.perf_counter()-start})
        print(f"{case_id}: FAILED {stage}: {exc}", flush=True)
        return None


def summarize(ctx, write=True):
    """Read validated compact records only. Cannot generate, score or optimize."""
    check_sources(ctx); records = [r for case in ALL_CASES if (r := find_success(ctx, case)) is not None]
    tables = ctx["root"] / TABLES / ctx["manifest"]["experiment_fingerprint"][:16]
    if write: tables.mkdir(parents=True, exist_ok=True)
    outputs = {}; noise = [r for r in records if r["case_id"] != "canonical"]
    for kind, selected in (("canonical", [r for r in records if r["case_id"] == "canonical"]), ("noise", noise)):
        if not selected: continue
        for name in ("policies", "groups", "paired", "overlap"):
            frame = pd.DataFrame([row for record in selected for row in record[name]]).sort_values(["case_id", *( {"policies":["policy"],"groups":["policy","B"],"paired":["comparison"],"overlap":["policy_first","policy_second","allocation_category"]}[name])]).reset_index(drop=True)
            path = tables / f"{kind}_{name}.csv"
            # Consolidated views are deterministic and replaceable; immutable case evidence remains untouched.
            if write:
                temporary = path.with_name(path.name + ".pending")
                if temporary.exists(): temporary.unlink()
                prior.write_csv(temporary, frame); temporary.replace(path)
            if path.exists(): outputs[str(path.relative_to(ctx["root"]))] = file_hash(path)
    if noise:
        frame = science.summarize_noise(pd.DataFrame([row for record in noise for row in record["paired"]]))
        path = tables / "noise_summary.csv"
        if write:
            temporary = path.with_name(path.name+".pending")
            if temporary.exists(): temporary.unlink()
            prior.write_csv(temporary, frame); temporary.replace(path)
        if path.exists(): outputs[str(path.relative_to(ctx["root"]))] = file_hash(path)
    complete = len(records) == 201
    if complete:
        paired = pd.DataFrame([row for r in noise for row in r["paired"]]); historical = paired.loc[paired.policy_first.eq(science.F) & paired.policy_second.eq(science.W), "true_profit_difference_USD"]
        if int((historical < 0).sum()) != 47: raise ValueError("historical 47/200 blind disadvantages not reproduced")
    verify_originals(ctx["root"])
    report = {"schema_version": science.SCHEMA, "status": "complete_201_cases" if complete else "incomplete",
              "experiment_fingerprint": ctx["manifest"]["experiment_fingerprint"], "successful_cases": len(records),
              "noise_draws": len(noise), "canonical_complete": any(r["case_id"]=="canonical" for r in records),
              "missing_cases": [case for case in ALL_CASES if case not in {r["case_id"] for r in records}],
              "policy_cells": 3*len(records), "cumulative_success_runtime_seconds": math.fsum(r["runtime_seconds"] for r in records),
              "failed_attempts": sum(checked(p)["status"]=="failed" for p in (ctx["directory"] / "attempts").glob("*.json")),
              "new_fits": 0, "new_noise_draws": 0, "reference_cutoff_searches": 0,
              "original_files_preserved": len(read_json(ctx["root"] / METRICS / "preservation_manifest.json")["original_checksums"]),
              "outputs": outputs, "scientific_suite_rerun": False}
    if write:
        path = ctx["directory"] / "latest_report.json"; temporary = path.with_name("latest_report.pending")
        if temporary.exists(): temporary.unlink()
        publish(temporary, report); temporary.replace(path)
    return report
