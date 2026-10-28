"""Six-case raw-ordering diagnostic: replay, freeze bank decisions, then evaluate.

Historical Step 10B records are hashed but not parsed until the new freeze.
Reference cutoffs are applied, never searched. Only new raw policies optimize.
"""
from __future__ import annotations

from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
from pathlib import Path
import subprocess
import time
import traceback
import uuid

import numpy as np
import pandas as pd
import yaml

from . import raw_ordering as science
from . import fixed_coefficient as fixed, fixed_coefficient_io as base
from .config import PROJECT_ROOT, configuration_fingerprint as fp
from .noise_pilot import frame_fingerprint
from .risk_io import file_hash

CONFIG = PROJECT_ROOT / "configs/professor_model/raw_ordering.yaml"
METRICS = Path("results/metrics/professor_model/raw_ordering")
TABLES = Path("results/tables/professor_model/raw_ordering")
DATA = Path("data/processed/professor_model/raw_ordering")
FIGURES = Path("results/figures/professor_model/raw_ordering")
SOURCE_EXPERIMENT = "cd7f3ee268cfc4d04ab906f0dbb6caab0de730aa21fdd96743c7eae815efebc7"


def load_config(path=CONFIG):
    config = yaml.safe_load(Path(path).read_text())
    expected = {"schema_version": science.SCHEMA, "study": "selected_mechanism_diagnostics_not_random_uncertainty_sample",
        "cases": list(science.CASES), "source_experiment": SOURCE_EXPERIMENT, "source_policies": list(science.POLICIES),
        "ordering_score": "frozen_raw_score", "accounting_PD": "unchanged_clip_raw_0_1", "capital_USD": 1826087220.7466326,
        "full_requests": True, "exact_raw_ties": "indivisible", "reject_all": "separate_action",
        "objective": "unchanged_perceived_expected_profit", "objective_tie_rule": "reject_all_first_then_lowest_raw_cutoff",
        "independent_audit": "Fraction_every_raw_endpoint", "refit": False, "new_noise": False,
        "baseline_reoptimization": False, "persist_applicant_rows": "canonical_new_policies_only", "full_200_draw_extension": False}
    if config != expected:
        raise ValueError("only the declared six-case raw-ordering diagnostic is supported")
    return config


def verify_originals(root):
    saved = base.read_json(root / METRICS / "preservation_manifest.json")
    if saved["schema_version"] != "professor-raw-ordering-preservation-v1":
        raise ValueError("wrong preservation schema")
    base.prior.verify_hashes(saved["original_checksums"], root)
    index = base.read_json(root / "results/metrics/professor_model/professor_model_evidence_index.json")
    if fp({k: v for k, v in index.items() if k != "extensions"}) != saved["prior_index_core_fingerprint"]:
        raise ValueError("historical evidence index core changed")
    if any(fp(index["extensions"][k]) != digest for k, digest in saved["prior_index_extensions"].items()):
        raise ValueError("historical evidence index extension changed")
    return saved


def source_hashes(root):
    names = set(base.source_hashes())
    names.update(f"src/fair_lending/professor_model/{n}.py" for n in ("raw_ordering", "raw_ordering_io", "raw_ordering_figures"))
    names.update(("experiments/run_professor_raw_ordering.py", "tests/test_professor_raw_ordering.py"))
    return {name: file_hash(root / name) for name in sorted(names)}


def prepare(root=PROJECT_ROOT, config_path=CONFIG, readonly=False):
    config = load_config(config_path); originals = verify_originals(root)
    old = base.prepare(root=root, readonly=True)
    if old["manifest"]["experiment_fingerprint"] != SOURCE_EXPERIMENT:
        raise ValueError("frozen Step 10B experiment differs")
    validation_path = old["directory"] / "final_validation.json"
    validation = base.read_json(validation_path)
    base.prior.verify_hashes(validation["artifact_hashes"], root)
    sources = {}
    for case in science.CASES:
        freezes = list((old["directory"] / "freezes").glob(f"{case}_*.json"))
        records = list((old["directory"] / "records").glob(f"{case}_*.json"))
        if len(freezes) != 1 or len(records) != 1:
            raise ValueError("one frozen successful source case required")
        frozen = base.checked(freezes[0])
        if frozen["case_id"] != case or frozen["experiment_fingerprint"] != SOURCE_EXPERIMENT or frozen["status"] != "bank_frozen_before_evaluation":
            raise ValueError("source bank freeze identity differs")
        # Record bytes stay opaque here. Their registered hashes bind the later
        # evaluator release to the exact historical source, including Z.
        sources[case] = {"freeze_path": str(freezes[0].relative_to(root)), "freeze_sha256": file_hash(freezes[0]),
                         "record_path": str(records[0].relative_to(root)), "record_sha256": file_hash(records[0])}
        base.prior.verify_hashes({p: validation["artifact_hashes"][p] for p in (sources[case]["freeze_path"], sources[case]["record_path"])}, root)
    identity = {"configuration": config, "config_sha256": file_hash(config_path), "source_hashes": source_hashes(root),
        "source_experiment": SOURCE_EXPERIMENT, "source_bindings": sources,
        "source_validation_sha256": file_hash(validation_path), "preservation_sha256": file_hash(root / METRICS / "preservation_manifest.json"),
        "dependencies": {**old["manifest"]["identity"]["dependencies"], "matplotlib": version("matplotlib")},
        "python": old["manifest"]["identity"]["python"]}
    digest = fp(identity); directory = root / METRICS / digest[:16]
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = base.checked(manifest_path)
        if manifest["identity"] != identity or manifest["experiment_fingerprint"] != digest:
            raise ValueError("raw-ordering identity mismatch")
    elif readonly:
        raise FileNotFoundError("no matching raw-ordering manifest; summary-only creates nothing")
    else:
        value = {"schema_version": science.SCHEMA, "experiment_fingerprint": digest, "identity": identity,
                 "expected_cases": list(science.CASES), "created_at_utc": datetime.now(timezone.utc).isoformat(),
                 "git_revision": subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip(),
                 "git_dirty": bool(subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True).stdout)}
        base.publish(manifest_path, value); manifest = base.checked(manifest_path)
    snapshot = {**identity["source_hashes"], "configs/professor_model/raw_ordering.yaml": identity["config_sha256"]}
    for name, digest in snapshot.items():
        path = directory / "source_snapshot" / name
        if path.exists():
            if file_hash(path) != digest: raise ValueError("archived source content differs")
        elif readonly:
            raise FileNotFoundError("missing archived source; summary-only creates nothing")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            original = Path(config_path) if name == "configs/professor_model/raw_ordering.yaml" else root / name
            with path.open("xb") as stream: stream.write(original.read_bytes())
            if file_hash(path) != digest: raise ValueError("source changed while archiving")
    return {"root": root, "config": config, "old": old, "manifest": manifest, "directory": directory,
            "observed": old["observed"], "contracts": old["contracts"], "budget": old["budget"], "protected_count": len(originals["original_checksums"])}


def check_sources(ctx):
    if source_hashes(ctx["root"]) != ctx["manifest"]["identity"]["source_hashes"]:
        raise ValueError("scientific implementation changed during run")


def replay_baselines(ctx, case):
    """Replay all three frozen policies without calling any old cutoff search."""
    entry = ctx["manifest"]["identity"]["source_bindings"][case]; root = ctx["root"]
    base.prior.verify_hashes({entry["freeze_path"]: entry["freeze_sha256"], entry["record_path"]: entry["record_sha256"]}, root)
    saved = base.checked(root / entry["freeze_path"]); refs = {}
    for p in (fixed.W, fixed.F):
        b = saved["banks"][p]
        payload = {**b, "model": b["source_model"]}
        refs[p] = fixed.replay_reference(ctx["observed"], ctx["contracts"], payload, p)
    pred, intervention = fixed.counterfactual_scores(ctx["observed"], refs[fixed.W]["predictions"], refs[fixed.W]["model"])
    if intervention != saved["banks"][fixed.Z]["intervention"]:
        raise ValueError("frozen coefficient intervention changed")
    # This helper creates bank assessments only; choose_counterfactual is never
    # called. The Z cutoff, just like W/F cutoffs, is read from its saved freeze.
    _, profit = fixed.optimize_counterfactual(ctx["contracts"], pred)
    profit["model_id"] = fixed.MODEL_ID
    selection = saved["banks"][fixed.Z]["selection"]
    dec = fixed.reconstruct_decisions(pred, ctx["contracts"], selection, fixed.Z)
    refs[fixed.Z] = {"predictions": pred, "profit": profit, "decisions": dec, "selection": selection}
    for p, bank in refs.items():
        for name in ("predictions", "profit", "decisions"):
            if frame_fingerprint(bank[name]) != saved["banks"][p][f"{name}_fingerprint"]:
                raise ValueError(f"frozen {p} {name} changed")
        if bank["selection"]["available_bank_capital"] != ctx["budget"]:
            raise ValueError("reference capital differs")
    return refs


def bank_case(ctx, case):
    refs = replay_baselines(ctx, case); banks = {}
    for policy in fixed.POLICIES:
        ref = refs[policy]; pred = fixed.normalized_predictions(ref, policy)
        ids = pred.applicant_id
        inputs = pred[["applicant_id"]].assign(source_policy=policy, ordering_score=pred.bank_PD_raw.to_numpy(),
            accounting_PD=pred.bank_PD_bounded.to_numpy(),
            requested_principal=ctx["contracts"].set_index("applicant_id").loc[ids, "L_i"].to_numpy(),
            expected_profit_perceived=ref["profit"].set_index("applicant_id").loc[ids, "expected_profit_bank"].to_numpy()).loc[:, science.INPUT_COLUMNS]
        print(f"{case}/{policy}: raw cutoff and independent exact audit", flush=True)
        result = science.choose_raw_cutoff(inputs, ctx["budget"])
        audit = science.audit_raw_cutoff(inputs, result, ctx["budget"])
        mapping, nesting = science.bounded_prefix_mapping(inputs, result, ref["decisions"], ref["selection"])
        banks[policy] = {"inputs": inputs, "baseline": ref, "raw": result, "mapping": mapping,
                         "diagnostics": {**audit, **nesting,
                            "zero_PD_applicants": int(inputs.accounting_PD.eq(0).sum()),
                            "zero_PD_principal_USD": math.fsum(inputs.loc[inputs.accounting_PD.eq(0), "requested_principal"]),
                            "accounting_probabilities_and_per_request_profits_unchanged": True}}
    return banks


def bank_payload(banks):
    return {p: {"baseline_fingerprints": {name: frame_fingerprint(b["baseline"][name]) for name in ("predictions", "profit", "decisions")},
                "input_fingerprint": frame_fingerprint(b["inputs"]), "mapping_fingerprint": frame_fingerprint(b["mapping"]),
                "raw_fingerprints": {name: frame_fingerprint(b["raw"][name]) for name in ("decisions", "candidates")},
                "selection": b["raw"]["selection"], "diagnostics": b["diagnostics"]} for p, b in banks.items()}


def freeze_case(ctx, case, banks, attempt):
    check_sources(ctx); artifacts = {}
    if case == "canonical":
        directory = ctx["root"] / DATA / ctx["manifest"]["experiment_fingerprint"][:16] / attempt
        directory.mkdir(parents=True, exist_ok=True)
        for name, get_frame in (("bank_inputs", lambda b: b["inputs"]), ("raw_order_decisions", lambda b: b["raw"]["decisions"]),
                                ("raw_cutoff_candidates", lambda b: b["raw"]["candidates"]), ("bounded_prefix_mapping", lambda b: b["mapping"])):
            frame = pd.concat([get_frame(b).assign(source_policy=p) for p, b in sorted(banks.items())], ignore_index=True)
            for suffix in (("csv", "parquet") if name in ("bank_inputs", "raw_order_decisions") else ("csv",)):
                path = directory / f"{name}.{suffix}"
                if suffix == "csv": base.prior.write_csv(path, frame)
                else:
                    with path.open("xb") as stream: frame.to_parquet(stream, index=False)
                    pd.testing.assert_frame_equal(pd.read_parquet(path), frame, check_exact=True)
                artifacts[str(path.relative_to(ctx["root"]))] = file_hash(path)
    frozen = {"schema_version": science.SCHEMA, "status": "bank_frozen_before_evaluation", "case_id": case,
        "experiment_fingerprint": ctx["manifest"]["experiment_fingerprint"],
        "source_binding": ctx["manifest"]["identity"]["source_bindings"][case], "banks": bank_payload(banks),
        "artifacts": artifacts, "frozen_at_utc": datetime.now(timezone.utc).isoformat()}
    path = ctx["directory"] / "freezes" / f"{case}_{attempt}.json"
    base.publish(path, frozen)
    return path


def load_evaluator(ctx, case, banks, frozen_path):
    check_sources(ctx); frozen = base.checked(frozen_path)
    if frozen["status"] != "bank_frozen_before_evaluation" or frozen["case_id"] != case or frozen["experiment_fingerprint"] != ctx["manifest"]["experiment_fingerprint"] or frozen["banks"] != bank_payload(banks) or frozen["source_binding"] != ctx["manifest"]["identity"]["source_bindings"][case]:
        raise ValueError("bank freeze does not match current experiment/case/policies")
    base.prior.verify_hashes(frozen["artifacts"], ctx["root"])
    entry = frozen["source_binding"]
    base.prior.verify_hashes({entry["record_path"]: entry["record_sha256"]}, ctx["root"])
    # First parse of any evaluator-bearing source record in this case.
    previous = base.validate_record(ctx["old"], base.checked(ctx["root"] / entry["record_path"]))
    truth_path = Path("data/processed/professor_model/true_expected_profit.parquet")
    checksum = base.read_json(ctx["root"] / METRICS / "preservation_manifest.json")["original_checksums"][str(truth_path)]
    base.prior.verify_hashes({str(truth_path): checksum}, ctx["root"])
    truth = pd.read_parquet(ctx["root"] / truth_path)
    return truth, previous


def verify_reference_metrics(result, previous):
    rows = result["policies"].set_index("policy")
    for before in previous["policies"]:
        now = rows.loc[f"{before['policy']}__{science.BOUNDED}"]
        for key in ("number_approved", "funded_principal_USD", "perceived_profit_USD", "true_profit_USD", "approval_gap_pp"):
            if now[key] != before[key]: raise ValueError(f"frozen baseline evaluator metric changed: {key}")


def validate_record(ctx, record):
    if record["schema_version"] != science.SCHEMA or record["status"] != "success" or record["case_id"] not in science.CASES or record["experiment_fingerprint"] != ctx["manifest"]["experiment_fingerprint"]:
        raise ValueError("invalid raw-ordering record identity")
    path = ctx["root"] / record["freeze_path"]
    if not path.resolve().is_relative_to(ctx["directory"].resolve()) or file_hash(path) != record["freeze_sha256"]:
        raise ValueError("raw-ordering freeze path/hash differs")
    frozen = base.checked(path); case = record["case_id"]
    if frozen["case_id"] != case or frozen["experiment_fingerprint"] != record["experiment_fingerprint"] or frozen["status"] != "bank_frozen_before_evaluation" or frozen["source_binding"] != ctx["manifest"]["identity"]["source_bindings"][case]:
        raise ValueError("raw-ordering result/freeze/source binding differs")
    base.prior.verify_hashes(frozen["artifacts"], ctx["root"])
    if any(record[k] != 0 for k in ("new_fits", "new_draws", "baseline_cutoff_searches")) or record["raw_cutoff_searches"] != 3 or not math.isfinite(record["runtime_seconds"]) or record["runtime_seconds"] < 0:
        raise ValueError("scope/runtime contract differs")
    for name, n, keys in (("policies", 6, ["policy"]), ("groups", 12, ["policy", "B"]),
                          ("comparisons", 9, ["policy_first", "policy_second"]), ("overlap", 36, ["policy_first", "policy_second", "allocation_category"])):
        f = pd.DataFrame(record[name])
        if len(f) != n or f.duplicated(keys).any() or not f.case_id.eq(case).all():
            raise ValueError(f"incomplete/duplicate {name} cells")
    pol = pd.DataFrame(record["policies"]).set_index("policy")
    expected = {f"{p}__{r}" for p in science.POLICIES for r in (science.BOUNDED, science.RAW)}
    if set(pol.index) != expected: raise ValueError("policy coverage differs")
    groups = pd.DataFrame(record["groups"])
    if set(zip(groups.policy, groups.B)) != {(p, b) for p in expected for b in (0, 1)}:
        raise ValueError("group coverage differs")
    expected_pairs = {(f"{p}__{science.RAW}", f"{p}__{science.BOUNDED}"): "within_model_ordering" for p in science.POLICIES}
    expected_pairs.update({(f"{a}__{r}", f"{b}__{r}"): "between_models_same_ordering" for r in (science.BOUNDED, science.RAW) for a,b in fixed.PAIRS})
    actual_pairs = {(x["policy_first"], x["policy_second"]):x["comparison_kind"] for x in record["comparisons"]}
    if actual_pairs != expected_pairs:
        raise ValueError("comparison coverage/kinds differ")
    expected_overlap = {(a,b,c) for a,b in expected_pairs for c in ("approved_by_both", "first_only", "second_only", "rejected_by_both")}
    if {(x["policy_first"],x["policy_second"],x["allocation_category"]) for x in record["overlap"]} != expected_overlap:
        raise ValueError("overlap coverage differs")
    for key, row in pol.iterrows():
        if not all(math.isfinite(row[k]) for k in ("funded_principal_USD", "perceived_profit_USD", "true_profit_USD", "approval_gap_pp")) or not 0 <= row.funded_principal_USD <= ctx["budget"] or row.available_capital_USD != ctx["budget"]:
            raise ValueError("invalid economic/feasibility metrics")
        g = groups[groups.policy.eq(key)].set_index("B")
        if g.n_applicants.sum() != row.n_applicants or g.number_approved.sum() != row.number_approved or row.approval_rate != row.number_approved / row.n_applicants:
            raise ValueError("group/overall counts differ")
        if any(x.approval_rate != x.number_approved/x.n_applicants or not 0 <= x.number_approved <= x.n_applicants for x in g.itertuples()):
            raise ValueError("group approval arithmetic differs")
        if not np.isclose(math.fsum(g.true_profit_USD), row.true_profit_USD, rtol=1e-15, atol=1e-6) or not np.isclose(100*(g.loc[1, "approval_rate"]-g.loc[0, "approval_rate"]), row.approval_gap_pp, rtol=0, atol=1e-12):
            raise ValueError("group profit/gap arithmetic differs")
        if row.ordering_regime == science.RAW:
            selection = frozen["banks"][row.source_policy]["selection"]
            if any(row[a] != selection[b] for a, b in (("number_approved", "number_approved"), ("funded_principal_USD", "funded_principal_USD"), ("perceived_profit_USD", "perceived_profit_USD"))):
                raise ValueError("result differs from frozen selection")
    for row in record["comparisons"]:
        a, b = pol.loc[row["policy_first"]], pol.loc[row["policy_second"]]
        if row["true_profit_difference_USD"] != a.true_profit_USD-b.true_profit_USD or row["perceived_profit_difference_USD"] != a.perceived_profit_USD-b.perceived_profit_USD:
            raise ValueError("paired profit arithmetic differs")
        if row["comparison_kind"] == "within_model_ordering" and row["perceived_profit_difference_USD"] < 0:
            raise ValueError("expanded-family perceived optimum decreased")
        overlap = [x for x in record["overlap"] if x["policy_first"]==row["policy_first"] and x["policy_second"]==row["policy_second"]]
        if {x["allocation_category"] for x in overlap} != {"approved_by_both", "first_only", "second_only", "rejected_by_both"} or sum(x["n_applicants"] for x in overlap) != a.n_applicants:
            raise ValueError("overlap is not a complete partition")
        counts = {x["allocation_category"]:x["n_applicants"] for x in overlap}
        if counts["approved_by_both"]+counts["first_only"] != a.number_approved or counts["approved_by_both"]+counts["second_only"] != b.number_approved:
            raise ValueError("overlap approval counts differ")
    for p in science.POLICIES:
        if not all(frozen["banks"][p]["diagnostics"][k] is True for k in ("every_raw_endpoint_fraction_checked", "all_bounded_prefixes_in_raw_family", "perceived_objective_nondecreasing_exact", "accounting_probabilities_and_per_request_profits_unchanged")):
            raise ValueError("missing exact/matching-accounting checks")
    return record


def find_success(ctx, case):
    found = []
    for path in sorted((ctx["directory"] / "records").glob(f"{case}_*.json")):
        # Retain corrupt records as evidence. They are not resumable successes.
        try: found.append(validate_record(ctx, base.checked(path)))
        except (ValueError, KeyError, TypeError, OSError, AssertionError): continue
    if len(found) > 1: raise ValueError("duplicate validated successes")
    return found[0] if found else None


def run_case(ctx, case, resume=False):
    if case not in science.CASES: raise ValueError("case outside six-case diagnostic design")
    check_sources(ctx); existing = find_success(ctx, case)
    if existing is not None:
        if not resume: raise FileExistsError("case already complete; use --resume")
        print(f"{case}: resumed validated success", flush=True); return existing
    start = time.perf_counter(); attempt = uuid.uuid4().hex; stage = "bank_replay_and_raw_optimization"
    try:
        banks = bank_case(ctx, case)
        stage = "bank_freeze"; frozen = freeze_case(ctx, case, banks, attempt)
        print(f"{case}: all three raw policies frozen; releasing evaluator", flush=True)
        stage = "evaluation"; truth, previous = load_evaluator(ctx, case, banks, frozen)
        result = science.evaluate_case(banks, ctx["observed"][["applicant_id", "B"]], truth, case)
        verify_reference_metrics(result, previous)
        record = {"schema_version": science.SCHEMA, "status": "success", "case_id": case,
            "experiment_fingerprint": ctx["manifest"]["experiment_fingerprint"], "freeze_path": str(frozen.relative_to(ctx["root"])),
            "freeze_sha256": file_hash(frozen), "new_fits": 0, "new_draws": 0, "baseline_cutoff_searches": 0,
            "raw_cutoff_searches": 3, "runtime_seconds": time.perf_counter()-start,
            **{k: base.rows(v) for k, v in result.items()}}
        stage = "validation"; validate_record(ctx, record)
        base.publish(ctx["directory"] / "records" / f"{case}_{attempt}.json", record)
        base.publish(ctx["directory"] / "attempts" / f"{case}_{attempt}.json", {"status": "success", "case_id": case, "runtime_seconds": record["runtime_seconds"]})
        print(f"{case}: complete in {record['runtime_seconds']:.2f}s", flush=True)
        return record
    except Exception as exc:
        base.publish(ctx["directory"] / "attempts" / f"{case}_{attempt}.json", {"status": "failed", "case_id": case,
            "stage": stage, "exception": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(), "runtime_seconds": time.perf_counter()-start})
        print(f"{case}: FAILED {stage}: {exc}", flush=True); return None


def summarize(ctx, write=False):
    """Validate saved records only. Cannot score, optimize or launch replications."""
    check_sources(ctx); verify_originals(ctx["root"])
    records = [r for case in science.CASES if (r := find_success(ctx, case)) is not None]
    outputs = {}; tables = ctx["root"] / TABLES / ctx["manifest"]["experiment_fingerprint"][:16]
    for kind in ("policies", "groups", "comparisons", "overlap"):
        if not records: continue
        keys = {"policies": ["policy"], "groups": ["policy", "B"], "comparisons": ["policy_first", "policy_second"], "overlap": ["policy_first", "policy_second", "allocation_category"]}[kind]
        frame = pd.DataFrame([x for r in records for x in r[kind]]).sort_values(["case_id", *keys]).reset_index(drop=True)
        path = tables / f"{kind}.csv"
        if write:
            tables.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(".pending")
            if temporary.exists(): raise FileExistsError("unfinished summary output retained")
            base.prior.write_csv(temporary, frame); temporary.replace(path)
        if path.exists(): outputs[str(path.relative_to(ctx["root"]))] = file_hash(path)
    report = {"schema_version": science.SCHEMA, "experiment_fingerprint": ctx["manifest"]["experiment_fingerprint"],
        "status": "complete_six_diagnostic_cases" if len(records)==6 else "incomplete", "successful_cases": len(records),
        "missing_cases": [case for case in science.CASES if case not in {r["case_id"] for r in records}],
        "failed_attempts": sum(base.checked(p)["status"]=="failed" for p in (ctx["directory"] / "attempts").glob("*.json")),
        "policy_cells": len(records)*6, "group_cells": len(records)*12, "comparison_cells": len(records)*9,
        "overlap_cells": len(records)*36, "new_fits": 0, "new_draws": 0, "baseline_cutoff_searches": 0,
        "raw_cutoff_searches": len(records)*3, "selected_cases_not_random_uncertainty_sample": True,
        "full_200_draw_raw_ordering_run": False, "cumulative_success_runtime_seconds": math.fsum(r["runtime_seconds"] for r in records),
        "protected_original_files_unchanged": ctx["protected_count"], "outputs": outputs}
    if write:
        path = ctx["directory"] / "latest_report.json"; temporary = path.with_suffix(".pending")
        if temporary.exists(): raise FileExistsError("unfinished report retained")
        base.publish(temporary, report); temporary.replace(path)
    return report
