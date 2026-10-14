"""Immutable pilot evidence, legacy preservation, and evaluator-only reporting."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import platform
import subprocess
import tempfile
from importlib.metadata import version

import numpy as np
import pandas as pd
import yaml

from . import group_blind_io as legacy
from . import noise_pilot as pilot
from .bank_regression import validate_ids
from .config import PROJECT_ROOT, configuration_fingerprint
from .risk_io import file_hash, verify_input_hashes

CONFIG_PATH = PROJECT_ROOT / "configs/professor_model/noise_robustness_pilot.yaml"
RUN_DIRECTORY = Path("results/metrics/professor_model/noise_robustness_pilot")
NEW_FILES = (
    "configs/professor_model/noise_robustness_pilot.yaml",
    "src/fair_lending/professor_model/noise_pilot.py", "src/fair_lending/professor_model/noise_pilot_io.py",
    "experiments/run_professor_noise_pilot.py", "tests/test_professor_noise_pilot.py",
)


def load_config(path=CONFIG_PATH):
    expected = {"schema_version": "professor-noise-pilot-v1", "master_seed": 499408, "replications": 5,
                "population_rows": 100000, "training_rows": 80000, "evaluation_rows": 20000,
                "signal_scale": .8, "noise": {"mean": 0., "sd": .1}, "bit_generator": "PCG64",
                "applicant_order": "ascending_lexicographic_applicant_id",
                "budget_source": "results/metrics/professor_model/lending_selection.json",
                "budget_field": "selection.available_bank_capital"}
    config = yaml.safe_load(Path(path).read_text())
    if config != expected:
        raise ValueError("Step 8A is exactly five fixed-population/noise-only replications")
    return config


def immutable_json(path, payload):
    """Publish a complete JSON atomically with no replacement of an existing file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=".pending_", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)  # atomic O_EXCL-like publication; never os.replace
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def preservation_snapshot(root=PROJECT_ROOT):
    """All pre-pilot professor artifacts plus existing scientific code/config/tests.

    Never inspect HMDA or unrelated raw data. Newly authored Step 8A files and
    pilot outputs are excluded so a restart yields the same original inventory.
    """
    folders = ("data/synthetic/professor_model", "data/processed/professor_model", "results/tables/professor_model",
               "results/metrics/professor_model", "results/figures/professor_model", "configs", "src", "experiments", "tests")
    result = {}
    for folder in folders:
        for path in (root / folder).rglob("*"):
            relative = path.relative_to(root)
            if not path.is_file() or "__pycache__" in path.parts or path.suffix in (".pyc", ".DS_Store") or str(relative) in NEW_FILES or relative.is_relative_to(RUN_DIRECTORY) or path.name.startswith("noise_pilot_"):
                continue
            result[str(relative)] = file_hash(path)
    return dict(sorted(result.items()))


def load_fixed_inputs(config, root=PROJECT_ROOT):
    """Validate all Step 7B provenance; do not generate any original variables."""
    old_config_path = root / "configs/professor_model/group_blind_comparison.yaml"
    old_config = legacy.load_config(old_config_path)
    inputs, old_hashes = legacy.load_inputs(old_config, root)
    legacy.verify_freeze(old_config, old_hashes, root=root, config_path=old_config_path)
    meta = json.loads((root / legacy.PROCESSED / "group_blind_comparison_metadata.json").read_text())
    if meta["status"] != "Step_7B_validated" or meta["configuration_fingerprint"] != configuration_fingerprint(old_config) or meta["source_code_fingerprint"] != configuration_fingerprint(meta["source_code_hashes"]) or meta["source_code_hashes"] != legacy.source_hashes():
        raise ValueError("prior group-blind provenance differs")
    for item in meta["artifacts"].values():
        if file_hash(root / item["path"]) != item["sha256"]:
            raise ValueError("prior group-blind output checksum mismatch")
    observed = inputs["observed"].loc[:, pilot.OBSERVED_COLUMNS].copy()
    if len(observed) != config["population_rows"] or observed.split.value_counts().to_dict() != {"train": config["training_rows"], "evaluation": config["evaluation_rows"]}:
        raise ValueError("frozen population/split counts differ")
    full_truth = pd.read_parquet(root / legacy.SYNTHETIC / "simulation_truth.parquet")
    validate_ids(full_truth, "simulator truth")
    if tuple(full_truth.columns) != ("applicant_id", "P_income", "P_other", "true_PD") or set(full_truth.applicant_id) != set(observed.applicant_id):
        raise ValueError("saved full-population truth mismatch")
    budget = inputs["bank_selection"]["available_bank_capital"]
    if budget != 1826087220.7466326 or budget != meta["exact_saved_budget"]:
        raise ValueError("must use the unchanged saved absolute capital budget")
    return observed, full_truth[["applicant_id", "true_PD"]], inputs["contracts"], budget


def prepare_manifest(config, *, root=PROJECT_ROOT, config_path=CONFIG_PATH, resume=False):
    original = preservation_snapshot(root)
    sources = {str(p.relative_to(PROJECT_ROOT)): file_hash(p) for p in sorted((PROJECT_ROOT / "src/fair_lending/professor_model").glob("*.py"))}
    sources["experiments/run_professor_noise_pilot.py"] = file_hash(PROJECT_ROOT / "experiments/run_professor_noise_pilot.py")
    identity = {"configuration": config, "configuration_file_sha256": file_hash(config_path), "original_checksums": original,
                "source_code_hashes": sources, "dependencies": {name: version(name) for name in ("numpy", "pandas", "scipy", "statsmodels", "pyarrow", "PyYAML", "matplotlib")}, "python": platform.python_version()}
    fingerprint = configuration_fingerprint(identity)
    path = root / RUN_DIRECTORY / "manifest.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if not resume:
            raise FileExistsError("Pilot evidence exists; --resume reuses matching records")
        if saved["experiment_fingerprint"] != fingerprint or saved["identity"] != identity:
            raise ValueError("pilot source/config/dependency/original-input provenance differs")
        return saved
    if resume:
        raise FileNotFoundError("No pilot manifest exists to resume/summarize")
    git = lambda *args: subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    manifest = {"schema_version": "professor-noise-pilot-v1", "experiment_fingerprint": fingerprint, "identity": identity,
                "git_revision": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
                "replication_indices": list(range(5)), "spawn_keys": [list(s.spawn_key) for s in pilot.streams()],
                "scope": "conditional noise-only pilot; fixed population, truth, split, contracts, budget, oracle"}
    immutable_json(path, manifest)
    return manifest


def verify_originals(manifest, root=PROJECT_ROOT):
    if not verify_input_hashes(manifest["identity"]["original_checksums"], root):
        raise ValueError("original scientific evidence changed during pilot")


def record_path(replication, root=PROJECT_ROOT):
    if type(replication) is not int or not 0 <= replication < 5:
        raise ValueError("only five pilot replication IDs allowed")
    return root / RUN_DIRECTORY / f"replication_{replication:02d}.json"


def freeze_bank(banks, replication, stream, manifest, root=PROJECT_ROOT):
    verify_originals(manifest, root)
    payload = {"status": "bank_decisions_frozen_before_profit_evaluation", "replication": replication,
               "experiment_fingerprint": manifest["experiment_fingerprint"], "stream": stream, "banks": pilot.bank_freeze_payload(banks)}
    path = root / RUN_DIRECTORY / f"bank_freeze_{replication:02d}.json"
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise ValueError("immutable bank freeze differs from deterministic reconstruction")
    else:
        immutable_json(path, payload)
    return payload


def evaluate_frozen(banks, replication, stream, observed, targets, contracts, budget, manifest, root=PROJECT_ROOT):
    path = root / RUN_DIRECTORY / f"bank_freeze_{replication:02d}.json"
    frozen = json.loads(path.read_text())
    if frozen["status"] != "bank_decisions_frozen_before_profit_evaluation" or frozen["replication"] != replication or frozen["stream"] != stream or frozen["experiment_fingerprint"] != manifest["experiment_fingerprint"] or frozen["banks"] != pilot.bank_freeze_payload(banks):
        raise ValueError("allocation must be frozen before true-profit evaluation")
    verify_originals(manifest, root)
    # Only the simulator has read full-population true_PD for target generation;
    # this is the first true-profit/oracle read in the replication lifecycle.
    truth = pd.read_parquet(root / legacy.PROCESSED / "true_expected_profit.parquet")
    oracle = pd.read_parquet(root / legacy.PROCESSED / "oracle_lending_decisions.parquet")
    oracle_selection = json.loads((root / legacy.METRICS / "oracle_comparison_validation.json").read_text())["oracle_selection"]
    oracle_reference = pd.read_csv(root / legacy.TABLES / "bank_vs_oracle_summary.csv", float_precision="round_trip").query("policy == 'oracle'").iloc[0]
    metrics, groups, comparison = pilot.evaluate_replication(banks, observed, targets, contracts, truth, oracle, oracle_selection, budget, frozen["banks"])
    # Check fixed oracle value without reoptimization; no true cashflows redrawn.
    actual_profit = math.fsum(truth.set_index("applicant_id").loc[oracle.loc[oracle.approved, "applicant_id"], "expected_profit_true"])
    if actual_profit != oracle_reference.true_expected_portfolio_profit or oracle_selection["available_bank_capital"] != budget:
        raise ValueError("frozen oracle reference value/budget differs")
    oracle_value = {"true_expected_portfolio_profit": actual_profit, "number_approved": int(oracle.approved.sum()), "optimal_tau": oracle_selection["optimal_tau"], "decision_fingerprint": pilot.frame_fingerprint(oracle)}
    return metrics, groups, comparison, oracle_value


def save_record(record, manifest, root=PROJECT_ROOT):
    verify_originals(manifest, root)
    payload = {**record, "record_fingerprint": configuration_fingerprint(record)}
    immutable_json(record_path(record["replication"], root), payload)
    return payload


def load_record(replication, manifest, root=PROJECT_ROOT):
    path = record_path(replication, root)
    saved = json.loads(path.read_text())
    payload = {k: v for k, v in saved.items() if k != "record_fingerprint"}
    if saved.get("record_fingerprint") != configuration_fingerprint(payload) or saved.get("status") != "success" or saved.get("replication") != replication or saved.get("experiment_fingerprint") != manifest["experiment_fingerprint"] or saved["stream"]["spawn_key"] != [replication]:
        raise ValueError("invalid/incompatible immutable replication record")
    if len(saved["policy_metrics"]) != 2 or {m["policy"] for m in saved["policy_metrics"]} != set(pilot.POLICIES) or len(saved["group_approval"]) != 4 or {(g["policy"], g["B"]) for g in saved["group_approval"]} != {(p, b) for p in pilot.POLICIES for b in (0, 1)}:
        raise ValueError("incomplete/duplicate policy/group coverage")
    frozen = json.loads((root / RUN_DIRECTORY / f"bank_freeze_{replication:02d}.json").read_text())
    if frozen["banks"] != saved["banks"] or frozen["stream"] != saved["stream"] or frozen["experiment_fingerprint"] != manifest["experiment_fingerprint"]:
        raise ValueError("record does not match immutable bank freeze")
    for m in saved["policy_metrics"]:
        if m["maximum_budget_violation_USD"] != 0 or not m["independent_exact_enumeration_passed"] or not np.isfinite([m["true_expected_portfolio_profit"], m["raw_PN_RMSE"], m["raw_true_PD_MAE"], m["bounded_true_PD_RMSE"]]).all():
            raise ValueError("invalid numeric/feasibility metrics")
    verify_originals(manifest, root)
    return saved


def _figures(replications, root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory = root / legacy.FIGURES
    directory.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.6, 4.6), layout="constrained")
    ax.axhline(0, color=".4", linewidth=1)
    ax.plot(replications.replication, replications.beta_B_hat, marker="o", linewidth=1.5, color="#243e59")
    ax.set(xticks=range(5), xlabel="Noise replication", ylabel="OLS B coefficient (raw PN units)", title="Group coefficient under five independent noise draws")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(directory / "noise_pilot_group_coefficient.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    x = replications.replication.to_numpy()
    ax.axhline(0, color=".4", linewidth=1)
    for shift, column, label, color in ((-.18, "aware_approval_gap_pp", "Group-aware", "#243e59"), (.18, "blind_approval_gap_pp", "Group-blind", "#288679")):
        ax.bar(x + shift, replications[column], width=.34, color=color, label=label)
    ax.set(xticks=range(5), xlabel="Noise replication", ylabel="B=1 minus B=0 approval gap (percentage points)", title="Approval-gap variation on one fixed population")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(directory / "noise_pilot_approval_gaps.png", dpi=180)
    plt.close(fig)


def summarize(manifest, root=PROJECT_ROOT):
    """Read five records only. This function cannot generate targets or fit models."""
    records = [load_record(i, manifest, root) for i in range(5)]
    tables = pilot.result_tables(records)
    outputs = {}
    for name, frame in zip(("replications", "coefficients", "group_approval", "summary"), tables):
        path = root / legacy.TABLES / f"noise_pilot_{name}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        outputs[str(path.relative_to(root))] = file_hash(path)
    _figures(tables[0], root)
    for stem in ("group_coefficient", "approval_gaps"):
        relative = legacy.FIGURES / f"noise_pilot_{stem}.png"
        outputs[str(relative)] = file_hash(root / relative)
    beta = tables[0].beta_B_hat
    report = {"status": "five_replication_pilot_complete", "experiment_fingerprint": manifest["experiment_fingerprint"],
              "n_replications": 5, "new_OLS_fits": 10, "beta_B_positive": int((beta > 0).sum()), "beta_B_negative": int((beta < 0).sum()),
              "beta_B_zero": int((beta == 0).sum()), "beta_B_changes_sign": bool((beta > 0).any() and (beta < 0).any()),
              "original_files_preserved": len(manifest["identity"]["original_checksums"]), "outputs": outputs,
              "oracle_reference": records[0]["oracle_reference"], "replication_record_hashes": {str(i): file_hash(record_path(i, root)) for i in range(5)},
              "all_threshold_audits_passed": all(m["independent_exact_enumeration_passed"] for r in records for m in r["policy_metrics"]),
              "full_Monte_Carlo_started": False, "limitation": "Only conditional noise variation on one fixed population; five draws do not establish statistical robustness/significance."}
    if any(r["oracle_reference"] != records[0]["oracle_reference"] for r in records):
        raise ValueError("oracle changed across pilot replications")
    path = root / RUN_DIRECTORY / "summary_metadata.json"
    if path.exists():
        if json.loads(path.read_text()) != report:
            raise ValueError("summary differs from immutable completed pilot metadata")
    else:
        immutable_json(path, report)
    verify_originals(manifest, root)
    return report
