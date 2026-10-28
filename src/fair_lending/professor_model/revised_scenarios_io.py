"""Read frozen observables/fits; persist separate Step 9A risk-only artifacts."""
from __future__ import annotations

from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
import pandas as pd
import yaml

from . import revised_scenarios as rs
from .bank_regression import ids_fingerprint, predict_bank_ols, validate_ids
from .config import PROJECT_ROOT, configuration_fingerprint
from .contracts import validate_contracts
from .group_blind import predict_group_blind
from .risk_io import file_hash, verify_input_hashes

CONFIG_PATH = PROJECT_ROOT / "configs/professor_model/revised_scenarios.yaml"
SYNTHETIC = Path("data/synthetic/professor_model/baseline")
PROCESSED = Path("data/processed/professor_model")
METRICS = Path("results/metrics/professor_model")
TABLES = Path("results/tables/professor_model")
FIGURES = Path("results/figures/professor_model")


def load_config(path=CONFIG_PATH):
    config = yaml.safe_load(Path(path).read_text())
    rs.validate_config(config)
    return config


def output_paths(root=PROJECT_ROOT):
    return {key: root / path for key, path in {
        "risk_csv": PROCESSED / "revised_scenarios/scenario_risk_estimates.csv",
        "risk_parquet": PROCESSED / "revised_scenarios/scenario_risk_estimates.parquet",
        "preview": PROCESSED / "revised_scenarios/scenario_risk_preview_EVALUATOR_ONLY.csv",
        "metadata": PROCESSED / "revised_scenarios/scenario_risk_metadata.json",
        "summary": TABLES / "revised_scenarios_risk_summary.csv",
        "groups": TABLES / "revised_scenarios_group_summary.csv",
        "ties": TABLES / "revised_scenarios_zero_tie_diagnostics.csv",
        "validation": METRICS / "revised_scenarios_validation.json",
        "figure": FIGURES / "revised_scenarios_raw_bounded_risk.png",
    }.items()}


def require_new_outputs(root=PROJECT_ROOT):
    existing = [str(p) for p in output_paths(root).values() if p.exists()]
    if existing:
        raise FileExistsError(f"Step 9A never overwrites existing outputs: {existing}")


def original_snapshot(root=PROJECT_ROOT):
    """Protect tracked science and every existing professor data/result file.

    The snapshot is byte-level preservation, not numerical access to private
    tables. New Step 9A source/configuration belongs in separate provenance.
    """
    tracked = subprocess.run(["git", "ls-files", "src", "configs", "experiments", "tests"],
                             cwd=root, capture_output=True, text=True, check=True)
    files = {root / name for name in tracked.stdout.splitlines()}
    for directory in (SYNTHETIC.parent, PROCESSED, TABLES, METRICS, FIGURES):
        files.update(p for p in (root / directory).rglob("*") if p.is_file()
                     and p.name != ".DS_Store" and "__pycache__" not in p.parts)
    return {str(p.relative_to(root)): file_hash(p) for p in sorted(files)}


def _register(hashes, root, relative, expected=None):
    path = Path(relative)
    allowed = ("src/fair_lending/professor_model", "configs/professor_model", "experiments",
               "data/synthetic/professor_model", "data/processed/professor_model",
               "results/tables/professor_model", "results/metrics/professor_model",
               "results/figures/professor_model")
    if path.is_absolute() or ".." in path.parts or not any(path.is_relative_to(prefix) for prefix in allowed) or not (root / path).resolve().is_relative_to(root.resolve()):
        raise ValueError("unexpected frozen-artifact path")
    actual = file_hash(root / path)
    if (expected is not None and actual != expected) or (str(path) in hashes and hashes[str(path)] != actual):
        raise ValueError(f"frozen artifact checksum mismatch: {relative}")
    hashes[str(path)] = actual


def _metadata(root, hashes, relative, config_name, status):
    """Validate recorded source paths, never a broadened current directory glob."""
    metadata = json.loads((root / relative).read_text())
    _register(hashes, root, relative)
    config_path = Path("configs/professor_model") / config_name
    if metadata["status"] != status or configuration_fingerprint(metadata["resolved_configuration"]) != metadata["configuration_fingerprint"]:
        raise ValueError("invalid frozen model metadata/configuration identity")
    if yaml.safe_load((root / config_path).read_text()) != metadata["resolved_configuration"]:
        raise ValueError("frozen configuration contents changed")
    _register(hashes, root, config_path, metadata["configuration_file_sha256"])
    if configuration_fingerprint(metadata["source_code_hashes"]) != metadata["source_code_fingerprint"]:
        raise ValueError("invalid frozen source fingerprint")
    for field in ("input_checksums", "source_code_hashes"):
        for path, expected in metadata[field].items():
            _register(hashes, root, path, expected)
    for collection in (metadata["artifacts"], metadata["freeze_checkpoint"]["artifacts"]):
        for artifact in collection.values():
            _register(hashes, root, artifact["path"], artifact["sha256"])
    if metadata["freeze_checkpoint"]["configuration_file_sha256"] != metadata["configuration_file_sha256"]:
        raise ValueError("frozen configuration checkpoint mismatch")
    return metadata


def _read_registered(relative, root, hashes):
    if str(relative) not in hashes:
        raise ValueError(f"unregistered scenario input: {relative}")
    path = root / relative
    return pd.read_csv(path, float_precision="round_trip") if path.suffix == ".csv" else pd.read_parquet(path)


def load_frozen_inputs(config, root=PROJECT_ROOT):
    """No PN, truth, private covariate, profit, or outcome table is parsed."""
    rs.validate_config(config)
    hashes = {}
    aware_meta = _metadata(root, hashes, SYNTHETIC / "bank_regression_metadata.json",
                           "bank_regression.yaml", "Step_5_validated")
    blind_meta = _metadata(root, hashes, PROCESSED / "group_blind_comparison_metadata.json",
                           "group_blind_comparison.yaml", "Step_7B_validated")
    checkpoint_path = METRICS / "group_blind_freeze_checkpoint.json"
    if str(checkpoint_path) not in hashes or json.loads((root / checkpoint_path).read_text()) != blind_meta["freeze_checkpoint"]:
        raise ValueError("standalone and metadata group-blind freezes differ")
    bank = _read_registered(SYNTHETIC / "applicants_bank_view.parquet", root, hashes)
    csv_bank = _read_registered(SYNTHETIC / "applicants_bank_view.csv", root, hashes)
    # Original Parquet B is int8; CSV infers int64. Require exact values, not
    # an integer width that the CSV format does not encode.
    pd.testing.assert_frame_equal(bank, csv_bank, check_exact=True, check_dtype=False)
    rs._observed(bank)  # Enforce the exact observable allowlist, before joining.
    split = _read_registered(SYNTHETIC / "applicant_splits.csv", root, hashes)
    validate_ids(split, "authoritative split")
    if tuple(split.columns) != ("applicant_id", "split") or split.split.value_counts().to_dict() != {"train": config["training_rows"], "evaluation": config["evaluation_rows"]} or len(bank) != config["population_rows"] or set(split.applicant_id) != set(bank.applicant_id):
        raise ValueError("original split and population must agree exactly")
    training_ids = split.loc[split.split == "train", "applicant_id"]
    eval_ids = split.loc[split.split == "evaluation", "applicant_id"]
    observed = bank.merge(split, on="applicant_id", validate="one_to_one")
    evaluation = observed.loc[observed.split == "evaluation", rs.OBSERVED_COLUMNS].sort_values("applicant_id").reset_index(drop=True)
    models, predictions = {}, {}
    for key, meta, stem, model_file, coef_file, raw_col, predictor in (
        ("aware", aware_meta, "bank_risk_estimates", "bank_regression_model.json", "bank_regression_coefficients.csv", "bank_PD_raw", predict_bank_ols),
        ("blind", blind_meta, "bank_risk_estimates_group_blind", "group_blind_regression_model.json", "group_blind_regression_coefficients.csv", "bank_PD_group_blind_raw", predict_group_blind),
    ):
        model_path = METRICS / model_file
        if str(model_path) not in hashes:
            raise ValueError("unregistered fitted model")
        model = json.loads((root / model_path).read_text())
        if model["configuration_fingerprint"] != meta["configuration_fingerprint"] or model["frozen_before_evaluator_truth"] is not True or model["training_count"] != config["training_rows"] or model["training_ID_fingerprint"] != ids_fingerprint(training_ids) or model["evaluation_ID_fingerprint"] != ids_fingerprint(eval_ids):
            raise ValueError("model configuration/cohort fingerprint mismatch")
        if model["family"] != "statsmodels_OLS" or model["solver"] != "pinv" or model["regularization"] is not None or model["target"] != "PN" or model["target_transform"] != "none" or model["prediction_rescaling"] != "none" or model["bounding"] != "clip_0_1_separate_column":
            raise ValueError("frozen estimator contract changed")
        coefficients = _read_registered(TABLES / coef_file, root, hashes)
        if tuple(coefficients.columns) != ("term", "coefficient", "predictor_units") or coefficients.term.tolist() != model["design_order"] or coefficients.coefficient.tolist() != [model["coefficients"][t] for t in model["design_order"]] or coefficients.predictor_units.tolist() != [model["units"][t] for t in model["design_order"]]:
            raise ValueError("coefficient CSV/model JSON mismatch")
        parquet = _read_registered(SYNTHETIC / f"{stem}.parquet", root, hashes)
        csv = _read_registered(SYNTHETIC / f"{stem}.csv", root, hashes)
        pd.testing.assert_frame_equal(parquet, csv, check_exact=True)
        validate_ids(parquet, "saved full predictions")
        pd.testing.assert_frame_equal(parquet[["applicant_id", "split"]].sort_values("applicant_id").reset_index(drop=True), split.sort_values("applicant_id").reset_index(drop=True), check_exact=True)
        evaluated = parquet.loc[parquet.split == "evaluation"].sort_values("applicant_id").reset_index(drop=True)
        if not np.array_equal(predictor(evaluation, model), evaluated[raw_col]):
            raise ValueError("frozen coefficients do not exactly replay evaluation predictions")
        models[key], predictions[key] = model, evaluated
    if models["aware"]["training_mean_PN"] != models["blind"]["training_mean_PN"]:
        raise ValueError("frozen historical training identities differ")
    # Only saved terms/principals and the saved capital value are read here.
    contracts = _read_registered(PROCESSED / "loan_contracts.parquet", root, hashes)
    pd.testing.assert_frame_equal(contracts, _read_registered(PROCESSED / "loan_contracts.csv", root, hashes), check_exact=True)
    validate_contracts(contracts)
    contracts = contracts.sort_values("applicant_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(contracts[["applicant_id", "Y"]], evaluation[["applicant_id", "Y"]], check_exact=True)
    if not contracts["T"].eq(30).all() or not contracts.theta.eq(.1).all() or not contracts.r.eq(.05).all():
        raise ValueError("original contract terms changed")
    budget_path = Path(config["budget_source"])
    if str(budget_path) not in hashes:
        raise ValueError("unregistered frozen budget")
    budget = json.loads((root / budget_path).read_text())["selection"]["available_bank_capital"]
    if budget != config["exact_saved_budget_USD"] or budget != blind_meta["exact_saved_budget"]:
        raise ValueError("original exact capital budget changed")
    if not verify_input_hashes(hashes, root):
        raise ValueError("frozen inputs changed during reading")
    rs.build_scenario_risk_estimates(evaluation, predictions["aware"], predictions["blind"])
    return {"applicants": evaluation, **predictions, "models": models, "contracts": contracts,
            "budget": budget, "training_ID_fingerprint": ids_fingerprint(training_ids),
            "evaluation_ID_fingerprint": ids_fingerprint(eval_ids)}, hashes


def save_risk_figure(estimates, path):
    """One figure, two full-range panels; no hidden risks, outcomes, or loans."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    labels, data, colors = [], [[], []], []
    names = ("No-direct: aware", "Direct: blind − 0.1B", "Blind reference")
    for definition, label in zip(rs.SCENARIOS, names):
        for group in (0, 1):
            frame = estimates.loc[(estimates.scenario_id == definition["scenario_id"]) & (estimates.B == group)]
            if frame.empty:
                continue
            labels.append(f"{label}\nB={group}")
            colors.append("#253746" if group == 0 else "#E5751F")
            for items, column in zip(data, ("estimated_PD_adjusted_raw", "estimated_PD_bounded")):
                items.append(100 * frame[column].to_numpy())
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.8), sharey=True, layout="constrained")
    for ax, items, title in zip(axes, data, ("Adjusted raw PD (negatives retained)", "Bounded PD (clipped, not calibrated)")):
        boxes = ax.boxplot(items, orientation="horizontal", patch_artist=True, whis=(0, 100),
                           medianprops={"color": "black", "linewidth": 1.4})
        for patch, color in zip(boxes["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(.65)
        ax.set_yticks(np.arange(1, len(labels) + 1), labels)
        ax.axvline(0, color="#68737D", linestyle="--", linewidth=.8)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("PD in percentage-point units (100 × raw probability)")
        ax.grid(axis="x", alpha=.2)
    axes[0].invert_yaxis()
    fig.suptitle("Frozen evaluation applicants: revised risk-only scenarios", fontsize=14)
    fig.legend(handles=[Patch(facecolor="#253746", label="B=0"), Patch(facecolor="#E5751F", label="B=1")], loc="outside upper right", ncol=2)
    fig.supxlabel("Separate horizontal scales. Boxes: quartiles/median. Whiskers: full min–max. No lending decisions.", fontsize=9)
    with path.open("xb") as stream:
        fig.savefig(stream, format="png", dpi=170)
    plt.close(fig)


def _write_json(path, data):
    with path.open("x") as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def save_artifacts(inputs, estimates, config, hashes, *, root=PROJECT_ROOT,
                   config_path=CONFIG_PATH, originals=None):
    """Validate first, refuse overwrites, then save new Step 9A files only."""
    rs.validate_config(config)
    require_new_outputs(root)
    originals = hashes if originals is None else originals
    if not verify_input_hashes(hashes, root) or not verify_input_hashes(originals, root):
        raise ValueError("original evidence changed before export")
    report = rs.validate_scenario_estimates(estimates, inputs["applicants"], inputs["aware"], inputs["blind"])
    summary, groups = rs.summarize_risks(estimates)
    ties = rs.zero_tie_diagnostics(estimates, inputs["contracts"][["applicant_id", "L_i"]], inputs["budget"])
    preview = rs.evaluator_preview(inputs["applicants"], estimates, config["preview_applicants_per_group"])
    output = output_paths(root)
    if any(str(p.relative_to(root)) in originals or str(p.relative_to(root)) in hashes for p in output.values()):
        raise ValueError("new outputs collide with original evidence")
    for path in output.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    estimates.to_csv(output["risk_csv"], index=False, mode="x")
    with output["risk_parquet"].open("xb") as stream:
        estimates.to_parquet(stream, index=False)
    for name, frame in (("preview", preview), ("summary", summary), ("groups", groups), ("ties", ties)):
        frame.to_csv(output[name], index=False, mode="x")
        pd.testing.assert_frame_equal(pd.read_csv(output[name], float_precision="round_trip"), frame, check_exact=True)
    pd.testing.assert_frame_equal(pd.read_csv(output["risk_csv"], float_precision="round_trip"), estimates, check_exact=True)
    pd.testing.assert_frame_equal(pd.read_parquet(output["risk_parquet"]), estimates, check_exact=True)
    save_risk_figure(estimates, output["figure"])
    if not verify_input_hashes(originals, root) or not verify_input_hashes(hashes, root):
        raise RuntimeError("original evidence changed during export")
    report.update({"original_files_preserved": len(originals), "frozen_input_files_verified": len(hashes),
                   "original_files_unchanged": True, "model_fits": 0, "random_draws": 0,
                   "lending_optimizations": 0, "profit_experiments": 0,
                   "hidden_tables_numerically_read": False, "csv_parquet_exact_agreement": True,
                   "frozen_coefficient_replay_exact": True, "zero_tie_diagnostics": ties.to_dict("records")})
    _write_json(output["validation"], report)
    sources = ["src/fair_lending/professor_model/revised_scenarios.py",
               "src/fair_lending/professor_model/revised_scenarios_io.py",
               "experiments/build_professor_revised_scenarios.py",
               "src/fair_lending/professor_model/bank_regression.py",
               "src/fair_lending/professor_model/group_blind.py",
               "src/fair_lending/professor_model/contracts.py",
               "src/fair_lending/professor_model/config.py", "src/fair_lending/professor_model/risk_io.py"]
    # Tests may provide a separate root with toy data but source remains this repo.
    source_hashes = {p: file_hash(PROJECT_ROOT / p) for p in sources}
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout)
    metadata = {
        "status": "Step_9A_validated_risk_only", "schema_version": rs.SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "git_revision": revision, "git_dirty": dirty,
        "resolved_configuration": config, "configuration_fingerprint": configuration_fingerprint(config),
        "configuration_file_sha256": file_hash(config_path), "source_code_hashes": source_hashes,
        "source_code_fingerprint": configuration_fingerprint(source_hashes),
        "dependencies": {p: version(p) for p in ("numpy", "pandas", "pyarrow", "matplotlib", "statsmodels", "PyYAML")},
        "python": platform.python_version(), "input_checksums": hashes, "original_checksums": originals,
        "training_ID_fingerprint": inputs["training_ID_fingerprint"],
        "evaluation_ID_fingerprint": inputs["evaluation_ID_fingerprint"],
        "frozen_models": {key: {"model_id": model["model_id"], "configuration_fingerprint": model["configuration_fingerprint"],
                               "coefficients": model["coefficients"]} for key, model in inputs["models"].items()},
        "summary_probability_units": "Risk/adjustment *_pp fields equal 100 times raw probability; *_share fields are fractions.",
        "information_boundary": "No PN, epsilon, omega, eta, true_PD, outcome or profit table parsed. Private bytes only hashed for preservation.",
        "model_fits": 0, "random_draws": 0, "lending_optimizations": 0,
        "artifacts": {key: {"path": str(path.relative_to(root)), "sha256": file_hash(path)}
                      for key, path in output.items() if key != "metadata"},
    }
    _write_json(output["metadata"], metadata)
    return report
