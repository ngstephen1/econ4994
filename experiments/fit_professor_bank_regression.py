"""Fit Step 5 on existing observed data, freeze, then evaluate hidden truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fair_lending.professor_model.bank_io import (
    CONFIG_PATH, freeze_bank_artifacts, load_bank_config, load_bank_inputs,
    load_evaluator_truth, output_paths, save_bank_evaluation,
)
from fair_lending.professor_model.bank_regression import (
    attach_splits, evaluate_bank_risk, fit_bank_ols, make_splits, risk_estimates,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--overwrite", action="store_true", help="Replace only Step 5 outputs; never original inputs")
    args = parser.parse_args()
    config = load_bank_config(args.config)
    joined, input_hashes, schemas = load_bank_inputs(config)
    for schema in schemas:
        print(schema)
    print("First 10 joined observed inputs BEFORE fitting:")
    print(joined.head(10).to_string(index=False))
    print("Raw PN summary (sample SD):")
    print(joined.PN.describe(percentiles=[.01, .05, .5, .95, .99]).to_string())
    splits = make_splits(joined.applicant_id, config["split"])
    applicants = attach_splits(joined, splits)
    model = fit_bank_ols(applicants)
    coefficients = model["coefficients"]
    print("bank_PD_raw = " + f"{coefficients['const']:.17g}" + " ".join(f" {coefficients[name]:+.17g} * {name}" for name in model["feature_order"]))
    predictions = risk_estimates(applicants, model)
    checkpoint = freeze_bank_artifacts(applicants, splits, predictions, model, config, input_hashes, config_path=args.config, overwrite=args.overwrite)
    print("Model and all predictions saved and frozen. Loading truth for EVALUATOR ONLY.", flush=True)
    truth = load_evaluator_truth(config, checkpoint, input_hashes)
    report = save_bank_evaluation(applicants, predictions, truth, model, config, checkpoint, input_hashes, config_path=args.config)
    _, _, preview, _ = evaluate_bank_risk(applicants, predictions, truth, model)
    print("First 10 EVALUATOR-ONLY preview rows:")
    print(preview.head(10).to_string(index=False, float_format=lambda x: f"{x:.10g}"))
    print(json.dumps({"artifacts": {key: str(path) for key, path in output_paths(config).items()}, "validation": report}, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
