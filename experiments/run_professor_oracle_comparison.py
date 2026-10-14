"""Step 7A evaluator-only true-risk oracle versus the frozen Step 6B bank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fair_lending.professor_model.oracle import optimize_oracle
from fair_lending.professor_model.oracle_io import CONFIG_PATH, load_oracle_config, load_oracle_inputs, output_paths, save_oracle_results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace only Step 7A outputs")
    args = parser.parse_args()
    config = load_oracle_config(args.config)
    inputs, hashes = load_oracle_inputs(config)
    for name in ("contracts", "bank", "truth", "bank_decisions"):
        print(name, inputs[name].shape, list(inputs[name].columns))
    print("Frozen original bank tau:", repr(inputs["bank_selection"]["optimal_tau"]))
    print("Exact saved capital budget:", repr(inputs["bank_selection"]["available_bank_capital"]))
    print("First 10 EVALUATOR-ONLY true-risk/contract rows:")
    print(inputs["contracts"][["applicant_id", "L_i"]].merge(inputs["truth"], on="applicant_id", validate="one_to_one").head(10).to_string(index=False))
    decisions, candidates, selection = optimize_oracle(inputs["contracts"], inputs["truth"], inputs["bank_selection"]["available_bank_capital"])
    report = save_oracle_results(inputs, decisions, candidates, selection, config, hashes, config_path=args.config, overwrite=args.overwrite)
    print(json.dumps({"artifacts": {key: str(p) for key, p in output_paths(config).items()}, "validation": report}, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
