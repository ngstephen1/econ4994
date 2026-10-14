"""Step 6B: exact common-cutoff policy; freeze decisions before evaluator truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fair_lending.professor_model.lending import capital_budget, choose_cutoff
from fair_lending.professor_model.lending_io import (
    CONFIG_PATH, freeze_lending_decisions, load_evaluator_inputs, load_lending_config,
    load_lending_inputs, output_paths, save_evaluator_results,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace only Step 6B outputs")
    args = parser.parse_args()
    config = load_lending_config(args.config)
    bank, hashes, schemas = load_lending_inputs(config)
    for item in schemas:
        print(item)
    print("First 10 saved bank-only inputs BEFORE selection:")
    print(bank.head(10).to_string(index=False))
    total, budget = capital_budget(bank, config["policy"]["budget_fraction"])
    print(f"Total requested principal={total!r}; exact operational budget={budget!r}")
    decisions, candidates, summary = choose_cutoff(bank, budget)
    checkpoint = freeze_lending_decisions(bank, decisions, candidates, summary, config, hashes, config_path=args.config, overwrite=args.overwrite)
    print("Bank cutoff and all decisions SAVED AND FROZEN. Now loading evaluator-only true profit.", flush=True)
    audit, truth = load_evaluator_inputs(bank, checkpoint, config, hashes)
    report = save_evaluator_results(bank, decisions, candidates, summary, audit, truth, checkpoint, config, hashes, config_path=args.config)
    print(json.dumps({"artifacts": {key: str(path) for key, path in output_paths(config).items()}, "validation": report}, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
