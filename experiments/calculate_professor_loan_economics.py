"""Step 6A: value existing held-out requests, without refitting or allocation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fair_lending.professor_model.contracts import make_loan_contracts
from fair_lending.professor_model.economics_io import (
    CONFIG_PATH, load_economic_inputs, load_economics_config,
    load_true_risk_for_economics, save_economic_artifacts,
)
from fair_lending.professor_model.profit import BANK_RISK_COLUMNS, assess_bank_profit, assess_true_profit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace only Step 6A outputs")
    args = parser.parse_args()
    config = load_economics_config(args.config)
    evaluation, hashes, schemas = load_economic_inputs(config)
    for item in schemas:
        print(item)
    print(f"Existing held-out evaluation applicants: {len(evaluation):,}")
    print(evaluation.head(10).to_string(index=False))
    terms = config["contract"]
    contracts = make_loan_contracts(evaluation, T=terms["T"], theta=terms["theta"], r=terms["r"])
    bank = assess_bank_profit(contracts, evaluation.loc[:, BANK_RISK_COLUMNS])
    print("Bank receipts/profits calculated from bounded predictions only; now loading EVALUATOR truth.", flush=True)
    true_risk = load_true_risk_for_economics(evaluation, config, hashes)
    truth = assess_true_profit(contracts, true_risk)
    paths, report, preview = save_economic_artifacts(evaluation, contracts, bank, truth, config, hashes, config_path=args.config, overwrite=args.overwrite)
    print("First 10 EVALUATOR-ONLY preview rows (USD; probabilities in raw units):")
    print(preview.head(10).to_string(index=False, float_format=lambda x: f"{x:.10g}"))
    print(json.dumps({"artifacts": {key: str(path) for key, path in paths.items()}, "validation": report}, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
