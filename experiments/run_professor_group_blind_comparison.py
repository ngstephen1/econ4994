"""Step 7B: one new group-blind OLS; preserve the original bank and oracle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fair_lending.professor_model import group_blind as gb
from fair_lending.professor_model import group_blind_io as io


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=io.CONFIG_PATH)
    parser.add_argument("--overwrite", action="store_true", help="Replace Step 7B outputs only")
    args = parser.parse_args()
    config = io.load_config(args.config)
    if not args.overwrite and any(p.exists() for p in io.paths().values()):
        raise FileExistsError("Step 7B outputs exist; use --overwrite explicitly")
    inputs, hashes = io.load_inputs(config)
    observed = inputs["observed"]
    print("Saved observed rows:", observed.shape, "cohorts:", observed.split.value_counts().to_dict())
    print(observed.head(10).to_string(index=False))
    print("Raw training PN:", observed.loc[observed.split == "train", "PN"].describe().to_dict())
    model = gb.fit_group_blind(observed.loc[:, gb.FIT_COLUMNS])
    predictions = gb.risk_estimates(observed, model)
    bounded = predictions.loc[predictions.split == "evaluation", ["applicant_id", "model_id", gb.BOUNDED]]
    profit = gb.assess_profit(inputs["contracts"], bounded)
    decisions, candidates, selection = gb.optimize_cutoff(inputs["contracts"], profit, inputs["bank_selection"]["available_bank_capital"])
    io.freeze(inputs, model, predictions, profit, decisions, candidates, selection, config, hashes, config_path=args.config, overwrite=args.overwrite)
    print("New model, predictions, accounting and cutoff frozen; now loading evaluator truth.")
    report = io.save_evaluation(inputs, predictions, profit, decisions, selection, config, hashes, config_path=args.config)
    print(json.dumps({"model": model, "validation": report, "outputs": {k: str(v) for k, v in io.paths().items()}}, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
