"""Step 9B only: frozen-risk cutoff policies, bank freeze, evaluator comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fair_lending.professor_model import revised_lending as lending
from fair_lending.professor_model import revised_lending_io as io


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=io.CONFIG_PATH)
    args = parser.parse_args()
    config = io.load_config(args.config)
    io.require_new_outputs()
    originals = io.preservation_manifest()
    inputs, hashes = io.load_risk_inputs(config, originals)
    print(f"Verified {len(originals)} original files and frozen evaluation risks.", flush=True)
    banks = {}
    for scenario in lending.SCENARIO_IDS:
        risk = inputs["risks"].loc[inputs["risks"].scenario_id == scenario, lending.RISK_VIEW]
        banks[scenario] = lending.optimize_scenario(inputs["contracts"], risk, inputs["budget"])
        selection = banks[scenario]["selection"]
        print(f"{scenario}: tau={selection['optimal_tau']!r}, approved={selection['number_approved']}; exact enumeration passed.", flush=True)
    io.freeze_banks(inputs, banks, config, hashes, originals, config_path=args.config)
    print("All three policies saved and frozen. Releasing evaluator-only truth/references.", flush=True)
    policies, truth, _ = io.load_evaluator_inputs(inputs, config, config_path=args.config)
    report = io.save_evaluation(inputs, policies, truth, config, originals, config_path=args.config)
    print(json.dumps({"validation": report, "outputs": {k: str(v) for k, v in io.paths().items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
