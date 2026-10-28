"""Step 9A: replay frozen risks, apply exact -0.1B, and audit without lending."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fair_lending.professor_model.revised_scenarios import build_scenario_risk_estimates
from fair_lending.professor_model.revised_scenarios_io import (
    CONFIG_PATH, load_config, load_frozen_inputs, original_snapshot, output_paths,
    require_new_outputs, save_artifacts,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    config = load_config(args.config)
    require_new_outputs()  # Fail before loading or building if any output exists.
    originals = original_snapshot()
    inputs, hashes = load_frozen_inputs(config)
    print("Verified frozen models:", {k: v["model_id"] for k, v in inputs["models"].items()}, flush=True)
    print("Evaluation bank view:", inputs["applicants"].shape, list(inputs["applicants"].columns), flush=True)
    print("Original capital (diagnostic only):", repr(inputs["budget"]), flush=True)
    estimates = build_scenario_risk_estimates(inputs["applicants"], inputs["aware"], inputs["blind"])
    report = save_artifacts(inputs, estimates, config, hashes, originals=originals, config_path=args.config)
    print(json.dumps({"validation": report, "outputs": {k: str(v) for k, v in output_paths().items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
