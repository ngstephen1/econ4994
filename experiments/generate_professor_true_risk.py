"""Calculate professor-model true risk from frozen Step 2 data, without draws."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fair_lending.professor_model.risk import TRUE_RISK_CONFIG_PATH, compute_true_risk, load_true_risk_config
from fair_lending.professor_model.risk_io import load_existing_applicants, save_true_risk_artifacts
from fair_lending.professor_model.risk_validation import validate_true_risk


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=TRUE_RISK_CONFIG_PATH)
    parser.add_argument("--overwrite", action="store_true", help="Replace only existing Step 3 artifacts")
    args = parser.parse_args()
    config = load_true_risk_config(args.config)
    applicants, input_hashes = load_existing_applicants(config)
    truth = compute_true_risk(applicants, config)
    report = validate_true_risk(truth, applicants, config)
    paths = save_true_risk_artifacts(applicants, truth, report, config, input_hashes, overwrite=args.overwrite, config_path=args.config)
    print(json.dumps({"artifacts": {name: str(path) for name, path in paths.items()}, "validation": report}, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
