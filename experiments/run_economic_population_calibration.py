"""Generate and validate the Version 2 baseline economic population."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fair_lending.economic_lending.config import PROJECT_ROOT, load_economic_config
from fair_lending.economic_lending.population import generate_economic_population


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--calibration", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    config = load_economic_config()
    default_rows = (
        config["artifacts"]["calibration_rows"]
        if args.calibration
        else config["artifacts"]["development_rows"]
    )
    rows = int(default_rows if args.rows is None else args.rows)
    seed = int(config["randomness"]["seed"] if args.seed is None else args.seed)
    result = generate_economic_population(rows, config, seed=seed)

    if not args.no_save:
        data_directory = PROJECT_ROOT / config["artifacts"]["data_directory"]
        data_directory.mkdir(parents=True, exist_ok=True)
        result.applicants.to_parquet(data_directory / "applicants.parquet", index=False)
        result.loan_options.to_parquet(data_directory / "loan_options.parquet", index=False)
        result.simulation_truth.to_parquet(
            data_directory / "simulation_truth.parquet", index=False
        )
        result.loan_outcomes.to_parquet(data_directory / "loan_outcomes.parquet", index=False)

        report_path = PROJECT_ROOT / config["artifacts"]["validation_report"]
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(result.validation_report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        group_path = PROJECT_ROOT / config["artifacts"]["group_diagnostics"]
        group_path.parent.mkdir(parents=True, exist_ok=True)
        result.group_diagnostics.to_csv(group_path, index=False)

    print(json.dumps(result.validation_report, indent=2, sort_keys=True))
    return 1 if result.validation_report["warnings"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
