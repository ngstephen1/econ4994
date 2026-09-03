"""Minimal command-line entry point for reproducible synthetic generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fair_lending.simulation.config import ALLOWED_SCENARIOS, EFFECT_LEVELS
from fair_lending.simulation.generator import (
    generate_synthetic_data,
    save_synthetic_dataset,
)
from fair_lending.simulation.validation import (
    save_validation_outputs,
    validate_generated_data,
    validation_summary_row,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate synthetic mortgage applications")
    parser.add_argument("--scenario", choices=ALLOWED_SCENARIOS, default="fair_baseline")
    parser.add_argument("--effect-level", choices=EFFECT_LEVELS, default="moderate")
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=4_994)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--recalibrate-intercept", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data, metadata = generate_synthetic_data(
        scenario=args.scenario,
        effect_level=args.effect_level,
        n_rows=args.rows,
        seed=args.seed,
        recalibrate_intercept=args.recalibrate_intercept,
    )
    dataset_path, metadata_path, saved_metadata = save_synthetic_dataset(
        data, metadata, args.output
    )
    report = validate_generated_data(data, saved_metadata)
    metrics_path, table_path = save_validation_outputs(report, dataset_path)
    output = validation_summary_row(report)
    output.update(
        {
            "dataset_path": str(dataset_path),
            "metadata_path": str(metadata_path),
            "validation_metrics_path": str(metrics_path),
            "validation_table_path": str(table_path),
        }
    )
    print(json.dumps(output, indent=2, sort_keys=True))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
