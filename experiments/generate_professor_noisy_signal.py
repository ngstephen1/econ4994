"""Step 4: inspect existing data, generate raw PN, validate, and export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fair_lending.professor_model.noisy_signal import (
    SIGNAL_CONFIG_PATH, generate_noisy_signal, load_signal_config, noisy_audit_preview,
)
from fair_lending.professor_model.signal_io import load_signal_inputs, save_signal_artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=SIGNAL_CONFIG_PATH)
    parser.add_argument("--overwrite", action="store_true", help="Replace only existing Step 4 outputs")
    args = parser.parse_args()
    config = load_signal_config(args.config)
    applicants, input_hashes, preview_ids, schemas = load_signal_inputs(config)
    for item in schemas:
        print(item)
    print("First 10 joined input rows BEFORE noisy-measurement calculation:")
    print(applicants.head(10).to_string(index=False))
    targets, noise, _ = generate_noisy_signal(applicants, config)
    print("First 10 newly generated audit rows (EVALUATOR ONLY):")
    print(noisy_audit_preview(applicants, targets, noise, preview_ids).head(10).to_string(index=False))
    paths, report = save_signal_artifacts(applicants, targets, noise, config, input_hashes, preview_ids, overwrite=args.overwrite, config_path=args.config)
    print(json.dumps({"artifacts": {key: str(path) for key, path in paths.items()}, "validation": report}, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
