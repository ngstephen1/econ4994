"""Generate and validate only professor-model applicant characteristics and C."""

from __future__ import annotations

import argparse
import json

from fair_lending.professor_model import generate_applicants, load_population_config, save_population
from fair_lending.professor_model.config import DEFAULT_CONFIG_PATH


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true", help="Explicitly regenerate existing professor-model outputs")
    args = parser.parse_args()
    config = load_population_config(args.config)
    bank, private, metadata = generate_applicants(config, n_samples=args.rows, seed=args.seed)
    # These are identity checks, not a repeated-seed research experiment.
    same_bank, same_private, _ = generate_applicants(metadata["resolved_configuration"])
    different_bank, different_private, _ = generate_applicants(metadata["resolved_configuration"], seed=metadata["random_seed"] + 1)
    checks = {
        "same_seed_identical_bank": bank.equals(same_bank),
        "same_seed_identical_private": private.equals(same_private),
        "different_seed_changes_bank": not bank.equals(different_bank),
        "different_seed_changes_private": not private.equals(different_private),
    }
    del same_bank, same_private, different_bank, different_private
    paths, report = save_population(bank, private, metadata, reproducibility_checks=checks, overwrite=args.overwrite)
    print(json.dumps({"artifacts": {name: str(path) for name, path in paths.items()}, "validation": report}, indent=2, sort_keys=True, allow_nan=False))
    # Permitted untruncated DTI tails are warnings, not validation failures.
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
