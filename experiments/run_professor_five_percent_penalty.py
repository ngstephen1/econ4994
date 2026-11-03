"""Recalculate only the five-percent-penalty policies, or read verified saved evidence."""
from __future__ import annotations
import argparse
import json
from fair_lending.professor_model import five_percent_penalty_io as io


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(io.CONFIG_PATH))
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args(argv)
    result = io.summarize_only(args.config) if args.summarize_only else io.run_experiment(args.config)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == io.COMPLETE else 2


if __name__ == "__main__":
    raise SystemExit(main())
