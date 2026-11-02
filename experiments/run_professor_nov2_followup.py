"""Run the two approved November worlds, or read their validated saved summary."""
from __future__ import annotations

import argparse
import json

from fair_lending.professor_model import nov2_followup_io as io


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(io.CONFIG_PATH))
    parser.add_argument("--summarize-only", action="store_true", help="Read-only: no simulation, fit, selection, models, or writes")
    args = parser.parse_args(argv)
    report = io.summarize_only(args.config) if args.summarize_only else io.run_experiment(args.config)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
