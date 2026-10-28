"""Step 10B: canonical plus saved draws; never fit or regenerate noise."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fair_lending.professor_model import fixed_coefficient_io as io


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=io.CONFIG)
    parser.add_argument("--case", choices=("canonical", "noise", "all"), default="all")
    parser.add_argument("--replication", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.replication is not None and (args.case != "noise" or not 0 <= args.replication < 200): parser.error("--replication requires --case noise and index 0..199")
    ctx = io.prepare(config_path=args.config, readonly=args.summarize_only)
    if not args.summarize_only:
        cases = ["canonical"] if args.case == "canonical" else [f"noise_{args.replication:03d}"] if args.replication is not None else list(io.ALL_CASES[1:]) if args.case == "noise" else list(io.ALL_CASES)
        for case in cases:
            if io.run_case(ctx, case, resume=args.resume) is None:
                print(json.dumps(io.summarize(ctx), indent=2)); return 1
    report = io.summarize(ctx, write=not args.summarize_only)
    print(json.dumps({"report": report, "metrics_directory": str(ctx["directory"])}, indent=2))
    return 0 if report["status"] == "complete_201_cases" or args.case != "all" else 2


if __name__ == "__main__":
    raise SystemExit(main())
