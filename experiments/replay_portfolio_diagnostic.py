"""Replay saved solver inputs only: no applicant generation or model fitting."""

import argparse
import json
from pathlib import Path

import pandas as pd

from fair_lending.economic_lending.portfolio import solve_fixed_request_portfolio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--time-limit", type=float, default=60)
    args = parser.parse_args()
    case = json.loads(args.case.read_text())
    solution = solve_fixed_request_portfolio(
        pd.DataFrame(case["loans"]), pd.DataFrame(case["assessments"]), case["bank_budget"],
        world_id=case["world_id"], policy_id=case["policy_id"], budget_id=case["budget_id"],
        time_limit=args.time_limit, diagnostic_directory=args.output_directory,
    )
    print(json.dumps(solution.solver_metadata, indent=2))


if __name__ == "__main__":
    main()
