"""Bounded replay of original dollar-scale MILPs from recovery checkpoints.

Diagnostic only: never replaces production decisions or Monte Carlo records.
Stops at the first invalid result and saves enough evidence to inspect it.
"""

import argparse
import json
from decimal import Decimal
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from fair_lending.economic_lending.recovery import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = [(p, json.loads(p.read_text())) for p in args.directory.glob("*.json")]
    # Match original loop order, so the first failure is directly interpretable.
    policies = {"true_risk_reference": 0, "traditional": 1, "ml": 2}
    budgets = {"nonbinding_100pct": 0, "moderate_40pct": 1, "tight_20pct": 2}
    def key(item):
        c = item[1]
        family = next(f for f in policies if f"_{f}_" in c["policy_id"])
        return c["world_id"], policies[family], budgets[c["budget_id"]]
    records = []
    for path, case in sorted(cases, key=key):
        p = np.array([r["requested_principal"] for r in case["loans"]])
        v = np.array([r["expected_profit_perceived"] for r in case["assessments"]])
        budget = case["bank_budget"]
        result = milp(c=-v, integrality=np.ones(len(p)), bounds=Bounds(np.zeros(len(p)), (v > 0).astype(float)),
                      constraints=LinearConstraint(p.reshape(1, -1), lb=-np.inf, ub=budget),
                      options={"disp": False, "mip_rel_gap": 0., "time_limit": 60.})
        selected = result.x > .5 if result.x is not None else None
        exact_residual = float(sum((Decimal(str(x)) for x in p[selected]), Decimal(0)) - Decimal(str(budget))) if selected is not None else None
        row = {"case": str(path), "world_id": case["world_id"], "policy_id": case["policy_id"],
               "budget_id": case["budget_id"], "status": int(result.status), "message": str(result.message),
               "budget_residual_dollars": exact_residual,
               "original_tolerance_dollars": max(1e-6, abs(budget)*1e-10),
               "candidate": result.x.tolist() if result.x is not None else None,
               "integrality_error": float(np.max(np.abs(result.x - np.round(result.x)))) if result.x is not None else None}
        records.append(row)
        failed = not result.success or exact_residual > row["original_tolerance_dollars"]
        atomic_json(args.output, {"diagnostic_only": True, "attempted": len(records), "original_validation_failure_reproduced": bool(failed), "records": records})
        print(f"{case['world_id']} {case['policy_id']} {case['budget_id']}: status={result.status}, residual=${exact_residual}", flush=True)
        if failed:
            break


if __name__ == "__main__":
    main()
