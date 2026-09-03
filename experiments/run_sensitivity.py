"""Run resumable synthetic sensitivity and Monte Carlo experiments."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
import statsmodels

from fair_lending.sensitivity.design import (
    ANALYSIS_VERSION,
    DETECTION_SIZE_GRID,
    DIRECT_GRID,
    MIXED_DIRECT_GRID,
    MIXED_UPSTREAM_GRID,
    SAMPLE_SIZE_GRID,
    UPSTREAM_GRID,
    build_experiment_design,
)
from fair_lending.sensitivity.figures import generate_figures
from fair_lending.sensitivity.outputs import write_output_tables
from fair_lending.sensitivity.runner import execute_design
from fair_lending.simulation.calibration import load_calibration_artifact
from fair_lending.simulation.config import PROJECT_ROOT, git_revision, package_version


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--experiment",
        choices=("direct", "upstream", "mixed", "sample-size", "all"),
        default="all",
    )
    value.add_argument("--replications", type=int)
    value.add_argument("--rows", type=int)
    value.add_argument("--seed", type=int, default=4_994)
    value.add_argument("--resume", action="store_true")
    value.add_argument("--quick", action="store_true")
    value.add_argument("--workers", type=int, default=1)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    experiment = args.experiment.replace("-", "_")
    design = build_experiment_design(
        experiment,
        replications=args.replications,
        rows=args.rows,
        master_seed=args.seed,
        quick=args.quick,
    )
    calibration = load_calibration_artifact()
    intercept = float(calibration["intercept"])
    revision, dirty = git_revision()
    started_at = datetime.now(timezone.utc)
    started = perf_counter()
    print(
        f"Sensitivity design: {len(design)} unique runs; workers={args.workers}; "
        f"resume={args.resume}; quick={args.quick}",
        flush=True,
    )
    completed, failed, resumed = execute_design(
        design,
        intercept=intercept,
        revision=revision,
        dirty=dirty,
        resume=args.resume,
        workers=args.workers,
    )
    table_paths = write_output_tables(completed)
    figure_paths = generate_figures(completed) if completed else {}
    finished_at = datetime.now(timezone.utc)
    metadata_path = PROJECT_ROOT / "results" / "metrics" / "sensitivity_experiment_metadata.json"
    metadata = {
        "experiment": experiment,
        "analysis_version": ANALYSIS_VERSION,
        "quick": args.quick,
        "requested_replications": args.replications,
        "requested_rows": args.rows,
        "master_seed": args.seed,
        "workers": args.workers,
        "resume": args.resume,
        "unique_design_runs": len(design),
        "completed_runs": len(completed),
        "failed_runs": len(failed),
        "resumed_runs": resumed,
        "runtime_seconds": perf_counter() - started,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "frozen_intercept": intercept,
        "calibration_fingerprint": calibration["config_fingerprint"],
        "grids": {
            "direct": list(DIRECT_GRID),
            "upstream": list(UPSTREAM_GRID),
            "mixed_direct": list(MIXED_DIRECT_GRID),
            "mixed_upstream": list(MIXED_UPSTREAM_GRID),
            "sample_size": list(SAMPLE_SIZE_GRID),
            "detection_size": list(DETECTION_SIZE_GRID),
        },
        "package_version": package_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "sklearn_version": sklearn.__version__,
        "statsmodels_version": statsmodels.__version__,
        "git_revision": revision,
        "git_worktree_dirty": dirty,
        "tables": {key: str(path.resolve()) for key, path in table_paths.items()},
        "figures": {key: str(path.resolve()) for key, path in figure_paths.items()},
        "failed_run_ids": [record["run_id"] for record in failed],
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "metadata": str(metadata_path),
                **{key: str(value) for key, value in table_paths.items()},
                **{f"figure_{key}": str(value) for key, value in figure_paths.items()},
            },
            indent=2,
        )
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
