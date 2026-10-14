"""Exactly five noise-only replications; no population generation or oracle fit."""
from __future__ import annotations

import argparse
import json
import time

from fair_lending.professor_model import noise_pilot as pilot
from fair_lending.professor_model import noise_pilot_io as io


def _rows(frame):
    return frame.astype(object).where(frame.notna(), None).to_dict("records")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true", help="Reuse matching immutable successes; run missing indices only")
    mode.add_argument("--summarize-only", action="store_true", help="Read saved records; no draws, fits or optimization")
    args = parser.parse_args()
    config = io.load_config()
    manifest = io.prepare_manifest(config, resume=args.resume or args.summarize_only)
    if args.summarize_only:
        print(json.dumps(io.summarize(manifest), indent=2, allow_nan=False))
        return 0
    observed, simulator_truth, contracts, budget = io.load_fixed_inputs(config)
    print(f"Fixed applicants: {len(observed):,}; cohorts: {observed.split.value_counts().to_dict()}", flush=True)
    print(f"Fixed saved capital: {budget!r}; preserved files: {len(manifest['identity']['original_checksums'])}", flush=True)
    for i in range(5):
        if io.record_path(i).exists():
            io.load_record(i, manifest)
            print(f"Replication {i}: reused validated immutable result", flush=True)
            continue
        start = time.perf_counter()
        targets, stream = pilot.draw_targets(simulator_truth, i)
        print(f"Replication {i}, child {stream['spawn_key']}: fitting both OLS models and cutoffs", flush=True)
        banks = pilot.fit_bank_policies(observed, targets, contracts, budget)
        frozen = io.freeze_bank(banks, i, stream, manifest)
        print(f"Replication {i}: bank decisions frozen; evaluating true profit", flush=True)
        metrics, groups, comparison, oracle = io.evaluate_frozen(banks, i, stream, observed, targets, contracts, budget, manifest)
        record = {"status": "success", "schema_version": config["schema_version"], "replication": i,
                  "experiment_fingerprint": manifest["experiment_fingerprint"], "stream": stream,
                  "banks": frozen["banks"], "policy_metrics": _rows(metrics), "group_approval": _rows(groups),
                  "comparison": comparison, "oracle_reference": oracle, "runtime_seconds": time.perf_counter() - start}
        io.save_record(record, manifest)
        print(f"Replication {i}: complete in {record['runtime_seconds']:.2f}s", flush=True)
        del targets, banks, metrics, groups
    print(json.dumps(io.summarize(manifest), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
