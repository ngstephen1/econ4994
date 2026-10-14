"""Exactly 200 fixed-population noise replications, with validated pilot reuse."""
from __future__ import annotations

import argparse
import json
import time
import uuid

from fair_lending.professor_model import noise_full_io as io
from fair_lending.professor_model import noise_pilot_io as pio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--resume", action="store_true")
    modes.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    config = io.load_config()
    if args.summarize_only:
        manifest = io.prepare_manifest(config, resume=True)
        report = io.summarize(manifest)
        print(json.dumps({k: v for k, v in report.items() if k != "replication_result_hashes"}, indent=2))
        return 0
    observed, truth, contracts, budget = pio.load_fixed_inputs(config)
    manifest = io.prepare_manifest(config, resume=args.resume)
    reference, mask, row = io.prepare_reference(observed, truth, manifest)
    started, computed, reused = time.perf_counter(), 0, 0
    try:
        for i in range(200):
            old = io.resume_record(i, manifest) if args.resume else None
            if old is not None:
                reused += 1
            else:
                record = io.run_one(i, observed, truth, contracts, budget, reference, mask, row, manifest)
                computed += 1
                if record is None:
                    print(f"Replication {i}: FAILED; immutable attempt diagnostics retained", flush=True)
                    if i < 5:
                        raise RuntimeError("First-five pilot reproduction gate failed; broad computation stopped")
                else:
                    print(f"Replication {i}: success ({record['origin']}) in {record['runtime_seconds']:.2f}s", flush=True)
            if (i + 1) % 10 == 0:
                print(f"Progress: {i + 1}/200 indices processed; {time.perf_counter() - started:.1f}s this invocation", flush=True)
        report = io.summarize(manifest)
        print(json.dumps({k: v for k, v in report.items() if k not in ("replication_result_hashes", "frequencies")}, indent=2))
        return 0
    finally:
        pio.immutable_json(io.PROJECT_ROOT / io.RUN_DIRECTORY / "sessions" / f"session_{uuid.uuid4().hex}.json",
                           {"wall_seconds": time.perf_counter() - started, "computed_or_adopted": computed, "resumed_valid_records": reused,
                            "experiment_fingerprint": manifest["experiment_fingerprint"]})


if __name__ == "__main__":
    raise SystemExit(main())
