"""Step 9C only: frozen score transformations, cutoff audit, then truth release."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fair_lending.professor_model import mechanism_audit as audit
from fair_lending.professor_model import mechanism_audit_io as io


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=io.CONFIG_PATH)
    args = parser.parse_args()
    config = io.load_config(args.config)
    io.require_new_outputs()
    originals = io.preservation_manifest()
    inputs, hashes = io.load_inputs(config, originals)
    print(f"Verified {len(originals)} pre-existing files and frozen evaluation risks.", flush=True)
    risks, derivation = audit.build_variants(inputs["risks"])
    print("Predeclared 1% zero-only floor (derived without truth): " + json.dumps(derivation), flush=True)
    banks = {}
    for variant in audit.POLICIES:
        view = risks.loc[risks.variant_id == variant, ["applicant_id", "variant_id", "score"]].rename(columns={"score": "estimated_PD_bounded"})
        banks[variant] = audit.optimize_policy(inputs["contracts"], view, inputs["budget"])
        selection = banks[variant]["selection"]
        print(f"{variant}: tau={selection['optimal_tau']!r}, approved={selection['number_approved']}; exact enumeration passed.", flush=True)
    io.freeze_banks(inputs, risks, derivation, banks, config, hashes, originals, config_path=args.config)
    print("All A/D/E decisions saved and frozen. Releasing evaluator-only truth and frozen references.", flush=True)
    saved_banks, references, truth = io.load_evaluator(inputs, config, config_path=args.config)
    report = io.save_evaluation(inputs, risks, derivation, saved_banks, references, truth, config, originals, config_path=args.config)
    print(json.dumps({"validation": report, "outputs": {k: str(v) for k, v in io.paths().items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
