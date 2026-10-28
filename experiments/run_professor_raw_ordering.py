"""Run only the canonical and five predeclared raw-ordering diagnostic cases."""
import argparse
import json

from fair_lending.professor_model import raw_ordering_io as io
from fair_lending.professor_model import raw_ordering as science
from fair_lending.professor_model.raw_ordering_figures import render


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=science.CASES, help="single diagnostic case; default is all six")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summarize-only", action="store_true", help="read-only; never creates outputs or starts computation")
    args = parser.parse_args(argv)
    ctx = io.prepare(readonly=args.summarize_only)
    if not args.summarize_only:
        for case in ([args.case] if args.case else science.CASES):
            if io.run_case(ctx, case, resume=args.resume) is None:
                print(json.dumps(io.summarize(ctx, write=True), indent=2)); return 1
    report = io.summarize(ctx, write=not args.summarize_only)
    if not args.summarize_only and report["status"] == "complete_six_diagnostic_cases":
        short = ctx["manifest"]["experiment_fingerprint"][:16]
        figure = render(ctx["root"] / io.TABLES / short, ctx["root"] / io.FIGURES / short / "diagnostic_portfolio_changes.png")
        print(f"Figure: {figure}")
    print(json.dumps({"report": report, "metrics_directory": str(ctx["directory"])}, indent=2))
    return 0 if report["status"] == "complete_six_diagnostic_cases" or args.case else 2


if __name__ == "__main__":
    raise SystemExit(main())
