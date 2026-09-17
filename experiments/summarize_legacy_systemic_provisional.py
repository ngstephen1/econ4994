"""Inspect legacy records without treating them as provenance-validated runs.

Writes ONLY to results/recovery/legacy_provisional. No model or solver calls.
"""

import importlib.util
import json
from pathlib import Path

from fair_lending.economic_lending.config import PROJECT_ROOT
from fair_lending.economic_lending.recovery import atomic_json
from fair_lending.economic_lending.systemic_mc import monte_carlo_summary, records_to_frames


def main():
    records = [json.loads(p.read_text()) for p in sorted((PROJECT_ROOT / "results/metrics/systemic_mc_runs").glob("*.json"))]
    output = PROJECT_ROOT / "results/recovery/legacy_provisional"
    output.mkdir(parents=True, exist_ok=True)
    frames = records_to_frames(records)
    runs = frames["runs"]
    summary = monte_carlo_summary(runs)
    failed_indices = [r["replication_index"] for r in records if r["status"] != "success"]
    successful_count = sum(r["status"] == "success" for r in records)
    label = f"PROVISIONAL legacy successes only; missing indices {failed_indices}; source provenance not validated"
    summary["evidence_status"] = label
    summary["legacy_attempted_replications"] = len(records)
    summary.to_csv(output / "summary.csv", index=False)
    report = {
        "evidence_status": label,
        "attempted": len(records),
        "marked_successful": successful_count,
        "failed_indices": failed_indices,
        "portfolio_rows": len(runs),
        "missing_source_content_hashes": sum("provenance" not in r for r in records),
        "reported_over_budget_rows": int((runs.total_principal_funded > runs.bank_budget + 1e-6).sum()),
        "denominator_min": int(summary.n_successful.min()),
        "denominator_max": int(summary.n_successful.max()),
        "reuse_authorized": False,
    }
    # Use the corrected plotting code but label every plot as provisional.
    spec = importlib.util.spec_from_file_location("mc_runner", PROJECT_ROOT / "experiments/run_systemic_monte_carlo.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    original = runner.plt.Figure.savefig
    def marked_save(fig, *args, **kwargs):
        tick_count = max(len(ax.get_xticklabels()) for ax in fig.axes)
        if tick_count > 12:
            fig.set_size_inches(max(14, .95 * tick_count), 6)
            for ax in fig.axes:
                labels = [label.get_text().replace("moderate_40pct", "40% budget")
                          .replace("nonbinding_100pct", "100% budget")
                          .replace("tight_20pct", "20% budget") for label in ax.get_xticklabels()]
                ax.set_xticks(ax.get_xticks(), labels)
        fig.suptitle(f"PROVISIONAL: {successful_count} legacy successes; failed replications excluded; not validated for reuse", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, .94))
        return original(fig, *args, **kwargs)
    runner.plt.Figure.savefig = marked_save
    try:
        report["figures"] = runner._render_figures(frames, output)
    finally:
        runner.plt.Figure.savefig = original
    atomic_json(output / "audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
