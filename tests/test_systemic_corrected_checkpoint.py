"""Focused tests for the five-run corrected checkpoint reporter."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fair_lending.economic_lending.config import PROJECT_ROOT


def checkpoint_module():
    spec = importlib.util.spec_from_file_location(
        "checkpoint", PROJECT_ROOT / "experiments/summarize_systemic_corrected_checkpoint.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checkpoint_indices_were_predeclared_without_outcome_selection():
    module = checkpoint_module()
    assert module.CHECKPOINT_INDICES == (0, 1, 2, 3, 13)


def test_corrected_replication_has_exactly_72_unique_complete_cells():
    module = checkpoint_module()
    records, _ = module.load_corrected_records(indices=(13,))
    frame = pd.DataFrame(records[0]["runs"])
    assert len(frame) == 72
    assert not frame.duplicated(list(module.REQUIRED_RUN_KEY)).any()
    assert frame[list(module.PRIMARY_DELTA_METRICS)].notna().all().all()
    assert np.isfinite(frame[list(module.PRIMARY_DELTA_METRICS)].to_numpy()).all()


def test_corrected_and_legacy_statuses_are_never_combined(monkeypatch):
    module = checkpoint_module()
    corrected = [{"status": "success", "runs": [], "pathway": [], "switcher_lending": [],
                  "model_performance": [], "ml_selections": [], "controlled_audits": [],
                  "run_id": "corrected", "replication_index": 1}]
    captured = module.records_to_frames(corrected)
    assert set(captured["replications"].run_id) == {"corrected"}
    assert module.EVIDENCE_STATUS.startswith("INTERIM CORRECTED")


def test_checkpoint_summary_rejects_nonfinite_input():
    module = checkpoint_module()
    with pytest.raises(ValueError, match="nonfinite"):
        module._stats(pd.Series([1.0, np.nan]))


def test_replication13_provenance_is_internally_consistent():
    module = checkpoint_module()
    records, _ = module.load_corrected_records()
    result = module.provenance_summary(records)
    assert result["all_consistent"]
    assert result["analysis_schema_version"] == "systemic-opportunity-mc-v2-recovery"


def test_legacy_comparison_is_diagnostic_and_never_authorizes_reuse():
    module = checkpoint_module()
    records, _ = module.load_corrected_records(indices=(13,))
    frame, report = module.compare_legacy(records)
    assert frame.loc[0, "legacy_status"] == "failed_solver"
    assert frame.loc[0, "corrected_status"] == "success"
    assert not report["legacy_scientifically_reusable"]
    assert "unavailable" in report["portfolio_decisions_comparison"]


def test_interim_summary_uses_five_corrected_records_only():
    module = checkpoint_module()
    records, _ = module.load_corrected_records()
    summary = module.build_interim_summary(records)
    assert set(summary.evidence_status) == {module.EVIDENCE_STATUS}
    assert summary.n_corrected.eq(5).all()
    assert "provisional_legacy" not in summary.to_csv(index=False)
