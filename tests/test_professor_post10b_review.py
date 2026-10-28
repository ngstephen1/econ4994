"""Independent accounting/partition checks for the post-hoc saved-file review."""
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_spec = spec_from_file_location("post10b_review", Path(__file__).resolve().parents[1] / "experiments/review_professor_fixed_coefficient.py")
review = module_from_spec(_spec)
_spec.loader.exec_module(review)


@pytest.fixture
def panel():
    rows = []
    for case, profits in (("draw_0", (0., 100., 90.)), ("draw_1", (80., 70., 85.))):
        for policy, profit in zip(review.POLICIES, profits):
            rows.append({"case_id": case, "policy": policy, "true_profit_USD": profit,
                         "number_approved": int(profit > 0), "reject_all_selected": profit == 0,
                         "zero_block_feasible": profit > 0, "zero_block_principal_USD": 11. if profit == 0 else 5.,
                         "funded_principal_USD": 0. if profit == 0 else 9., "available_capital_USD": 10.})
    return pd.DataFrame(rows)


def paired(panel, first=review.Z, second=review.W):
    left = panel[panel.policy.eq(first)].set_index("case_id")
    right = panel[panel.policy.eq(second)].set_index("case_id")
    return pd.DataFrame({"case_id": left.index, "comparison": f"{first}_minus_{second}",
                         "policy_first": first, "policy_second": second,
                         "true_profit_difference_USD": (left.true_profit_USD - right.true_profit_USD).to_numpy()})


def test_full_denominator_retains_tail_and_negative_outcome(panel):
    result = review.tail_attribution(panel, paired(panel))[0]
    assert result["n"] == 2
    assert result["full_mean_USD"] == 45.  # (100 - 10)/2
    assert result["discordant_reject_all_cases"] == ["draw_0"]
    assert result["discordant_contribution_to_full_mean_USD"] == 50.
    assert result["remaining_contribution_to_full_mean_USD"] == -5.
    assert result["remaining_conditional_mean_USD"] == -10.


def test_all_discordant_cases_have_undefined_remaining_mean(panel):
    p = panel[panel.case_id.eq("draw_0")]
    assert review.tail_attribution(p, paired(p))[0]["remaining_conditional_mean_USD"] is None


def test_partition_rejects_incorrect_saved_pair_arithmetic(panel):
    q = paired(panel); q.loc[0, "true_profit_difference_USD"] += 1
    with pytest.raises(AssertionError):
        review.tail_attribution(panel, q)


def test_partition_rejects_duplicate_draws(panel):
    q = paired(panel)
    with pytest.raises(ValueError, match="duplicate"):
        review.tail_attribution(panel, pd.concat([q, q.iloc[:1]]))


@pytest.mark.parametrize("change", ["duplicate", "missing", "over_budget", "nonfinite", "string_boolean", "wrong_reject_all"])
def test_policy_panel_integrity(panel, change):
    p = panel.copy()
    if change == "duplicate": p = pd.concat([p, p.iloc[:1]])
    if change == "missing": p = p.iloc[1:]
    if change == "over_budget": p.loc[0, "funded_principal_USD"] = 11.
    if change == "nonfinite": p.loc[0, "true_profit_USD"] = np.nan
    if change == "string_boolean": p["reject_all_selected"] = p.reject_all_selected.astype(str)
    if change == "wrong_reject_all": p.loc[0, "reject_all_selected"] = False
    with pytest.raises(ValueError):
        review.validate_policy_cells(p, {"draw_0", "draw_1"})


@pytest.mark.parametrize("beta", [.2, -.2])
def test_projection_from_saved_coefficients_preserves_signed_ols_identity(beta):
    aware = {"const": .1, "Y": -.001, "D": .01, "C": .02, "B": beta}
    delta = np.array([.15, .0001, -.01, .005])
    blind = {k: aware[k] + beta * d for k, d in zip(review.ORDER, delta)}
    np.testing.assert_allclose(review.implied_group_projection(aware, blind), delta, atol=1e-14, rtol=0)


def test_zero_B_coefficient_does_not_invent_projection():
    assert review.implied_group_projection({"B": 0.}, {}) is None


def test_tampered_evidence_rejected(tmp_path):
    p = tmp_path / "evidence.json"
    p.write_text(json.dumps({"value": 2, "content_fingerprint": review.fingerprint({"value": 1})}))
    with pytest.raises(ValueError, match="fingerprint"):
        review.sealed_json(p)


def test_new_review_is_immutable_and_deterministic(tmp_path):
    report = {"scope": "toy", "n": 2}
    p = review.write_review(report, tmp_path)
    before = (p.stat().st_mtime_ns, p.read_bytes())
    assert review.write_review(report, tmp_path) == p
    assert (p.stat().st_mtime_ns, p.read_bytes()) == before
    p.write_text("tampered")
    with pytest.raises(FileExistsError):
        review.write_review(report, tmp_path)


def test_default_cli_cannot_write(monkeypatch, capsys):
    sample = {k: {} for k in ("scope", "checks", "economics", "tail_attribution")}
    monkeypatch.setattr(review, "build_review", lambda: sample)
    def forbidden(*args, **kwargs):
        raise AssertionError("read-only invocation attempted a write")
    monkeypatch.setattr(review, "write_review", forbidden)
    review.main([])
    assert "scope" in capsys.readouterr().out
