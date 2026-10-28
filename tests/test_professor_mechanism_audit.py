"""Step 9C small deterministic fixtures; no population generation or fitting."""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import json
import math

import numpy as np
import pandas as pd
import pytest
import yaml

from fair_lending.professor_model import mechanism_audit as audit
from fair_lending.professor_model import mechanism_audit_io as io
from fair_lending.professor_model import revised_lending as prior
from fair_lending.professor_model import revised_scenarios as scenarios
from fair_lending.professor_model import group_blind
from fair_lending.professor_model.bank_regression import MODEL_ID, ids_fingerprint
from fair_lending.professor_model.contracts import make_loan_contracts
from fair_lending.professor_model.oracle import optimize_oracle
from fair_lending.professor_model.risk_io import file_hash


def frozen(applicants, raw):
    aware = applicants[["applicant_id"]].assign(split="evaluation", model_id=MODEL_ID,
        bank_PD_raw=.02, bank_PD_bounded=.02, prediction_out_of_bounds=False)
    blind = applicants[["applicant_id"]].assign(split="evaluation", model_id=group_blind.MODEL_ID,
        bank_PD_group_blind_raw=raw, bank_PD_group_blind_bounded=np.clip(raw, 0, 1),
        prediction_out_of_bounds=(np.asarray(raw) < 0) | (np.asarray(raw) > 1))
    return scenarios.build_scenario_risk_estimates(applicants, aware, blind)


@pytest.fixture
def toy():
    applicants = pd.DataFrame({"applicant_id": [f"audit_{i}" for i in range(8)],
        "Y": np.arange(20, 100, 10) * 1000., "D": .3, "B": np.arange(8) % 2, "C": .1})
    contracts = make_loan_contracts(applicants[["applicant_id", "Y"]], T=30, theta=.1, r=.05)
    true_profit = np.arange(1, 9) * 100.
    return {"applicants": applicants, "contracts": contracts,
        "risks": frozen(applicants, np.array([-.004, .015, .003, .025, .001, .035, .04, .06])),
        "audit": applicants[["applicant_id", "B"]], "budget": .8 * math.fsum(contracts.L_i),
        "truth": pd.DataFrame({"applicant_id": applicants.applicant_id, "true_PD": np.arange(1, 9) / 1000,
            "expected_receipts_true": contracts.L_i + true_profit, "expected_profit_true": true_profit})}


def policy_view(risks, variant):
    return risks.loc[risks.variant_id == variant, ["applicant_id", "variant_id", "score"]].rename(columns={"score": "estimated_PD_bounded"})


def banks_for(toy, risks=None, budget=None):
    risks = audit.build_variants(toy["risks"])[0] if risks is None else risks
    return {v: audit.optimize_policy(toy["contracts"], policy_view(risks, v), toy["budget"] if budget is None else budget) for v in audit.POLICIES}


def old_policies(toy, budget):
    result = {}
    for s in prior.SCENARIO_IDS:
        bank = prior.optimize_scenario(toy["contracts"], toy["risks"].loc[toy["risks"].scenario_id == s, prior.RISK_VIEW], budget)
        result[s] = prior.evaluation_policy(bank["decisions"], bank["selection"], score_column="estimated_PD_bounded", role="frozen_reference")
    for s, ref in ((prior.SCENARIO_IDS[0], prior.REFERENCE_IDS[0]), (prior.SCENARIO_IDS[2], prior.REFERENCE_IDS[1])):
        result[ref] = deepcopy(result[s])
    decisions, _, selection = optimize_oracle(toy["contracts"], toy["truth"], budget)
    result[prior.REFERENCE_IDS[2]] = prior.evaluation_policy(decisions, selection, score_column="true_PD", role="frozen_reference", oracle=True)
    return result


@pytest.fixture
def saved(toy, tmp_path, monkeypatch):
    config = io.load_config(); config["evaluation_rows"] = 8
    cp = tmp_path / "configs/professor_model/mechanism_audit.yaml"
    cp.parent.mkdir(parents=True); cp.write_text(yaml.safe_dump(config))
    source = tmp_path / "frozen_evidence.txt"; source.write_text("Unchanged source evidence.\n")
    originals = {"frozen_evidence.txt": file_hash(source)}
    io.prior.write_json(io.paths(tmp_path)["preservation"], {"schema_version": "professor-mechanism-preservation-v1", "checksums": originals})
    inputs = {**toy, "budget": scenarios.EXACT_BUDGET, "old_config": {}, "old_config_path": cp,
        "evaluation_ID_fingerprint": ids_fingerprint(toy["contracts"].applicant_id)}
    policies = old_policies(toy, inputs["budget"])
    # Stub the PREVIOUS step's independently tested truth/reference loader only.
    # New freeze, hashes, data round trips and truth-access gate remain real.
    monkeypatch.setattr(io.prior, "load_evaluator_inputs", lambda *a, **k: (policies, toy["truth"], {}))
    return tmp_path, config, cp, inputs, originals.copy(), originals


def test_predeclared_config():
    config = io.load_config()
    assert config["expected_capital_USD"] == scenarios.EXACT_BUDGET
    assert config["positive_floor"]["quantile"] == .01
    assert config["positive_floor"]["replace"] == "exact_zeros_of_frozen_direct_bounded_scores_only"
    assert config["positive_floor"]["outcome_tuning"] is False


@pytest.mark.parametrize("change", ["quantile", "replace", "budget"])
def test_no_floor_or_budget_retuning(tmp_path, change):
    config = io.load_config()
    if change == "budget": config["expected_capital_USD"] -= 1
    else: config["positive_floor"][change] = .05 if change == "quantile" else "all_low_scores"
    path = tmp_path / "config.yaml"; path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError): io.load_config(path)


def test_exact_A_and_E_reproduction(toy):
    risks, derivation = audit.build_variants(toy["risks"])
    for variant, scenario in ((audit.A, prior.SCENARIO_IDS[1]), (audit.E, prior.SCENARIO_IDS[2])):
        np.testing.assert_array_equal(risks.loc[risks.variant_id == variant, "score"], toy["risks"].loc[toy["risks"].scenario_id == scenario, "estimated_PD_bounded"])
    assert audit.validate_variants(risks, toy["risks"], derivation)["exact_A_and_E_reproduction"]


def test_raw_diagnostic_and_group_adjustment(toy):
    risks, _ = audit.build_variants(toy["risks"])
    raw = risks.loc[risks.variant_id == audit.B]
    base = toy["risks"].loc[toy["risks"].scenario_id == prior.SCENARIO_IDS[2]]
    np.testing.assert_array_equal(raw.score.to_numpy(), base.estimated_PD_adjusted_raw.to_numpy() - .1 * base.B.to_numpy())
    assert (raw.score < 0).sum() == 5
    assert not raw.eligible_for_lending.any()
    assert not raw.lower_clipped.any()


def test_floor_derived_from_positive_control_and_only_zeros_change(toy):
    risks, d = audit.build_variants(toy["risks"])
    a = risks.loc[risks.variant_id == audit.A]; e = risks.loc[risks.variant_id == audit.E]
    variant = risks.loc[risks.variant_id == audit.D]
    expected = np.quantile(e.loc[e.score > 0, "score"], .01, method="linear")
    assert expected == d["floor_raw_probability"]
    assert expected == pytest.approx(.00112, abs=1e-18)
    assert d["strictly_positive_source_count"] == 7
    assert d["affected_applicants"] == 5 and d["B1_affected"] == 4 and d["B0_affected"] == 1
    zero = a.score.to_numpy() == 0
    np.testing.assert_array_equal(variant.score.to_numpy()[~zero], a.score.to_numpy()[~zero])
    assert (variant.score.to_numpy()[zero] == expected).all()
    assert ((variant.score > 0) & (variant.score < expected)).sum() == 1


def test_floor_undefined_if_no_positive_control(toy):
    bad = frozen(toy["applicants"], np.full(8, -.01))
    with pytest.raises(ValueError, match="floor undefined"): audit.build_variants(bad)


def test_C_equals_A_without_upper_exceedance(toy):
    risks, d = audit.build_variants(toy["risks"])
    assert d["C_exactly_equals_A"] and d["direct_raw_above_one"] == 0
    np.testing.assert_array_equal(risks.loc[risks.variant_id == audit.A, "score"], risks.loc[risks.variant_id == audit.C, "score"])
    assert not risks.loc[risks.variant_id == audit.C, "eligible_for_lending"].any()


def test_upper_clipping_and_C_keeps_invalid_upper_diagnostic(toy):
    original = frozen(toy["applicants"], np.array([1.2, 1.3, .003, .025, .001, .035, .04, .06]))
    risks, d = audit.build_variants(original)
    assert d["direct_raw_above_one"] == 2 and not d["C_exactly_equals_A"]
    assert risks.loc[risks.variant_id == audit.A, "score"].max() == 1
    c = risks.loc[risks.variant_id == audit.C]
    assert (c.score > 1).sum() == 2 and not c.eligible_for_lending.any()
    assert not c.loc[c.score > 1, "is_valid_probability"].any()
    assert not c.upper_clipped.any()


@pytest.mark.parametrize("variant", [audit.B, audit.C])
def test_diagnostic_never_becomes_policy_even_if_numbers_are_bounded(toy, variant):
    view = pd.DataFrame({"applicant_id": toy["contracts"].applicant_id, "variant_id": variant, "estimated_PD_bounded": .01})
    with pytest.raises(ValueError, match="A/D/E"): audit.optimize_policy(toy["contracts"], view, toy["budget"])


@pytest.mark.parametrize("column", ["B", "true_PD", "PN", "epsilon", "expected_profit_true"])
def test_policy_allowlist_rejects_group_or_hidden_data(toy, column):
    risks, _ = audit.build_variants(toy["risks"])
    view = policy_view(risks, audit.A).assign(**{column: 0})
    with pytest.raises(ValueError): audit.optimize_policy(toy["contracts"], view, toy["budget"])


def test_variants_have_same_IDs_and_valid_policy_probabilities(toy):
    risks, _ = audit.build_variants(toy["risks"])
    assert len(risks) == 40 and not risks.duplicated(["variant_id", "applicant_id"]).any()
    for variant in audit.VARIANTS:
        assert set(risks.loc[risks.variant_id == variant, "applicant_id"]) == set(toy["contracts"].applicant_id)
    assert risks.loc[risks.variant_id.isin(audit.POLICIES), "is_valid_probability"].all()


def test_reproducible_without_mutating_inputs(toy):
    before = {k: v.copy(deep=True) for k, v in toy.items() if isinstance(v, pd.DataFrame)}
    first, d1 = audit.build_variants(toy["risks"])
    second, d2 = audit.build_variants(toy["risks"].sample(frac=1, random_state=4994))
    pd.testing.assert_frame_equal(first, second, check_exact=True); assert d1 == d2
    banks = banks_for(toy, first); rerun = banks_for(toy, second)
    for v in audit.POLICIES:
        assert banks[v]["selection"] == rerun[v]["selection"]
        pd.testing.assert_frame_equal(banks[v]["decisions"], rerun[v]["decisions"], check_exact=True)
    for k, frame in before.items(): pd.testing.assert_frame_equal(frame, toy[k], check_exact=True)


def test_exact_budget_and_tied_block_boundary(toy):
    risks, d = audit.build_variants(toy["risks"])
    zero_ids = risks.loc[(risks.variant_id == audit.A) & risks.score.eq(0), "applicant_id"]
    exact = sum((Fraction(float(v)) for v in toy["contracts"].loc[toy["contracts"].applicant_id.isin(zero_ids), "L_i"]), Fraction())
    budget = float(exact)
    bank = audit.optimize_policy(toy["contracts"], policy_view(risks, audit.A), budget)
    assert bank["selection"]["total_funded_principal"] == budget
    assert bank["selection"]["number_approved"] == len(zero_ids)
    assert bank["selection"]["selected_cutoff_group_size"] == len(zero_ids)
    under = audit.optimize_policy(toy["contracts"], policy_view(risks, audit.A), np.nextafter(budget, 0.))
    assert under["selection"]["optimal_tau"] is None and not under["decisions"].approved.any()
    assert bank["checks"]["independent_exact_enumeration_passed"]


def test_positive_floor_preserves_shared_tie_and_prefix_cost(toy):
    risks, d = audit.build_variants(toy["risks"]); banks = banks_for(toy, risks)
    decisions = pd.concat([b["decisions"] for b in banks.values()], ignore_index=True)
    table = audit.tie_diagnostics(risks, decisions, toy["contracts"], banks, toy["budget"], d["floor_raw_probability"]).set_index("variant_id")
    assert table.loc[audit.A, "zero_block_count"] == table.loc[audit.D, "floor_block_count"] == 5
    assert table.loc[audit.D, "zero_block_count"] == 0
    assert table.loc[audit.D, "positive_scores_below_floor"] == 1
    assert table.loc[audit.D, "floor_prefix_applicants"] == 6
    assert table.loc[audit.D, "floor_prefix_principal_USD"] > table.loc[audit.D, "floor_block_principal_USD"]
    assert table.loc[audit.D, "floor_block_approved"] in (0, 5)


def test_floor_block_feasible_alone_but_prefix_infeasible(toy):
    risks, d = audit.build_variants(toy["risks"])
    group = risks.loc[(risks.variant_id == audit.D) & risks.score.eq(d["floor_raw_probability"])]
    block_cost = math.fsum(toy["contracts"].loc[toy["contracts"].applicant_id.isin(group.applicant_id), "L_i"])
    budget = block_cost + 1
    banks = banks_for(toy, risks, budget)
    decisions = pd.concat([b["decisions"] for b in banks.values()], ignore_index=True)
    row = audit.tie_diagnostics(risks, decisions, toy["contracts"], banks, budget, d["floor_raw_probability"]).set_index("variant_id").loc[audit.D]
    assert row.floor_block_feasible and not row.floor_prefix_feasible
    assert row.floor_block_action == "excluded_whole" and row.number_approved == 1
    assert not row.reject_all_selected  # Lower positive scores can still be funded.


def test_risk_diagnostic_units_and_counts(toy):
    risks, _ = audit.build_variants(toy["risks"])
    table = audit.risk_diagnostics(risks).set_index(["variant_id", "scope"])
    row = table.loc[(audit.B, "B=1")]
    raw = risks.loc[(risks.variant_id == audit.B) & risks.B.eq(1), "score"]
    assert row.mean_score_pp == 100 * raw.mean()
    assert row.negative_score_count == 4
    assert len(table) == 15


def test_evaluation_aggregates_group_gaps_profit_and_overlap(toy):
    banks = banks_for(toy)
    references = {k: v for k, v in old_policies(toy, toy["budget"]).items() if k in prior.REFERENCE_IDS}
    portfolio, groups, overlap = audit.evaluate_policies(banks, references, toy["contracts"], toy["audit"], toy["truth"], toy["budget"])
    assert len(portfolio) == 6 and len(groups) == 12 and len(overlap) == 60
    for v in audit.POLICIES:
        decisions = banks[v]["decisions"].set_index("applicant_id")
        ids = set(decisions.index[decisions.approved]); row = portfolio.set_index("policy_id").loc[v]
        assert row.true_expected_portfolio_profit_USD == math.fsum(toy["truth"].loc[toy["truth"].applicant_id.isin(ids), "expected_profit_true"])
        g = groups.loc[groups.policy_id == v].set_index("B")
        assert row.approval_gap_pp == 100 * (g.loc[1, "approval_rate"] - g.loc[0, "approval_rate"])
    pair = overlap.loc[(overlap.policy_first == audit.D) & (overlap.policy_second == audit.A)].set_index("allocation_category")
    a = set(banks[audit.D]["decisions"].loc[lambda f: f.approved, "applicant_id"])
    b = set(banks[audit.A]["decisions"].loc[lambda f: f.approved, "applicant_id"])
    assert pair.loc["approved_by_both", "n_applicants"] == len(a & b)
    assert pair.loc["first_only", "n_applicants"] == len(a - b)
    assert pair.loc["second_only", "n_applicants"] == len(b - a)
    assert pair.loc["approved_by_both", "jaccard_overlap"] == len(a & b) / len(a | b)
    assert pair.loc["rejected_by_both", "profit_scope"] == "hypothetical_unfunded_by_both"
    assert pair.loc["rejected_by_both", "first_policy_expected_profit_contribution_USD"] == 0


def test_truth_blocked_without_new_freeze(saved, monkeypatch):
    root, config, cp, inputs, _, _ = saved
    monkeypatch.setattr(io.prior, "load_evaluator_inputs", lambda *a, **k: pytest.fail("truth reached before new freeze"))
    with pytest.raises(ValueError, match="frozen before truth"): io.load_evaluator(inputs, config, root=root, config_path=cp)


def test_lifecycle_roundtrips_hashes_preservation_and_no_reoptimization(saved, monkeypatch):
    root, config, cp, inputs, hashes, originals = saved
    risks, d = audit.build_variants(inputs["risks"]); banks = banks_for(inputs, risks)
    io.freeze_banks(inputs, risks, d, banks, config, hashes, originals, root=root, config_path=cp)
    monkeypatch.setattr(prior, "choose_cutoff", lambda *a, **k: pytest.fail("post-freeze optimization"))
    saved_banks, refs, truth = io.load_evaluator(inputs, config, root=root, config_path=cp)
    report = io.save_evaluation(inputs, risks, d, saved_banks, refs, truth, config, originals, root=root, config_path=cp)
    assert report["all_originals_byte_identical"] and report["pair_count"] == 15
    assert report["model_fits"] == report["random_draws"] == report["oracle_reoptimizations"] == 0
    assert all(path.exists() for path in io.paths(root).values())
    manifest = json.loads(io.paths(root)["manifest"].read_text())
    for item in manifest["artifacts"].values(): assert file_hash(root / item["path"]) == item["sha256"]
    io.prior.verify_hashes(originals, root)
    from PIL import Image
    for name in ("risk_figure", "tie_figure", "outcome_figure"):
        with Image.open(io.paths(root)[name]) as image: image.verify()
    with pytest.raises(FileExistsError): io.require_new_outputs(root)


def test_tampered_decisions_block_truth(saved, monkeypatch):
    root, config, cp, inputs, hashes, originals = saved
    risks, d = audit.build_variants(inputs["risks"])
    io.freeze_banks(inputs, risks, d, banks_for(inputs, risks), config, hashes, originals, root=root, config_path=cp)
    with io.paths(root)["decisions_csv"].open("a") as stream: stream.write("tampered\n")
    monkeypatch.setattr(io.prior, "load_evaluator_inputs", lambda *a, **k: pytest.fail("truth gate failed"))
    with pytest.raises(ValueError, match="checksum mismatch"): io.load_evaluator(inputs, config, root=root, config_path=cp)


@pytest.mark.parametrize("field", ["decisions", "assessments", "derivation"])
def test_mutated_in_memory_bundle_rejected_before_writes(saved, field):
    root, config, cp, inputs, hashes, originals = saved
    risks, d = audit.build_variants(inputs["risks"]); banks = banks_for(inputs, risks)
    if field == "decisions": banks[audit.A][field].loc[0, "approved"] = not banks[audit.A][field].loc[0, "approved"]
    elif field == "assessments": banks[audit.A][field].loc[0, "expected_profit_bank"] += 100
    else: d["floor_raw_probability"] *= 2
    with pytest.raises((ValueError, AssertionError)):
        io.freeze_banks(inputs, risks, d, banks, config, hashes, originals, root=root, config_path=cp)
    assert not io.paths(root)["freeze"].exists()
    assert not io.paths(root)["risk_csv"].exists()


def test_changed_original_rejected(saved):
    root, _, _, _, _, _ = saved
    assert len(io.preservation_manifest(root)) == 1
    (root / "frozen_evidence.txt").write_text("changed")
    with pytest.raises(ValueError, match="checksum mismatch"): io.preservation_manifest(root)


def test_changed_source_invalidates_truth_access(saved, monkeypatch):
    root, config, cp, inputs, hashes, originals = saved
    risks, d = audit.build_variants(inputs["risks"])
    io.freeze_banks(inputs, risks, d, banks_for(inputs, risks), config, hashes, originals, root=root, config_path=cp)
    monkeypatch.setattr(io, "source_hashes", lambda: {"changed_implementation": "different_hash"})
    monkeypatch.setattr(io.prior, "load_evaluator_inputs", lambda *a, **k: pytest.fail("truth gate failed"))
    with pytest.raises(ValueError, match="source fingerprint"): io.load_evaluator(inputs, config, root=root, config_path=cp)


def test_no_numerical_truth_or_reference_reads_in_new_bank_loader(monkeypatch):
    # Read the ACTUAL frozen reference inputs, with a guard prohibiting parsing
    # truth, repayment/noise outcomes, or decision masks during the bank stage.
    read_parquet, read_csv = pd.read_parquet, pd.read_csv
    def guard(reader):
        def guarded(path, *args, **kwargs):
            name = str(path)
            assert not any(token in name for token in ("true_expected_profit", "true_risk_components", "lending_decisions", "loan_outcomes", "bank_training_target", "private_noise"))
            return reader(path, *args, **kwargs)
        return guarded
    monkeypatch.setattr(pd, "read_parquet", guard(read_parquet))
    monkeypatch.setattr(pd, "read_csv", guard(read_csv))
    originals = io.preservation_manifest()
    inputs, hashes = io.load_inputs(io.load_config(), originals)
    assert len(inputs["contracts"]) == 20000
    assert hashes and "true_PD" not in inputs["risks"].columns
