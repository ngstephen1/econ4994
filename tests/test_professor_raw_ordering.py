"""Deterministic raw-ordering accounting, feasible-family and freeze tests."""
from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fair_lending.professor_model import raw_ordering as s, raw_ordering_io as io
from fair_lending.professor_model import fixed_coefficient as fixed, bank_regression as aware, group_blind as blind, lending
from fair_lending.professor_model.contracts import make_loan_contracts
from fair_lending.professor_model.profit import assess_bank_profit, assess_true_profit
from fair_lending.professor_model.noise_pilot import frame_fingerprint
from fair_lending.professor_model.risk_io import file_hash

_check_sources = io.check_sources


def bank(raw=(-.2, -.1, .1), costs=(10., 10., 10.), profits=(5., 5., 5.), policy=fixed.W):
    return pd.DataFrame({"applicant_id": [f"toy_{i}" for i in range(len(raw))], "source_policy": policy,
        "ordering_score": raw, "accounting_PD": np.clip(raw, 0, 1), "requested_principal": costs,
        "expected_profit_perceived": profits}).loc[:, s.INPUT_COLUMNS]


def solve(inp, budget):
    result = s.choose_raw_cutoff(inp, budget)
    assert s.audit_raw_cutoff(inp, result, budget)["every_raw_endpoint_fraction_checked"]
    return result


def test_negative_ordering_is_not_negative_accounting_PD():
    b = bank(); r = solve(b, 10.)
    assert r["selection"]["cutoff_score"] == -.2
    assert r["decisions"].approved.tolist() == [True, False, False]
    assert r["decisions"].accounting_PD.tolist() == [0., 0., .1]


@pytest.mark.parametrize("column", ["group", "B", "true_PD", "expected_profit_true", "PN", "default_period"])
def test_cutoff_allowlist_excludes_groups_truth_and_outcomes(column):
    with pytest.raises(ValueError, match="allowlist"):
        s.choose_raw_cutoff(bank().assign(**{column: 0}), 10.)


@pytest.mark.parametrize("change", ["duplicate", "wrong_clip", "nan", "nonpositive_principal", "mixed_sources"])
def test_invalid_bank_inputs_rejected(change):
    b = bank()
    if change == "duplicate": b.loc[1, "applicant_id"] = b.loc[0, "applicant_id"]
    if change == "wrong_clip": b.loc[0, "accounting_PD"] = -.2
    if change == "nan": b.loc[0, "ordering_score"] = np.nan
    if change == "nonpositive_principal": b.loc[0, "requested_principal"] = 0.
    if change == "mixed_sources": b.loc[0, "source_policy"] = fixed.F
    with pytest.raises(ValueError): s.choose_raw_cutoff(b, 10.)


@pytest.mark.parametrize("budget", [-1., float("inf"), float("nan"), True])
def test_invalid_budgets_rejected(budget):
    with pytest.raises(ValueError): s.choose_raw_cutoff(bank(), budget)


def test_exact_raw_tie_is_not_broken_by_applicant_id():
    b = bank(raw=(-.2, -.2, .1))
    r = solve(b, 10.)
    assert r["selection"]["cutoff_score"] is None
    assert r["selection"]["number_approved"] == 0
    assert r["candidates"].number_approved.tolist() == [0, 2, 3]


def test_all_zero_objectives_choose_reject_all_first():
    assert solve(bank(profits=(0., 0., 0.)), 30.)["selection"]["cutoff_score"] is None


def test_exact_objective_tie_chooses_lowest_raw_cutoff():
    r = solve(bank(profits=(5., 0., -1.)), 30.)
    assert r["selection"]["number_approved"] == 1


def test_negative_increment_does_not_force_largest_prefix():
    assert solve(bank(profits=(5., -6., -7.)), 30.)["selection"]["number_approved"] == 1


def test_can_include_negative_profit_if_prefix_total_is_best():
    assert solve(bank(profits=(-1., 4., -5.)), 30.)["selection"]["number_approved"] == 2


def test_zero_budget_gives_separate_reject_all():
    assert solve(bank(), 0.)["selection"]["reject_all_selected"]


def test_large_dollar_exact_budget_boundary():
    b = bank(raw=(-.2, -.1), costs=(100_000_000., 100_000_000.), profits=(5., 5.))
    assert solve(b, 200_000_000.)["selection"]["funded_principal_USD"] == 200_000_000.


def test_binary_float_budget_does_not_receive_hidden_slack():
    b = bank(raw=(-.2, -.1), costs=(.1, .2), profits=(5., 5.))
    assert solve(b, .3)["selection"]["number_approved"] == 1


def test_row_order_does_not_change_policy():
    b = bank(); first = solve(b, 20.); second = solve(b.iloc[::-1], 20.)
    for name in ("decisions", "candidates"):
        pd.testing.assert_frame_equal(first[name], second[name], check_exact=True)
    assert first["selection"] == second["selection"]


def test_above_one_scores_rank_but_accounting_stays_one():
    b = bank(raw=(1.1, 1.2), costs=(10., 10.), profits=(-10., -10.))
    assert b.accounting_PD.eq(1).all()
    assert solve(b, 20.)["selection"]["number_approved"] == 0


def test_rational_audit_detects_candidate_tampering():
    b = bank(); r = solve(b, 10.); r["candidates"].loc[1, "funded_principal"] += 1
    with pytest.raises(ValueError, match="rational"):
        s.audit_raw_cutoff(b, r, 10.)


def test_audit_rejects_misreported_ordering_score():
    b = bank(); r = solve(b, 10.); r["decisions"].loc[0,"ordering_score"] = -100.
    with pytest.raises(AssertionError):s.audit_raw_cutoff(b,r,10.)


def test_bounded_family_mapping_includes_zero_and_upper_clipped_blocks():
    b = bank(raw=(-.2, -.1, .5, 1.1, 1.2), costs=(10.,)*5, profits=(5.,)*5)
    raw = solve(b, 30.)
    baseline = {"optimal_tau": .5, "number_approved": 3, "total_funded_principal": 30., "bank_expected_portfolio_profit": 15.}
    dec = b[["applicant_id"]].assign(approved=[True, True, True, False, False])
    mapping, checks = s.bounded_prefix_mapping(b, raw, dec, baseline)
    assert mapping.number_approved.tolist() == [0, 2, 3, 5]
    assert checks["additional_raw_endpoints"] == 2
    assert checks["perceived_objective_nondecreasing_exact"]


def test_true_profit_can_fall_despite_perceived_improvement():
    banks = {}
    for p in s.POLICIES:
        b = bank(policy=p, profits=(10., 10., 10.)); raw = solve(b, 10.)
        baseline = {"selection": {"optimal_tau": None, "available_bank_capital": 10.},
                    "decisions": b[["applicant_id"]].assign(approved=False, funded_principal=0.)}
        banks[p] = {"inputs": b, "baseline": baseline, "raw": raw}
    observed = pd.DataFrame({"applicant_id": bank().applicant_id, "B": [0, 1, 0]})
    truth = observed[["applicant_id"]].assign(true_PD=[.8, .01, .01], expected_receipts_true=[5., 30., 40.], expected_profit_true=[-5., 20., 30.])
    result = s.evaluate_case(banks, observed, truth, "canonical")
    within = result["comparisons"].query("comparison_kind == 'within_model_ordering'")
    assert within.perceived_profit_difference_USD.eq(10.).all()
    assert within.true_profit_difference_USD.eq(-5.).all()
    assert result["groups"].groupby("policy").number_approved.sum().sort_index().equals(result["policies"].set_index("policy").number_approved.sort_index())
    assert result["overlap"].groupby(["policy_first", "policy_second"]).n_applicants.sum().eq(3).all()


def toy_model(policy):
    m = {"family": "statsmodels_OLS", "solver": "pinv", "intercept": True, "target": "PN", "target_transform": "none",
         "prediction_rescaling": "none", "regularization": None, "bounding": "clip_0_1_separate_column",
         "feature_order": list(aware.FEATURES if policy==fixed.W else blind.FEATURES),
         "design_order": list(aware.DESIGN_COLUMNS if policy==fixed.W else blind.DESIGN),
         "model_id": aware.MODEL_ID if policy==fixed.W else blind.MODEL_ID,
         "coefficients": {"const": .001 if policy==fixed.W else .0011, "Y": -3e-8, "D": .001, "C": .001}}
    if policy == fixed.W: m["coefficients"]["B"] = .0002
    return m


@pytest.fixture
def saved(tmp_path, monkeypatch):
    obs = pd.DataFrame({"applicant_id": [f"toy_{i}" for i in range(6)], "Y": np.arange(20, 80, 10)*1000.,
        "D": [.2,.3,.4,.2,.3,.4], "B": [0,1,0,1,0,1], "C": [.0,.1,.2,.0,.1,.2], "split": "evaluation"}).loc[:, fixed.OBSERVED]
    contracts = make_loan_contracts(obs[["applicant_id", "Y"]], T=30, theta=.1, r=.05)
    budget = .4*sum(contracts.L_i); refs = {}
    for p in (fixed.W, fixed.F):
        mod = aware if p==fixed.W else blind; model = toy_model(p); pred = mod.risk_estimates(obs, model)
        if p == fixed.W:
            profit = assess_bank_profit(contracts, pred[["applicant_id","model_id","bank_PD_bounded"]])
            inp = contracts[["applicant_id","L_i"]].merge(profit,on="applicant_id").rename(columns={"L_i":"requested_principal"})
            dec, _, sel = lending.choose_cutoff(inp.loc[:, lending.BANK_INPUT_COLUMNS], budget)
        else:
            profit = blind.assess_profit(contracts, pred[["applicant_id","model_id",blind.BOUNDED]])
            dec, _, sel = blind.optimize_cutoff(contracts, profit, budget)
        refs[p] = {"model": model, "predictions": pred, "profit": profit, "decisions": dec, "selection": sel}
    pred, intervention = fixed.counterfactual_scores(obs, refs[fixed.W]["predictions"], refs[fixed.W]["model"])
    refs[fixed.Z] = fixed.choose_counterfactual(contracts, pred, budget); refs[fixed.Z]["intervention"] = intervention
    truth = assess_true_profit(contracts, obs[["applicant_id"]].assign(true_PD=.002))
    source = tmp_path / "old"; source.mkdir()
    freeze = {"case_id": "canonical", "status": "bank_frozen_before_evaluation", "banks": io.base.bank_payload(refs)}
    fp = source / "freeze.json"; io.base.publish(fp, freeze)
    result = fixed.evaluate_case(refs, obs, contracts, truth, "canonical")
    rp = source / "record.json"; io.base.publish(rp, {k: io.base.rows(v) for k,v in result.items()})
    truth_path = tmp_path / "data/processed/professor_model/true_expected_profit.parquet"
    truth_path.parent.mkdir(parents=True); truth.to_parquet(truth_path,index=False)
    preservation = tmp_path / io.METRICS / "preservation_manifest.json"; preservation.parent.mkdir(parents=True)
    preservation.write_text(json.dumps({"original_checksums": {str(truth_path.relative_to(tmp_path)): file_hash(truth_path)}}))
    binding = {"freeze_path":str(fp.relative_to(tmp_path)), "freeze_sha256":file_hash(fp),
               "record_path":str(rp.relative_to(tmp_path)), "record_sha256":file_hash(rp)}
    ctx = {"root":tmp_path, "directory":tmp_path/io.METRICS/"toy", "old":{}, "observed":obs, "contracts":contracts,
           "budget":budget, "protected_count":1, "manifest":{"experiment_fingerprint":"toy",
           "identity":{"source_bindings":{"canonical":binding},"source_hashes":{"toy":"original"}}}}
    monkeypatch.setattr(io,"check_sources",lambda ctx:None)
    monkeypatch.setattr(io,"verify_originals",lambda root:{})
    monkeypatch.setattr(io.base,"validate_record",lambda ctx,record:record)
    return ctx


def test_replay_never_searches_a_baseline(saved,monkeypatch):
    def forbidden(*a,**k): raise AssertionError("baseline was reoptimized")
    monkeypatch.setattr(lending,"choose_cutoff",forbidden)
    monkeypatch.setattr(blind,"optimize_cutoff",forbidden)
    monkeypatch.setattr(fixed,"choose_counterfactual",forbidden)
    refs = io.replay_baselines(saved,"canonical")
    assert set(refs)==set(s.POLICIES)


def test_bank_case_holds_accounting_and_profit_fixed(saved):
    banks = io.bank_case(saved,"canonical")
    for p,b in banks.items():
        expected = fixed.normalized_predictions(b['baseline'],p)
        np.testing.assert_array_equal(b['inputs'].accounting_PD,expected.bank_PD_bounded)
        np.testing.assert_array_equal(b['inputs'].expected_profit_perceived,b['baseline']['profit'].sort_values('applicant_id').expected_profit_bank)
        assert b['diagnostics']['all_bounded_prefixes_in_raw_family']


def test_missing_freeze_prevents_evaluator_read(saved,monkeypatch):
    def forbidden(*a,**k): raise AssertionError("truth read before freeze")
    monkeypatch.setattr(pd,"read_parquet",forbidden)
    with pytest.raises(FileNotFoundError): io.load_evaluator(saved,"canonical",{},saved['directory']/'absent.json')


def test_source_record_parsed_only_after_new_freeze(saved,monkeypatch):
    original = io.base.checked; source_record = saved['root']/saved['manifest']['identity']['source_bindings']['canonical']['record_path']
    calls = []
    def checked(path):
        if Path(path)==source_record:
            assert list((saved['directory']/'freezes').glob('canonical_*.json'))
            calls.append('source_evaluator_record')
        return original(path)
    monkeypatch.setattr(io.base,'checked',checked)
    record = io.run_case(saved,'canonical')
    assert record is not None and calls==['source_evaluator_record']
    assert record['raw_cutoff_searches']==3 and record['baseline_cutoff_searches']==0


def test_tampered_freeze_blocks_truth(saved):
    banks = io.bank_case(saved,'canonical'); path = io.freeze_case(saved,'canonical',banks,'toy')
    value = json.loads(path.read_text());value['status']='changed';path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='fingerprint'):io.load_evaluator(saved,'canonical',banks,path)


def test_resume_and_summary_are_read_only(saved,monkeypatch):
    record = io.run_case(saved,'canonical'); assert record is not None
    before = {str(p):file_hash(p) for p in saved['root'].rglob('*') if p.is_file()}
    def forbidden(*a,**k):raise AssertionError('resume/summary started computation or write')
    monkeypatch.setattr(io,'bank_case',forbidden);monkeypatch.setattr(io.base,'publish',forbidden)
    assert io.run_case(saved,'canonical',resume=True)==record
    report = io.summarize(saved,write=False)
    assert report['successful_cases']==1 and report['status']=='incomplete'
    assert before=={str(p):file_hash(p) for p in saved['root'].rglob('*') if p.is_file()}


def test_corrupt_record_is_not_resumed(saved):
    record = io.run_case(saved,'canonical'); assert record is not None
    path = next((saved['directory']/'records').glob('*.json'))
    value = json.loads(path.read_text());value['policies']=[];path.write_text(json.dumps(value))
    assert io.find_success(saved,'canonical') is None
    assert path.exists()


def test_incomplete_cells_rejected_even_if_record_checksum_is_rebuilt(saved):
    record=io.run_case(saved,'canonical');record['groups']=record['groups'][:-1]
    with pytest.raises(ValueError,match='incomplete'):io.validate_record(saved,record)


def test_wrong_comparison_family_rejected(saved):
    record=io.run_case(saved,'canonical');record['comparisons'][-1]['policy_second']=record['comparisons'][-1]['policy_first']
    with pytest.raises(ValueError,match='comparison coverage'):io.validate_record(saved,record)


def test_changed_source_invalidates_identity(saved,monkeypatch):
    monkeypatch.setattr(io,'source_hashes',lambda root:{'toy':'changed'})
    with pytest.raises(ValueError,match='changed'):_check_sources(saved)


def test_failures_retain_stage_and_do_not_count_as_success(saved,monkeypatch):
    def fail(*a,**k):raise ValueError('toy bank failure')
    monkeypatch.setattr(io,'bank_case',fail)
    assert io.run_case(saved,'canonical') is None
    attempt=io.base.checked(next((saved['directory']/'attempts').glob('*.json')))
    assert attempt['status']=='failed' and attempt['stage']=='bank_replay_and_raw_optimization'
    assert io.find_success(saved,'canonical') is None


def test_only_six_declared_cases_and_no_full_noise_extension(saved):
    config=io.load_config()
    assert len(config['cases'])==6 and config['full_200_draw_extension'] is False
    with pytest.raises(ValueError,match='outside'):io.run_case(saved,'noise_000')


def test_production_cli_summary_never_runs_or_renders(saved,monkeypatch):
    spec=spec_from_file_location('raw_cli',Path(__file__).resolve().parents[1]/'experiments/run_professor_raw_ordering.py')
    cli=module_from_spec(spec);spec.loader.exec_module(cli)
    monkeypatch.setattr(io,'prepare',lambda readonly:saved if readonly else pytest.fail('write preflight'))
    def forbidden(*a,**k):raise AssertionError('summary launched work')
    monkeypatch.setattr(io,'run_case',forbidden);monkeypatch.setattr(cli,'render',forbidden)
    monkeypatch.setattr(io.base,'publish',forbidden)
    assert cli.main(['--summarize-only'])==2  # explicit incomplete report
