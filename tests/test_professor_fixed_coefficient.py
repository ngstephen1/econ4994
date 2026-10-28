"""Deterministic Step 10B fixtures; no fitting, population or noise generation."""
from copy import deepcopy
from fractions import Fraction
import json
import math

import numpy as np
import pandas as pd
import pytest
import yaml

from fair_lending.professor_model import fixed_coefficient as s, fixed_coefficient_io as io
from fair_lending.professor_model import bank_regression as aware, group_blind as blind, lending
from fair_lending.professor_model.config import configuration_fingerprint as fp
from fair_lending.professor_model.contracts import make_loan_contracts
from fair_lending.professor_model.noise_pilot import frame_fingerprint
from fair_lending.professor_model.profit import assess_bank_profit, assess_true_profit
from fair_lending.professor_model.risk_io import file_hash


def model(policy=s.W, beta=.006, intercept=.01):
    mod = aware if policy == s.W else blind
    coefficients = {"const": intercept, "Y": -1e-8, "D": .004, "C": .001}
    if policy == s.W: coefficients["B"] = beta
    return {"family": "statsmodels_OLS", "solver": "pinv", "intercept": True, "target": "PN",
            "target_transform": "none", "prediction_rescaling": "none", "regularization": None,
            "bounding": "clip_0_1_separate_column", "model_id": mod.MODEL_ID,
            "feature_order": list(mod.FEATURES), "design_order": ["const", *mod.FEATURES],
            "coefficients": coefficients, "training_count": 80, "training_ID_fingerprint": "toy_training"}


def reference(observed, contracts, policy, beta=.006, intercept=.01, budget=None):
    mod = aware if policy == s.W else blind; m = model(policy, beta, intercept)
    pred = mod.risk_estimates(observed, m); bound = "bank_PD_bounded" if policy == s.W else blind.BOUNDED
    profit = (assess_bank_profit if policy == s.W else blind.assess_profit)(contracts,pred[["applicant_id","model_id",bound]])
    if policy == s.W:
        bank = contracts[["applicant_id","L_i"]].merge(profit,on="applicant_id").rename(columns={"L_i":"requested_principal"})
        decisions, _, selection = lending.choose_cutoff(bank.loc[:,lending.BANK_INPUT_COLUMNS],budget)
    else: decisions, _, selection = blind.optimize_cutoff(contracts,profit,budget)
    payload = {"model":m,"selection":selection,**{f"{k}_fingerprint":frame_fingerprint(v) for k,v in [("predictions",pred),("profit",profit),("decisions",decisions)]}}
    return payload


@pytest.fixture
def toy():
    observed = pd.DataFrame({"applicant_id":[f"toy_{i}" for i in range(8)],"Y":np.arange(20,100,10)*1000.,
        "D":[.2,.3,.4,.25,.35,.45,.28,.38],"B":np.arange(8)%2,"C":np.arange(8)*.05,"split":"evaluation"}).loc[:,s.OBSERVED]
    contracts=make_loan_contracts(observed[["applicant_id","Y"]],T=30,theta=.1,r=.05);budget=.4*math.fsum(contracts.L_i)
    payloads={p:reference(observed,contracts,p,budget=budget) for p in (s.W,s.F)}
    truth=assess_true_profit(contracts,observed[["applicant_id"]].assign(true_PD=np.arange(8)*.001+.01))
    targets=observed[["applicant_id"]].assign(PN=np.arange(8)*.002+.01)
    return {"observed":observed,"contracts":contracts,"budget":budget,"payloads":payloads,"truth":truth,"targets":targets}


def banks(toy):
    result={p:s.replay_reference(toy["observed"],toy["contracts"],toy["payloads"][p],p) for p in (s.W,s.F)}
    pred,artifact=s.counterfactual_scores(toy["observed"],result[s.W]["predictions"],result[s.W]["model"])
    result[s.Z]=s.choose_counterfactual(toy["contracts"],pred,toy["budget"]);result[s.Z]["intervention"]=artifact
    return result


@pytest.fixture
def saved(toy,tmp_path,monkeypatch):
    sources=io.source_hashes(); ctx={"root":tmp_path,"observed":toy["observed"],"contracts":toy["contracts"],"budget":toy["budget"],
        "directory":tmp_path/io.METRICS/"toy_experiment","manifest":{"experiment_fingerprint":"toy_experiment","identity":{"source_hashes":sources}}}
    for path,frame in [(tmp_path/io.PROCESSED/'true_expected_profit.parquet',toy['truth']),
                       (tmp_path/io.SYNTHETIC/'training_signals.parquet',toy['targets'])]:
        path.parent.mkdir(parents=True,exist_ok=True);frame.to_parquet(path,index=False)
    io.immutable_json(tmp_path/io.METRICS/'preservation_manifest.json',{'original_checksums':{'toy_source':'hash'}})
    monkeypatch.setattr(io,'verify_originals',lambda *a,**k: {})
    monkeypatch.setattr(io,'load_banks',lambda *a,**k:(banks(toy),{'kind':'toy_canonical'}))
    return ctx


def test_frozen_configuration():
    c=io.load_config();assert c['saved_noise_replications']==200 and c['positive_floor'] is None and c['noise_reconstruction'] is False


@pytest.mark.parametrize('field,value',[('positive_floor',.0001),('saved_noise_replications',5),('raw_replay_absolute_tolerance',1e-5),('noise_reconstruction',True)])
def test_no_configuration_retuning(tmp_path,field,value):
    c=io.load_config();c[field]=value;p=tmp_path/'config.yaml';p.write_text(yaml.safe_dump(c))
    with pytest.raises(ValueError):io.load_config(p)


@pytest.mark.parametrize('beta',[.006,-.006,0.])
def test_only_saved_B_contribution_removed(toy,beta):
    m=model(beta=beta);before=deepcopy(m);p=aware.risk_estimates(toy['observed'],m)
    z,artifact=s.counterfactual_scores(toy['observed'],p,m)
    np.testing.assert_allclose(z.bank_PD_raw-p.bank_PD_raw,-beta*toy['observed'].B,atol=1e-15,rtol=0)
    np.testing.assert_array_equal(z.loc[toy['observed'].B==0,'bank_PD_raw'],p.loc[toy['observed'].B==0,'bank_PD_raw'])
    assert m==before and artifact['coefficients_used']=={**m['coefficients'],'B':0.}
    assert artifact['source_coefficients']==before['coefficients'] and artifact['new_fits']==0


@pytest.mark.parametrize('intercept,beta',[(-.1,.05),(1.1,-.2)])
def test_clipping_preserves_raw_and_does_not_add_positive_floor(toy,intercept,beta):
    m=model(beta=beta,intercept=intercept);p=aware.risk_estimates(toy['observed'],m)
    z,_=s.counterfactual_scores(toy['observed'],p,m)
    np.testing.assert_array_equal(z.bank_PD_bounded,np.clip(z.bank_PD_raw,0,1))
    assert (z.bank_PD_bounded==0).any() if intercept<0 else (z.bank_PD_bounded==1).any()


@pytest.mark.parametrize('change',['interaction','transform','offset','features','rescaling','coefficient'])
def test_incompatible_models_fail_clearly(toy,change):
    m=model()
    if change=='interaction':m['coefficients']['B:C']=.1
    elif change=='transform':m['transforms']={'Y':'log'}
    elif change=='offset':m['offset']=0.
    elif change=='features':m['feature_order']=['Y','D','C']
    elif change=='rescaling':m['prediction_rescaling']='divide_by_0.8'
    else:m['coefficients']['B']=np.nan
    with pytest.raises(ValueError):s.counterfactual_scores(toy['observed'],aware.risk_estimates(toy['observed'],model()),m)


def test_saved_reference_replay_never_optimizes(toy,monkeypatch):
    monkeypatch.setattr(lending,'choose_cutoff',lambda *a,**k:pytest.fail('reference reoptimization'))
    for p in (s.W,s.F):s.replay_reference(toy['observed'],toy['contracts'],toy['payloads'][p],p)


def test_changed_reference_prediction_fails(toy):
    payload=deepcopy(toy['payloads'][s.W]);payload['predictions_fingerprint']='wrong'
    with pytest.raises(ValueError,match='fingerprint'):s.replay_reference(toy['observed'],toy['contracts'],payload,s.W)


@pytest.mark.parametrize('extra',['true_PD','PN','epsilon'])
def test_bank_observable_allowlist(toy,extra):
    view=toy['observed'].assign(**{extra:0.})
    with pytest.raises(ValueError,match='observed-only'):s.replay_reference(view,toy['contracts'],toy['payloads'][s.W],s.W)


def test_missing_IDs_fail(toy):
    with pytest.raises(ValueError):s.replay_reference(toy['observed'].iloc[:-1],toy['contracts'],toy['payloads'][s.W],s.W)


def test_deterministic_and_original_inputs_unchanged(toy):
    original={k:v.copy(deep=True) for k,v in toy.items() if isinstance(v,pd.DataFrame)};payloads=deepcopy(toy['payloads'])
    a=banks(toy);b=banks(toy)
    for p in s.POLICIES:
        assert a[p]['selection']==b[p]['selection'];pd.testing.assert_frame_equal(a[p]['decisions'],b[p]['decisions'],check_exact=True)
    assert toy['payloads']==payloads
    for k,v in original.items():pd.testing.assert_frame_equal(v,toy[k],check_exact=True)


def test_exact_budget_and_indivisible_oversized_zero_block(toy):
    m=model(beta=0.,intercept=-.01);pred=aware.risk_estimates(toy['observed'],m);z,_=s.counterfactual_scores(toy['observed'],pred,m)
    total=math.fsum(toy['contracts'].L_i)
    exact=s.choose_counterfactual(toy['contracts'],z,total)
    assert exact['selection']['number_approved']==8 and exact['selection']['exact_budget_exhausted']
    under=s.choose_counterfactual(toy['contracts'],z,np.nextafter(total,0.))
    assert under['selection']['optimal_tau'] is None and not under['decisions'].approved.any()
    assert under['diagnostics']['independent_exact_enumeration_passed']


def test_evaluation_group_profit_overlap_and_signed_arithmetic(toy):
    b=banks(toy);result=s.evaluate_case(b,toy['observed'],toy['contracts'],toy['truth'],'canonical',toy['targets']);p=result['policies'].set_index('policy')
    assert len(p)==3 and len(result['groups'])==6 and len(result['overlap'])==12
    for policy in s.POLICIES:
        approved=set(b[policy]['decisions'].loc[lambda d:d.approved,'applicant_id'])
        assert p.loc[policy,'true_profit_USD']==math.fsum(toy['truth'].loc[toy['truth'].applicant_id.isin(approved),'expected_profit_true'])
        g=result['groups'].loc[lambda d:d.policy==policy].set_index('B')
        assert p.loc[policy,'approval_gap_pp']==100*(g.loc[1,'approval_rate']-g.loc[0,'approval_rate'])
    for r in result['paired'].itertuples():assert r.true_profit_difference_USD==p.loc[r.policy_first,'true_profit_USD']-p.loc[r.policy_second,'true_profit_USD']
    assert result['overlap'].loc[lambda d:d.allocation_category=='rejected_by_both','first_policy_profit_contribution_USD'].eq(0).all()


def test_no_PN_replay_for_noise_results(toy):
    r=s.evaluate_case(banks(toy),toy['observed'],toy['contracts'],toy['truth'],'noise_000')
    assert r['policies'].PN_metrics_status.eq('unavailable_no_noise_replay').all()
    assert not any('_PN_' in c for c in r['policies'])


def test_toy_summary_units_and_counts(toy):
    frame=s.evaluate_case(banks(toy),toy['observed'],toy['contracts'],toy['truth'],'noise_000')['paired']
    second=frame.copy();second['case_id']='noise_001'
    for metric in s.SUMMARY_UNITS:frame[metric]=1.;second[metric]=-1.
    summary=s.summarize_noise(pd.concat([frame,second],ignore_index=True))
    assert len(summary)==24 and summary.n_defined.eq(2).all() and summary['mean'].eq(0).all()
    assert summary.positive_count.eq(1).all() and summary.negative_count.eq(1).all()
    assert np.allclose(summary.MCSE_mean,1.)
    assert set(summary.loc[summary.metric=='approval_gap_difference_pp','unit'])=={'pp'}


def test_summary_rejects_canonical_or_duplicate_cells(toy):
    frame=s.evaluate_case(banks(toy),toy['observed'],toy['contracts'],toy['truth'],'canonical')['paired']
    with pytest.raises(ValueError):s.summarize_noise(frame)
    frame['case_id']='noise_000'
    with pytest.raises(ValueError):s.summarize_noise(pd.concat([frame,frame],ignore_index=True))


def test_evaluator_requires_new_freeze(saved,toy,monkeypatch):
    b=banks(toy)
    monkeypatch.setattr(pd,'read_parquet',lambda *a,**k:pytest.fail('truth before freeze'))
    with pytest.raises(FileNotFoundError):io.load_evaluator(saved,'canonical',b,saved['directory']/'absent.json')


def test_immutable_lifecycle_and_resume(saved,monkeypatch):
    record=io.run_case(saved,'canonical');assert record and record['status']=='success'
    before={str(p):file_hash(p) for p in saved['root'].rglob('*') if p.is_file()}
    monkeypatch.setattr(io,'load_banks',lambda *a,**k:pytest.fail('resume computed a case'))
    assert io.run_case(saved,'canonical',resume=True)==record
    assert before=={str(p):file_hash(p) for p in saved['root'].rglob('*') if p.is_file()}
    with pytest.raises(FileExistsError):io.run_case(saved,'canonical')


def test_summary_only_does_not_generate_or_write(saved,monkeypatch):
    assert io.run_case(saved,'canonical')
    io.summarize(saved)
    before={str(p):file_hash(p) for p in saved['root'].rglob('*') if p.is_file()}
    monkeypatch.setattr(io,'load_banks',lambda *a,**k:pytest.fail('summary computed a case'))
    report=io.summarize(saved,write=False)
    assert report['status']=='incomplete' and report['successful_cases']==1
    assert before=={str(p):file_hash(p) for p in saved['root'].rglob('*') if p.is_file()}


def test_failed_attempt_retained_and_retried(saved,monkeypatch):
    original=io.load_banks
    monkeypatch.setattr(io,'load_banks',lambda *a,**k:(_ for _ in ()).throw(ValueError('diagnostic_failure')))
    assert io.run_case(saved,'canonical') is None
    failed=list((saved['directory']/'attempts').glob('*.json'));assert len(failed)==1 and io.checked(failed[0])['stage']=='bank_replay'
    monkeypatch.setattr(io,'load_banks',original)
    assert io.run_case(saved,'canonical',resume=True)
    assert len(list((saved['directory']/'attempts').glob('*.json')))==2


def test_tampered_freeze_and_malformed_record_not_resumed(saved):
    record=io.run_case(saved,'canonical');assert record
    path=saved['root']/record['freeze_path'];path.write_text(path.read_text()+' ')
    assert io.find_success(saved,'canonical') is None


def test_source_change_invalidates_context(saved,monkeypatch):
    monkeypatch.setattr(io,'source_hashes',lambda:{'changed':'source'})
    with pytest.raises(ValueError,match='implementation changed'):io.check_sources(saved)


@pytest.mark.parametrize('change',['duplicate','group_rate','paired_profit','fits'])
def test_semantic_record_validation(saved,change):
    record=deepcopy(io.run_case(saved,'canonical'));assert record
    if change=='duplicate':record['policies'][1]=deepcopy(record['policies'][0])
    elif change=='group_rate':record['groups'][0]['approval_rate']+=.1
    elif change=='paired_profit':record['paired'][0]['true_profit_difference_USD']+=1
    else:record['new_fits']=1
    with pytest.raises(ValueError):io.validate_record(saved,record)
