"""Read-only evidence audit and presentation assets. Never runs experiments."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
FULL = ROOT / 'results/metrics/professor_model/noise_robustness_full'
TABLES = ROOT / 'results/tables/professor_model'
SYN = ROOT / 'data/synthetic/professor_model/baseline'
MAROON, ORANGE, SLATE, GRAY = '#861F41', '#E5751F', '#253746', '#68737D'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def protected_snapshot():
    folders = ['src', 'configs', 'experiments', 'tests', 'docs', 'dashboard',
               'data/synthetic/professor_model', 'data/processed/professor_model',
               'results/tables/professor_model', 'results/metrics/professor_model',
               'results/figures/professor_model']
    return {str(p.relative_to(ROOT)): sha(p) for f in folders for p in sorted((ROOT/f).rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.name != '.DS_Store'}

def dist(values):
    x = np.asarray(values, dtype=float)
    sd = x.std(ddof=1)
    return dict(n=len(x), mean=x.mean(), median=np.median(x), SD=sd, MCSE=sd/np.sqrt(len(x)),
                p2_5=np.quantile(x, .025), p97_5=np.quantile(x, .975), minimum=x.min(), maximum=x.max())

def json_safe(value):
    if isinstance(value, dict):
        return {k:json_safe(v) for k,v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None  # E.g. oracle has no perceived-bank profit. Never fabricate zero.
    if isinstance(value, np.generic):
        return value.item()
    return value

def main():
    (OUT/'assets').mkdir(exist_ok=True)
    snapshot = protected_snapshot()
    (OUT/'_build/protected_before.json').write_text(json.dumps(snapshot,indent=2))
    manifest = json.loads((FULL/'manifest.json').read_text())
    identity = manifest['identity']
    assert fingerprint(identity) == manifest['experiment_fingerprint']
    for mapping in ['original_checksums', 'source_code_hashes']:
        assert all(sha(ROOT/p)==h for p,h in identity[mapping].items())
    records = [json.loads(p.read_text()) for p in sorted((FULL/'replications').glob('replication_*.json'))]
    assert len(records)==200 and {r['replication'] for r in records}==set(range(200))
    assert sum(r['origin']=='validated_pilot_reuse' for r in records)==5
    metric_rows=[]
    for r in records:
        assert r['status']=='success'
        assert r['experiment_fingerprint']==manifest['experiment_fingerprint']
        assert fingerprint({k:v for k,v in r.items() if k!='content_fingerprint'})==r['content_fingerprint']
        assert r['stream']['master_seed']==499408 and r['stream']['spawn_key']==[r['replication']]
        assert sha(ROOT/r['freeze_path'])==r['freeze_sha256']
        assert {p['policy'] for p in r['policy_metrics']}=={'group_aware','group_blind'}
        assert r['decomposition']['valid']
        for p in r['policy_metrics']:
            assert p['maximum_budget_violation_USD']==0 and p['independent_exact_enumeration_passed']
            assert p['zero_risk_approved'] in (0,p['zero_risk_applicants'])
            assert p['available_bank_capital']==identity['fixed_budget']
            assert np.isclose(p['approval_gap_pp'],100*(p['B1_approval_rate']-p['B0_approval_rate']),atol=1e-12)
            metric_rows.append({'replication':r['replication'], **p})
    mr = pd.DataFrame(metric_rows)
    assert len(mr)==400 and not mr.duplicated(['replication','policy']).any()
    tidy=pd.read_csv(TABLES/'noise_robustness_full/noise_full_results.csv',float_precision='round_trip')
    assert len(tidy)==400 and not tidy.duplicated(['replication','policy']).any()
    merged=tidy.merge(mr,on=['replication','policy'],validate='one_to_one',suffixes=('_csv','_record'))
    for col in ['approval_gap_pp','true_expected_portfolio_profit','number_approved','zero_block_principal_USD']:
        np.testing.assert_allclose(merged[col+'_csv'],merged[col+'_record'],rtol=1e-13,atol=1e-10)
    w=mr.pivot(index='replication',columns='policy',values='approval_gap_pp')
    profits=mr.pivot(index='replication',columns='policy',values='true_expected_portfolio_profit')
    b=np.array([r['banks']['group_aware']['model']['coefficients']['B'] for r in records])
    aware=w.group_aware.to_numpy(); blind=w.group_blind.to_numpy()
    delta=(profits.group_blind-profits.group_aware).to_numpy()
    summary=pd.read_csv(TABLES/'noise_robustness_full/noise_full_summary.csv',float_precision='round_trip')
    stats={k:dist(x) for k,x in [('beta_B_hat',b),('aware_gap_pp',aware),('blind_gap_pp',blind),('true_profit_change_USD',delta)]}
    for k,v in stats.items():
        row=summary.loc[summary.metric.eq(k)&summary.scope.eq('paired_replications')].iloc[0]
        for a,c in [('mean','mean'),('SD','sample_SD'),('MCSE','MCSE_mean'),('p2_5','p2.5'),('p97_5','p97.5')]:
            assert np.isclose(v[a],row[c],rtol=1e-13,atol=1e-13)
    coeff_summary=pd.read_csv(TABLES/'noise_robustness_full/noise_full_coefficient_summary.csv')
    groups_summary=pd.read_csv(TABLES/'noise_robustness_full/noise_full_group_outcome_summary.csv')
    tie_csv=pd.read_csv(TABLES/'noise_robustness_full/noise_full_tie_diagnostics.csv')
    assert (len(coeff_summary),len(groups_summary),len(tie_csv))==(9,16,400)
    events=dict(beta_positive=int((b>0).sum()),beta_negative=int((b<0).sum()),
                aware_positive=int((aware>0).sum()),aware_negative=int((aware<0).sum()),
                blind_smaller_absolute_gap=int((abs(blind)<abs(aware)).sum()),
                aware_at_least_original=int((abs(aware)>=14.385705).sum()),
                blind_profit_higher=int((delta>0).sum()),blind_profit_lower=int((delta<0).sum()),profit_equal=int((delta==0).sum()))
    assert events['blind_smaller_absolute_gap']==183 and events['aware_at_least_original']==131
    assert (events['beta_positive'],events['beta_negative'])==(105,95)
    assert (events['blind_profit_higher'],events['blind_profit_lower'],events['profit_equal'])==(141,47,12)
    ties={p:{'reject_all':int(g.reject_all_selected.sum()),
             'zero_block_over_90pct':int((g.zero_block_principal_USD/g.available_bank_capital>.9).sum()),
             'zero_block_infeasible':int((~g.zero_block_feasible).sum())}
          for p,g in mr.groupby('policy')}
    assert ties=={'group_aware':{'reject_all':13,'zero_block_over_90pct':17,'zero_block_infeasible':13},
                  'group_blind':{'reject_all':11,'zero_block_over_90pct':17,'zero_block_infeasible':11}}
    base=pd.read_csv(TABLES/'group_aware_vs_blind_portfolio.csv',float_precision='round_trip')
    audit=pd.read_csv(TABLES/'group_aware_vs_blind_group_audit.csv',float_precision='round_trip')
    for _,row in base.iterrows():
        rates=audit.loc[audit.policy.eq(row.policy)].set_index('B')
        assert np.isclose(row.approval_gap_pp,100*(rates.loc[1,'approval_rate']-rates.loc[0,'approval_rate']),atol=1e-12)
    earlier={name:pd.read_csv(TABLES/(name+'.csv')).to_dict('records') for name in
             ['bank_vs_oracle_summary','bank_vs_oracle_group_audit','group_aware_vs_blind_risk']}
    original_beta=json.loads((ROOT/'results/metrics/professor_model/bank_regression_model.json').read_text())['coefficients']['B']
    oracle=base.loc[base.policy.eq('oracle'),'true_expected_portfolio_profit'].item()
    assert oracle==records[0]['oracle_reference']['true_expected_portfolio_profit']
    assert len({r['oracle_reference']['decision_fingerprint'] for r in records})==1
    signal=records[0]['decomposition']['beta_B_signal']
    assert all(r['decomposition']['beta_B_signal']==signal for r in records)
    bank=pd.read_csv(SYN/'applicants_bank_view.csv',float_precision='round_trip')
    truth=pd.read_csv(SYN/'simulation_truth.csv',float_precision='round_trip')
    pn=pd.read_csv(SYN/'training_signals.csv',float_precision='round_trip')
    eta=pd.read_csv(SYN/'risk_measurement_noise_private.csv',float_precision='round_trip')
    split=pd.read_csv(SYN/'applicant_splits.csv')
    assert len(bank)==len(truth)==len(pn)==100000
    assert split.split.value_counts().to_dict()=={'train':80000,'evaluation':20000}
    snippet=bank.merge(truth,on='applicant_id',validate='one_to_one').merge(pn,on='applicant_id',validate='one_to_one').sort_values('applicant_id').head(5)
    snippet=snippet[['applicant_id','B','Y','D','C','true_PD','PN']]
    snippet.to_csv(OUT/'assets/applicant_preview_EVALUATOR_ONLY.csv',index=False)
    extremes=[{'replication':int(i),'profit_change_USD':float(delta[i]),
               'aware_approved':int(mr.loc[(mr.replication==i)&mr.policy.eq('group_aware'),'number_approved'].item()),
               'blind_approved':int(mr.loc[(mr.replication==i)&mr.policy.eq('group_blind'),'number_approved'].item())}
              for i in np.flatnonzero(delta>2.36e9)]
    assert len(extremes)==2
    excerpt=tidy.drop_duplicates('replication').set_index('replication').loc[[0,1,2],['aware_gap_pp','blind_gap_pp','gap_change_pp']].reset_index()
    payload=dict(statistics=stats,events=events,ties=ties,canonical=base.to_dict('records'),
                 original_beta_B=original_beta,oracle_true_profit=oracle,signal_beta_B=signal,
                 mean_true_PD=truth.true_PD.mean(),noise_signal_SD_ratio=eta.eta.std(ddof=1)/(.8*truth.true_PD).std(ddof=1),
                 group_B1_share=bank.B.mean(),snippet=snippet.to_dict('records'),first_three=excerpt.to_dict('records'),
                 extreme_cases=extremes,earlier_inspected=earlier,n_successful=200,n_policy_rows=400,pilot_reuses=5,new_replications=195,
                 fixed_population_source=manifest['identity']['original_checksums'],experiment_fingerprint=manifest['experiment_fingerprint'])
    (OUT/'verified_evidence.json').write_text(json.dumps(json_safe(payload),indent=2,allow_nan=False))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.labelcolor':SLATE,'text.color':SLATE,
                         'axes.edgecolor':'#D7DBDF','xtick.color':GRAY,'ytick.color':GRAY,'axes.spines.top':False,'axes.spines.right':False,
                         'svg.fonttype':'none','pdf.fonttype':42})
    def save(fig,name):
        fig.savefig(OUT/f'assets/{name}.png',dpi=240,facecolor='white')
        fig.savefig(OUT/f'assets/{name}.svg',facecolor='white')
        plt.close(fig)
    fig,axs=plt.subplots(2,1,figsize=(6.6,4.2),layout='constrained')
    for ax,x,label,color,bins in [(axs[0],aware,'Group-aware',MAROON,np.linspace(-80,100,28)),
                                 (axs[1],blind,'Group-blind',ORANGE,np.linspace(-2,2,24))]:
        ax.hist(x,bins=bins,color=color,edgecolor='white',linewidth=.6)
        ax.axvline(0,color=SLATE,lw=1)
        ax.set_ylabel('Draws');ax.set_title(label+' (n=200)',loc='left',fontweight='bold',fontsize=13)
        ax.set_xlabel('B=1 minus B=0 approval gap (pp)')
    axs[0].axvline(-14.385705,color=GRAY,ls='--',lw=1.5)
    axs[0].text(-14.385705,axs[0].get_ylim()[1]*.90,'Original\n−14.39',ha='right',va='top',fontsize=10,color=GRAY)
    save(fig,'approval_gap_panels')
    fig,ax=plt.subplots(figsize=(4.8,4.2),layout='constrained')
    ax.hist(b*1e3,bins=24,color=MAROON,edgecolor='white',linewidth=.6)
    ax.axvline(0,color=SLATE,lw=1.3,label='Zero')
    ax.axvline(original_beta*1e3,color=ORANGE,ls='--',lw=1.8,label='Original draw')
    ax.axvline(b.mean()*1e3,color=GRAY,ls=':',lw=1.6,label='200-draw mean')
    ax.set_xlabel('Fitted B coefficient (×10⁻³ PN units)');ax.set_ylabel('Draws')
    ax.set_title('Group coefficient (n=200)',loc='left',fontsize=13,fontweight='bold')
    ax.legend(frameon=False,fontsize=10)
    save(fig,'coefficient_distribution')
    fig,(top,bottom)=plt.subplots(2,1,figsize=(7.2,4.1),gridspec_kw={'height_ratios':[1,2]},layout='constrained')
    top.scatter(delta/1e6,np.arange(200),s=12,color=MAROON,alpha=.6)
    top.scatter(delta[delta>2.36e9]/1e6,np.flatnonzero(delta>2.36e9),s=45,color=ORANGE,zorder=3)
    top.set_xlim(-80,2500);top.set_ylim(-10,210);top.set_yticks([0,100,199]);top.set_ylabel('Draw index',fontsize=10)
    top.set_xticks([0,500,1000,1500,2000,2500]);top.set_xlabel('Full range: blind minus aware true profit ($M)',fontsize=11)
    top.set_title('All 200 draws, including both >$2.36B cases',loc='left',fontsize=12,fontweight='bold')
    bottom.hist(delta/1e6,bins=np.linspace(-8,10,28),color=MAROON,edgecolor='white',linewidth=.5)
    bottom.axvline(0,color=SLATE,lw=1)
    bottom.axvline(np.median(delta)/1e6,color=ORANGE,lw=2,label='Median +$0.652M')
    bottom.set_xlim(-8,10);bottom.set_xlabel('Central zoom: true-profit change ($M)');bottom.set_ylabel('Draws')
    bottom.set_title('Central window shows 198/200 draws. Both extremes appear above.',loc='left',fontsize=11)
    bottom.legend(frameon=False,fontsize=10)
    save(fig,'profit_full_range_and_zoom')
    mapping=[]
    def add(slide,metric,source,column,computation,rounding):
        mapping.append(dict(slide=slide,metric=metric,source=source,column=column,computation=computation,rounding=rounding))
    s='results/tables/professor_model/noise_robustness_full/noise_full_results.csv'
    add(1,'Applicant count',str((SYN/'applicants_bank_view.csv').relative_to(ROOT)),'applicant_id','unique IDs','integer')
    add(1,'200 draws / 91.5%',s,'replication, aware_gap_pp, blind_gap_pp','200 unique pairs, 183 abs(blind)<abs(aware)','1 decimal percent')
    add(2,'Risk and accounting pipeline','docs/professor_loan_economics.md; docs/professor_bank_regression.md','equations','retain exact working definitions','coefficients unchanged')
    add(3,'Data design and signal/noise','configs/professor_model/baseline.yaml; data/synthetic/professor_model/baseline/simulation_truth.csv; risk_measurement_noise_private.csv','true_PD, eta','100*mean(PD), SD(eta)/SD(.8PD)','4 decimals percent, nearest integer ratio')
    add(3,'Evaluator preview',str((SYN/'applicants_bank_view.csv').relative_to(ROOT))+'; '+str((SYN/'simulation_truth.csv').relative_to(ROOT))+'; '+str((SYN/'training_signals.csv').relative_to(ROOT)),'applicant_id,B,Y,D,C,true_PD,PN','1:1 ID join, first five lexicographic IDs','Y whole dollars, D/C 3 decimals, PD% 4 decimals, raw PN 4 decimals')
    add(4,'Canonical gaps and true profits','results/tables/professor_model/group_aware_vs_blind_portfolio.csv; group_aware_vs_blind_group_audit.csv','approval_gap_pp,true_expected_portfolio_profit,approval_rate','cross-check gaps from B1-B0 group rates','gaps 2 decimals pp, profits 4 decimals billions')
    add(5,'Completion and first-three excerpt',s+'; results/metrics/professor_model/noise_robustness_full/replications/','replication,policy,aware_gap_pp,blind_gap_pp,gap_change_pp,origin','pair completeness; indices 0,1,2 only','3 decimals pp, integer counts')
    add(6,'Gap and coefficient plots/statistics',s+'; replication_000.json through replication_199.json','beta_B_hat,aware_gap_pp,blind_gap_pp','independent record aggregation, ddof=1, SD/sqrt(200), linear quantiles','3 decimals pp, 8 decimals coefficient, 1 decimal frequencies')
    add(6,'Noiseless B projection','results/metrics/professor_model/noise_robustness_full/signal_reference_EVALUATOR_ONLY.json','reference.beta_B_signal','read evaluator-only saved reference','12 decimals')
    add(7,'Profit distribution and reject-all',s,'true_profit_change_USD,number_approved,reject_all_selected,zero_block_principal_USD,available_bank_capital','200 paired differences, extrema retained; per-policy denominators 200','mean 3 decimals M, median nearest USD, counts integer')
    add(8,'Takeaways and repository status',s+'; docs/PROJECT_STATUS.md','paired metrics, latest documented suite','reuse audited findings, no fresh test run','documented 624 passed / 4 failures / 32 warnings')
    sources=set()
    for folder in [SYN,TABLES,FULL,ROOT/'docs',ROOT/'configs/professor_model',ROOT/'results/figures/professor_model']:
        for path in folder.rglob('*'):
            if path.is_file() and path.name!='.DS_Store':sources.add(path)
    evidence_manifest=dict(slides=mapping,snippet_applicant_ids=snippet.applicant_id.tolist(),
                           source_hashes={str(p.relative_to(ROOT)):sha(p) for p in sorted(sources)},
                           scientific_audit={'successful_replications':200,'unique_indices':200,'policy_rows':400,'pilot_reuses':5,'new_replications':195,
                                             'all_paired':True,'duplicate_cells':0,'budget_violations':0,'zero_groups_split':0,
                                             'canonical_separate_from_MC':True,'fixed_original_hashes_verified':285},
                           discrepancies=['Step 8A heading states Step 8B not started as historical pilot status. Full manifest, records and current PROJECT_STATUS confirm complete.',
                                          'Step 1 annual-rate suggestion is historical. Implemented contract uses r=.05 per period. Slides use the implemented convention.'],
                           experiment_fingerprint=manifest['experiment_fingerprint'],
                           documented_tests={'passed':624,'failed':4,'warnings':32,'fresh_tests_run_for_deck':False})
    (OUT/'evidence_manifest.json').write_text(json.dumps(evidence_manifest,indent=2))
    assert snapshot==protected_snapshot()
    print(json.dumps({'statistics':stats,'events':events,'ties':ties,'extremes':extremes,'snippet':snippet.to_dict('records'),
                      'protected_files':len(snapshot),'discrepancies':evidence_manifest['discrepancies']},indent=2))

if __name__=='__main__':main()
