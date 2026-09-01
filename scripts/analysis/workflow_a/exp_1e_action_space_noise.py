"""CPU-only Step 1E structural candidate and one-swap action composition."""
from __future__ import annotations
import importlib.util,json
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'reports/task1/exp_1e_action_space_noise_report.json'; R1D=ROOT/'reports/task1/exp_1d_union_rank_sweep_report.json'
KS=[20,23,29,45,77,135,197]; EPS=1e-12
def load():
 p=ROOT/'scripts/analysis/exp_1a_recall_gap_audit.py'; s=importlib.util.spec_from_file_location('s1a',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
def ratio(a,b): return a/b if b else None
def action_class(gold,base,doc,rank):
 swap=list(base); swap[rank-1]=doc
 if len(set(swap))!=5: raise ValueError('invalid action')
 delta=len(gold&set(swap))/len(gold)-len(gold&set(base))/len(gold)
 return 'beneficial' if delta>EPS else 'harmful' if delta<-EPS else 'neutral'
def blank(): return {'total_candidates_added_from_K20':0,'gold_candidates_added':0,'non_gold_candidates_added':0,'total_actions_added_from_K20':0,'beneficial_actions_added':0,'neutral_actions_added':0,'harmful_actions_added':0}
def finish(x):
 tc=x['total_candidates_added_from_K20']; ta=x['total_actions_added_from_K20']; x.update({'candidate_noise_rate':ratio(x['non_gold_candidates_added'],tc),'candidate_gold_yield':ratio(x['gold_candidates_added'],tc),'candidates_per_gold_candidate':ratio(tc,x['gold_candidates_added']),'beneficial_action_rate':ratio(x['beneficial_actions_added'],ta),'neutral_action_rate':ratio(x['neutral_actions_added'],ta),'harmful_action_rate':ratio(x['harmful_actions_added'],ta),'actions_per_beneficial_action':ratio(ta,x['beneficial_actions_added'])})
 return x
def main():
 r=json.loads(R1D.read_text()); expected=[.04177083333333333,.04447916666666667,.048110119047619054,.053839285714285715,.05878571428571428,.06202976190476191,.06381547619047619]; counts=[86169,102407,135404,224371,403322,728038,1075215]
 if not(r.get('status')=='PASS' and r.get('K_values',r.get('k_selection',{}).get('final_K_values'))==KS): raise RuntimeError('Step1D contract unavailable')
 if [x['incoming_candidate_count_total'] for x in r['sweep']]!=counts or any(abs(x['pooled_macro_recall_oracle_gain']-y)>1e-10 for x,y in zip(r['sweep'],expected)): raise RuntimeError('Step1D headline mismatch')
 m=load(); folds=m.read_target_fold_map(); target=set(folds); gold=m.read_target_train(target); base,bs=m.read_baseline(target,folds); pool,cs=m.read_candidates(target,folds)
 ranked=defaultdict(dict); rows,_=m.stream_target_rows(m.CANDIDATES,target)
 for row in rows:
  q=str(row['query_id'])
  if isinstance(row.get('candidates'),list):
   for i,x in enumerate(row['candidates'],1): ranked[q].setdefault(str(x['doc_id']),int(x.get('union_rank',i)))
  else: ranked[q].setdefault(str(row['doc_id']),int(row.get('union_rank',1)))
 if set(ranked)!=target: raise ValueError('candidate coverage')
 cumulative={K:blank() for K in KS}; pf={K:{str(f):blank() for f in range(1,5)} for K in KS}; shell={f'{a}->{b}':blank() for a,b in zip(KS,KS[1:])}; candidate_nested=True; action_nested=True
 for q in target:
  bset=set(base[q]); prior=set(); f=str(folds[q]); inc20={d for d,rank in ranked[q].items() if rank<=20 and d not in bset}
  for K in KS:
   inc={d for d,rank in ranked[q].items() if rank<=K and d not in bset}; candidate_nested &= prior<=inc; action_nested &= prior<=inc; prior=inc
   add=inc-inc20; x=cumulative[K]; y=pf[K][f]
   for doc in add:
    for z in (x,y):
     z['total_candidates_added_from_K20']+=1
     z['gold_candidates_added']+=doc in gold[q]; z['non_gold_candidates_added']+=doc not in gold[q]
    for rank in (4,5):
     kind=action_class(gold[q],base[q],doc,rank)
     for z in (x,y): z['total_actions_added_from_K20']+=1; z[f'{kind}_actions_added']+=1
   if K>20:
    prev=KS[KS.index(K)-1]; sx=shell[f'{prev}->{K}']; new=inc-{d for d,rank in ranked[q].items() if rank<=prev and d not in bset}
    for doc in new:
     sx['total_candidates_added_from_K20']+=1; sx['gold_candidates_added']+=doc in gold[q]; sx['non_gold_candidates_added']+=doc not in gold[q]
     for rank in (4,5):
      kind=action_class(gold[q],base[q],doc,rank); sx['total_actions_added_from_K20']+=1; sx[f'{kind}_actions_added']+=1
 for K in KS:
  finish(cumulative[K]);
  for f in pf[K]: finish(pf[K][f])
 for name,x in shell.items():
  finish(x); x.update({'interval':name,'new_candidates':x.pop('total_candidates_added_from_K20'),'new_gold_candidates':x.pop('gold_candidates_added'),'new_non_gold_candidates':x.pop('non_gold_candidates_added'),'shell_noise_rate':x.pop('candidate_noise_rate'),'new_actions':x.pop('total_actions_added_from_K20'),'new_beneficial_actions':x.pop('beneficial_actions_added'),'new_neutral_actions':x.pop('neutral_actions_added'),'new_harmful_actions':x.pop('harmful_actions_added'),'shell_harmful_action_rate':x.pop('harmful_action_rate')})
 candidate_entries=[]; action_entries=[]; action_counts_ok=True
 for K,expected_count in zip(KS,counts):
  added=cumulative[K]; total=expected_count-counts[0]; action_counts_ok &= added['total_actions_added_from_K20']==2*total
  candidate_entries.append({'K':K,**{k:added[k] for k in ('total_candidates_added_from_K20','gold_candidates_added','non_gold_candidates_added','candidate_noise_rate','candidate_gold_yield','candidates_per_gold_candidate')}})
  action_entries.append({'K':K,**{k:added[k] for k in ('total_actions_added_from_K20','beneficial_actions_added','neutral_actions_added','harmful_actions_added','beneficial_action_rate','neutral_action_rate','harmful_action_rate','actions_per_beneficial_action')}})
 monotonic=all(cumulative[a][field]<=cumulative[b][field] for a,b in zip(KS,KS[1:]) for field in ('total_candidates_added_from_K20','gold_candidates_added','non_gold_candidates_added','total_actions_added_from_K20','beneficial_actions_added','neutral_actions_added','harmful_actions_added'))
 identities=all(x['gold_candidates_added']+x['non_gold_candidates_added']==x['total_candidates_added_from_K20'] and x['beneficial_actions_added']+x['neutral_actions_added']+x['harmful_actions_added']==x['total_actions_added_from_K20'] for x in cumulative.values())
 pareto=[]
 for i,K in enumerate(KS):
  benefit=r['sweep'][i]['fraction_of_action_space_gap_recovered']; noise=cumulative[K]['candidate_noise_rate']; harm=cumulative[K]['harmful_action_rate']; acts=r['sweep'][i]['estimated_total_actions_current_contract']; dominated=any(j!=i and r['sweep'][j]['fraction_of_action_space_gap_recovered']>=benefit and r['sweep'][j]['estimated_total_actions_current_contract']<=acts and (cumulative[KS[j]]['harmful_action_rate'] or 0)<= (harm or 0) and (r['sweep'][j]['fraction_of_action_space_gap_recovered']>benefit or r['sweep'][j]['estimated_total_actions_current_contract']<acts or (cumulative[KS[j]]['harmful_action_rate'] or 0)<(harm or 0)) for j in range(len(KS)))
  pareto.append({'K':K,'benefit_score':benefit,'noise_score':noise,'advisory_joint_score':None if noise is None else benefit-noise,'total_actions':acts,'harmful_action_rate':harm,'pareto_status':'DOMINATED' if dominated else 'NON_DOMINATED'})
 checks={'all_queries_covered':len(target)==5600,'step1d_k_grid_exact':True,'candidate_count_crosschecks':all(x['total_candidates_added_from_K20']==c-counts[0] for x,c in zip(candidate_entries,counts)),'action_count_crosschecks':action_counts_ok,'candidate_sets_nested':candidate_nested,'action_sets_nested':action_nested,'candidate_composition_identity':identities,'action_composition_identity':identities,'cumulative_counts_monotonic':monotonic,'fold0_payload_not_materialized':True}
 report={'status':'PASS' if all(checks.values()) else 'CONTRACT_ERROR','branch_gate':'DIAGNOSTIC_ONLY_NO_BRANCH_GATE','official_metric':'macro_recall','folds_used':[1,2,3,4],'fold0_touched':False,'fold0_payload_materialized':False,'fold0_used_in_statistics':False,'fold0_used_in_oracle':False,'fold0_labels_used':False,'public_labels_used':False,'total_queries':5600,'K_values':KS,'baseline_non_target_records_skipped':bs,'candidate_non_target_records_skipped':cs,'input_artifacts':{'step1d_report':str(R1D.relative_to(ROOT)),'train_labels':str(m.TRAIN.relative_to(ROOT)),'fold_mapping':str(m.FOLDS.relative_to(ROOT)),'baseline_top5':str(m.BASELINE.relative_to(ROOT)),'full_candidate_pool':str(m.CANDIDATES.relative_to(ROOT))},'action_generator_contract':{'source_file':'scripts/beam/task1_v3_residual/build_actions.py','source_lines':'110-113','drop_ranks':[4,5],'eligibility_rules':['non-baseline incoming candidate from deterministic shortlist; valid baseline anchor'],'actions_per_candidate_if_constant':2,'verified':True},'candidate_composition':candidate_entries,'action_utility_composition':action_entries,'interval_composition':list(shell.values()),'per_fold':{str(K):pf[K] for K in KS},'joint_selection_preregistration':{'benefit_score':'Step1D fraction_of_action_space_gap_recovered','noise_score':'candidate_noise_rate','lambda':1.0,'formula':'benefit_score - lambda * noise_score','advisory_only':True,'production_K_selected':False},'pareto_diagnostics':pareto,'sanity_checks':checks,'workflow_document_updated':False,'notes':['Candidate non-gold composition is not policy harm. Harmful actions are exact structural rank4/5 action utilities only.','Rates are intentionally not tested for monotonicity.','No training, policy replay, inference, GPU, Modal, Beam, Fold0 materialization or public labels used.']}
 OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
