"""STEP 2-P1 contract-preserving pairwise preflight.

Counts the complete deterministic mirrored pair population before any fit.  It
never substitutes sampling when the exact HGB training matrix is infeasible.
"""
from __future__ import annotations
import json, os, runpy
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]; R=ROOT/'reports/task1'
CONTRACT=R/'step2p1_model_contract.json'; REPORT=R/'step2p1_realizability_report.json'; DEC=R/'step2p1_oof_policy_decisions.jsonl'
def pair_matrix(rows, cols):
 import numpy as np
 grouped=defaultdict(list)
 for a in rows: grouped[a['query_id']].append(a)
 n=0
 for xs in grouped.values():
  c=Counter(a['label'] for a in xs);n+=2*(c['BENEFICIAL']*c['NEUTRAL']+c['BENEFICIAL']*c['HARMFUL']+c['NEUTRAL']*c['HARMFUL'])
 X=np.empty((n,len(cols)),dtype=np.float32);y=np.empty(n,dtype=np.uint8); pos=0; rank={'BENEFICIAL':2,'NEUTRAL':1,'HARMFUL':0}
 for xs in grouped.values():
  xs=sorted(xs,key=lambda a:(a['incoming_union_rank'],a['incoming_doc_id'],a['drop_rank']))
  for i,a in enumerate(xs):
   for b in xs[i+1:]:
    if a['label']==b['label']:continue
    hi,lo=(a,b) if rank[a['label']]>rank[b['label']] else (b,a)
    d=np.asarray([hi['features'][c]-lo['features'][c] for c in cols],dtype=np.float32);X[pos]=d;y[pos]=1;X[pos+1]=-d;y[pos+1]=0;pos+=2
 if pos!=n or int(y.sum())*2!=n:raise RuntimeError('CONTRACT_ERROR pair balance')
 return X,y
def top_scores(model, rows, cols):
 import numpy as np
 grouped=defaultdict(list)
 for a in rows:grouped[a['query_id']].append(a)
 out={}
 for q,xs in grouped.items():
  m=len(xs); X=np.empty((m*(m-1),len(cols)),dtype=np.float32);owner=[];p=0
  for i,a in enumerate(xs):
   for j,b in enumerate(xs):
    if i==j:continue
    X[p]=[a['features'][c]-b['features'][c] for c in cols];owner.append(i);p+=1
  probs=model.predict_proba(X)[:,1]; sums=np.zeros(m,dtype=np.float64);np.add.at(sums,owner,probs);sc=sums/(m-1)
  idx=sorted(range(m),key=lambda i:(-sc[i],-xs[i]['drop_rank'],xs[i]['incoming_union_rank'],xs[i]['incoming_doc_id']))[0];out[q]=(xs[idx],float(sc[idx]))
 return out
def main():
 ns=runpy.run_path(str(ROOT/'scripts/analysis/step2_k77_oof_policy_realizability.py'),run_name='p1_parent')
 v2=json.loads((R/'step2_k77_model_contract_v2.json').read_text()); r1=json.loads((R/'step2_k77_v2_oof_policy_realizability_report.json').read_text()); f2=json.loads((R/'step2f2_rescoring_diagnostic_report.json').read_text()); f3=json.loads((R/'step2f3_feature_sufficiency_report.json').read_text())
 cols=v2['feature_columns']; pre=(r1['status']=='PASS' and r1['scientific_gate']['result']=='FAIL' and r1['pooled']['D77_policy_gain']==0.0 and f2['status']=='PASS' and f2['full_action_rows']==806644 and [f2['trs'][x]['query_count'] for x in ('S_SUCCESS','T_THRESHOLD_LOSS','R_RANKING_DISCRIMINATION_LOSS')]==[7,39,380] and f3['status']=='PASS' and f3['canonical']['feature_count']==36 and f3['upstream_signal_analysis']['pattern']=='SEPARATION_VISIBLE' and len(cols)==len(set(cols))==36)
 if not pre: raise RuntimeError('BLOCKED canonical precondition mismatch')
 folds=ns['target_folds'](); target=set(folds); records,gold=ns['load_gold'](target); base,_=ns['load_baseline'](target,folds); candidates,_=ns['load_candidates'](target,folds); actions,incoming=ns['make_actions'](folds,records,gold,base,candidates,cols)
 if incoming!=403322 or sum(len(x) for x in actions.values())!=806644:raise RuntimeError('CONTRACT_ERROR K77 action contract')
 counts={}; totals=Counter(); min_actions=[]
 for fold in range(1,5):
  per=defaultdict(Counter)
  for a in actions[fold]:per[a['query_id']][a['label']]+=1
  c=Counter()
  for q,x in per.items():
   if sum(x.values())<2:min_actions.append(q)
   c['B_vs_N_unordered_pairs']+=x['BENEFICIAL']*x['NEUTRAL'];c['B_vs_H_unordered_pairs']+=x['BENEFICIAL']*x['HARMFUL'];c['N_vs_H_unordered_pairs']+=x['NEUTRAL']*x['HARMFUL']
  c['unordered_total']=c['B_vs_N_unordered_pairs']+c['B_vs_H_unordered_pairs']+c['N_vs_H_unordered_pairs'];c['mirrored_training_rows_total']=2*c['unordered_total'];counts[str(fold)]=dict(c);totals.update(c)
 # Largest final outer fit uses exactly the union of three folds; float32 feature
 # matrix alone is a hard lower bound, excluding labels/weights/HGB workspace.
 outer={str(h):sum(counts[str(f)]['mirrored_training_rows_total'] for f in range(1,5) if f!=h) for h in range(1,5)}
 maxrows=max(outer.values()); matrix_bytes=maxrows*36*4; minimum_bytes=matrix_bytes+maxrows*1+maxrows*8
 contract={'status':'FROZEN_PRE_FIT','contract_version':'p1','created_before_first_fit':True,'experiment':'STEP 2-P1 — Controlled Query-Aware (Pairwise) Policy Objective Experiment at Fixed K77','parent_feature_contract':'reports/task1/step2_k77_model_contract_v2.json','feature_columns':cols,'feature_order':'exact_v2_order','fixed_K':77,'drop_ranks':[4,5],'pairwise_truth_order':'BENEFICIAL>NEUTRAL>HARMFUL','pair_generation':'all unordered cross-class action pairs within each training query; deterministic rank/doc/drop ordering; two mirrored rows per pair','mirroring':True,'pair_weights':'uniform_1.0','model_family':'sklearn.ensemble.HistGradientBoostingClassifier','hyperparameters':v2['hyperparameters'],'random_seed':20260827,'pairwise_action_score':'mean P(a outranks b) over all other actions; score both ordered directions','tie_break':'drop_rank5_then_lower_union_rank_then_lexicographic_doc_id','outer_oof_protocol':v2['outer_oof_protocol'],'inner_threshold_protocol':'same nested inner-OOF candidate threshold protocol as Step2-R1','original_gate_comparison':'non_binding original Step2 gate','oof_caveat':{'policy_oof_strict':True,'end_to_end_selection_oof':False,'k77_selected_exploratorily_on_folds_1_4':True,'interpretation_scope':'PAIRWISE_POLICY_REALIZABILITY_CONDITIONAL_ON_EXPLORATORY_FIXED_K77'},'fold0_allowed':False,'public_labels_allowed':False,'pair_preflight_counts':counts,'outer_final_fit_mirrored_rows':outer,'largest_final_fit_matrix_float32_bytes':matrix_bytes,'largest_final_fit_minimum_data_bytes':minimum_bytes}
 CONTRACT.write_text(json.dumps(contract,indent=2)+'\n',encoding='utf-8')
 # HGB has no streaming fit interface: it requires a single complete dense X.
 # We explicitly decline to create an enormous matrix when its lower-bound size
 # alone exceeds 16 GiB, rather than sample/change the frozen experiment.
 feasible=minimum_bytes<=16*1024**3 and not min_actions
 status='PREFLIGHT_PASS' if feasible else 'BLOCKED'
 notes=['Full pair population was counted exactly before fit; no pair sampling or weighting change was used.','HistGradientBoostingClassifier has no streaming training API; exact fits use complete dense matrices.']
 if feasible:
  import numpy as np
  from sklearn.ensemble import HistGradientBoostingClassifier
  def fit(rows):
   X,y=pair_matrix(rows,cols);model=HistGradientBoostingClassifier(**v2['hyperparameters']);model.fit(X,y,sample_weight=np.ones(len(y)));return model,len(y)
  def choose_threshold(bestmap):
   pairs=list(bestmap.values()); choices=[]
   for t in [float('inf')]+sorted({s for _,s in pairs}):
    take=[a for a,s in pairs if s>=t];gain=sum(a['truth_delta'] for a in take)/len(pairs);pd=sum((1 if a['label']=='BENEFICIAL' else -1 if a['label']=='HARMFUL' else 0)/5 for a in take)/len(pairs);choices.append((gain,pd,t))
   return max(choices,key=lambda z:(z[0],z[1],z[2]))[2]
  decisions=[];outerruns=[];inner_runs=[]
  for held in range(1,5):
   tr=[f for f in range(1,5) if f!=held];inner={}
   for valid in tr:
    itr=[f for f in tr if f!=valid];model,nrows=fit([a for f in itr for a in actions[f]]); tops=top_scores(model,actions[valid],cols);inner.update(tops);inner_runs.append({'outer_held_out':held,'inner_validation_fold':valid,'train_folds':itr,'mirrored_training_rows_total':nrows})
   tau=choose_threshold(inner);model,nrows=fit([a for f in tr for a in actions[f]]);tops=top_scores(model,actions[held],cols);outerruns.append({'outer_fold_held_out':held,'train_folds':tr,'threshold':'+Infinity' if tau==float('inf') else tau,'mirrored_training_rows_total':nrows})
   for q,(a,s) in tops.items():
    take=s>=tau; final=list(base[q]);
    if take:final[a['drop_rank']-1]=a['incoming_doc_id']
    br=ns['recall'](gold[q],base[q]);pr=ns['recall'](gold[q],final);bp=len(gold[q]&set(base[q]))/5;pp=len(gold[q]&set(final))/5
    decisions.append({'query_id':q,'fold':held,'action_space_size':sum(x['query_id']==q for x in actions[held]),'outer_threshold':'+Infinity' if tau==float('inf') else tau,'query_top_action':{'incoming_doc_id':a['incoming_doc_id'],'incoming_union_rank':a['incoming_union_rank'],'drop_rank':a['drop_rank'],'dropped_doc_id':a['dropped_doc_id'],'pairwise_action_score':s},'executed_action':None if not take else {'incoming_doc_id':a['incoming_doc_id'],'incoming_union_rank':a['incoming_union_rank'],'drop_rank':a['drop_rank'],'dropped_doc_id':a['dropped_doc_id'],'pairwise_action_score':s},'decision':'EXECUTE' if take else 'NO_OP','baseline_top5':base[q],'final_top5':final,'baseline_recall':br,'policy_recall':pr,'recall_delta':pr-br,'baseline_precision':bp,'policy_precision':pp,'precision_delta':pp-bp,'selected_action_class':a['label'] if take else 'NO_OP'})
  with DEC.open('w',encoding='utf-8',newline='\n') as h:
   for z in decisions:h.write(json.dumps(z)+'\n')
  ds=[json.loads(x) for x in DEC.open(encoding='utf-8') if x.strip()]
  def met(xs):
   n=len(xs);cnt=Counter(x['selected_action_class'] for x in xs);br=sum(x['baseline_recall'] for x in xs)/n;pr=sum(x['policy_recall'] for x in xs)/n;bp=sum(x['baseline_precision'] for x in xs)/n;pp=sum(x['policy_precision'] for x in xs)/n
   return {'query_count':n,'baseline_macro_recall':br,'pairwise_policy_macro_recall':pr,'D77_pairwise':pr-br,'baseline_macro_precision':bp,'pairwise_policy_macro_precision':pp,'precision_delta':pp-bp,'queries_execute':n-cnt['NO_OP'],'queries_no_op':cnt['NO_OP'],'executed_beneficial':cnt['BENEFICIAL'],'executed_neutral':cnt['NEUTRAL'],'executed_harmful':cnt['HARMFUL'],'positive_gain_queries':sum(x['recall_delta']>1e-12 for x in xs),'harmful_gain_queries':sum(x['recall_delta']<-1e-12 for x in xs)}
  pooled=met(ds);pooled.update({'C77':.058785714285714274,'C_minus_D':.058785714285714274-pooled['D77_pairwise'],'realization_ratio':pooled['D77_pairwise']/.058785714285714274});per={str(f):met([x for x in ds if x['fold']==f]) for f in range(1,5)}
  gate={'gate_binding':False,'pooled_D_gt_0':pooled['D77_pairwise']>0,'all_fold_D_ge_0':all(x['D77_pairwise']>=0 for x in per.values()),'precision_guardrail':pooled['precision_delta']>=-.5*pooled['D77_pairwise']};gate['result']='PASS' if all(gate[x] for x in ('pooled_D_gt_0','all_fold_D_ge_0','precision_guardrail')) else 'FAIL'
  status='PASS' if len(ds)==5600 and len({x['query_id'] for x in ds})==5600 and pooled['D77_pairwise']<=.058785714285714274+1e-12 else 'CONTRACT_ERROR'
  report={'status':status,'experiment':'STEP 2-P1 — Controlled Query-Aware (Pairwise) Policy Objective Experiment at Fixed K77','experiment_type':'CONTROLLED_DIAGNOSTIC_TRAINING','scientific_status':'DIAGNOSTIC_ONLY','adopted_research_K':77,'production_K_selected':False,'policy_oof_strict':True,'end_to_end_selection_oof':False,'k77_selected_exploratorily_on_folds_1_4':True,'interpretation_scope':'PAIRWISE_POLICY_REALIZABILITY_CONDITIONAL_ON_EXPLORATORY_FIXED_K77','folds_used':[1,2,3,4],'fold0_touched':False,'fold0_payload_materialized':False,'fold0_used_in_training':False,'fold0_used_in_threshold_selection':False,'fold0_used_in_evaluation':False,'fold0_labels_used':False,'public_labels_used':False,'model_contract_path':'reports/task1/step2p1_model_contract.json','oof_decisions_path':'reports/task1/step2p1_oof_policy_decisions.jsonl','fixed_contract':{'feature_count':36,'K':77,'drop_ranks':[4,5],'model':'HistGradientBoostingClassifier','objective':'PAIRWISE_DIFFERENCE_BINARY_LOGLOSS','truth_order':'BENEFICIAL>NEUTRAL>HARMFUL','action_score':'MEAN_PAIRWISE_WIN_PROBABILITY'},'pair_counts':{'inner_runs':inner_runs,'outer_runs':outerruns,'all_folds':dict(totals)},'outer_runs':outerruns,'pooled':pooled,'per_fold':per,'comparators':{'pointwise_step2r1_D77':0.0,'pairwise_minus_pointwise_D77':pooled['D77_pairwise']},'original_step2_gate_comparison':gate,'sanity_checks':{'canonical_preconditions_verified':pre,'feature_count_36':len(cols)==36,'feature_columns_unique':len(set(cols))==36,'k77_incoming_403322':incoming==403322,'k77_actions_806644':sum(len(x) for x in actions.values())==806644,'all_queries_ge_two_actions':not min_actions,'full_pair_counts_exact':True,'pair_contract_written_before_fit':True,'final_metrics_recomputed_from_serialized_decisions':True,'oof_rows_5600':len(ds)==5600,'unique_query_ids_5600':len({x['query_id'] for x in ds})==5600,'D_le_C':pooled['D77_pairwise']<=.058785714285714274+1e-12,'fold0_payload_not_materialized':True,'public_labels_not_used':True},'workflow_document_updated':False,'notes':['All pairwise fits use complete deterministic mirrored cross-class pairs and uniform weights.']}
  REPORT.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps({'status':status,'pooled':pooled,'per_fold':{k:v['D77_pairwise'] for k,v in per.items()}},indent=2));return
 report={'status':status,'experiment':'STEP 2-P1 — Controlled Query-Aware (Pairwise) Policy Objective Experiment at Fixed K77','experiment_type':'CONTROLLED_DIAGNOSTIC_TRAINING','scientific_status':'DIAGNOSTIC_ONLY','adopted_research_K':77,'production_K_selected':False,'policy_oof_strict':True,'end_to_end_selection_oof':False,'k77_selected_exploratorily_on_folds_1_4':True,'interpretation_scope':'PAIRWISE_POLICY_REALIZABILITY_CONDITIONAL_ON_EXPLORATORY_FIXED_K77','folds_used':[1,2,3,4],'fold0_touched':False,'fold0_payload_materialized':False,'fold0_used_in_training':False,'fold0_used_in_threshold_selection':False,'fold0_used_in_evaluation':False,'fold0_labels_used':False,'public_labels_used':False,'model_contract_path':'reports/task1/step2p1_model_contract.json','oof_decisions_path':None,'fixed_contract':{'feature_count':36,'K':77,'drop_ranks':[4,5],'model':'HistGradientBoostingClassifier','objective':'PAIRWISE_DIFFERENCE_BINARY_LOGLOSS','truth_order':'BENEFICIAL>NEUTRAL>HARMFUL','action_score':'MEAN_PAIRWISE_WIN_PROBABILITY'},'pair_counts':{'per_fold_full_population':counts,'all_folds':dict(totals),'outer_final_fit_mirrored_rows':outer},'feasibility':{'all_queries_have_at_least_two_actions':not min_actions,'queries_with_lt_two_actions':min_actions,'largest_final_fit_mirrored_rows':maxrows,'largest_final_fit_matrix_float32_bytes':matrix_bytes,'largest_final_fit_minimum_data_bytes':minimum_bytes,'HGB_streaming_fit_available':False,'full_exact_fit_executed':False},'outer_runs':[],'pooled':{},'per_fold':{},'comparators':{'pointwise_step2r1_D77':0.0,'pairwise_minus_pointwise_D77':None},'original_step2_gate_comparison':{'gate_binding':False,'pooled_D_gt_0':None,'all_fold_D_ge_0':None,'precision_guardrail':None,'result':'NOT_RUN_BLOCKED'},'sanity_checks':{'canonical_preconditions_verified':pre,'feature_count_36':len(cols)==36,'feature_columns_unique':len(set(cols))==36,'k77_incoming_403322':incoming==403322,'k77_actions_806644':sum(len(x) for x in actions.values())==806644,'all_queries_ge_two_actions':not min_actions,'full_pair_counts_exact':True,'pair_contract_written_before_fit':True,'no_model_fit_executed':True,'fold0_payload_not_materialized':True,'public_labels_not_used':True},'workflow_document_updated':False,'notes':notes}
 REPORT.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps({'status':status,'pair_counts':report['pair_counts'],'largest_final_fit_mirrored_rows':maxrows,'minimum_data_bytes':minimum_bytes},indent=2))
if __name__=='__main__':main()
