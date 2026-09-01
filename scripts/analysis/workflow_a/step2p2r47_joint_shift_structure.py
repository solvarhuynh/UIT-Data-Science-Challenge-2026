"""Read-only STEP 2-P2-R47 diagnostic on the fixed rank-4--7 population."""
from __future__ import annotations
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]; R=ROOT/'reports/task1'; OUT=R/'step2p2r47_joint_shift_structure_report.json'; EPS=1e-12
RB=0.8318562423746685; RN=0.80171758726799
def read(n): return json.loads((R/n).read_text(encoding='utf-8'))
def dump(x): OUT.write_text(json.dumps(x,indent=2,allow_nan=False,default=lambda v: v.item() if isinstance(v,np.generic) else str(v))+'\n',encoding='utf-8')
def ident(x): return [x['query_id'],x['incoming_doc_id'],x['drop_rank']]
def stats(rows):
 if not rows:return {'count':0,'median_best_B_score':None,'mean_best_B_score':None,'median_best_N_score':None,'mean_best_N_score':None,'median_B_minus_N':None}
 return {'count':len(rows),'median_best_B_score':float(np.median([x['best_B_score'] for x in rows])),'mean_best_B_score':float(np.mean([x['best_B_score'] for x in rows])),'median_best_N_score':float(np.median([x['best_N_score'] for x in rows])),'mean_best_N_score':float(np.mean([x['best_N_score'] for x in rows])),'median_B_minus_N':float(np.median([x['B_minus_N'] for x in rows]))}
def ranks(v):
 out=[0.]*len(v); order=sorted(range(len(v)),key=lambda i:v[i]);i=0
 while i<len(order):
  j=i+1
  while j<len(order) and v[order[j]]==v[order[i]]:j+=1
  for k in order[i:j]:out[k]=(i+1+j)/2
  i=j
 return out
def spearman(rows):
 if len(rows)<2:return {'n':len(rows),'Spearman_rho':None}
 a=np.asarray(ranks([x['best_B_score'] for x in rows]));b=np.asarray(ranks([x['best_N_score'] for x in rows]));da=a-a.mean();db=b-b.mean();den=np.sqrt(np.dot(da,da)*np.dot(db,db));return {'n':len(rows),'Spearman_rho':None if den==0 else float(np.dot(da,db)/den)}
def main():
 p2s=read('step2p2s_shallow_failure_isolation_report.json');p2=read('step2p2_ranking_failure_localization_report.json');c=read('step2p1_model_contract.json');pre={'p2s_pass':p2s['status']=='PASS','p2_pass':p2['status']=='PASS','K77':c['fixed_K']==77,'p2s_counts':p2s['canonical_reproduction']['shallow_success']==69 and p2s['canonical_reproduction']['shallow_failure']==232}
 g=defaultdict(list);nr=0
 with (R/'step2p1f_full_action_scores.jsonl').open(encoding='utf-8') as h:
  for line in h:x=json.loads(line);g[x['query_id']].append(x);nr+=1
 pre.update({'rows_806644':nr==806644,'queries_5600':len(g)==5600})
 rows=[]
 for q,xs in g.items():
  if max([x['truth_recall_delta'] for x in xs]+[0.])<=0:continue
  b=max((x for x in xs if x['true_action_class']=='BENEFICIAL'),key=lambda x:(x['pairwise_policy_score'],-x['drop_rank'],-x['incoming_union_rank'],x['incoming_doc_id']))
  if not 4<=b['incoming_union_rank']<=7:continue
  n=max((x for x in xs if x['true_action_class']=='NEUTRAL'),key=lambda x:(x['pairwise_policy_score'],-x['drop_rank'],-x['incoming_union_rank'],x['incoming_doc_id']))
  gap=b['pairwise_policy_score']-n['pairwise_policy_score'];out='SUCCESS' if gap>EPS else 'FAILURE' if gap<-EPS else 'TIE';bd=bool(b['pairwise_policy_score']<RB);nu=bool(n['pairwise_policy_score']>RN)
  cat='JOINT_B_DOWN_N_UP' if bd and nu else 'B_DOWN_ONLY' if bd else 'N_UP_ONLY' if nu else 'NEITHER_ADVERSE'
  rows.append({'query_id':q,'fold':b['fold'],'ranking_outcome':out,'best_B_action_identity':ident(b),'best_N_action_identity':ident(n),'best_B_incoming_union_rank':b['incoming_union_rank'],'best_B_score':b['pairwise_policy_score'],'best_N_score':n['pairwise_policy_score'],'B_minus_N':gap,'B_down':bd,'N_up':nu,'joint_shift_category':cat})
 s=[x for x in rows if x['ranking_outcome']=='SUCCESS'];f=[x for x in rows if x['ranking_outcome']=='FAILURE'];t=[x for x in rows if x['ranking_outcome']=='TIE']
 pre.update({'rank4_7_total_74':len(rows)==74,'success_14':len(s)==14,'failure_60':len(f)==60,'ties_0':len(t)==0,'success_B_median':np.median([x['best_B_score'] for x in s])==RB,'failure_B_median':np.median([x['best_B_score'] for x in f])==.7647645260416488,'success_N_median':np.median([x['best_N_score'] for x in s])==RN,'failure_N_median':np.median([x['best_N_score'] for x in f])==.8226402236119117})
 if not all(pre.values()):dump({'status':'CONTRACT_ERROR','experiment':'STEP 2-P2-R47 — Rank-Bin-4-7 Joint B/N Score-Shift Structure Diagnostic','preconditions':pre});print(json.dumps(pre,indent=2));return
 cats=('JOINT_B_DOWN_N_UP','B_DOWN_ONLY','N_UP_ONLY','NEITHER_ADVERSE')
 def classify(z):return {k:{'query_count':sum(x['joint_shift_category']==k for x in z),'fraction_within_outcome':sum(x['joint_shift_category']==k for x in z)/len(z)} for k in cats}
 cl={'SUCCESS':classify(s),'FAILURE':classify(f),'ALL':classify(rows)}
 fs=cl['FAILURE']; anyadv=fs['JOINT_B_DOWN_N_UP']['query_count']+fs['B_DOWN_ONLY']['query_count']+fs['N_UP_ONLY']['query_count'];maxn=max(v['query_count'] for v in fs.values());dominant=[k for k,v in fs.items() if v['query_count']==maxn]
 summaries={o:{k:stats([x for x in z if x['joint_shift_category']==k]) for k in cats} for o,z in [('SUCCESS',s),('FAILURE',f)]}
 report={'status':'PASS','experiment':'STEP 2-P2-R47 — Rank-Bin-4-7 Joint B/N Score-Shift Structure Diagnostic','scientific_status':'DIAGNOSTIC_ONLY_NO_BRANCH_GATE','training_executed':False,'new_scoring_executed':False,'pair_weighting_changed':False,'objective_changed':False,'aggregation_changed':False,'K_changed':False,'policy_oof_strict':True,'end_to_end_selection_oof':False,'k77_selected_exploratorily_on_folds_1_4':True,'interpretation_scope':'RANK_BIN_4_7_JOINT_B_N_SCORE_SHIFT_STRUCTURE_CONDITIONAL_ON_EXPLORATORY_FIXED_K77','K':77,'preconditions':pre,'reference_values':{'success_best_B_median':RB,'success_best_N_median':RN},'canonical_reproduction':{'rank4_7_total':74,'success':14,'failure':60,'ties':0},'classification':cl,'failure_structure':{'joint_count':fs[cats[0]]['query_count'],'joint_fraction':fs[cats[0]]['fraction_within_outcome'],'B_down_only_count':fs[cats[1]]['query_count'],'B_down_only_fraction':fs[cats[1]]['fraction_within_outcome'],'N_up_only_count':fs[cats[2]]['query_count'],'N_up_only_fraction':fs[cats[2]]['fraction_within_outcome'],'neither_count':fs[cats[3]]['query_count'],'neither_fraction':fs[cats[3]]['fraction_within_outcome'],'any_B_down_count':fs[cats[0]]['query_count']+fs[cats[1]]['query_count'],'any_N_up_count':fs[cats[0]]['query_count']+fs[cats[2]]['query_count'],'any_adverse_count':anyadv,'joint_share_among_any_adverse':fs[cats[0]]['query_count']/anyadv if anyadv else None},'spearman':{'SUCCESS':spearman(s),'FAILURE':spearman(f)},'category_score_summaries':summaries,'per_query':sorted(rows,key=lambda x:int(x['query_id'])),'descriptive_structure_summary':{'failure_joint_fraction':fs[cats[0]]['fraction_within_outcome'],'failure_B_down_only_fraction':fs[cats[1]]['fraction_within_outcome'],'failure_N_up_only_fraction':fs[cats[2]]['fraction_within_outcome'],'failure_neither_fraction':fs[cats[3]]['fraction_within_outcome'],'failure_joint_share_among_any_adverse':fs[cats[0]]['query_count']/anyadv if anyadv else None,'success_joint_fraction':cl['SUCCESS'][cats[0]]['fraction_within_outcome'],'success_B_down_only_fraction':cl['SUCCESS'][cats[1]]['fraction_within_outcome'],'success_N_up_only_fraction':cl['SUCCESS'][cats[2]]['fraction_within_outcome'],'success_neither_fraction':cl['SUCCESS'][cats[3]]['fraction_within_outcome'],'failure_Spearman_rho':spearman(f)['Spearman_rho'],'success_Spearman_rho':spearman(s)['Spearman_rho'],'dominant_failure_category':dominant[0] if len(dominant)==1 else 'TIE:'+'|'.join(dominant)},'preserved_status':{'ranking_bottleneck':'PRIMARY','ranking_mechanism':'RANK_CONDITIONED_B_VS_N_SCORE_DISTORTION','pair_family':'STRONGLY_WEAKENED','pair_reweighting':'NOT_AUTHORIZED','B_vs_N_only':'NOT_AUTHORIZED','rank_feature_intervention':'NOT_JUSTIFIED_FOR_CONTROLLED_INTERVENTION','raw_pairwise_dispersion':'DEFERRED','HGB':'UNRESOLVED','feature_expansion':'NOT_JUSTIFIED','K77':'NOT_REJECTED','smaller_K':'DEFERRED','workflow_B':'NOT_ACTIVE'},'notes':['Read-only fixed-population descriptive classification using frozen reference success medians.','Spearman is descriptive only; no causal conclusion or intervention recommendation is made.']}
 dump(report);print(json.dumps({'status':'PASS','failure':report['failure_structure'],'spearman':report['spearman'],'dominant':report['descriptive_structure_summary']['dominant_failure_category']},indent=2))
if __name__=='__main__':main()
