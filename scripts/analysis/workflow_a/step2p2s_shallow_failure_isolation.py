"""Read-only STEP 2-P2-S shallow ranking-failure isolation diagnostic."""
from __future__ import annotations
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]; R=ROOT/'reports/task1'; OUT=R/'step2p2s_shallow_failure_isolation_report.json'; EPS=1e-12
def read(n): return json.loads((R/n).read_text(encoding='utf-8'))
def dump(x): OUT.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def dist(v):
 if not v:return {'count':0,**{k:None for k in ('min','p10','p25','median','mean','p75','p90','max')}}
 a=np.asarray(v,float);return {'count':len(a),'min':float(a.min()),'p10':float(np.percentile(a,10)),'p25':float(np.percentile(a,25)),'median':float(np.median(a)),'mean':float(a.mean()),'p75':float(np.percentile(a,75)),'p90':float(np.percentile(a,90)),'max':float(a.max())}
def auc(s,f):
 w=sum(a>b for a in s for b in f);t=sum(a==b for a in s for b in f);z=len(s)*len(f);raw=(w+.5*t)/z
 return {'ROC_AUC_raw':raw,'ROC_AUC_directional':max(raw,1-raw),'P_success_gt_failure':w/z,'ties':t,'tie_probability':t/z}
def compare(rows,field):
 s=[x[field] for x in rows if x['outcome']=='SUCCESS'];f=[x[field] for x in rows if x['outcome']=='FAILURE'];return {'success':dist(s),'failure':dist(f),**auc(s,f)}
def gap_bucket(g):
 if abs(g)<=EPS:return 'TIE'
 if g<-.2:return 'DECISIVELY_WRONG'
 if g<-.05:return 'MODERATELY_WRONG'
 if g<0:return 'NEAR_TIE_WRONG'
 if g<=.05:return 'NARROW_SUCCESS'
 if g<=.2:return 'MODERATE_SUCCESS'
 return 'DECISIVE_SUCCESS'
def main():
 p2=read('step2p2_ranking_failure_localization_report.json');contract=read('step2p1_model_contract.json')
 pre={'p2_pass':p2['status']=='PASS','K77':contract['fixed_K']==77,'p2_success_failure':p2['canonical_reproduction']['success']==69 and p2['canonical_reproduction']['failure']==357,'p2_auc_comparators':p2['success_failure_descriptive_auc']['best_B_score']['ROC_AUC_directional']==.8102951325457719 and p2['success_failure_descriptive_auc']['incoming_union_rank_best_B']['ROC_AUC_directional']==.9573133601266594}
 groups=defaultdict(list);rows=0
 with (R/'step2p1f_full_action_scores.jsonl').open(encoding='utf-8') as h:
  for l in h:
   x=json.loads(l);groups[x['query_id']].append(x);rows+=1
 pre.update({'rows_806644':rows==806644,'queries_5600':len(groups)==5600})
 states=[]
 for q,rs in groups.items():
  oracle=max([x['truth_recall_delta'] for x in rs]+[0.])
  if oracle<=0:continue
  cls={c:[x for x in rs if x['true_action_class']==c] for c in ('BENEFICIAL','NEUTRAL','HARMFUL')}
  b=max(cls['BENEFICIAL'],key=lambda x:(x['pairwise_policy_score'],-x['drop_rank'],-x['incoming_union_rank'],x['incoming_doc_id']))
  n=max(cls['NEUTRAL'],key=lambda x:(x['pairwise_policy_score'],-x['drop_rank'],-x['incoming_union_rank'],x['incoming_doc_id']))
  top=max(rs,key=lambda x:(x['pairwise_policy_score'],-x['drop_rank'],-x['incoming_union_rank'],x['incoming_doc_id']))
  gap=b['pairwise_policy_score']-n['pairwise_policy_score'];out='SUCCESS' if gap>EPS else 'FAILURE' if gap<-EPS else 'TIE'
  states.append({'query_id':q,'fold':b['fold'],'oracle':oracle,'outcome':out,'depth':b['incoming_union_rank'],'best_B_score':b['pairwise_policy_score'],'best_N_score':n['pairwise_policy_score'],'B_minus_N':gap,'incoming_union_rank_best_B':b['incoming_union_rank'],'incoming_union_rank_best_N':n['incoming_union_rank'],'drop_rank_best_B':b['drop_rank'],'drop_rank_best_N':n['drop_rank'],'top_class':top['true_action_class'],'total_actions':len(rs),'unique_incoming_doc_ids':len({x['incoming_doc_id'] for x in rs}),'drop_rank_alternatives':len({x['drop_rank'] for x in rs})})
 shallow=[x for x in states if x['depth']<=20];deep=[x for x in states if x['depth']>20];s=[x for x in shallow if x['outcome']=='SUCCESS'];f=[x for x in shallow if x['outcome']=='FAILURE']
 pre.update({'oracle_positive_426':len(states)==426,'shallow_301':len(shallow)==301,'shallow_success_69':len(s)==69,'shallow_failure_232':len(f)==232,'deep_success_0':sum(x['outcome']=='SUCCESS' for x in deep)==0,'deep_failure_125':sum(x['outcome']=='FAILURE' for x in deep)==125,'headroom_total':sum(x['oracle'] for x in shallow)==233.91666666666666,'headroom_success':sum(x['oracle'] for x in s)==55.5,'headroom_failure':sum(x['oracle'] for x in f)==178.41666666666666,'top_NH_229_3':Counter(x['top_class'] for x in f)==Counter({'NEUTRAL':229,'HARMFUL':3})})
 if not all(pre.values()):dump({'status':'CONTRACT_ERROR','experiment':'STEP 2-P2-S — Shallow Ranking-Failure Isolation Diagnostic','preconditions':pre});print(json.dumps(pre,indent=2));return
 for x in shallow:x['gap_bucket']=gap_bucket(x['B_minus_N'])
 bscore=compare(shallow,'best_B_score');nscore=compare(shallow,'best_N_score');brank=compare(shallow,'incoming_union_rank_best_B');nrank=compare(shallow,'incoming_union_rank_best_N');db=compare(shallow,'drop_rank_best_B');dn=compare(shallow,'drop_rank_best_N')
 def persistence(a):return 'PERSISTS_STRONGLY' if a>=.80 else 'PERSISTS_VISIBLY' if a>=.70 else 'WEAKENS' if a>=.60 else 'COLLAPSES'
 def rankstatus(a):return 'SHALLOW_RANK_GRADIENT_PERSISTS' if a>=.75 else 'SHALLOW_RANK_GRADIENT_WEAKENS' if a>=.60 else 'SHALLOW_RANK_GRADIENT_COLLAPSES'
 gb={}
 for g in ('DECISIVELY_WRONG','MODERATELY_WRONG','NEAR_TIE_WRONG','TIE','NARROW_SUCCESS','MODERATE_SUCCESS','DECISIVE_SUCCESS'):
  z=[x for x in shallow if x['gap_bucket']==g];gb[g]={'query_count':len(z),'failure_share':len(z)/232 if 'WRONG' in g else None,'oracle_headroom_sum':sum(x['oracle'] for x in z),'fold_distribution':dict(Counter(str(x['fold']) for x in z))}
 bins={}; supported=[]
 for name,lo,hi in [('1-3',1,3),('4-7',4,7),('8-12',8,12),('13-20',13,20)]:
  z=[x for x in shallow if lo<=x['depth']<=hi];zs=[x for x in z if x['outcome']=='SUCCESS'];zf=[x for x in z if x['outcome']=='FAILURE'];mbs=dist([x['best_B_score'] for x in zs])['median'];mbf=dist([x['best_B_score'] for x in zf])['median'];mns=dist([x['best_N_score'] for x in zs])['median'];mnf=dist([x['best_N_score'] for x in zf])['median'];bins[name]={'success_count':len(zs),'failure_count':len(zf),'failure_rate':len(zf)/len(z),'median_best_B_score_success':mbs,'median_best_B_score_failure':mbf,'median_best_N_score_success':mns,'median_best_N_score_failure':mnf,'median_B_minus_N':dist([x['B_minus_N'] for x in z])['median']}
  if mbs is not None and mbf is not None:supported.append(mbf-mbs)
 # Fixed descriptive decision: persistent negative B-score shift in every supported bin => beyond depth;
 # all shifts <0.02 absolute => primarily depth; mixed signs/strength => mixed.
 if not supported:within='INSUFFICIENT_WITHIN_BIN_SUPPORT'
 elif all(v<=-.02 for v in supported):within='SCORE_SUPPRESSION_BEYOND_DEPTH_VISIBLE'
 elif all(abs(v)<.02 for v in supported):within='SUPPRESSION_PRIMARILY_DEPTH_CONDITIONED'
 else:within='MIXED_DEPTH_AND_SCORE_EFFECT'
 delta_b=bscore['failure']['median']-bscore['success']['median'];delta_n=nscore['failure']['median']-nscore['success']['median']
 pattern='NO_CLEAR_SCORE_SHIFT' if abs(delta_b)<.02 and abs(delta_n)<.02 else 'B_SUPPRESSION_DOMINANT' if abs(delta_b)>=1.5*abs(delta_n) else 'N_INFLATION_DOMINANT' if abs(delta_n)>=1.5*abs(delta_b) else 'BOTH_COMPARABLE'
 folds={}
 for fold in range(1,5):
  z=[x for x in shallow if x['fold']==fold];zs=[x for x in z if x['outcome']=='SUCCESS'];zf=[x for x in z if x['outcome']=='FAILURE'];folds[str(fold)]={'shallow_oracle_positive_count':len(z),'shallow_success':len(zs),'shallow_failure':len(zf),'shallow_failure_rate':len(zf)/len(z),'median_best_B_score_success':dist([x['best_B_score'] for x in zs])['median'],'median_best_B_score_failure':dist([x['best_B_score'] for x in zf])['median'],'median_best_N_score_success':dist([x['best_N_score'] for x in zs])['median'],'median_best_N_score_failure':dist([x['best_N_score'] for x in zf])['median'],'median_best_B_rank_success':dist([x['depth'] for x in zs])['median'],'median_best_B_rank_failure':dist([x['depth'] for x in zf])['median'],'gap_bucket_counts':dict(Counter(x['gap_bucket'] for x in z))}
 ac={k:compare(shallow,k) for k in ('total_actions','unique_incoming_doc_ids','drop_rank_alternatives')};action_explanatory=any(v['ROC_AUC_directional']>=.60 for v in ac.values())
 summary='SHALLOW_B_SCORE_SUPPRESSION_PERSISTS' if bscore['ROC_AUC_directional']>=.75 and within=='SCORE_SUPPRESSION_BEYOND_DEPTH_VISIBLE' else 'SHALLOW_FAILURE_MAINLY_EXPLAINED_BY_RESIDUAL_RANK_DIFFERENCE' if brank['ROC_AUC_directional']>=.75 and within=='SUPPRESSION_PRIMARILY_DEPTH_CONDITIONED' else 'SHALLOW_FAILURE_BOTH_SCORE_AND_RANK_DEPENDENT' if bscore['ROC_AUC_directional']>=.70 and brank['ROC_AUC_directional']>=.70 and within=='MIXED_DEPTH_AND_SCORE_EFFECT' else 'SHALLOW_MECHANISM_UNRESOLVED'
 report={'status':'PASS','experiment':'STEP 2-P2-S — Shallow Ranking-Failure Isolation Diagnostic','scientific_status':'DIAGNOSTIC_ONLY_NO_BRANCH_GATE','training_executed':False,'new_scoring_executed':False,'pair_weighting_changed':False,'objective_changed':False,'aggregation_changed':False,'K_changed':False,'policy_oof_strict':True,'end_to_end_selection_oof':False,'k77_selected_exploratorily_on_folds_1_4':True,'interpretation_scope':'SHALLOW_PAIRWISE_RANKING_FAILURE_ISOLATION_CONDITIONAL_ON_EXPLORATORY_FIXED_K77','K':77,'preconditions':pre,'canonical_reproduction':{'oracle_positive':426,'overall_success':69,'overall_failure':357,'shallow_total':301,'shallow_success':69,'shallow_failure':232,'deep_success':0,'deep_failure':125},'best_B_score_analysis':{**bscore,'pooled_directional_AUC_comparator':.8102951325457719,'persistence_status':persistence(bscore['ROC_AUC_directional'])},'best_N_score_analysis':{**nscore,'failure_associated_N_score_elevation':'VISIBLE' if nscore['failure']['median']>nscore['success']['median'] else 'NOT_VISIBLE'},'gap_analysis':{'buckets':gb,'failure_bucket_sum':sum(gb[k]['query_count'] for k in ('DECISIVELY_WRONG','MODERATELY_WRONG','NEAR_TIE_WRONG')),'success_bucket_sum':sum(gb[k]['query_count'] for k in ('NARROW_SUCCESS','MODERATE_SUCCESS','DECISIVE_SUCCESS')),'tautology_warning':'Gap sign defines outcome; buckets characterize severity only.'},'best_B_incoming_rank_analysis':{**brank,'pooled_directional_AUC_comparator':.9573133601266594,'rank_gradient_status':rankstatus(brank['ROC_AUC_directional'])},'best_N_incoming_rank_analysis':nrank,'drop_rank_analysis':{'best_B':db,'best_N':dn,'status':'NO_DESCRIPTIVE_SEPARATION' if db['ROC_AUC_directional']<.60 and dn['ROC_AUC_directional']<.60 else 'DESCRIPTIVE_SEPARATION_PRESENT'},'score_decomposition':{'delta_B_median_failure_minus_success':delta_b,'delta_N_median_failure_minus_success':delta_n,'abs_delta_B_median':abs(delta_b),'abs_delta_N_median':abs(delta_n),'pattern':pattern},'rank_conditioned_score_analysis':{'fixed_bins':bins,'supported_bin_B_score_failure_minus_success':supported},'oracle_headroom':{'total':233.91666666666666,'success':55.5,'failure':178.41666666666666,'failure_share':178.41666666666666/233.91666666666666},'per_fold':folds,'shallow_failure_top_class':dict(Counter(x['top_class'] for x in f)),'action_count_control':{**ac,'action_count_explanatory':action_explanatory,'status':'ACTION_COUNT_NOT_EXPLANATORY_WITHIN_SHALLOW' if not action_explanatory else 'DESCRIPTIVE_ACTION_COUNT_SEPARATION'},'within_rank_bin_mechanism':within,'shallow_mechanism_summary':summary,'preserved_status':{'ranking_bottleneck':'PRIMARY','pair_family':'STRONGLY_WEAKENED','pair_reweighting':'NOT_AUTHORIZED','B_vs_N_only':'NOT_AUTHORIZED','raw_pairwise_dispersion':'DEFERRED','HGB':'UNRESOLVED','feature_expansion':'NOT_JUSTIFIED','K77':'NOT_REJECTED','smaller_K':'DEFERRED','workflow_B':'NOT_ACTIVE'},'notes':['Read-only diagnostic using persisted frozen MEAN action scores.','Within-bin classification is descriptive association, not causal attribution or intervention authorization.']}
 dump(report);print(json.dumps({'status':'PASS','best_B_auc':bscore['ROC_AUC_directional'],'best_N_auc':nscore['ROC_AUC_directional'],'rank_auc':brank['ROC_AUC_directional'],'within_bin':within,'summary':summary},indent=2))
if __name__=='__main__':main()
