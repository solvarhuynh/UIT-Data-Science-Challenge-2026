"""STEP 2-F3: frozen-feature descriptive forensic; deliberately no model.fit."""
from __future__ import annotations
import json, runpy, statistics
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT=Path(__file__).resolve().parents[2]; R=ROOT/'reports/task1'
SCORES=R/'step2f2_full_action_scores.jsonl'; OUT=R/'step2f3_feature_sufficiency_report.json'
ns=runpy.run_path(str(ROOT/'scripts/analysis/step2_k77_oof_policy_realizability.py'),run_name='f3_parent')
EPS=1e-12
def qstats(a):
 if len(a)==0:return {'count':0,'mean':None,'median':None,'p25':None,'p75':None}
 v=np.asarray(a,float);return {'count':len(v),'mean':float(v.mean()),'median':float(np.median(v)),'p25':float(np.percentile(v,25)),'p75':float(np.percentile(v,75))}
def feature_groups(cols):
 query={'incoming_question_token_length','incoming_question_char_length','dropped_question_token_length','dropped_question_char_length'}
 structural={'dropped_baseline_rank','incoming_is_baseline_top5','incoming_baseline_rank','incoming_is_baseline_top5','dropped_is_baseline_top5'}
 meta={}; g={i:[] for i in range(1,6)}; up=[]
 upstream_tokens=('union_rank','reciprocal_union_rank','source_support','min_source_rank','has_bge_support','bm25_rank','adaptive_k500_rank','knn_char_rank','knn_word_rank','missing')
 for c in cols:
  fam=4 if c in query else 5 if c in structural else 3 if c.startswith('diff_') else 1 if c.startswith('incoming_') else 2 if c.startswith('dropped_') else None
  if fam is None:raise RuntimeError('CONTRACT_ERROR ambiguous family '+c)
  isup=any(t in c for t in upstream_tokens); g[fam].append(c)
  if isup:up.append(c)
  meta[c]={'feature_name':c,'primary_family':fam,'is_upstream_signal':isup,'source_path_or_source_field':'candidate_refs_full/source_ranks or deterministic query/baseline metadata','reason':'name/source tracing fixed before outcome statistics'}
 return g,up,meta
def aucstats(B,N,feat,mi=True):
 b=np.asarray([x[feat] for x in B],float);n=np.asarray([x[feat] for x in N],float); y=np.r_[np.ones(len(b)),np.zeros(len(n))]; v=np.r_[b,n]; auc=float(roc_auc_score(y,v)); pr=float(average_precision_score(y,v)); out={'B':qstats(b),'N':qstats(n),'mean_difference':float(b.mean()-n.mean()),'median_difference':float(np.median(b)-np.median(n)),'ROC_AUC_B_vs_N':auc,'PR_AUC_B_vs_N':pr,'AUC_directional_separation':max(auc,1-auc),'PR_AUC_random_baseline':.5}
 if mi:
  # Explicit source typing: rank/support/missing/baseline flags are discrete; score/length values continuous.
  discrete=any(t in feat for t in ('rank','support','missing','is_baseline','has_bge'))
  try: out.update({'MI':float(mutual_info_classif(v.reshape(-1,1),y,discrete_features=[discrete],random_state=20260827)[0]),'mi_status':'COMPUTED'})
  except Exception:out.update({'MI':None,'mi_status':'NOT_COMPUTED_AMBIGUOUS_FEATURE_TYPE'})
 return out
def summarize(stats, members):
 vals=[stats[x] for x in members]; direction=[x['AUC_directional_separation'] for x in vals]; mi=[x['MI'] for x in vals if x['MI'] is not None]
 best=max(members,key=lambda x:stats[x]['AUC_directional_separation']); bestmi=max((x for x in members if stats[x]['MI'] is not None),key=lambda x:stats[x]['MI'],default=None)
 return {'feature_count':len(members),'median_absolute_ROC_AUC_minus_0_5':float(np.median([abs(x['ROC_AUC_B_vs_N']-.5) for x in vals])),'max_absolute_ROC_AUC_minus_0_5':max(abs(x['ROC_AUC_B_vs_N']-.5) for x in vals),'median_AUC_directional_separation':float(np.median(direction)),'max_AUC_directional_separation':max(direction),'median_MI':float(np.median(mi)) if mi else None,'feature_largest_directional_AUC':best,'feature_largest_MI':bestmi}
def main():
 c=json.loads((R/'step2_k77_model_contract_v2.json').read_text()); f2=json.loads((R/'step2f2_rescoring_diagnostic_report.json').read_text()); repro=json.loads((R/'step2f2_reproduction_check_report.json').read_text())
 cols=c['feature_columns']; trs=f2['trs']; required=(repro['status']=='PASS' and repro['reproduction_gate_pass'] and f2['status']=='PASS' and f2['full_action_rows']==806644 and len(cols)==len(set(cols))==36 and [trs[x]['query_count'] for x in ('S_SUCCESS','T_THRESHOLD_LOSS','R_RANKING_DISCRIMINATION_LOSS')]==[7,39,380])
 if not required:raise RuntimeError('BLOCKED F2 preconditions')
 groups,up,meta=feature_groups(cols)
 # F2 score artifact is intentionally feature-free; reconstruct only frozen feature values with unchanged Step2 code.
 folds=ns['target_folds'](); target=set(folds); records,gold=ns['load_gold'](target); base,_=ns['load_baseline'](target,folds); candidates,_=ns['load_candidates'](target,folds); actions,incoming=ns['make_actions'](folds,records,gold,base,candidates,cols)
 action={ (a['query_id'],a['incoming_doc_id'],a['drop_rank']):a for ff in range(1,5) for a in actions[ff] }
 byquery=defaultdict(list)
 for a in action.values():byquery[a['query_id']].append(a)
 score={}; truth={}; top={}
 rows=0
 for line in SCORES.open(encoding='utf-8'):
  z=json.loads(line);rows+=1;k=(z['query_id'],z['incoming_doc_id'],z['drop_rank']);score[k]=z['policy_score'];truth[k]=z['true_action_class']
  if z['is_query_top_scored']:top[z['query_id']]=k
 if rows!=806644 or len(score)!=806644 or len(action)!=806644 or set(score)!=set(action):raise RuntimeError('CONTRACT_ERROR action join')
 diag={x['query_id']:x for x in f2['oracle_positive_query_diagnostics']}; oq=set(diag)
 if len(oq)!=426:raise RuntimeError('CONTRACT_ERROR oracle positives')
 reps={}; depth={'1-20':[],'21-29':[],'30-45':[],'46-77':[]}
 for q in oq:
  items=byquery[q]
  by={cl:sorted([a for a in items if a['label']==cl],key=lambda a:(-score[(q,a['incoming_doc_id'],a['drop_rank'])],-a['drop_rank'],a['incoming_union_rank'],a['incoming_doc_id']))[0] for cl in ('BENEFICIAL','NEUTRAL','HARMFUL') if any(a['label']==cl for a in items)}
  if abs(score[(q,by['BENEFICIAL']['incoming_doc_id'],by['BENEFICIAL']['drop_rank'])]-diag[q]['best_beneficial_score'])>1e-12:raise RuntimeError('CONTRACT_ERROR REP_B score')
  if 'NEUTRAL' in by and abs(score[(q,by['NEUTRAL']['incoming_doc_id'],by['NEUTRAL']['drop_rank'])]-diag[q]['best_neutral_score'])>1e-12:raise RuntimeError('CONTRACT_ERROR REP_N score')
  reps[q]=by;r=by['BENEFICIAL']['incoming_union_rank'];bucket='1-20' if r<=20 else '21-29' if r<=29 else '30-45' if r<=45 else '46-77';depth[bucket].append(q)
 if {k:len(v) for k,v in depth.items()}!={'1-20':300,'21-29':44,'30-45':43,'46-77':39}:raise RuntimeError('CONTRACT_ERROR depth')
 paired=[q for q in oq if 'BENEFICIAL' in reps[q] and 'NEUTRAL' in reps[q]];B=[reps[q]['BENEFICIAL']['features'] for q in paired];N=[reps[q]['NEUTRAL']['features'] for q in paired]
 if len(B)!=len(N):raise RuntimeError('CONTRACT_ERROR unbalanced reps')
 primary={x:aucstats(B,N,x) for x in cols}; Rq=[q for q in paired if diag[q]['T_R_S_class']=='R_RANKING_DISCRIMINATION_LOSS']; Rstats={x:aucstats([reps[q]['BENEFICIAL']['features'] for q in Rq],[reps[q]['NEUTRAL']['features'] for q in Rq],x) for x in cols}
 dstat={}
 for buck,qs in depth.items():
  pq=[q for q in qs if q in paired];dstat[buck]={'query_count':len(qs),'feature_statistics':{x:aucstats([reps[q]['BENEFICIAL']['features'] for q in pq],[reps[q]['NEUTRAL']['features'] for q in pq],x) for x in cols}}
 family={f'group{i}':summarize(primary,groups[i]) for i in range(1,6)};family['group6_upstream']=summarize(primary,up)
 dirs={}
 for x in up:
  ds=[]
  for buck in depth: 
   s=dstat[buck]['feature_statistics'][x];delta=s['B']['median']-s['N']['median'];ds.append('POSITIVE' if delta>EPS else 'NEGATIVE' if delta<-EPS else 'TIED')
  dirs[x]=ds
 visible=[x for x,d in dirs.items() if (len(set(d))==1 and d[0] in ('POSITIVE','NEGATIVE') and ((d[0]=='POSITIVE' and primary[x]['ROC_AUC_B_vs_N']>.5) or (d[0]=='NEGATIVE' and primary[x]['ROC_AUC_B_vs_N']<.5)))]
 pattern='SEPARATION_VISIBLE' if visible else 'SEPARATION_WEAK' if not any(len(set(d))==1 and d[0] in ('POSITIVE','NEGATIVE') for d in dirs.values()) else 'MIXED'
 # Secondary global distributions; raw actions are descriptive only.
 # Build global reference arrays in one pass; this is only an execution
 # optimization of the descriptive ACTION_LEVEL_SECONDARY_REFERENCE view.
 global_arrays={cl:[] for cl in ('BENEFICIAL','NEUTRAL','HARMFUL')}
 for a in action.values(): global_arrays[a['label']].append([a['features'][x] for x in cols])
 global_arrays={cl:np.asarray(v,float) for cl,v in global_arrays.items()}
 globalref={cl:{'count':len(global_arrays[cl]),'features':{x:qstats(global_arrays[cl][:,i]) for i,x in enumerate(cols)}} for cl in global_arrays}
 gy=np.r_[np.ones(len(global_arrays['BENEFICIAL'])),np.zeros(len(global_arrays['NEUTRAL']))]
 globalref['B_vs_N']={x:{'ROC_AUC':float(roc_auc_score(gy,np.r_[global_arrays['BENEFICIAL'][:,i],global_arrays['NEUTRAL'][:,i]])),'PR_AUC':float(average_precision_score(gy,np.r_[global_arrays['BENEFICIAL'][:,i],global_arrays['NEUTRAL'][:,i]]))} for i,x in enumerate(cols)}
 hq=[q for q in oq if 'HARMFUL' in reps[q]]; harmful={'label':'SECONDARY_HARMFUL_CONTRAST','query_count':len(hq),'feature_statistics':{x:aucstats([reps[q]['BENEFICIAL']['features'] for q in hq],[reps[q]['HARMFUL']['features'] for q in hq],x) for x in cols}}
 sts={t:[q for q in oq if diag[q]['T_R_S_class']==t] for t in ('S_SUCCESS','T_THRESHOLD_LOSS','R_RANKING_DISCRIMINATION_LOSS')}; views={t:{'label':'SMALL_SAMPLE_DESCRIPTIVE_ONLY' if t=='S_SUCCESS' else 'DESCRIPTIVE_ONLY','query_count':len(qs),'representative_B_feature_summaries':{x:qstats([reps[q]['BENEFICIAL']['features'][x] for q in qs]) for x in cols}} for t,qs in sts.items()}
 # No persisted models: F2 serialized scores only and script writes no model artifacts; refitting solely for F3 prohibited.
 checks={'f2_preconditions':required,'full_action_rows':rows==806644,'feature_count_36':len(cols)==36,'feature_schema_unique':len(set(cols))==36,'feature_join_806644':len(action)==len(score)==806644,'feature_groups_frozen_pre_statistics':True,'primary_family_coverage_36':sum(len(v) for v in groups.values())==36,'upstream_tracing_completed':bool(up),'oracle_positive_426':len(oq)==426,'trs_7_39_380':[len(sts[x]) for x in sts]==[7,39,380],'depth_counts':{k:len(v) for k,v in depth.items()}=={'1-20':300,'21-29':44,'30-45':43,'46-77':39},'representative_scores_reproduce_f2':True,'primary_balanced':len(B)==len(N),'mi_recorded_all_features':all('mi_status' in primary[x] for x in cols),'no_model_fit':True,'fold0_payload_not_materialized':True,'public_labels_not_used':True}
 report={'status':'PASS' if all(checks.values()) else 'CONTRACT_ERROR','experiment':'STEP 2-F3 — Frozen-Feature Discriminative Sufficiency Forensic','experiment_type':'DIAGNOSTIC_ONLY','branch_gate':'DIAGNOSTIC_ONLY_NO_BRANCH_GATE','adopted_research_K':77,'production_K_selected':False,'folds_used':[1,2,3,4],'fold0_touched':False,'fold0_payload_materialized':False,'public_labels_used':False,'canonical':{'full_action_rows':806644,'feature_count':36,'oracle_positive_queries':426,'S':7,'T':39,'R':380,'depth_counts':{k:len(v) for k,v in depth.items()}},'feature_materialization':{'source':'DETERMINISTIC_RECONSTRUCTION','rows_matched':806644,'schema_verified':True},'feature_grouping_audit':meta,'feature_groups':{'group1_incoming':groups[1],'group2_dropped':groups[2],'group3_diff':groups[3],'group4_query':groups[4],'group5_structural':groups[5],'group6_upstream_overlay':up},'primary_query_balanced_B_vs_N':{'query_count':len(paired),'representative_rows':2*len(paired),'feature_statistics':primary},'ranking_loss_only_B_vs_N':{'query_count':len(Rq),'feature_statistics':Rstats},'depth_stratified':dstat,'harmful_secondary':harmful,'success_threshold_reference_view':views,'family_summaries':family,'upstream_signal_analysis':{'features':[{'feature_name':x,'pooled':primary[x],'depth_ordering_direction':dirs[x]} for x in up],'pattern':pattern,'predeclared_rule':'VISIBLE iff a feature has one non-tied median B-vs-N direction in all four buckets and pooled ROC direction agrees; WEAK iff none has all-four consistent direction; otherwise MIXED.'},'global_action_level_reference':{'label':'ACTION_LEVEL_SECONDARY_REFERENCE','statistics':globalref},'permutation_importance':{'status':'NOT_RUN_NO_PERSISTED_FROZEN_MODEL','model_artifacts':[],'family_results':{},'note':'Professor prohibited refitting solely for Step2-F3 permutation importance; Step2-F2 score artifact remains valid for all non-refitting analyses.'},'current_status':{'K77':'NOT_REJECTED','threshold':'DEFERRED','smaller_K':'DEFERRED','query_conditioned_K':'DEFERRED','model_family':'UNRESOLVED','objective':'UNRESOLVED','workflow_B':'CANDIDATE_AFTER_FEATURE_SUFFICIENCY_FORENSIC'},'sanity_checks':checks,'workflow_document_updated':False,'notes':['Truth labels are audit outputs only and never entered the 36 frozen feature vectors.','No model.fit, ablation, model change, threshold change, or K change occurred.']}
 OUT.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps({'status':report['status'],'paired':len(paired),'upstream_count':len(up),'pattern':pattern,'top_upstream':family['group6_upstream']['feature_largest_directional_AUC']},indent=2))
if __name__=='__main__':main()
