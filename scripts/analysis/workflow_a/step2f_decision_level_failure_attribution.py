"""STEP 2-F: deterministic decision-level attribution; no model operations."""
from __future__ import annotations
import importlib.util,json,math,statistics
from collections import Counter,defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]; R=ROOT/'reports/task1'
DEC=R/'step2_k77_v2_oof_policy_decisions.jsonl'; OUT=R/'step2f_failure_attribution_report.json'
C77=.058785714285714274; TOTAL=329.2; F_SUM={1:68.91666666666667,2:84.58333333333334,3:85.5,4:90.2}
CATS=('A_ORACLE_POSITIVE_NO_OP','B_ORACLE_POSITIVE_BENEFICIAL','C_ORACLE_POSITIVE_NEUTRAL','D_ORACLE_POSITIVE_HARMFUL','E_ORACLE_ZERO_NO_OP','F_ORACLE_ZERO_NEUTRAL','G_ORACLE_ZERO_HARMFUL')
def stats(values, extra=('p25','p75')):
 if not values:return {'count':0,'min':None,'median':None,'mean':None,'p90':None,'max':None,**{x:None for x in extra}}
 s=sorted(values); q=lambda p:s[math.ceil(p*len(s))-1]
 return {'count':len(s),'min':s[0],'median':statistics.median(s),'mean':statistics.fmean(s),'p90':q(.9),'max':s[-1],**({'p25':q(.25),'p75':q(.75)} if 'p25' in extra else {})}
def main():
 rep=json.loads((R/'step2_k77_v2_oof_policy_realizability_report.json').read_text())
 p=rep.get('pooled',{}); valid=(rep.get('status')=='PASS' and rep.get('scientific_gate',{}).get('result')=='FAIL' and rep.get('adopted_research_K')==77 and rep.get('production_K_selected') is False and p.get('D77_policy_gain')==0.0 and p.get('queries_selecting_action')==362 and p.get('queries_no_op')==5238 and (p.get('selected_beneficial'),p.get('selected_neutral'),p.get('selected_harmful'))==(7,348,7) and p.get('oracle_positive_queries')==426)
 if not valid:raise RuntimeError('BLOCKED Step2-R1 contract mismatch')
 decisions=[json.loads(x) for x in DEC.open(encoding='utf-8') if x.strip()]
 if len(decisions)!=5600 or len({x['query_id'] for x in decisions})!=5600 or set(x['fold'] for x in decisions)!={1,2,3,4}:raise RuntimeError('CONTRACT_ERROR decision artifact')
 # Approved Step1A reader supplies target-only structural Fold0 skipping.
 spec=importlib.util.spec_from_file_location('s1a',ROOT/'scripts/analysis/exp_1a_recall_gap_audit.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
 folds=m.read_target_fold_map(); target=set(folds); gold=m.read_target_train(target); base,_=m.read_baseline(target,folds); pool,_=m.read_candidates(target,folds)
 # Candidate rank lookup is target-only decoded by the approved structural reader.
 rows,_=m.stream_target_rows(m.CANDIDATES,target); ranks=defaultdict(dict)
 for row in rows:
  q=str(row['query_id']); xs=row.get('candidates') if isinstance(row.get('candidates'),list) else [row]
  for i,x in enumerate(xs,1):ranks[q].setdefault(str(x['doc_id']),int(x.get('union_rank',i)))
 oracle={}
 for q in target:
  before=m.recall(gold[q],base[q]); best=0.
  for doc,rk in ranks[q].items():
   if rk<=77 and doc not in base[q]:
    for drop in (4,5):
     top=list(base[q]);top[drop-1]=doc;best=max(best,m.recall(gold[q],top)-before)
  oracle[q]=max(0.,best)
 if len(oracle)!=5600 or sum(x>1e-12 for x in oracle.values())!=426 or abs(sum(oracle.values())-TOTAL)>1e-9 or abs(sum(oracle.values())/5600-C77)>1e-12 or any(abs(sum(oracle[q] for q in target if folds[q]==f)-F_SUM[f])>1e-9 for f in F_SUM):raise RuntimeError('CONTRACT_ERROR oracle crosscheck')
 byq={x['query_id']:x for x in decisions}; cats={c:[] for c in CATS}; selected=[]
 for q,d in byq.items():
  pos=oracle[q]>1e-12; cl=d['selected_action_class']
  if pos: cat={'NO_OP':CATS[0],'BENEFICIAL':CATS[1],'NEUTRAL':CATS[2],'HARMFUL':CATS[3]}[cl]
  else:
   if cl=='BENEFICIAL':raise RuntimeError('CONTRACT_ERROR oracle-zero beneficial')
   cat={'NO_OP':CATS[4],'NEUTRAL':CATS[5],'HARMFUL':CATS[6]}[cl]
  d['_oracle']=oracle[q];d['_cat']=cat;cats[cat].append(d)
  if cl!='NO_OP':
   a=d['selected_action'];rk=ranks[q].get(a['incoming_doc_id'])
   if rk is None or rk>77 or a['drop_rank'] not in (4,5) or a['incoming_doc_id'] in d['baseline_top5']:raise RuntimeError('CONTRACT_ERROR selected lookup')
   d['_union_rank']=rk;selected.append(d)
 def catmetrics(xs):
  h=sum(x['_oracle'] for x in xs);g=sum(x['recall_delta'] for x in xs);miss=h-g
  return {'query_count':len(xs),'oracle_headroom_sum':h,'oracle_headroom_macro':h/5600,'oracle_headroom_share_of_total':h/TOTAL,'realized_gain_sum':g,'realized_gain_macro':g/5600,'missed_headroom_sum':miss,'missed_headroom_macro':miss/5600,'missed_headroom_share_of_total_gap':miss/TOTAL,'mean_relevant_count':statistics.fmean(x['relevant_count'] for x in xs) if xs else None,'fold_counts':{str(f):sum(x['fold']==f for x in xs) for f in range(1,5)}}
 catout={c:catmetrics(cats[c]) for c in CATS}
 def foldout(f):
  xs=[x for x in decisions if x['fold']==f];cm={c:catmetrics([x for x in xs if x['_cat']==c]) for c in CATS}; ben=[x for x in xs if x['selected_action_class']=='BENEFICIAL'];harm=[x for x in xs if x['selected_action_class']=='HARMFUL']
  return {'query_count':len(xs),'threshold':xs[0]['no_op_threshold_used'],'threshold_is_infinite':xs[0]['no_op_threshold_used']=='+Infinity','category_counts':{c:cm[c]['query_count'] for c in CATS},'category_metrics':cm,'oracle_positive_count':sum(x['_oracle']>1e-12 for x in xs),'policy_selected_count':sum(x['selected_action_class']!='NO_OP' for x in xs),'policy_noop_count':sum(x['selected_action_class']=='NO_OP' for x in xs),'beneficial_selected':len(ben),'neutral_selected':sum(x['selected_action_class']=='NEUTRAL' for x in xs),'harmful_selected':len(harm),'beneficial_recall_delta_sum':sum(x['recall_delta'] for x in ben),'harmful_recall_delta_sum':sum(x['recall_delta'] for x in harm)}
 per={str(f):foldout(f) for f in range(1,5)}
 def delta_diag(xs,cl):
  z=[x for x in xs if x['selected_action_class']==cl];ds=[x['recall_delta'] for x in z];return {'count':len(z),'recall_delta_sum':sum(ds),'mean_recall_delta':statistics.fmean(ds) if ds else None}
 relevant={}
 for scope,xs in {'pooled':selected,**{f'fold_{f}':[x for x in selected if x['fold']==f] for f in range(1,5)}}.items():relevant[scope]={cl:{str(k):delta_diag([x for x in xs if x['relevant_count']==k],cl) for k in range(1,6)} for cl in ('BENEFICIAL','HARMFUL')}
 score={}
 for scope,xs in {'pooled':decisions,**{f'fold_{f}':[x for x in decisions if x['fold']==f] for f in range(1,5)}}.items():
  finite_noop=[x['policy_score_of_selected_action'] for x in xs if x['selected_action_class']=='NO_OP' and isinstance(x['policy_score_of_selected_action'],(int,float))]
  score[scope]={'executed':{cl:stats([x['policy_score_of_selected_action'] for x in xs if x['selected_action_class']==cl],()) for cl in ('BENEFICIAL','NEUTRAL','HARMFUL')},'no_op':stats(finite_noop,()),'no_op_count':sum(x['selected_action_class']=='NO_OP' for x in xs),'no_op_finite_score_count':len(finite_noop),'threshold':xs[0]['no_op_threshold_used'] if scope!='pooled' else None,'threshold_is_infinite':xs[0]['no_op_threshold_used']=='+Infinity' if scope!='pooled' else None}
 union={}; buckets=((1,20),(21,29),(30,45),(46,77))
 for cl in ('BENEFICIAL','NEUTRAL','HARMFUL'):
  xs=[x for x in selected if x['selected_action_class']==cl]; union[cl]={'summary':stats([x['_union_rank'] for x in xs]),'bucket_by_fold':{str(f):{f'{a}-{b}':sum(x['fold']==f and a<=x['_union_rank']<=b for x in xs) for a,b in buckets} for f in range(1,5)}}
 drop={'pooled':{cl:{str(k):sum(x['selected_action_class']==cl and x['selected_action']['drop_rank']==k for x in selected) for k in (4,5)} for cl in ('BENEFICIAL','NEUTRAL','HARMFUL')},'per_fold':{str(f):{cl:{str(k):sum(x['fold']==f and x['selected_action_class']==cl and x['selected_action']['drop_rank']==k for x in selected) for k in (4,5)} for cl in ('BENEFICIAL','NEUTRAL','HARMFUL')} for f in range(1,5)}}
 hs=[x for x in selected if x['selected_action_class']=='HARMFUL']; hsum=sum(x['recall_delta'] for x in hs); realized=sum(x['recall_delta'] for x in decisions); missed=sum(x['_oracle']-x['recall_delta'] for x in decisions)
 checks={'valid_step2r1_report':valid,'decision_rows_5600':len(decisions)==5600,'unique_query_ids_5600':len(byq)==5600,'folds_1_4_only':set(x['fold'] for x in decisions)=={1,2,3,4},'oracle_map_complete':len(oracle)==5600,'oracle_positive_426':sum(x>1e-12 for x in oracle.values())==426,'oracle_mean_c77':abs(sum(oracle.values())/5600-C77)<=1e-12,'oracle_sum_329_2':abs(sum(oracle.values())-TOTAL)<=1e-9,'per_fold_oracle_sums':all(abs(sum(oracle[q] for q in target if folds[q]==f)-F_SUM[f])<=1e-9 for f in F_SUM),'seven_categories_exhaustive':sum(len(x) for x in cats.values())==5600,'category_positive_total':sum(len(cats[c]) for c in CATS[:4])==426,'selected_noop_total':len(cats[CATS[0]])+len(cats[CATS[4]])==5238,'beneficial_total':len(cats[CATS[1]])==7,'neutral_total':len(cats[CATS[2]])+len(cats[CATS[5]])==348,'harmful_total':len(cats[CATS[3]])+len(cats[CATS[6]])==7,'realized_sum_zero':abs(realized)<=1e-9,'missed_sum_329_2':abs(missed-TOTAL)<=1e-9,'selected_lookup_complete':len(selected)==362,'selected_union_rank_le_77':all(x['_union_rank']<=77 for x in selected),'drop_ranks_valid':all(x['selected_action']['drop_rank'] in (4,5) for x in selected),'no_model_training_or_inference':True,'fold0_payload_not_materialized':True,'public_labels_not_used':True}
 status='PASS' if all(checks.values()) else 'CONTRACT_ERROR'
 selected_delta={'pooled':{'beneficial':delta_diag(selected,'BENEFICIAL'),'harmful':delta_diag(selected,'HARMFUL')},'per_fold':{str(f):{'beneficial':delta_diag([x for x in selected if x['fold']==f],'BENEFICIAL'),'harmful':delta_diag([x for x in selected if x['fold']==f],'HARMFUL')} for f in range(1,5)}}
 for value in selected_delta['per_fold'].values(): value['net_selected_nonzero_delta_sum']=value['beneficial']['recall_delta_sum']+value['harmful']['recall_delta_sum']
 selected_delta['pooled']['net_selected_nonzero_delta_sum']=selected_delta['pooled']['beneficial']['recall_delta_sum']+selected_delta['pooled']['harmful']['recall_delta_sum']
 report={'status':status,'experiment':'STEP 2-F — Decision-Level Failure Attribution for Step2-R1','experiment_type':'DIAGNOSTIC_ONLY','branch_gate':'DIAGNOSTIC_ONLY_NO_BRANCH_GATE','adopted_research_K':77,'production_K_selected':False,'folds_used':[1,2,3,4],'fold0_touched':False,'fold0_payload_materialized':False,'public_labels_used':False,'total_queries':5600,'canonical':{'C77_macro_gain':C77,'C77_sum_gain':TOTAL,'D77_macro_gain':0.0,'decision_policy_gap':C77,'oracle_positive_queries':426},'oracle_map':{'source':'deterministic_reconstruction','query_count':5600,'positive_count':426,'pooled_mean':sum(oracle.values())/5600,'total_sum':sum(oracle.values()),'per_fold':{str(f):{'sum':sum(oracle[q] for q in target if folds[q]==f),'mean':sum(oracle[q] for q in target if folds[q]==f)/1400} for f in range(1,5)}},'categories':catout,'pooled_gain_decomposition':{'oracle_headroom_sum':sum(oracle.values()),'realized_gain_sum':realized,'missed_headroom_sum':missed,'category_A_oracle_headroom_sum':catout[CATS[0]]['oracle_headroom_sum'],'category_A_headroom_share':catout[CATS[0]]['oracle_headroom_share_of_total'],'category_A_missed_headroom_share':catout[CATS[0]]['missed_headroom_share_of_total_gap'],'harmful_query_count':len(hs),'sum_harmful_recall_delta':hsum,'harmful_absolute_loss':-hsum,'harmful_loss_share_of_total_gap':-hsum/TOTAL},'per_fold':per,'selected_nonzero_delta_diagnostics':selected_delta,'relevant_count_diagnostics':relevant,'selected_score_diagnostics':score,'selected_union_rank_diagnostics':union,'drop_rank_diagnostics':drop,'attribution_limits':{'threshold_failure':'UNRESOLVED_REQUIRES_STEP2_F2_FULL_ACTION_RESCORING','ranking_discrimination':'UNRESOLVED_REQUIRES_STEP2_F2_FULL_ACTION_RESCORING','full_action_rescoring_performed':False},'current_status':{'K77':'NOT_REJECTED','rank1_3':'CLOSED_AT_K77','step2_policy':'REJECTED_BY_VALID_STEP2_R1','feature_change':'NOT_AUTHORIZED','classifier_change':'NOT_AUTHORIZED','smaller_K':'NOT_AUTHORIZED','query_conditioned_K':'NOT_AUTHORIZED','workflow_B':'NOT_YET'},'sanity_checks':checks,'workflow_document_updated':False,'notes':['Uses only the valid v2 serialized policy decisions plus deterministic target-only K77 oracle reconstruction.','No model training, model loading, scoring, or threshold retuning was performed.']}
 OUT.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps({'status':status,'A':catout[CATS[0]],'categories':{c:len(cats[c]) for c in CATS},'oracle_sum':sum(oracle.values()),'realized':realized,'missed':missed},indent=2))
if __name__=='__main__':main()
