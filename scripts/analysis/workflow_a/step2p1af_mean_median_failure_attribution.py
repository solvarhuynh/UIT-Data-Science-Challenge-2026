"""Read-only P1A-F attribution from persisted MEAN and MEDIAN action artifacts."""
from __future__ import annotations
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]; R=ROOT/'reports/task1'
TOP=R/'step2p1af_topaction_change_report.json'; BREAK=R/'step2p1af_fold_and_depth_breakdown.json'
MEAN_T={1:.8871766063648394,2:.9158352964999534,3:.8885338990548798,4:.9072451320464932}; MED_T={1:float('inf'),2:float('inf'),3:.9837219720799008,4:float('inf')}
ORDER={'BENEFICIAL':2,'NEUTRAL':1,'HARMFUL':0}

def read(name):return json.loads((R/name).read_text(encoding='utf-8'))
def loadjsonl(name):return [json.loads(x) for x in (R/name).read_text(encoding='utf-8').splitlines() if x.strip()]
def key(x):return (x['query_id'],x['incoming_doc_id'],x['drop_rank'])
def dist(v):
 if not v:return {'count':0,'min':None,'p10':None,'p25':None,'median':None,'mean':None,'p75':None,'p90':None,'max':None}
 a=np.asarray(v,float);return {'count':len(a),'min':float(a.min()),'p10':float(np.percentile(a,10)),'p25':float(np.percentile(a,25)),'median':float(np.median(a)),'mean':float(a.mean()),'p75':float(np.percentile(a,75)),'p90':float(np.percentile(a,90)),'max':float(a.max())}
def matrix(rows):return {a+'->'+b:sum(x['mean']['true_action_class']==a and x['median']['true_action_class']==b for x in rows) for a in ORDER for b in ORDER}
def sums(rows):
 return {'query_count':len(rows),'oracle_positive_count':sum(x['oracle']>0 for x in rows),'mean_realized_delta_sum':sum(x['mean_realized'] for x in rows),'median_realized_delta_sum':sum(x['median_realized'] for x in rows),'delta_realized_sum':sum(x['delta'] for x in rows),'mean_harmful_loss_sum':sum(x['mean_realized'] for x in rows if x['mean']['true_action_class']=='HARMFUL'),'median_harmful_loss_sum':sum(x['median_realized'] for x in rows if x['median']['true_action_class']=='HARMFUL'),'mean_beneficial_gain_sum':sum(x['mean_realized'] for x in rows if x['mean']['true_action_class']=='BENEFICIAL'),'median_beneficial_gain_sum':sum(x['median_realized'] for x in rows if x['median']['true_action_class']=='BENEFICIAL')}
def cat(x):
 a,b=x['mean']['true_action_class'],x['median']['true_action_class']
 if a=='HARMFUL' and b in ('BENEFICIAL','NEUTRAL'):return 'HARM_AVOIDANCE_TOP_FLIP'
 if a=='BENEFICIAL' and b in ('NEUTRAL','HARMFUL'):return 'OPPORTUNITY_LOSS_TOP_FLIP'
 if a in ('NEUTRAL','HARMFUL') and b=='BENEFICIAL':return 'OPPORTUNITY_DISCOVERY_TOP_FLIP'
 if a in ('BENEFICIAL','NEUTRAL') and b=='HARMFUL':return 'HARM_INTRODUCTION_TOP_FLIP'
 return 'SAME_CLASS_RESHUFFLE'
def depth(rank):
 if rank<=20:return '1-20'
 if rank<=29:return '21-29'
 if rank<=45:return '30-45'
 return '46-77'
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
def main():
 p1a=read('step2p1a_realizability_report.json');attr=read('step2p1f_failure_attribution_report.json');p1t=read('step2p1t_threshold_calibration_report.json');p1c=read('step2p1c_realizability_report.json');contract=read('step2p1_model_contract.json')
 mean=loadjsonl('step2p1f_full_action_scores.jsonl');med=loadjsonl('step2p1a_median_scores.jsonl')
 md={key(x):x for x in mean};dd={key(x):x for x in med};checks={'p1a_pass':p1a['status']=='PASS','rows_mean':len(mean)==806644,'rows_median':len(med)==806644,'join':len(md)==len(dd)==806644 and set(md)==set(dd),'K77':contract['fixed_K']==77,'oracle':attr['oracle_positive_queries']==426 and attr['C77']==.058785714285714274,'p1c_pass':p1c['status']=='PASS','p1t_pass':p1t['status']=='PASS'}
 if not all(checks.values()):dump(TOP,{'status':'CONTRACT_ERROR','experiment':'STEP 2-P1A-F — MEAN-vs-MEDIAN Aggregation Failure Attribution','preconditions':checks});return
 by=defaultdict(list)
 for k,m in md.items():by[m['query_id']].append((m,dd[k]))
 oracle={q:max([m['truth_recall_delta'] for m,_ in xs]+[0.]) for q,xs in by.items()}
 states=[]
 for q,xs in by.items():
  mt=next(m for m,d in xs if m['is_query_top_scored']);dt=next(d for m,d in xs if d['is_median_top_action']);mexec=mt['pairwise_policy_score']>=MEAN_T[mt['fold']];dexec=dt['median_pairwise_policy_score']>=MED_T[dt['fold']]
  bestb=max((m for m,d in xs if m['true_action_class']=='BENEFICIAL'),key=lambda x:(x['pairwise_policy_score'],-x['drop_rank'],-x['incoming_union_rank'],x['incoming_doc_id'])) if oracle[q]>0 else None
  st={'q':q,'fold':mt['fold'],'mean':mt,'median':dt,'oracle':oracle[q],'mean_exec':mexec,'median_exec':dexec,'mean_realized':mt['truth_recall_delta'] if mexec else 0.,'median_realized':dt['truth_recall_delta'] if dexec else 0.,'top_changed':key(mt)!=key(dt),'depth':depth(bestb['incoming_union_rank']) if bestb else None}
  st['delta']=st['median_realized']-st['mean_realized'];st['decision_transition']=('EXECUTE' if mexec else 'NOOP')+'_TO_'+('EXECUTE' if dexec else 'NOOP');st['transition_cell']=mt['true_action_class']+'->'+dt['true_action_class'];st['headline_category']=cat(st) if st['top_changed'] else None
  st['mean_advantage_for_mean_top']=mt['pairwise_policy_score']-md[key(dt)]['pairwise_policy_score'];st['median_advantage_for_median_top']=dt['median_pairwise_policy_score']-dd[key(mt)]['median_pairwise_policy_score'];states.append(st)
 changed=[x for x in states if x['top_changed']];opchanged=[x for x in changed if x['oracle']>0]
 def executions(xs,which):
  return [x for x in xs if x[which+'_exec']]
 def trs(xs,which):
  op=[x for x in xs if x['oracle']>0];return {'S':sum(x[which]['true_action_class']=='BENEFICIAL' and x[which+'_exec'] for x in op),'T':sum(x[which]['true_action_class']=='BENEFICIAL' and not x[which+'_exec'] for x in op),'R':sum(x[which]['true_action_class']!='BENEFICIAL' for x in op)}
 mx,dx=executions(states,'mean'),executions(states,'median');mc,dc=Counter(x['mean']['true_action_class'] for x in mx),Counter(x['median']['true_action_class'] for x in dx)
 reproduce={'mean_gain_sum':sum(x['mean_realized'] for x in states),'median_gain_sum':sum(x['median_realized'] for x in states),'delta_gain_sum':sum(x['delta'] for x in states),'mean_executions':len(mx),'median_executions':len(dx),'mean_executed_BNH':[mc['BENEFICIAL'],mc['NEUTRAL'],mc['HARMFUL']],'median_executed_BNH':[dc['BENEFICIAL'],dc['NEUTRAL'],dc['HARMFUL']],'mean_STR':trs(states,'mean'),'median_STR':trs(states,'median'),'top_changes':len(changed),'oracle_positive_top_changes':len(opchanged),'oracle_positive_count':sum(x['oracle']>0 for x in states),'oracle_headroom_sum':sum(oracle.values())}
 expected=reproduce['mean_gain_sum']==-4.0 and reproduce['median_gain_sum']==1/3 and reproduce['delta_gain_sum']==13/3 and reproduce['mean_executions']==410 and reproduce['median_executions']==17 and reproduce['mean_executed_BNH']==[6,392,12] and reproduce['median_executed_BNH']==[1,16,0] and reproduce['mean_STR']=={'S':6,'T':63,'R':357} and reproduce['median_STR']=={'S':1,'T':60,'R':365} and len(changed)==1318 and len(opchanged)==95 and sum(oracle.values())==329.2
 if not expected:dump(TOP,{'status':'CONTRACT_ERROR','experiment':'STEP 2-P1A-F — MEAN-vs-MEDIAN Aggregation Failure Attribution','canonical_reproduction':reproduce});return
 categories={}
 for name in ('HARM_AVOIDANCE_TOP_FLIP','OPPORTUNITY_LOSS_TOP_FLIP','OPPORTUNITY_DISCOVERY_TOP_FLIP','HARM_INTRODUCTION_TOP_FLIP','SAME_CLASS_RESHUFFLE'):
  z=[x for x in changed if x['headline_category']==name];categories[name]={**sums(z),'oracle_headroom_sum':sum(x['oracle'] for x in z),'share_of_329_2':sum(x['oracle'] for x in z)/329.2}
 cells={}
 for same in (True,False):
  for trans in ('EXECUTE_TO_EXECUTE','EXECUTE_TO_NOOP','NOOP_TO_EXECUTE','NOOP_TO_NOOP'):
   cells[('TOP_SAME' if same else 'TOP_CHANGED')+'|'+trans]=sums([x for x in states if x['top_changed']!=same and x['decision_transition']==trans])
 harm=[]
 for x in mx:
  if x['mean']['true_action_class']!='HARMFUL':continue
  if not x['median_exec'] and not x['top_changed']:route='SAME_TOP_EXECUTE_TO_NOOP'
  elif not x['median_exec'] and x['median']['true_action_class']=='NEUTRAL':route='TOP_H_TO_N_NOOP'
  elif not x['median_exec'] and x['median']['true_action_class']=='BENEFICIAL':route='TOP_H_TO_B_NOOP'
  elif x['top_changed'] and x['median_exec']:route='TOP_CHANGED_AND_EXECUTES'
  else:route='OTHER_DETERMINISTIC_TRANSITION'
  harm.append({'query_id':x['q'],'fold':x['fold'],'mean_top_action_identity':key(x['mean']),'median_top_action_identity':key(x['median']),'mean_top_class':x['mean']['true_action_class'],'median_top_class':x['median']['true_action_class'],'mean_score':x['mean']['pairwise_policy_score'],'median_score':x['median']['median_pairwise_policy_score'],'mean_threshold':MEAN_T[x['fold']],'median_threshold':MED_T[x['fold']],'mean_execute':x['mean_exec'],'median_execute':x['median_exec'],'mean_truth_delta':x['mean']['truth_recall_delta'],'median_truth_delta':x['median']['truth_recall_delta'],'delta_realized':x['delta'],'mean_incoming_union_rank':x['mean']['incoming_union_rank'],'median_incoming_union_rank':x['median']['incoming_union_rank'],'mean_drop_rank':x['mean']['drop_rank'],'median_drop_rank':x['median']['drop_rank'],'route':route})
 ben=[]
 for x in mx:
  if x['mean']['true_action_class']!='BENEFICIAL':continue
  if x['median_exec'] and not x['top_changed']:route='RETAINS_BENEFICIAL_AND_EXECUTES'
  elif not x['median_exec'] and not x['top_changed']:route='RETAINS_BENEFICIAL_BUT_NOOPS'
  elif not x['median_exec'] and x['median']['true_action_class']=='NEUTRAL':route='B_TO_N_NOOP'
  elif not x['median_exec'] and x['median']['true_action_class']=='HARMFUL':route='B_TO_H_NOOP'
  else:route='OTHER_DETERMINISTIC_TRANSITION'
  ben.append({'query_id':x['q'],'fold':x['fold'],'mean_top_action_identity':key(x['mean']),'median_top_action_identity':key(x['median']),'mean_top_class':x['mean']['true_action_class'],'median_top_class':x['median']['true_action_class'],'mean_execute':x['mean_exec'],'median_execute':x['median_exec'],'mean_truth_delta':x['mean']['truth_recall_delta'],'median_truth_delta':x['median']['truth_recall_delta'],'delta_realized':x['delta'],'route':route})
 single=[x for x in dx if x['median']['true_action_class']=='BENEFICIAL'][0]
 def collision(z):return {'mean_advantage_for_mean_top':dist([x['mean_advantage_for_mean_top'] for x in z]),'median_advantage_for_median_top':dist([x['median_advantage_for_median_top'] for x in z])}
 feat={}
 for name in categories:
  z=[x for x in changed if x['headline_category']==name];feat[name]={'feature_status':'CORE_ACTION_METADATA_ONLY; exact frozen 36 feature values NOT_PERSISTED_FOR_POSTHOC_ANALYSIS','incoming_union_rank_median_minus_mean':dist([x['median']['incoming_union_rank']-x['mean']['incoming_union_rank'] for x in z]),'drop_rank_median_minus_mean':dist([x['median']['drop_rank']-x['mean']['drop_rank'] for x in z])}
 perq=[]
 for x in opchanged:perq.append({'query_id':x['q'],'fold':x['fold'],'oracle_headroom':x['oracle'],'mean_top_action_identity':key(x['mean']),'median_top_action_identity':key(x['median']),'mean_top_truth_class':x['mean']['true_action_class'],'median_top_truth_class':x['median']['true_action_class'],'transition_cell':x['transition_cell'],'headline_category':x['headline_category'],'mean_score_of_mean_top':x['mean']['pairwise_policy_score'],'median_score_of_mean_top':dd[key(x['mean'])]['median_pairwise_policy_score'],'mean_score_of_median_top':md[key(x['median'])]['pairwise_policy_score'],'median_score_of_median_top':x['median']['median_pairwise_policy_score'],'mean_execute':x['mean_exec'],'median_execute':x['median_exec'],'mean_realized_delta':x['mean_realized'],'median_realized_delta':x['median_realized'],'delta_realized':x['delta'],'mean_incoming_union_rank':x['mean']['incoming_union_rank'],'median_incoming_union_rank':x['median']['incoming_union_rank'],'mean_drop_rank':x['mean']['drop_rank'],'median_drop_rank':x['median']['drop_rank'],'canonical_depth_bucket':x['depth']})
 ranks={'improve':['HARMFUL->NEUTRAL','HARMFUL->BENEFICIAL','NEUTRAL->BENEFICIAL'],'worsen':['BENEFICIAL->NEUTRAL','BENEFICIAL->HARMFUL','NEUTRAL->HARMFUL'],'same':['BENEFICIAL->BENEFICIAL','NEUTRAL->NEUTRAL','HARMFUL->HARMFUL']};rank_summary={k:{'count':sum(x['transition_cell'] in v for x in opchanged),'oracle_headroom_sum':sum(x['oracle'] for x in opchanged if x['transition_cell'] in v)} for k,v in ranks.items()}
 topreport={'status':'PASS','experiment':'STEP 2-P1A-F — MEAN-vs-MEDIAN Aggregation Failure Attribution','scientific_status':'DIAGNOSTIC_ONLY_NO_BRANCH_GATE','training_executed':False,'new_scoring_executed':False,'new_formula_tested':False,'policy_oof_strict':True,'end_to_end_selection_oof':False,'k77_selected_exploratorily_on_folds_1_4':True,'interpretation_scope':'MEAN_VS_MEDIAN_AGGREGATION_FAILURE_ATTRIBUTION_CONDITIONAL_ON_EXPLORATORY_FIXED_K77','K':77,'canonical_reproduction':reproduce,'top_action_changes':{'all_queries':{'count':len(changed),'fraction':len(changed)/5600,'transition_matrix':matrix(changed)},'oracle_positive':{'count':len(opchanged),'transition_matrix':matrix(opchanged),'per_query':perq}},'headline_categories':categories,'top_change_vs_decision_change':cells,'harmful_execution_removal':{'count':len(harm),'details':harm,'removed_loss_total':-sum(x['mean_truth_delta'] for x in harm),'removed_by_top_action_change':-sum(x['mean_truth_delta'] for x in harm if x['mean_top_action_identity']!=x['median_top_action_identity']),'removed_by_same_top_execute_to_noop':-sum(x['mean_truth_delta'] for x in harm if x['route']=='SAME_TOP_EXECUTE_TO_NOOP')},'beneficial_execution_loss':{'count':len(ben),'details':ben,'mean_gain':sum(x['mean_truth_delta'] for x in ben),'median_realized_gain':sum(x['median_truth_delta'] if x['median_execute'] and x['median_top_class']=='BENEFICIAL' else 0 for x in ben),'lost_by_top_action_change':-sum(x['delta_realized'] for x in ben if x['mean_top_action_identity']!=x['median_top_action_identity']),'lost_by_same_top_execute_to_noop':-sum(x['delta_realized'] for x in ben if x['route']=='RETAINS_BENEFICIAL_BUT_NOOPS')},'median_single_beneficial_execution':{'query_id':single['q'],'fold':single['fold'],'mean_top_same':not single['top_changed'],'mean_top_class':single['mean']['true_action_class'],'median_top_class':single['median']['true_action_class'],'mean_execute':single['mean_exec'],'median_execute':single['median_exec'],'mean_score':single['mean']['pairwise_policy_score'],'median_score':single['median']['median_pairwise_policy_score'],'mean_threshold':MEAN_T[single['fold']],'median_threshold':MED_T[single['fold']],'truth_recall_delta':single['median']['truth_recall_delta'],'incoming_union_rank':single['median']['incoming_union_rank'],'drop_rank':single['median']['drop_rank'],'retained_or_discovered':'beneficial retained from MEAN' if single['mean']['true_action_class']=='BENEFICIAL' else 'newly discovered by MEDIAN'},'score_collision_distributions':{'all_changed':collision(changed),'oracle_positive_changed':collision(opchanged),'by_headline_category':{k:collision([x for x in changed if x['headline_category']==k]) for k in categories}},'feature_comparison':feat,'ranking_transition_summary':rank_summary,'preserved_status':{'remaining_ranking_bottleneck':'PRIMARY','aggregation_sensitivity':'STRONGLY_SUPPORTED','median_status':'MECHANISTICALLY_INFORMATIVE_BUT_NOT_REALIZABILITY_SUCCESS','raw_mean_scalar_threshold':'CLOSED','pair_family':'UNRESOLVED_NOT_ESTABLISHED','pair_reweighting':'NOT_AUTHORIZED','another_aggregation':'NOT_AUTHORIZED','HGB':'UNRESOLVED','feature_expansion':'NOT_JUSTIFIED','K77':'NOT_REJECTED','smaller_K':'DEFERRED','workflow_B':'NOT_ACTIVE'},'notes':['Pure post-hoc join; zero model fit, zero new scoring, zero formula/threshold change.']}
 # depth/fold breakdown
 depths={}
 for d in ('1-20','21-29','30-45','46-77'):
  z=[x for x in opchanged if x['depth']==d];depths[d]={**sums(z),'headline_category_counts':dict(Counter(x['headline_category'] for x in z)),'oracle_headroom_sum':sum(x['oracle'] for x in z),'mean_execute_count':sum(x['mean_exec'] for x in z),'median_execute_count':sum(x['median_exec'] for x in z)}
 folds={}
 for f in range(1,5):
  z=[x for x in states if x['fold']==f];ch=[x for x in z if x['top_changed']];op=[x for x in z if x['oracle']>0];opc=[x for x in ch if x['oracle']>0];folds[str(f)]={'total_queries':len(z),'top_action_change_count':len(ch),'top_action_change_fraction':len(ch)/len(z),'oracle_positive_count':len(op),'oracle_positive_top_change_count':len(opc),'transition_matrix':matrix(ch),'headline_category_counts':dict(Counter(x['headline_category'] for x in ch)),'top_change_vs_decision_change':{k:sums([x for x in z if x['top_changed']==(k.startswith('TOP_CHANGED')) and x['decision_transition']==k.split('|')[1]]) for k in cells},'mean_gain_sum':sum(x['mean_realized'] for x in z),'median_gain_sum':sum(x['median_realized'] for x in z),'delta_gain_sum':sum(x['delta'] for x in z),'mean_harmful_loss':sum(x['mean_realized'] for x in z if x['mean']['true_action_class']=='HARMFUL'),'median_harmful_loss':sum(x['median_realized'] for x in z if x['median']['true_action_class']=='HARMFUL'),'mean_beneficial_gain':sum(x['mean_realized'] for x in z if x['mean']['true_action_class']=='BENEFICIAL'),'median_beneficial_gain':sum(x['median_realized'] for x in z if x['median']['true_action_class']=='BENEFICIAL'),'mean_executions':sum(x['mean_exec'] for x in z),'median_executions':sum(x['median_exec'] for x in z),'median_threshold':'+Infinity' if np.isinf(MED_T[f]) else MED_T[f]}
 def special(z):
  ch=[x for x in z if x['top_changed']];op=[x for x in z if x['oracle']>0];opc=[x for x in ch if x['oracle']>0];return {'query_count':len(z),'top_change_rate':len(ch)/len(z),'oracle_positive_top_change_rate':len(opc)/len(op) if op else None,'harm_avoidance_flip_rate':sum(x['headline_category']=='HARM_AVOIDANCE_TOP_FLIP' for x in ch)/len(ch) if ch else None,'opportunity_loss_flip_rate':sum(x['headline_category']=='OPPORTUNITY_LOSS_TOP_FLIP' for x in ch)/len(ch) if ch else None,'same_top_EXECUTE_TO_NOOP_rate':sum(not x['top_changed'] and x['decision_transition']=='EXECUTE_TO_NOOP' for x in z)/len(z),'changed_top_EXECUTE_TO_NOOP_rate':sum(x['top_changed'] and x['decision_transition']=='EXECUTE_TO_NOOP' for x in z)/len(z),'mean_harmful_loss_removed':-sum(x['mean_realized'] for x in z if x['mean']['true_action_class']=='HARMFUL'),'beneficial_gain_lost':-sum(x['delta'] for x in z if x['mean']['true_action_class']=='BENEFICIAL' and x['mean_exec']),'median_beneficial_gain_realized':sum(x['median_realized'] for x in z if x['median']['true_action_class']=='BENEFICIAL')}
 br={'status':'PASS','experiment':'STEP 2-P1A-F — MEAN-vs-MEDIAN Aggregation Failure Attribution','depth_definition_source':'STEP 2-P1F-R2: highest-scoring MEAN P1 BENEFICIAL action incoming_union_rank; buckets 1-20, 21-29, 30-45, 46-77','depth_buckets':depths,'per_fold':folds,'F3_vs_other_folds':{'F3':special([x for x in states if x['fold']==3]),'F1_F2_F4_pooled':special([x for x in states if x['fold']!=3])},'notes':['Read-only post-hoc analysis.']}
 dump(TOP,topreport);dump(BREAK,br);print(json.dumps({'status':'PASS','changes':len(changed),'op_changes':len(opchanged),'harm':len(harm),'beneficial':len(ben),'delta_gain':reproduce['delta_gain_sum']},indent=2))
if __name__=='__main__':main()
