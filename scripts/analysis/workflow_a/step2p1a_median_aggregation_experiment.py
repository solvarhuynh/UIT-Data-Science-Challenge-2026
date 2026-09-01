"""P1A: deterministic posthoc MEAN->MEDIAN recombination; never calls model.fit."""
from __future__ import annotations
import json, runpy
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from joblib import load

ROOT=Path(__file__).resolve().parents[2]; R=ROOT/'reports/task1'
MEDIAN=R/'step2p1a_median_scores.jsonl'; INNER=R/'step2p1a_inner_oof_median_top_scores.jsonl'; THRESH=R/'step2p1a_threshold_selection_report.json'; OUT=R/'step2p1a_realizability_report.json'
MEAN_T={1:.8871766063648394,2:.9158352964999534,3:.8885338990548798,4:.9072451320464932}; MEAN_D={1:.0009523809523809524,2:0.,3:-.002380952380952381,4:-.0014285714285714286}

def read(name):return json.loads((R/name).read_text(encoding='utf-8'))
def ident(a):return (a['query_id'],a['incoming_doc_id'],a['drop_rank'])
def top(xs, score):return sorted(xs,key=lambda z:(-z[score],-z['action']['drop_rank'],z['action']['incoming_union_rank'],z['action']['incoming_doc_id']))[0]
def score_actions(model, rows, cols):
 grouped=defaultdict(list)
 for a in rows:grouped[a['query_id']].append(a)
 out={}
 for q,xs in grouped.items():
  m=len(xs);X=np.empty((m*(m-1),len(cols)),dtype=np.float32);owners=[];p=0
  for i,a in enumerate(xs):
   for j,b in enumerate(xs):
    if i==j:continue
    X[p]=[a['features'][c]-b['features'][c] for c in cols];owners.append(i);p+=1
  probs=model.predict_proba(X)[:,1]; mean=np.zeros(m,dtype=np.float64);np.add.at(mean,owners,probs);mean=mean/(m-1)
  values=[]
  for i,a in enumerate(xs): values.append({'action':a,'mean':float(mean[i]),'median':float(np.median(probs[np.asarray(owners)==i]))})
  out[q]=values
 return out
def curve(best):
 vals=list(best.values());out=[]
 for tau in [float('inf')]+sorted({x['median'] for x in vals}):
  take=[x for x in vals if x['median']>=tau];gain=sum(x['action']['truth_delta'] for x in take);pd=sum((1 if x['action']['label']=='BENEFICIAL' else -1 if x['action']['label']=='HARMFUL' else 0)/5 for x in take)
  out.append({'threshold':'+Infinity' if tau==float('inf') else tau,'gain_sum':gain,'D_inner':gain/len(vals),'execute_count':len(take),'executed_B':sum(x['action']['label']=='BENEFICIAL' for x in take),'executed_N':sum(x['action']['label']=='NEUTRAL' for x in take),'executed_H':sum(x['action']['label']=='HARMFUL' for x in take),'_tau':tau,'_pd':pd})
 return out
def choose(cv):return max(cv,key=lambda x:(x['gain_sum'],x['_pd'],x['_tau']))
def stats(v):
 if not v:return {'count':0,'min':None,'p10':None,'p25':None,'median':None,'mean':None,'p75':None,'p90':None,'max':None}
 a=np.asarray(v,dtype=float);return {'count':len(a),'min':float(a.min()),'p10':float(np.percentile(a,10)),'p25':float(np.percentile(a,25)),'median':float(np.median(a)),'mean':float(a.mean()),'p75':float(np.percentile(a,75)),'p90':float(np.percentile(a,90)),'max':float(a.max())}
def dump(path,obj):path.write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')
def main():
 c=read('step2p1_model_contract.json');r2=read('step2p1f_r2_reproduction_gate_report.json');a=read('step2p1f_failure_attribution_report.json');p1t=read('step2p1t_threshold_calibration_report.json');p1c=read('step2p1c_realizability_report.json')
 fullpath=R/'step2p1f_full_action_scores.jsonl'; innerpath=R/'step2p1c_inner_oof_top_scores.jsonl'
 outer_paths=[R/'step2p1f_models'/f'outer_fold{i}.joblib' for i in range(1,5)];inner_paths=list((R/'step2p1c_models'/'inner_fold_models').glob('outer_fold*/inner_valid_fold*.joblib'))
 checks={'p1c_pass':p1c['status']=='PASS','inner_12':len(inner_paths)==12 and p1c['inner_models_persisted']==12,'outer_4':len(outer_paths)==4 and all(p.exists() for p in outer_paths),'inner_rows':sum(1 for _ in innerpath.open(encoding='utf-8'))==16800,'outer_rows':sum(1 for _ in fullpath.open(encoding='utf-8'))==806644,'K77':c['fixed_K']==77,'features36':len(c['feature_columns'])==36,'mean_thresholds':p1c['thresholds']['old']=={str(k):v for k,v in MEAN_T.items()},'mean_policy':p1c['old_policy']['gain_sum']==-4.0 and p1c['old_policy']['D77']==-.0007142857142857143 and [p1c['old_policy'][k] for k in ('executed_B','executed_N','executed_H')]==[6,392,12] and [p1c['old_policy'][k] for k in ('S','T','R')]==[6,63,357],'best_BN':a['best_B_minus_N']['median']==-.0944695955058123,'oracle_C':a['oracle_positive_queries']==426 and a['C77']==.058785714285714274}
 if not all(checks.values()):dump(OUT,{'status':'CONTRACT_ERROR','experiment':'STEP 2-P1A — Controlled Median-Aggregation Action-Score Experiment','preconditions':checks,'model_fit_executed':False});return
 ns=runpy.run_path(str(ROOT/'scripts/analysis/step2_k77_oof_policy_realizability.py'),run_name='p1a_parent');folds=ns['target_folds']();records,gold=ns['load_gold'](set(folds));base,_=ns['load_baseline'](set(folds),folds);cand,_=ns['load_candidates'](set(folds),folds);actions,incoming=ns['make_actions'](folds,records,gold,base,cand,c['feature_columns'])
 if incoming!=403322 or sum(map(len,actions.values()))!=806644:raise RuntimeError('CONTRACT_ERROR actions')
 canonical={}; canonical_tops={}
 audit_queries=set()
 for line in fullpath.open(encoding='utf-8'):
  x=json.loads(line)
  if x['is_query_top_scored']:canonical_tops[x['query_id']]=ident(x)
  if len(audit_queries)<100:audit_queries.add(x['query_id'])
  if x['query_id'] in audit_queries:canonical[ident(x)]=x['pairwise_policy_score']
 cols=c['feature_columns'];outer={i:load(outer_paths[i-1])['model'] for i in range(1,5)}; outer_scores={};audit_abs=audit_rel=0.;audit_n=0;audit_top=0
 with MEDIAN.open('w',encoding='utf-8',newline='\n') as h:
  for f in range(1,5):
   res=score_actions(outer[f],actions[f],cols);outer_scores.update(res)
   for q,xs in res.items():
    mt=top(xs,'mean');dt=top(xs,'median');audit_top+=int(q not in audit_queries or ident(mt['action'])==canonical_tops[q])
    for z in xs:
     k=ident(z['action'])
     if k in canonical:
      e=abs(z['mean']-canonical[k]);audit_abs=max(audit_abs,e);audit_rel=max(audit_rel,e/max(abs(canonical[k]),1e-300));audit_n+=1
     x=z['action'];h.write(json.dumps({'query_id':q,'fold':f,'incoming_doc_id':x['incoming_doc_id'],'incoming_union_rank':x['incoming_union_rank'],'drop_rank':x['drop_rank'],'dropped_doc_id':x['dropped_doc_id'],'true_action_class':x['label'],'truth_recall_delta':x['truth_delta'],'mean_pairwise_policy_score':z['mean'],'median_pairwise_policy_score':z['median'],'is_mean_top_action':z is mt,'is_median_top_action':z is dt})+'\n')
 audit_ok=audit_n>=100 and (audit_abs<=1e-12 or audit_rel<=1e-10) and all(ident(top(outer_scores[q],'mean')['action'])==canonical_tops[q] for q in audit_queries)
 audit={'status':'PASS' if audit_ok else 'FAIL','queries_checked':len(audit_queries),'actions_checked':audit_n,'max_abs_error':audit_abs,'max_rel_error':audit_rel,'top_identities_matched':sum(ident(top(outer_scores[q],'mean')['action'])==canonical_tops[q] for q in audit_queries),'top_identities_expected':len(audit_queries),'tolerance':'abs<=1e-12 OR rel<=1e-10'}
 if not audit_ok:dump(OUT,{'status':'PAIRWISE_PROBABILITY_RECONSTRUCTION_MISMATCH','experiment':'STEP 2-P1A — Controlled Median-Aggregation Action-Score Experiment','model_fit_executed':False,'preconditions':checks,'pairwise_probability_reconstruction_audit':audit});return
 inner=[];thresholds={};selection={}
 with INNER.open('w',encoding='utf-8',newline='\n') as h:
  for held in range(1,5):
   best={}
   for valid in [f for f in range(1,5) if f!=held]:
    m=load(R/'step2p1c_models'/'inner_fold_models'/f'outer_fold{held}'/f'inner_valid_fold{valid}.joblib')['model'];res=score_actions(m,actions[valid],cols)
    for q,xs in res.items():
     z=top(xs,'median');best[q]=z;x=z['action'];row={'outer_held_fold':held,'inner_valid_fold':valid,'query_id':q,'median_top_action_identity':{'incoming_doc_id':x['incoming_doc_id'],'incoming_union_rank':x['incoming_union_rank'],'drop_rank':x['drop_rank'],'dropped_doc_id':x['dropped_doc_id']},'median_top_score':z['median'],'true_action_class':x['label'],'truth_recall_delta':x['truth_delta']};inner.append(row);h.write(json.dumps(row)+'\n')
   cv=curve(best);sel=choose(cv);thresholds[held]=sel['_tau'];selection[str(held)]={'outer_held_fold':held,'inner_oof_query_count':len(best),'candidate_count':len(cv),'mean_threshold_reference':MEAN_T[held],'median_selected_threshold':sel['_tau'],'selected_gain_sum':sel['gain_sum'],'selected_D_inner':sel['D_inner'],'selected_execute_count':sel['execute_count'],'tie_count_at_max_objective':sum(x['gain_sum']==sel['gain_sum'] for x in cv),'tie_break_rule':'max(gain_sum, precision_delta, threshold)','curve_entries':[{k:v for k,v in x.items() if not k.startswith('_')} for x in cv]}
 dump(THRESH,{'status':'PASS','experiment':'STEP 2-P1A — Controlled Median-Aggregation Action-Score Experiment','experiment_type':'DETERMINISTIC_POSTHOC_RECOMBINATION','model_fit_executed':False,'threshold_selection_uses_outer_labels':False,'folds':selection})
 tops=[];oracle={}
 for f in range(1,5):
  for q,xs in outer_scores.items():
   if xs[0]['action']['fold']!=f:continue
   mean=top(xs,'mean');med=top(xs,'median');oracle[q]=max([z['action']['truth_delta'] for z in xs]+[0.]);tops.append({'q':q,'fold':f,'mean':mean,'median':med,'take':med['median']>=thresholds[f]})
 def evaluate(which):
  z=[]
  for row in tops:
   x=row[which];take=(x['mean']>=MEAN_T[row['fold']]) if which=='mean' else row['take'];z.append((row,x,take))
  ex=[x for _,x,t in z if t];cc=Counter(x['action']['label'] for x in ex);op=[v for v in z if oracle[v[0]['q']]>0];S=[v for v in op if v[1]['action']['label']=='BENEFICIAL' and v[2]];T=[v for v in op if v[1]['action']['label']=='BENEFICIAL' and not v[2]];RR=[v for v in op if v[1]['action']['label']!='BENEFICIAL'];gain=sum(x['action']['truth_delta'] for x in ex);head=lambda v:sum(oracle[r['q']] for r,_,_ in v)
  return {'gain_sum':gain,'D77':gain/5600,'top_B':sum(x['action']['label']=='BENEFICIAL' for _,x,_ in z),'top_N':sum(x['action']['label']=='NEUTRAL' for _,x,_ in z),'top_H':sum(x['action']['label']=='HARMFUL' for _,x,_ in z),'executed_B':cc['BENEFICIAL'],'executed_N':cc['NEUTRAL'],'executed_H':cc['HARMFUL'],'S':len(S),'T':len(T),'R':len(RR),'S_headroom':head(S),'T_headroom':head(T),'R_headroom':head(RR),'beneficial_gain_sum':sum(x['action']['truth_delta'] for x in ex if x['action']['label']=='BENEFICIAL'),'harmful_loss_sum':sum(x['action']['truth_delta'] for x in ex if x['action']['label']=='HARMFUL')},z
 mean,meanz=evaluate('mean');med,medz=evaluate('median')
 def rates(p):return {c:{'executed':p['executed_'+c[0]],'top':p['top_'+c[0]],'rate':p['executed_'+c[0]]/p['top_'+c[0]]} for c in ('BENEFICIAL','NEUTRAL','HARMFUL')}
 gaps=[]
 for q in [q for q,v in oracle.items() if v>0]:
  xs=outer_scores[q];b=max((x for x in xs if x['action']['label']=='BENEFICIAL'),key=lambda x:(x['median'],-x['action']['drop_rank'],-x['action']['incoming_union_rank'],x['action']['incoming_doc_id']));n=max((x for x in xs if x['action']['label']=='NEUTRAL'),key=lambda x:(x['median'],-x['action']['drop_rank'],-x['action']['incoming_union_rank'],x['action']['incoming_doc_id']));gaps.append(b['median']-n['median'])
 changes=[r for r in tops if ident(r['mean']['action'])!=ident(r['median']['action'])];opchanges=[r for r in changes if oracle[r['q']]>0]
 per={}
 for f in range(1,5):
  mz=[x for x in meanz if x[0]['fold']==f];dz=[x for x in medz if x[0]['fold']==f]
  def fm(z):
   ex=[x for _,x,t in z if t];cc=Counter(x['action']['label'] for x in ex);gain=sum(x['action']['truth_delta'] for x in ex);op=[v for v in z if oracle[v[0]['q']]>0];return {'top_B':sum(x['action']['label']=='BENEFICIAL' for _,x,_ in z),'top_N':sum(x['action']['label']=='NEUTRAL' for _,x,_ in z),'top_H':sum(x['action']['label']=='HARMFUL' for _,x,_ in z),'executed_B':cc['BENEFICIAL'],'executed_N':cc['NEUTRAL'],'executed_H':cc['HARMFUL'],'S':sum(x['action']['label']=='BENEFICIAL' and t for _,x,t in op),'T':sum(x['action']['label']=='BENEFICIAL' and not t for _,x,t in op),'R':sum(x['action']['label']!='BENEFICIAL' for _,x,t in op),'gain_sum':gain,'D_fold':gain/1400}
  mo,md=fm(mz),fm(dz);fg=[g for q,g in zip([q for q,v in oracle.items() if v>0],gaps) if next(r['fold'] for r in tops if r['q']==q)==f];per[str(f)]={'mean_threshold':MEAN_T[f],'median_threshold':thresholds[f],'mean':mo,'median':md,'gain_delta':md['gain_sum']-mo['gain_sum'],'D_delta':md['D_fold']-mo['D_fold'],'median_best_B_minus_N':stats(fg)['median'],'median_negative_B_minus_N_count':sum(v<0 for v in fg),'top_action_change_count':sum(r['fold']==f for r in changes)}
 report={'status':'PASS','experiment':'STEP 2-P1A — Controlled Median-Aggregation Action-Score Experiment','experiment_type':'DETERMINISTIC_POSTHOC_RECOMBINATION','scientific_status':'DIAGNOSTIC_ONLY_NO_BRANCH_GATE','model_fit_executed':False,'base_model_training_executed':False,'inner_model_training_executed':False,'outer_model_training_executed':False,'policy_oof_strict':True,'end_to_end_selection_oof':False,'k77_selected_exploratorily_on_folds_1_4':True,'interpretation_scope':'MEDIAN_AGGREGATION_REALIZABILITY_CONDITIONAL_ON_EXPLORATORY_FIXED_K77','K':77,'feature_count':36,'variable_under_test':'ACTION_SCORE_AGGREGATION_MEAN_TO_MEDIAN','aggregation_sweep':False,'model_objective_changed':False,'pair_weighting_changed':False,'features_changed':False,'K_changed':False,'preconditions':checks,'pairwise_probability_reconstruction_audit':audit,'outer_median_action_rows':806644,'inner_median_oof_rows':len(inner),'thresholds':{'mean':{str(k):v for k,v in MEAN_T.items()},'median':{str(k):v for k,v in thresholds.items()}},'mean_policy':{**mean,'median_best_B_minus_N':-.0944695955058123},'median_policy':{**med,'median_best_B_minus_N':stats(gaps)['median']},'comparison':{'delta_D77_median_minus_mean':med['D77']-mean['D77'],'median_D77_gt_mean':med['D77']>mean['D77'],'median_D77_gt_0':med['D77']>0,'all_fold_median_D_ge_mean':all(per[str(f)]['median']['D_fold']>=per[str(f)]['mean']['D_fold'] for f in range(1,5)),'median_R_lt_mean_R':med['R']<mean['R'],'median_harmful_loss_less_severe':med['harmful_loss_sum']>mean['harmful_loss_sum'],'top_action_change_count':len(changes),'top_action_change_fraction':len(changes)/5600,'top_action_changes_new_classes':dict(Counter(x['median']['action']['label'] for x in changes)),'oracle_positive_top_action_change_count':len(opchanges),'oracle_positive_top_action_changes_new_classes':dict(Counter(x['median']['action']['label'] for x in opchanges))},'crossing_rates':{'mean':rates(mean),'median':rates(med)},'ranking_preservation':{'median_best_B_minus_N':stats(gaps),'gt0':sum(x>0 for x in gaps),'eq0_within_1e12':sum(abs(x)<=1e-12 for x in gaps),'lt0':sum(x<0 for x in gaps),'mean_median_best_B_minus_N':-.0944695955058123,'mean_negative_B_minus_N':'357/426','R_mean':357,'R_median':med['R']},'harmful_execution':{'mean_count':12,'mean_loss_sum':-7.666666666666667,'median_count':med['executed_H'],'median_loss_sum':med['harmful_loss_sum'],'median_beneficial_gain_sum':med['beneficial_gain_sum'],'median_neutral_execution_count':med['executed_N'],'median_net_gain_sum':med['gain_sum']},'per_fold':per,'preserved_status':{'threshold_scalar_branch':'CLOSED','absolute_score_selectivity':'DIRECTLY_MEASURED_WEAK','mean_aggregation_hypothesis':'STRONGLY_SUPPORTED_HYPOTHESIS','remaining_ranking_loss':'PRIMARY_REMAINING_BOTTLENECK','pair_family':'UNRESOLVED_NOT_ESTABLISHED','pair_reweighting':'NOT_AUTHORIZED','B_vs_N_only':'NOT_AUTHORIZED','HGB':'UNRESOLVED','feature_expansion':'NOT_JUSTIFIED','K77':'NOT_REJECTED','smaller_K':'DEFERRED','workflow_B':'NOT_ACTIVE'},'notes':['Only the final reducer changed from MEAN to MEDIAN; no model.fit or retraining was performed.','Median thresholds were selected exclusively from proper inner-OOF median scores; outer labels were not used for selection.']}
 dump(OUT,report);print(json.dumps({'status':'PASS','median_thresholds':thresholds,'mean_D77':mean['D77'],'median_D77':med['D77'],'delta':report['comparison']['delta_D77_median_minus_mean'],'top_changes':len(changes)},indent=2))
if __name__=='__main__':main()
