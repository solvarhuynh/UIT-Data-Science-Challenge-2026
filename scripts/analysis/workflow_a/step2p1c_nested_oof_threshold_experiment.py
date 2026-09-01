"""Authorized P1C: rebuild only inner OOF models/traces for frozen P1 scoring."""
from __future__ import annotations
import hashlib, json, runpy, sklearn
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from joblib import dump
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT=Path(__file__).resolve().parents[2]; R=ROOT/'reports/task1'; MD=R/'step2p1c_models'/'inner_fold_models'
TRACE=R/'step2p1c_inner_oof_top_scores.jsonl'; CURVES=R/'step2p1c_inner_oof_threshold_curves.json'; OUT=R/'step2p1c_realizability_report.json'
OLD={1:.8871766063648394,2:.9158352964999534,3:.8885338990548798,4:.9072451320464932}
OLD_D={1:.0009523809523809524,2:0.0,3:-.002380952380952381,4:-.0014285714285714286}

def read(name): return json.loads((R/name).read_text(encoding='utf-8'))
def pairs(rows,cols):
 g=defaultdict(list)
 for a in rows:g[a['query_id']].append(a)
 n=sum(2*(Counter(a['label'] for a in xs)['BENEFICIAL']*Counter(a['label'] for a in xs)['NEUTRAL']+Counter(a['label'] for a in xs)['BENEFICIAL']*Counter(a['label'] for a in xs)['HARMFUL']+Counter(a['label'] for a in xs)['NEUTRAL']*Counter(a['label'] for a in xs)['HARMFUL']) for xs in g.values())
 X=np.empty((n,len(cols)),dtype=np.float32);y=np.empty(n,dtype=np.uint8);p=0;rank={'BENEFICIAL':2,'NEUTRAL':1,'HARMFUL':0}
 for xs in g.values():
  xs=sorted(xs,key=lambda a:(a['incoming_union_rank'],a['incoming_doc_id'],a['drop_rank']))
  for i,a in enumerate(xs):
   for b in xs[i+1:]:
    if a['label']==b['label']:continue
    hi,lo=(a,b) if rank[a['label']]>rank[b['label']] else (b,a)
    X[p]=np.asarray([hi['features'][c]-lo['features'][c] for c in cols],dtype=np.float32);y[p]=1;X[p+1]=-X[p];y[p+1]=0;p+=2
 if p!=n or int(y.sum())*2!=n:raise RuntimeError('CONTRACT_ERROR pair balance')
 return X,y
def fit(rows,cols,hp):
 X,y=pairs(rows,cols);m=HistGradientBoostingClassifier(**hp);m.fit(X,y,sample_weight=np.ones(len(y)));return m,len(y)
def scores(model,rows,cols):
 g=defaultdict(list)
 for a in rows:g[a['query_id']].append(a)
 out={}
 for q,xs in g.items():
  m=len(xs);X=np.empty((m*(m-1),len(cols)),dtype=np.float32);owner=[];p=0
  for i,a in enumerate(xs):
   for j,b in enumerate(xs):
    if i==j:continue
    X[p]=[a['features'][c]-b['features'][c] for c in cols];owner.append(i);p+=1
  prob=model.predict_proba(X)[:,1]; total=np.zeros(m,dtype=np.float64);np.add.at(total,owner,prob);v=total/(m-1)
  out[q]=[(a,float(v[i])) for i,a in enumerate(xs)]
 return out
def top(xs):return sorted(xs,key=lambda z:(-z[1],-z[0]['drop_rank'],z[0]['incoming_union_rank'],z[0]['incoming_doc_id']))[0]
def curve(bestmap):
 vals=list(bestmap.values()); out=[]
 for tau in [float('inf')]+sorted({s for _,s in vals}):
  take=[a for a,s in vals if s>=tau]; gain=sum(a['truth_delta'] for a in take); pd=sum((1 if a['label']=='BENEFICIAL' else -1 if a['label']=='HARMFUL' else 0)/5 for a in take)
  out.append({'threshold':'+Infinity' if tau==float('inf') else tau,'gain_sum':gain,'D_inner':gain/len(vals),'execute_count':len(take),'executed_B':sum(a['label']=='BENEFICIAL' for a in take),'executed_N':sum(a['label']=='NEUTRAL' for a in take),'executed_H':sum(a['label']=='HARMFUL' for a in take),'_tau':tau,'_pd':pd})
 return out
def choose(entries): return max(entries,key=lambda x:(x['gain_sum'],x['_pd'],x['_tau']))
def dump_json(path,obj): path.write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')
def main():
 contract=read('step2p1_model_contract.json');r2=read('step2p1f_r2_reproduction_gate_report.json');p1t=read('step2p1t_threshold_calibration_report.json');attr=read('step2p1f_failure_attribution_report.json');metric=read('step2p1f_metric_contract_resolution_report.json')
 full=[json.loads(x) for x in (R/'step2p1f_full_action_scores.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
 checks={'r2_pass':r2['status']=='PASS' and r2['reproduction_gate_pass'],'p1t_pass':p1t['status']=='PASS','K77':contract['fixed_K']==77,'features36':len(contract['feature_columns'])==36,'full_rows':len(full)==806644,'oracle_positive':attr['oracle_positive_queries']==426,'trs': [attr['trs'][x]['query_count'] for x in ('S_SUCCESS','T_THRESHOLD_LOSS','R_RANKING_DISCRIMINATION_LOSS')]==[6,63,357],'old_gain_D':r2['canonical_gain_sum']==-4.0 and r2['canonical_D77']==-.0007142857142857143,'p1t_top':p1t['canonical_counts']['B_top']==69 and p1t['canonical_counts']['N_top']==5405 and p1t['canonical_counts']['H_top']==126,'p1t_T':p1t['canonical_counts']['T']==63 and p1t['beneficial_top_69']['T_distance_bins']['> 0.050']['count']==49,'thresholds':{int(k):v for k,v in p1t['fixed_thresholds'].items()}==OLD}
 if not all(checks.values()): dump_json(OUT,{'status':'CONTRACT_ERROR','experiment':'STEP 2-P1C — Nested-OOF Threshold/Calibration Experiment for Frozen Pairwise Score','preconditions':checks,'inner_models_fit':0,'inner_models_persisted':0});return
 ns=runpy.run_path(str(ROOT/'scripts/analysis/step2_k77_oof_policy_realizability.py'),run_name='p1c_parent');folds=ns['target_folds']();records,gold=ns['load_gold'](set(folds));base,_=ns['load_baseline'](set(folds),folds);cand,_=ns['load_candidates'](set(folds),folds);actions,incoming=ns['make_actions'](folds,records,gold,base,cand,contract['feature_columns'])
 if incoming!=403322 or sum(map(len,actions.values()))!=806644:raise RuntimeError('CONTRACT_ERROR action contract')
 cols=contract['feature_columns'];hp=contract['hyperparameters'];chash=hashlib.sha256((R/'step2p1_model_contract.json').read_bytes()).hexdigest(); trace=[];curves={};fit_count=0;persisted=[];repro={};
 for held in range(1,5):
  train=[f for f in range(1,5) if f!=held];bestmap={};mdir=MD/f'outer_fold{held}';mdir.mkdir(parents=True,exist_ok=True)
  for valid in train:
   innertrain=[f for f in train if f!=valid];model,nrows=fit([a for f in innertrain for a in actions[f]],cols,hp);fit_count+=1
   mp=mdir/f'inner_valid_fold{valid}.joblib';dump({'model':model,'artifact_type':'DIAGNOSTIC INNER-OOF ARTIFACT ONLY','outer_held_fold':held,'inner_valid_fold':valid,'inner_train_folds':innertrain,'ordered_feature_columns':cols,'model_parameters':hp,'random_seed':contract['random_seed'],'sklearn_version':sklearn.__version__,'contract_path':'reports/task1/step2p1_model_contract.json','contract_sha256':chash},mp);persisted.append(str(mp.relative_to(ROOT)))
   for q,xs in scores(model,actions[valid],cols).items():
    a,s=top(xs); bestmap[q]=(a,s); trace.append({'outer_held_fold':held,'inner_valid_fold':valid,'query_id':q,'top_action_identity':{'incoming_doc_id':a['incoming_doc_id'],'incoming_union_rank':a['incoming_union_rank'],'drop_rank':a['drop_rank'],'dropped_doc_id':a['dropped_doc_id']},'top_action_score':s,'true_action_class':a['label'],'truth_recall_delta':a['truth_delta'],'baseline_recall':ns['recall'](gold[q],base[q]),'resulting_recall_if_execute':ns['recall'](gold[q],list(base[q][:a['drop_rank']-1])+[a['incoming_doc_id']]+list(base[q][a['drop_rank']:])),'inner_train_folds':innertrain,'mirrored_training_rows_total':nrows})
  cv=curve(bestmap);selected=choose(cv);tau=selected['_tau'];same=tau==OLD[held];repro[str(held)]={'expected':OLD[held],'measured':tau,'pass':same,'candidate_count':len(cv),'selected_gain_sum':selected['gain_sum'],'selected_D_inner':selected['D_inner'],'tie_count_at_max_objective':sum(x['gain_sum']==selected['gain_sum'] for x in cv),'implementation_function':'scripts/analysis/step2p1_controlled_pairwise_policy.py::choose_threshold equivalent candidate/tie-break'}
  curves[str(held)]={'outer_held_fold':held,'inner_train_context':train,'inner_oof_query_count':len(bestmap),'candidate_count':len(cv),'historical_threshold_expected':OLD[held],'historical_threshold_reproduced':tau,'historical_reproduction_pass':same,'canonical_selected_threshold':tau,'same_as_historical_threshold':same,'selected_gain_sum':selected['gain_sum'],'selected_D_inner':selected['D_inner'],'selected_execute_count':selected['execute_count'],'selected_B_count':selected['executed_B'],'selected_N_count':selected['executed_N'],'selected_H_count':selected['executed_H'],'tie_count_at_max_objective':repro[str(held)]['tie_count_at_max_objective'],'tie_break_rule':'max(gain_sum, precision_delta, threshold)','curve_entries':[{k:v for k,v in x.items() if not k.startswith('_')} for x in cv]}
 with TRACE.open('w',encoding='utf-8',newline='\n') as h:
  for x in trace:h.write(json.dumps(x)+'\n')
 dump_json(CURVES,{'status':'PASS' if all(x['pass'] for x in repro.values()) else 'INNER_THRESHOLD_REPRODUCTION_MISMATCH','experiment':'STEP 2-P1C — Nested-OOF Threshold/Calibration Experiment for Frozen Pairwise Score','training_terminology':'AUTHORIZED_NESTED_OOF_THRESHOLD_EXPERIMENT_TRAINING','historical_threshold_reproduction':repro,'folds':curves})
 if fit_count!=12 or len(persisted)!=12 or len(trace)!=16800 or any(sum(x['outer_held_fold']==h for x in trace)!=4200 for h in range(1,5)) or not all(x['pass'] for x in repro.values()):
  dump_json(OUT,{'status':'INNER_THRESHOLD_REPRODUCTION_MISMATCH','experiment':'STEP 2-P1C — Nested-OOF Threshold/Calibration Experiment for Frozen Pairwise Score','experiment_type':'AUTHORIZED_NESTED_OOF_THRESHOLD_EXPERIMENT_TRAINING','training_terminology':'AUTHORIZED_NESTED_OOF_THRESHOLD_EXPERIMENT_TRAINING','preconditions':checks,'inner_models_fit':fit_count,'inner_models_persisted':len(persisted),'models_persisted':persisted,'historical_threshold_reproduction':{'pass':False,'folds':repro},'notes':['Stop: do not evaluate a new threshold policy.']});return
 new={f:curves[str(f)]['canonical_selected_threshold'] for f in range(1,5)}
 top_rows=[x for x in full if x['is_query_top_scored']]
 def policy(tmap):
  z=[]
  for x in top_rows:
   take=x['pairwise_policy_score']>=tmap[x['fold']];z.append(dict(x,take=take))
  return z
 oldp=policy(OLD);newp=policy(new)
 def metricp(z):
  ex=[x for x in z if x['take']];cc=Counter(x['true_action_class'] for x in ex);gain=sum(x['truth_recall_delta'] for x in ex);op=[x for x in z if x['true_action_class']=='BENEFICIAL'];S=[x for x in op if x['take']];T=[x for x in op if not x['take']]
  oracle={};
  for x in full:oracle[x['query_id']]=max(oracle.get(x['query_id'],0.0),x['truth_recall_delta'])
  head=lambda xs:sum(oracle[x['query_id']] for x in xs)
  R=[x for x in z if oracle[x['query_id']]>0 and x['true_action_class']!='BENEFICIAL']
  return {'gain_sum':gain,'D77':gain/5600,'executed_B':cc['BENEFICIAL'],'executed_N':cc['NEUTRAL'],'executed_H':cc['HARMFUL'],'S':len(S),'T':len(T),'R':len(R),'S_headroom':head(S),'T_headroom':head(T),'R_headroom':head(R),'harmful_loss_sum':sum(x['truth_recall_delta'] for x in ex if x['true_action_class']=='HARMFUL'),'beneficial_gain_sum':sum(x['truth_recall_delta'] for x in ex if x['true_action_class']=='BENEFICIAL')}
 om,nm=metricp(oldp),metricp(newp)
 def rates(z):
  return {c:{'executed':sum(x['take'] and x['true_action_class']==c for x in z),'total':sum(x['true_action_class']==c for x in z),'rate':sum(x['take'] and x['true_action_class']==c for x in z)/sum(x['true_action_class']==c for x in z)} for c in ('BENEFICIAL','NEUTRAL','HARMFUL')}
 per={}
 for f in range(1,5):
  oo=[x for x in oldp if x['fold']==f];nn=[x for x in newp if x['fold']==f];a,b=metricp(oo),metricp(nn);ro,rn=rates(oo),rates(nn);per[str(f)]={'historical_threshold':OLD[f],'new_inner_selected_threshold':new[f],'threshold_delta':new[f]-OLD[f],'old':a,'new':b,'gain_sum_delta':b['gain_sum']-a['gain_sum'],'D_delta':b['D77']-a['D77'],'B_top_count':ro['BENEFICIAL']['total'],'old_B_crossing_rate':ro['BENEFICIAL']['rate'],'new_B_crossing_rate':rn['BENEFICIAL']['rate'],'old_H_crossing_rate':ro['HARMFUL']['rate'],'new_H_crossing_rate':rn['HARMFUL']['rate']}
 report={'status':'PASS','experiment':'STEP 2-P1C — Nested-OOF Threshold/Calibration Experiment for Frozen Pairwise Score','experiment_type':'AUTHORIZED_NESTED_OOF_THRESHOLD_EXPERIMENT_TRAINING','training_terminology':'AUTHORIZED_NESTED_OOF_THRESHOLD_EXPERIMENT_TRAINING','scientific_status':'DIAGNOSTIC_ONLY_NO_BRANCH_GATE','policy_oof_strict':True,'end_to_end_selection_oof':False,'k77_selected_exploratorily_on_folds_1_4':True,'interpretation_scope':'NESTED_OOF_THRESHOLD_CALIBRATION_CONDITIONAL_ON_EXPLORATORY_FIXED_K77','K':77,'feature_count':36,'model_objective_changed':False,'pair_weighting_changed':False,'action_score_changed':False,'outer_models_retrained':False,'inner_models_fit':fit_count,'inner_models_persisted':len(persisted),'inner_model_paths':persisted,'inner_oof_top_score_rows':len(trace),'preconditions':checks,'historical_threshold_reproduction':{'pass':True,'folds':repro},'thresholds':{'old':{str(k):v for k,v in OLD.items()},'new':{str(k):v for k,v in new.items()}},'old_policy':om,'new_policy':nm,'comparison':{'delta_D77_new_minus_old':nm['D77']-om['D77'],'new_D77_gt_old':nm['D77']>om['D77'],'new_D77_gt_0':nm['D77']>0,'all_fold_D_new_ge_old':all(per[str(f)]['new']['D77']>=per[str(f)]['old']['D77'] for f in range(1,5))},'crossing_rates':{'old':rates(oldp),'new':rates(newp)},'harmful_execution':{'old_count':om['executed_H'],'old_loss_sum':om['harmful_loss_sum'],'new_count':nm['executed_H'],'new_loss_sum':nm['harmful_loss_sum']},'per_fold':per,'preserved_status':{'remaining_ranking_loss':'PRIMARY_REMAINING_BOTTLENECK','absolute_score_selectivity':'DIRECTLY_MEASURED_WEAK','simple_borderline_threshold_hypothesis':'WEAKENED','mean_pairwise_win_probability':'SUPPORTED_HYPOTHESIS','pair_family':'UNRESOLVED_NOT_ESTABLISHED','pair_reweighting':'NOT_AUTHORIZED','B_vs_N_only':'NOT_AUTHORIZED','HGB':'UNRESOLVED','feature_expansion':'NOT_JUSTIFIED','K77':'NOT_REJECTED','smaller_K':'DEFERRED','workflow_B':'NOT_ACTIVE'},'notes':['Only 12 inner models were fit; persisted outer models were not retrained.','Existing 806644 frozen outer action scores were reused for outer policy evaluation.','Outer labels were not used for threshold selection.']}
 dump_json(OUT,report);print(json.dumps({'status':'PASS','thresholds_new':new,'old':om,'new':nm,'delta_D77':report['comparison']['delta_D77_new_minus_old']},indent=2))
if __name__=='__main__':main()
