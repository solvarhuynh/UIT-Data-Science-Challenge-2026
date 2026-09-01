"""Authorized deterministic exact reconstruction of frozen STEP 2-P1."""
from __future__ import annotations
import json, math, runpy
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from joblib import dump
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT=Path(__file__).resolve().parents[2]; R=ROOT/'reports/task1'
GATE=R/'step2p1f_r2_reproduction_gate_report.json'; FULL=R/'step2p1f_full_action_scores.jsonl'
ATTR=R/'step2p1f_failure_attribution_report.json'; CORR=R/'step2p1f_pairfamily_correlation_report.json'
MODELS=R/'step2p1f_models'
EXPECTED_T={1:.8871766063648394,2:.9158352964999534,3:.8885338990548798,4:.9072451320464932}

def pairs(rows, cols):
 g=defaultdict(list)
 for a in rows:g[a['query_id']].append(a)
 n=sum(2*(Counter(a['label'] for a in xs)['BENEFICIAL']*Counter(a['label'] for a in xs)['NEUTRAL']+Counter(a['label'] for a in xs)['BENEFICIAL']*Counter(a['label'] for a in xs)['HARMFUL']+Counter(a['label'] for a in xs)['NEUTRAL']*Counter(a['label'] for a in xs)['HARMFUL']) for xs in g.values())
 X=np.empty((n,len(cols)),dtype=np.float32); y=np.empty(n,dtype=np.uint8); p=0; order={'BENEFICIAL':2,'NEUTRAL':1,'HARMFUL':0}
 for xs in g.values():
  xs=sorted(xs,key=lambda a:(a['incoming_union_rank'],a['incoming_doc_id'],a['drop_rank']))
  for i,a in enumerate(xs):
   for b in xs[i+1:]:
    if a['label']==b['label']:continue
    hi,lo=(a,b) if order[a['label']]>order[b['label']] else (b,a)
    X[p]=np.asarray([hi['features'][c]-lo['features'][c] for c in cols],dtype=np.float32);y[p]=1;X[p+1]=-X[p];y[p+1]=0;p+=2
 if p!=n or int(y.sum())*2!=n:raise RuntimeError('CONTRACT_ERROR pair balance')
 return X,y
def fit(rows,cols,hp):
 X,y=pairs(rows,cols);m=HistGradientBoostingClassifier(**hp);m.fit(X,y,sample_weight=np.ones(len(y)));return m
def scores(model,rows,cols):
 g=defaultdict(list)
 for a in rows:g[a['query_id']].append(a)
 out={}
 for q,xs in g.items():
  m=len(xs);X=np.empty((m*(m-1),len(cols)),dtype=np.float32);own=[];p=0
  for i,a in enumerate(xs):
   for j,b in enumerate(xs):
    if i==j:continue
    X[p]=[a['features'][c]-b['features'][c] for c in cols];own.append(i);p+=1
  prob=model.predict_proba(X)[:,1];su=np.zeros(m,dtype=np.float64);np.add.at(su,own,prob);sc=su/(m-1)
  out[q]=[(a,float(sc[i])) for i,a in enumerate(xs)]
 return out
def top(xs):return sorted(xs,key=lambda z:(-z[1],-z[0]['drop_rank'],z[0]['incoming_union_rank'],z[0]['incoming_doc_id']))[0]
def threshold(bestmap):
 z=[]
 for t in [float('inf')]+sorted({s for _,s in bestmap.values()}):
  take=[a for a,s in bestmap.values() if s>=t];gain=sum(a['truth_delta'] for a in take)/len(bestmap);pd=sum((1 if a['label']=='BENEFICIAL' else -1 if a['label']=='HARMFUL' else 0)/5 for a in take)/len(bestmap);z.append((gain,pd,t))
 return max(z,key=lambda x:(x[0],x[1],x[2]))[2]
def stats(v):
 a=np.asarray(v,dtype=float)
 return {'count':len(a),'min':float(a.min()),'p10':float(np.percentile(a,10)),'p25':float(np.percentile(a,25)),'median':float(np.median(a)),'mean':float(a.mean()),'p75':float(np.percentile(a,75)),'p90':float(np.percentile(a,90)),'max':float(a.max())}
def main():
 contract=json.loads((R/'step2p1_model_contract.json').read_text());canon=json.loads((R/'step2p1_realizability_report.json').read_text());original={x['query_id']:x for x in map(json.loads,(R/'step2p1_oof_policy_decisions.jsonl').read_text().splitlines())}
 if not(canon['status']=='PASS' and len(original)==5600 and contract['fixed_K']==77 and len(contract['feature_columns'])==36 and canon['pooled']['D77_pairwise']==-0.0007142857142857784):raise RuntimeError('BLOCKED canonical P1 mismatch')
 ns=runpy.run_path(str(ROOT/'scripts/analysis/step2_k77_oof_policy_realizability.py'),run_name='p1fr_parent'); folds=ns['target_folds'](); records,gold=ns['load_gold'](set(folds));base,_=ns['load_baseline'](set(folds),folds);cand,_=ns['load_candidates'](set(folds),folds);actions,incoming=ns['make_actions'](folds,records,gold,base,cand,contract['feature_columns'])
 if incoming!=403322 or sum(map(len,actions.values()))!=806644:raise RuntimeError('CONTRACT_ERROR action contract')
 cols=contract['feature_columns'];hp=contract['hyperparameters']; allsc={}; thresholds={}; models={}
 for held in range(1,5):
  tr=[f for f in range(1,5) if f!=held]; inner={}
  for valid in tr:
   m=fit([a for f in tr if f!=valid for a in actions[f]],cols,hp)
   inner.update({q:top(v) for q,v in scores(m,actions[valid],cols).items()})
  thresholds[held]=threshold(inner); models[held]=fit([a for f in tr for a in actions[f]],cols,hp);allsc.update(scores(models[held],actions[held],cols))
 # Gate data
 rec=[];maxabs=maxrel=0.0;execmatch=noopmatch=0
 for q,xs in allsc.items():
  a,s=top(xs);f=next(x['fold'] for x in original.values() if x['query_id']==q);tau=thresholds[f];take=s>=tau;o=original[q];os=o['query_top_action']['pairwise_action_score'];maxabs=max(maxabs,abs(s-os));maxrel=max(maxrel,abs(s-os)/max(abs(os),1e-300));
  if take and o['decision']=='EXECUTE' and all(a[k]==o['executed_action'][k] for k in ('incoming_doc_id','drop_rank','dropped_doc_id')):execmatch+=1
  if not take and o['decision']=='NO_OP':noopmatch+=1
  final=list(base[q]);
  if take:final[a['drop_rank']-1]=a['incoming_doc_id']
  rec.append({'query_id':q,'fold':f,'top':a,'score':s,'take':take,'br':ns['recall'](gold[q],base[q]),'pr':ns['recall'](gold[q],final)})
 def gain_sum(xs):return sum(float(original[x['query_id']]['recall_delta']) for x in xs)
 def d77(xs):return gain_sum(xs)/len(xs)
 per={str(f):d77([x for x in rec if x['fold']==f]) for f in range(1,5)};per_gain={str(f):gain_sum([x for x in rec if x['fold']==f]) for f in range(1,5)};d=d77(rec);gain=gain_sum(rec); cc=Counter(x['top']['label'] for x in rec if x['take'])
 score_match=sum(x['score']==original[x['query_id']]['query_top_action']['pairwise_action_score'] for x in rec)
 passgate=(all(thresholds[f]==EXPECTED_T[f] for f in thresholds) and execmatch==410 and noopmatch==5190 and cc==Counter({'NEUTRAL':392,'HARMFUL':12,'BENEFICIAL':6}) and gain==-4.0 and per_gain=={'1':1.3333333333333333,'2':0.0,'3':-3.3333333333333335,'4':-2.0} and score_match==5600 and maxabs==0.0 and maxrel==0.0)
 gate={'status':'PASS' if passgate else 'RECONSTRUCTION_MISMATCH','experiment':'STEP 2-P1F-R2 — Canonical-Metric Resolution and Pairwise Failure Attribution','training_terminology':'DETERMINISTIC_FROZEN_MODEL_RECONSTRUCTION','phase':'DETERMINISTIC_FROZEN_MODEL_RECONSTRUCTION','scientific_new_training':False,'mechanical_model_fit_executed':True,'policy_oof_strict':True,'end_to_end_selection_oof':False,'k77_selected_exploratorily_on_folds_1_4':True,'interpretation_scope':'PAIRWISE_POLICY_FAILURE_ATTRIBUTION_CONDITIONAL_ON_EXPLORATORY_FIXED_K77','primitive_gate':{'thresholds_exact':all(thresholds[f]==EXPECTED_T[f] for f in thresholds),'top_scores_exact':score_match==5600,'executed_actions_exact':execmatch==410,'noop_exact':noopmatch==5190,'class_composition_exact':cc==Counter({'NEUTRAL':392,'HARMFUL':12,'BENEFICIAL':6})},'decision_reproduction':{'total':5600,'executed_match':execmatch,'executed_expected':410,'noop_match':noopmatch,'noop_expected':5190},'class_counts':dict(cc),'canonical_gain_sum':gain,'canonical_D77':d,'per_fold_gain_sum':per_gain,'per_fold_D77':per,'historical_float_crosscheck':{'historical_D77':-.0007142857142857784,'canonical_D77':d,'absolute_difference':abs(d-(-.0007142857142857784))},'top_score_reproduction':{'score_rows_available':5600,'score_rows_matched':score_match,'max_abs_error':maxabs,'max_rel_error':maxrel},'models_persisted':[],'reproduction_gate_pass':passgate,'notes':['Authorized deterministic exact refit only; no new scientific training.']}
 if not passgate:GATE.write_text(json.dumps(gate,indent=2)+'\n');return
 MODELS.mkdir(exist_ok=True)
 for f,m in models.items():
  p=MODELS/f'outer_fold{f}.joblib';dump({'model':m,'feature_columns':cols,'outer_fold_held_out':f,'threshold':thresholds[f],'artifact_type':'DIAGNOSTIC_REPRODUCTION_ONLY'},p);gate['models_persisted'].append(str(p.relative_to(ROOT)))
 GATE.write_text(json.dumps(gate,indent=2)+'\n')
 # Full actions and audit maps
 rows=[];qmap={};family=Counter()
 for f in range(1,5):
  by=defaultdict(list)
  for a in actions[f]:by[a['query_id']].append(a)
  for q,xs in by.items():
   sc=dict((id(a),s) for a,s in allsc[q]);tx=top(allsc[q]);c=Counter(a['label'] for a in xs);family['B_N']+=c['BENEFICIAL']*c['NEUTRAL'];family['B_H']+=c['BENEFICIAL']*c['HARMFUL'];family['N_H']+=c['NEUTRAL']*c['HARMFUL'];qmap[q]={'fold':f,'xs':xs,'BN':c['BENEFICIAL']*c['NEUTRAL'],'BH':c['BENEFICIAL']*c['HARMFUL'],'NH':c['NEUTRAL']*c['HARMFUL'],'top':tx,'tau':thresholds[f]}
   for a in xs:rows.append({'query_id':q,'fold':f,'incoming_doc_id':a['incoming_doc_id'],'incoming_union_rank':a['incoming_union_rank'],'drop_rank':a['drop_rank'],'dropped_doc_id':a['dropped_doc_id'],'true_action_class':a['label'],'truth_recall_delta':a['truth_delta'],'pairwise_policy_score':sc[id(a)],'outer_threshold':thresholds[f],'is_query_top_scored':a is tx[0],'would_execute':a is tx[0] and tx[1]>=thresholds[f]})
 if family!=Counter({'B_N':125576,'B_H':1625,'N_H':1539444}):raise RuntimeError('CONTRACT_ERROR pair counts')
 with FULL.open('w',encoding='utf-8',newline='\n') as h:
  for x in rows:h.write(json.dumps(x)+'\n')
 oracle={q:max([a['truth_delta'] for a in z['xs']]+[0.0]) for q,z in qmap.items()};op=[q for q,v in oracle.items() if v>0]
 if len(rows)!=806644 or len(op)!=426 or abs(sum(oracle.values())-329.2)>1e-9:raise RuntimeError('CONTRACT_ERROR oracle')
 trs={k:{'query_count':0,'C77_headroom_sum':0.0,'per_fold':{str(f):{'query_count':0,'headroom_sum':0.0} for f in range(1,5)} } for k in ('S_SUCCESS','T_THRESHOLD_LOSS','R_RANKING_DISCRIMINATION_LOSS')}; gaps=[]; harmop=[]
 for q in op:
  z=qmap[q];a,s=z['top'];k='S_SUCCESS' if a['label']=='BENEFICIAL' and s>=z['tau'] else 'T_THRESHOLD_LOSS' if a['label']=='BENEFICIAL' else 'R_RANKING_DISCRIMINATION_LOSS';t=trs[k];t['query_count']+=1;t['C77_headroom_sum']+=oracle[q];t['per_fold'][str(z['fold'])]['query_count']+=1;t['per_fold'][str(z['fold'])]['headroom_sum']+=oracle[q]
  bs=max((x for x in allsc[q] if x[0]['label']=='BENEFICIAL'),key=lambda x:(x[1],-x[0]['drop_rank'],-x[0]['incoming_union_rank'],x[0]['incoming_doc_id']));bn=max((x for x in allsc[q] if x[0]['label']=='NEUTRAL'),key=lambda x:(x[1],-x[0]['drop_rank'],-x[0]['incoming_union_rank'],x[0]['incoming_doc_id']));bh=[x for x in allsc[q] if x[0]['label']=='HARMFUL'];gaps.append({'q':q,'fold':z['fold'],'bn':bs[1]-bn[1],'bh':bs[1]-max(bh,key=lambda x:x[1])[1] if bh else None,'logr':math.log((z['BN']+1)/(z['NH']+1)),'rank':bs[0]['incoming_union_rank'],'cat':k,'head':oracle[q]})
  if a['label']=='HARMFUL':harmop.append({'query_id':q,'fold':z['fold'],'C77_query':oracle[q],'top_score':s,'threshold':z['tau'],'would_execute':s>=z['tau'],'truth_recall_delta':a['truth_delta'],'incoming_union_rank':a['incoming_union_rank'],'drop_rank':a['drop_rank']})
 for x in trs.values():x['share_of_329_2']=x['C77_headroom_sum']/329.2
 bn=[x['bn'] for x in gaps];bhn=[x['bh'] for x in gaps if x['bh'] is not None]
 def breakdown(group):
  gg=[x for x in gaps if group(x)];return {'query_count':len(gg),'median_B_minus_N':float(np.median([x['bn'] for x in gg])) if gg else None,'mean_B_minus_N':float(np.mean([x['bn'] for x in gg])) if gg else None,'R_count':sum(x['cat']=='R_RANKING_DISCRIMINATION_LOSS' for x in gg),'R_fraction':sum(x['cat']=='R_RANKING_DISCRIMINATION_LOSS' for x in gg)/len(gg) if gg else None,'S_T_R_headroom':{k:sum(x['head'] for x in gg if x['cat']==k) for k in trs}}
 ordered=sorted(gaps,key=lambda x:x['logr']);quart=[]
 for i,chunk in enumerate(np.array_split(ordered,4),1):
  vals=list(chunk); quart.append({'quartile':i,**breakdown(lambda x,ids={id(v) for v in vals}:id(x) in ids),'median_log_ratio':float(np.median([x['logr'] for x in vals])),'mean_log_ratio':float(np.mean([x['logr'] for x in vals]))})
 fold={str(f):breakdown(lambda x,f=f:x['fold']==f) for f in range(1,5)}
 for f in range(1,5):
  fold[str(f)].update({'query_count':1400,'oracle_positive_count':sum(x['fold']==f for x in gaps),'D77':per[str(f)],'top_harmful_count':sum(x['fold']==f for x in harmop),'executed_harmful':sum(x['fold']==f and x['take'] and x['top']['label']=='HARMFUL' for x in rec),'executed_beneficial':sum(x['fold']==f and x['take'] and x['top']['label']=='BENEFICIAL' for x in rec),'executed_neutral':sum(x['fold']==f and x['take'] and x['top']['label']=='NEUTRAL' for x in rec),'B_minus_N_negative_count':sum(x['fold']==f and x['bn']<0 for x in gaps)})
 depth={name:breakdown(lambda x,lo=lo,hi=hi:lo<=x['rank']<=hi) for name,lo,hi in [('1-20',1,20),('21-29',21,29),('30-45',30,45),('46-77',46,77)]}
 executed_h=[x for x in rec if x['take'] and x['top']['label']=='HARMFUL']
 attr={'status':'PASS','experiment':'STEP 2-P1F-R2 — Canonical-Metric Resolution and Pairwise Failure Attribution','scientific_status':'DIAGNOSTIC_ONLY_NO_BRANCH_GATE','reproduction_gate_pass':True,'policy_oof_strict':True,'end_to_end_selection_oof':False,'k77_selected_exploratorily_on_folds_1_4':True,'interpretation_scope':'PAIRWISE_POLICY_FAILURE_ATTRIBUTION_CONDITIONAL_ON_EXPLORATORY_FIXED_K77','K':77,'canonical_P1_gain_sum':-4.0,'canonical_P1_D77':-.0007142857142857143,'C77':.058785714285714274,'oracle_positive_queries':426,'trs':trs,'best_B_minus_N':{**stats(bn),'gt0':sum(x>0 for x in bn),'eq0_within_1e12':sum(abs(x)<=1e-12 for x in bn),'lt0':sum(x<0 for x in bn)},'best_B_minus_H':stats(bhn),'pointwise_F2_comparator':{'R_count':380,'R_headroom':293.7,'R_share':.8921628189550426,'best_B_minus_N_median':-.26528508631863545,'negative_B_minus_N':'380/426'},'harmful_attribution':{'top_action_classes_all_5600':dict(Counter(x['top']['label'] for x in rec)),'top_action_classes_oracle_positive':dict(Counter(qmap[q]['top'][0]['label'] for q in op)),'oracle_positive_top_harmful':harmop,'executed_harmful':len(executed_h),'executed_harmful_recall_delta_sum':sum(x['pr']-x['br'] for x in executed_h),'executed_harmful_details':[{'query_id':x['query_id'],'fold':x['fold'],'score':x['score'],'threshold':thresholds[x['fold']],'truth_recall_delta':x['top']['truth_delta'],'incoming_union_rank':x['top']['incoming_union_rank'],'drop_rank':x['top']['drop_rank']} for x in executed_h]},'per_fold':fold,'depth_analysis':depth,'current_status':{'P1_formulation':'REJECTED','pairwise_principle':'UNRESOLVED','pair_reweighting':'NOT_AUTHORIZED','B_vs_N_only_training':'NOT_AUTHORIZED','HGB_family':'UNRESOLVED','K77':'NOT_REJECTED','threshold':'DEFERRED','workflow_B':'NOT_ACTIVE'},'notes':['Threshold attribution is descriptive only; no threshold was retuned.']}
 train={}
 for held in range(1,5):
  c=Counter()
  for f in range(1,5):
   if f==held:continue
   for q,z in qmap.items():
    if z['fold']==f:c.update({'B_N':z['BN'],'B_H':z['BH'],'N_H':z['NH']})
  total=sum(c.values());train[str(held)]={'train_folds':[f for f in range(1,5) if f!=held],'pair_counts':dict(c),'pair_fractions':{k:v/total for k,v in c.items()},'held_out_median_B_minus_N':fold[str(held)]['median_B_minus_N'],'held_out_R_share':fold[str(held)]['R_fraction'],'held_out_D77':per[str(held)]}
 corr={'status':'PASS','global_pair_counts':dict(family),'query_local_analysis':{'population':426,'ratio':'log((BN+1)/(NH+1))','spearman':float(spearmanr([x['logr'] for x in gaps],bn).statistic),'pearson':float(pearsonr([x['logr'] for x in gaps],bn).statistic),'quartiles':quart},'outer_training_exposure':{'label':'SMALL_N_FOLD_LEVEL_DESCRIPTIVE_ONLY','folds':train},'causal_limitation':'ASSOCIATION_ONLY_NOT_CAUSAL','notes':['Query-local pair-family ratio describes held-out action-space composition, not literal outer-model training exposure.']}
 ATTR.write_text(json.dumps(attr,indent=2)+'\n');CORR.write_text(json.dumps(corr,indent=2)+'\n')
 print(json.dumps({'status':'PASS','D77':d,'full_actions':len(rows),'trs':{k:(v['query_count'],v['C77_headroom_sum']) for k,v in trs.items()},'median_B_minus_N':attr['best_B_minus_N']['median'],'spearman':corr['query_local_analysis']['spearman']},indent=2))
if __name__=='__main__':main()
