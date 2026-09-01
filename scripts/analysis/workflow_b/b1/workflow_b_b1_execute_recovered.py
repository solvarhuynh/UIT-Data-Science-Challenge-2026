"""Memory-bounded execution of frozen B1 from equivalence-gated fold NPZ files."""
from __future__ import annotations
import gc, json, runpy
from collections import Counter
from pathlib import Path
import numpy as np
from lightgbm import LGBMRanker
ROOT=Path(__file__).resolve().parents[2]; R=ROOT/'reports/task1'; REC=R/'workflow_b_b1_recovery'; MODELS=R/'workflow_b_b1_models'
F=json.loads((R/'step2p1_model_contract.json').read_text())['feature_columns']; HP=json.loads((R/'workflow_b_b1_environment_manifest.json').read_text())['estimator_parameters']; HP={k:v for k,v in HP.items() if k!='n_jobs'}|{'n_jobs':1}
def load(f):
 z=np.load(REC/f'f{f}_actions.npz'); return {k:z[k] for k in z.files}
def groups(q):
 _,c=np.unique(q,return_counts=True); return c.tolist()
def fit(ds):
 X=np.concatenate([d['features'] for d in ds]); y=np.concatenate([d['label'] for d in ds]); g=sum((groups(d['query_id']) for d in ds),[]); m=LGBMRanker(**HP);m.fit(X,y,group=g,feature_name=F);return m
def tops(d,s):
 out=[]; start=0
 for n in groups(d['query_id']):
  ix=range(start,start+n); best=min(ix,key=lambda i:(-s[i],-int(d['drop_rank'][i]),int(d['incoming_union_rank'][i]),int(d['incoming_doc_id'][i])));out.append(best);start+=n
 return out
def threshold(d,s,ix):
 c=[float('inf')]+sorted({float(s[i]) for i in ix}); rows=[]
 for t in c:
  take=[i for i in ix if s[i]>=t]; gain=sum(float(d['truth_delta'][i]) for i in take); p=sum((1 if d['label'][i]==2 else -1 if d['label'][i]==0 else 0)/5 for i in take);rows.append((gain,p,t,len(take)))
 return max(rows,key=lambda x:(x[0],x[1],x[2]))
def main():
 MODELS.mkdir(exist_ok=True);ns=runpy.run_path(str(ROOT/'scripts/analysis/workflow_a/step2_k77_oof_policy_realizability.py'),run_name='b1rec_exec'); fmap=ns['target_folds'](); targets=set(fmap); _,gold=ns['load_gold'](targets);base,_=ns['load_baseline'](targets,fmap)
 scores_path=R/'workflow_b_b1_oof_action_scores.jsonl'; sf=scores_path.open('w',encoding='utf8',newline='\n'); thresholds={}; pred={}; selections={}
 for held in range(1,5):
  train=[x for x in range(1,5) if x!=held]; inner=[]
  for valid in train:
   tr=[x for x in train if x!=valid]; ds=[load(x) for x in tr]; m=fit(ds); vd=load(valid); s=m.predict(vd['features']); ix=tops(vd,s); inner.extend((vd,float(s[i]),i) for i in ix); del ds,m,vd,s;gc.collect()
  # Freeze inner top actions/scores; only now consume labels/deltas in threshold.
  vals=[float(s[i]) for d,s,i in inner]; cand=[float('inf')]+sorted(set(vals)); curve=[]
  for t in cand:
   take=[z for z in inner if z[1]>=t]; gain=sum(float(d['truth_delta'][i]) for d,s,i in take);p=sum((1 if d['label'][i]==2 else -1 if d['label'][i]==0 else 0)/5 for d,s,i in take);curve.append((gain,p,t,len(take)))
  gain,p,tau,n=max(curve,key=lambda x:(x[0],x[1],x[2])); thresholds[str(held)]={'threshold':'+Infinity' if tau==float('inf') else tau,'gain_sum':gain,'precision_delta':p,'execute_count':n,'candidate_count':len(curve)}
  ds=[load(x) for x in train];m=fit(ds);vd=load(held);s=m.predict(vd['features']);ix=tops(vd,s);m.booster_.save_model(str(MODELS/f'outer_f{held}_model.txt'));(MODELS/f'outer_f{held}_config.json').write_text(json.dumps({'parameters':HP,'features':F})+'\n')
  for i in range(len(s)):
   sf.write(json.dumps({'query_id':str(vd['query_id'][i]),'fold':held,'incoming_doc_id':str(vd['incoming_doc_id'][i]),'dropped_doc_id':str(vd['dropped_doc_id'][i]),'drop_rank':int(vd['drop_rank'][i]),'incoming_union_rank':int(vd['incoming_union_rank'][i]),'b1_score':float(s[i])})+'\n')
  for i in ix:
   q=str(vd['query_id'][i]); apply=bool(s[i]>=tau);final=list(base[q]);
   if apply:final[int(vd['drop_rank'][i])-1]=str(vd['incoming_doc_id'][i])
   pred[q]=final;selections[q]={'fold':held,'apply':apply,'score':float(s[i]),'threshold':tau,'i':i,'d':vd}
  del ds,m,vd,s;gc.collect()
 sf.close();
 # Persist predictions before joining truth for end-to-end metrics.
 (R/'workflow_b_b1_oof_predictions.json').write_text(json.dumps({'predictions':{q:{'answer':pred[q],'fold':selections[q]['fold'],'apply':selections[q]['apply']} for q in sorted(pred,key=int)}},indent=2)+'\n')
 p1t=json.loads((R/'step2p1t_threshold_calibration_report.json').read_text());pt={int(k):v for k,v in p1t['fixed_thresholds'].items()};p1={}
 for line in (R/'step2p1f_full_action_scores.jsonl').open(encoding='utf8'):
  x=json.loads(line)
  if x['is_query_top_scored']:
   q=x['query_id'];fin=list(base[q]);
   if x['pairwise_policy_score']>=pt[x['fold']]:fin[x['drop_rank']-1]=x['incoming_doc_id']
   p1[q]=fin
 ev=runpy.run_path(str(ROOT/'src/udsc2026/evaluation/legal_ir.py')); rec,prec=ev['legal_ir_recall'],ev['legal_ir_precision']
 def met(p,qs):return {'recall':sum(rec(p[q],gold[q]) for q in qs)/len(qs),'precision':sum(prec(p[q],gold[q]) for q in qs)/len(qs)}
 per={};
 for f in range(1,5):
  qs=[q for q in pred if fmap[q]==f];a,b=met(p1,qs),met(pred,qs);per[str(f)]={'p1':a,'b1':b,'recall_delta':b['recall']-a['recall']}
 a,b=met(p1,list(pred)),met(pred,list(pred));rd=b['recall']-a['recall'];pd=b['precision']-a['precision'];success={'pooled_recall_delta_gte_0_003':rd>=.003,'all_fold_recall_delta_nonnegative':all(x['recall_delta']>=0 for x in per.values()),'precision_delta_gte_minus_0_001':pd>=-.001};status='PASS' if all(success.values()) else 'FAIL'
 out={'status':status,'p1':a,'b1':b,'recall_delta':rd,'precision_delta':pd,'per_fold':per,'success':success};(R/'workflow_b_b1_evaluation.json').write_text(json.dumps(out,indent=2)+'\n');(R/'workflow_b_b1_nested_thresholds.json').write_text(json.dumps({'status':'PASS','thresholds':thresholds},indent=2)+'\n');(R/'workflow_b_b1_diagnostics.json').write_text(json.dumps({'apply_count':sum(x['apply'] for x in selections.values()),'no_op_count':sum(not x['apply'] for x in selections.values()),'models_fit':16},indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
