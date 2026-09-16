"""One fixed F1-F4 document-preference experiment; no retrieval or inference."""
from __future__ import annotations
import hashlib,json,math
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
from sklearn import __version__ as SKVER
from sklearn.linear_model import LogisticRegression
R=Path(__file__).resolve().parents[2]; OUT=R/'reports/task1/pairwise_document_preference_top5'
C=R/'artifacts/task1/recovery_096/inference_matched_reranker_v1/compact/candidate_refs_full.jsonl'; B=R/'artifacts/task1/recovery_096/baseline_093_oof/predictions.jsonl'; G=R/'artifacts/task1/recovery_096/baseline_093_oof/sources/f1to4_original_bge_chunk200_compact.jsonl'; Q=R/'artifacts/task1/workflow_b/tv2/b2a/qwen3-vl-reranker-2b/predictions.jsonl'; T=R/'data/raw/btc/LegalIR/train.json'
BH='7286ec481d26fc134711ecb03ceac2afe1a03bf0dabd842e1ed29ffb074e85c8'; QH='65ef5e500f6f0f3e9006da0aeaa6f07bff98e7123d7a5c2e09fd87db5eaabc2f'
F=('qwen_rank_percentile','qwen_missing','bge_rank_percentile','bge_missing','bm25_rank_percentile','bm25_missing','dense_rank_percentile','dense_missing','is_baseline_top5','baseline_rank_quality')
HP={'penalty':'l2','C':1.0,'fit_intercept':False,'solver':'lbfgs','max_iter':1000,'random_state':2026}; CAN=(.9259285714285714,.19739285714285715,{1:.9363690476190477,2:.9283333333333333,3:.919404761904762,4:.9196071428571428})
def jl(p):
 with open(p,encoding='utf8') as h:
  for x in h:
   if x.strip(): yield json.loads(x)
def hs(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def metric(pred,gold):
 r=[];p=[]
 for q,x in pred.items():
  z=set(map(str,gold[q]['answer']));r.append(len(z&set(x))/len(z));p.append(len(z&set(x))/5)
 return {'recall':float(np.mean(r)),'precision':float(np.mean(p)),'query_count':len(pred)}
def pct(vals,high=True):
 # returns rank percentile, available only; deterministic ID tie break handled upstream.
 a=sorted(vals.items(),key=lambda z:((-z[1],z[0]) if high else (z[1],z[0])));n=len(a)
 return {d:1. if n==1 else 1-i/(n-1) for i,(d,_) in enumerate(a)}
def cov(v):
 r=Counter()
 for x in v:r['both' if x['bp'] and x['qp'] else 'bge_only' if x['bp'] else 'qwen_only' if x['qp'] else 'neither']+=1
 return dict(r)
def main():
 if hs(G)!=BH or hs(Q)!=QH:raise RuntimeError('BLOCKED_FEATURE_PROVENANCE: BGE/Qwen SHA mismatch')
 gold=json.load(open(T,encoding='utf8')); cand={str(x['query_id']):x for x in jl(C) if x['fold']!=0}; base={str(x['query_id']):x for x in jl(B) if x['fold']!=0}
 if set(cand)!=set(base) or len(cand)!=5600:raise RuntimeError('BLOCKED_FEATURE_PROVENANCE: candidate/baseline identity')
 bg=defaultdict(dict)
 for x in jl(G):
  q=str(x['query_id'])
  for z in x['hits']:
   d=str(z['doc_id']);old=bg[q].get(d,(-np.inf,-np.inf));bg[q][d]=(max(old[0],float(z['bge_score'])),max(old[1],float(z['dense_score'])))
 qw={str(x['query_id']):{str(z['doc_id']):float(z['score']) for z in x['document_scores']} for x in jl(Q)}
 if set(qw)!=set(cand) or set(bg)!=set(cand):raise RuntimeError('BLOCKED_FEATURE_PROVENANCE: signal query identity')
 docs={}; insert_dups=0
 for q,c in cand.items():
  u={str(z['doc_id']):{'rank':int(z['union_rank']),'src':z['source_ranks']} for z in c['candidates'] if int(z['union_rank'])<=20}
  for i,d in enumerate(base[q]['top5'],1):
   d=str(d);insert_dups+=d in u;u.setdefault(d,{'rank':None,'src':{}})['br']=i
  raw={'qwen':{d:qw[q][d] for d in u if d in qw[q]},'bge':{d:bg[q][d][0] for d in u if d in bg[q]},'bm25':{d:v['src']['bm25'] for d,v in u.items() if 'bm25' in v['src']},'dense':{d:bg[q][d][1] for d in u if d in bg[q]}}
  pp={k:pct(v,k not in ('bm25',)) for k,v in raw.items()}
  docs[q]=[]
  for d,v in u.items():
   br=v.get('br',0); feat=[]
   for k in ('qwen','bge','bm25','dense'):feat += [pp[k].get(d,0.),float(d not in pp[k])]
   feat += [float(bool(br)),(6-br)/5 if br else 0.]
   docs[q].append({'q':q,'fold':c['fold'],'d':d,'rank':v['rank'],'br':br,'x':np.array(feat),'bp':d in bg[q],'qp':d in qw[q],'rel':int(d in set(map(str,gold[q]['answer'])) )})
 bm=metric({q:[str(d) for d in base[q]['top5']] for q in docs},gold);bf={f:metric({q:[str(d) for d in base[q]['top5']] for q in docs if cand[q]['fold']==f},gold) for f in range(1,5)}
 if not(math.isclose(bm['recall'],CAN[0],abs_tol=1e-15) and math.isclose(bm['precision'],CAN[1],abs_tol=1e-15)):raise RuntimeError('BLOCKED_BASELINE_MISMATCH')
 pairs={};pred={};outs=[];coef={};conc={};positions=defaultdict(Counter)
 for held in range(1,5):
  X=[];y=[];nrel=nneg=0
  for q,ds in docs.items():
   if cand[q]['fold']==held:continue
   a=[z for z in ds if z['rel']];n=[z for z in ds if not z['rel']];nrel+=len(a);nneg+=len(n)
   for r in a:
    for z in n:X += [r['x']-z['x'],z['x']-r['x']];y += [1,0]
  pairs[str(held)]={'training_queries':4200,'relevant_candidates':nrel,'nonrelevant_candidates':nneg,'preference_relations':len(y)//2,'mirrored_rows':len(y)}
  m=LogisticRegression(**HP).fit(np.asarray(X),np.asarray(y));coef[str(held)]=[float(x) for x in m.coef_[0]]
  good=tot=0
  for q,ds in docs.items():
   if cand[q]['fold']!=held:continue
   for z in ds:z['utility']=float(m.coef_[0]@z['x'])
   ordered=sorted(ds,key=lambda z:(-z['utility'],-bool(z['br']),z['br'] if z['br'] else 999,z['d']));final=[z['d'] for z in ordered[:5]];pred[q]=final
   for r in (z for z in ds if z['rel']):
    for n in (z for z in ds if not z['rel']):tot+=1;good+=r['utility']>n['utility']
   for i,z in enumerate(ordered,1):
    if z['rel']:positions[str(held)][str(i if i<=5 else '6-10' if i<=10 else '11-20' if i<=20 else '21+')]+=1
   for z in ds:outs.append({'query_id':q,'fold':held,'doc_id':z['d'],'utility':z['utility'],'relevant':z['rel'],'candidate_rank':z['rank'],'is_baseline_top5':bool(z['br']),'final_top5':final})
  conc[str(held)]=good/tot if tot else 0.
 rm=metric(pred,gold);rf={f:metric({q:pred[q] for q in pred if cand[q]['fold']==f},gold) for f in range(1,5)}; dr=rm['recall']-bm['recall'];dp=rm['precision']-bm['precision'];fd={f:rf[f]['recall']-bf[f]['recall'] for f in rf}
 ch=Counter();act=Counter();damage=Counter()
 for q,ds in docs.items():
  t=set(map(str,gold[q]['answer']));before=set(map(str,base[q]['top5']));after=set(pred[q]);delta=len(t&after)-len(t&before);changed=before!=after
  ch['changed']+=changed;ch['improved']+=changed and delta>0;ch['harmed']+=changed and delta<0;ch['neutral']+=changed and delta==0;ch['zero_hit_rescues']+=changed and not(t&before) and bool(t&after);ch['previously_correct_broken']+=changed and bool(t&before) and not(t&after);ch['net_relevant_document_change']+=delta
  for z in ds:
   if z['rel'] and not z['br']:
    act['total']+=1;act['moved']+=z['d'] in after;act['both_recovery']+=z['d'] in after and z['bp'] and z['qp'];act['qwen_only_recovery']+=z['d'] in after and not z['bp'] and z['qp'];act['band_'+('1_5' if z['rank']<=5 else '6_10' if z['rank']<=10 else '11_20')]+=1
   if z['rel'] and z['br']:
    if z['d'] in after:damage['preserved']+=1
    elif t&after:damage['displaced_by_relevant']+=1
    else:damage['displaced_by_nonrelevant']+=1
 gate={'pooled_positive':dr>0,'every_fold_nonnegative':all(x>=0 for x in fd.values()),'precision_guard':dp>=-.001,'complete_oof':len(pred)==5600,'provenance_clean':True};passed=all(gate.values())
 status=('PAIRWISE_DOCUMENT_PREFERENCE_FAIL' if not passed else 'PAIRWISE_DOCUMENT_TARGET_LEVEL_PASS' if dr>=.003 else 'PAIRWISE_DOCUMENT_STRONG_PASS' if dr>=.002 else 'PAIRWISE_DOCUMENT_MATERIAL_PASS' if dr>=.001 else 'PAIRWISE_DOCUMENT_WEAK_PASS');mat=('NEGATIVE_OR_ZERO' if not passed else 'TARGET_LEVEL' if dr>=.003 else 'STRONG_POSITIVE' if dr>=.002 else 'MATERIAL_POSITIVE' if dr>=.001 else 'WEAK_POSITIVE');interp='PAIRWISE_SIGNAL_EXISTS_BUT_TOP5_SELECTION_REMAINS_FRAGILE' if sum(conc.values())/4>=.5 else 'AVAILABLE_SIGNALS_DO_NOT_RESOLVE_WITHIN_QUERY_ORDERING'
 OUT.mkdir(parents=True,exist_ok=True)
 with open(OUT/'pairwise_document_oof_predictions.jsonl','w',encoding='utf8') as h:
  for z in sorted(outs,key=lambda x:(x['fold'],x['query_id'],x['doc_id'])):h.write(json.dumps(z,ensure_ascii=False,sort_keys=True)+'\n')
 ev={'status':status,'baseline':bm,'baseline_by_fold':bf,'result':rm,'result_by_fold':rf,'delta_vs_baseline':{'recall':dr,'precision':dp},'fold_recall_deltas':fd,'pairwise_concordance':{'pooled':sum(conc.values())/4,'by_fold':conc},'query_outcomes':dict(ch),'actionable_320':{**dict(act),'remained_outside':act['total']-act['moved']},'baseline_damage':dict(damage),'learned_rank_positions':positions,'interpretation':interp,'materiality':mat,'scientific_gate':{**gate,'overall':passed}}
 prov={'features':list(F),'BGE':{'path':str(G.relative_to(R)),'sha256':hs(G),'aggregation':'per-query/doc maximum for BGE and dense scores'},'Qwen':{'path':str(Q.relative_to(R)),'sha256':hs(Q)},'excluded_features':[],'normalization':'within-query rank percentile; labels not used','fold0_used':False,'public_labels_used':False}
 man={'experiment':'PAIRWISE_DOCUMENT_PREFERENCE_TOP5','estimator':'sklearn.linear_model.LogisticRegression','sklearn_version':SKVER,'parameters':HP,'random_state_note':'lbfgs determinism does not use random_state materially','pairs':pairs,'coefficients':coef,'primary_rows':sum(map(len,docs.values())),'baseline_overlap_insertions':insert_dups,'inputs':{str(x.relative_to(R)):hs(x) for x in (C,B,G,Q,T)}}
 for n,x in [('pairwise_document_evaluation.json',ev),('pairwise_document_feature_provenance.json',prov),('pairwise_document_training_manifest.json',man),('pairwise_document_failure_attribution.json',{'status':status,'interpretation':interp,'damage':dict(damage),'actionable':dict(act)})]:open(OUT/n,'w',encoding='utf8').write(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'status':status,'recall':rm['recall'],'delta':dr,'precision':rm['precision'],'pairs':pairs,'concordance':conc},indent=2))
if __name__=='__main__':main()
