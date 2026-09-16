"""Single fixed direct-document experiment with existing BGE and corrected Qwen scores."""
from __future__ import annotations
import hashlib, json, math
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from sklearn import __version__ as SKVER
from sklearn.ensemble import HistGradientBoostingClassifier

R=Path(__file__).resolve().parents[2]
S=R/'artifacts/task1/recovery_096/v3_residual/shortlist_evidence.jsonl'; C=R/'artifacts/task1/recovery_096/inference_matched_reranker_v1/compact/candidate_refs_full.jsonl'; CR=C.with_name('candidate_refs_full_report.json')
B=R/'artifacts/task1/recovery_096/baseline_093_oof/predictions.jsonl'; BG=R/'artifacts/task1/recovery_096/baseline_093_oof/sources/f1to4_original_bge_chunk200_compact.jsonl'; Q=R/'artifacts/task1/workflow_b/tv2/b2a/qwen3-vl-reranker-2b/predictions.jsonl'; F=R/'artifacts/task1/evaluation/strict_cv_v2/folds.json'; T=R/'data/raw/btc/LegalIR/train.json'; PREV=R/'reports/task1/direct_document_relevance_top5/direct_document_relevance_evaluation.json'; OUT=R/'reports/task1/direct_document_neural_signal_top5'
QHASH='65ef5e500f6f0f3e9006da0aeaa6f07bff98e7123d7a5c2e09fd87db5eaabc2f'
FEATURES=('bge_document_max','bge_missing','qwen_document_score','qwen_missing','is_baseline_top5','baseline_rank')
HP={'loss':'log_loss','learning_rate':.05,'max_iter':200,'max_leaf_nodes':31,'l2_regularization':1.,'early_stopping':False,'random_state':2026}
CAN=(.9259285714285714,.19739285714285715,{1:.9363690476190477,2:.9283333333333333,3:.919404761904762,4:.9196071428571428})
def jl(p):
 with p.open(encoding='utf-8') as h:
  for x in h:
   if x.strip(): yield json.loads(x)
def sh(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''): h.update(b)
 return h.hexdigest()
def met(pred,gold):
 rs=[]; ps=[]
 for q,x in pred.items():
  a={str(z) for z in gold[q]['answer']}; hit=len(a&set(x)); rs.append(hit/len(a) if a else 1.); ps.append(hit/5)
 return {'recall':float(np.mean(rs)),'precision':float(np.mean(ps)),'query_count':len(pred)}
def top(sc):
 return [x['doc_id'] for x in sorted(sc,key=lambda x:(-x['p'],-int(x['doc_id'] in x['base']),x['base'].index(x['doc_id'])+1 if x['doc_id'] in x['base'] else 10**9,x['doc_id']))[:5]]
def main():
 if sh(Q)!=QHASH: raise RuntimeError('BLOCKED_NEURAL_SIGNAL_PROVENANCE: corrected Qwen SHA mismatch')
 cr=json.loads(CR.read_text(encoding='utf-8'))
 if cr.get('no_gold_used_in_construction') is not True: raise RuntimeError('BLOCKED_NEURAL_SIGNAL_PROVENANCE: candidate safety')
 train=json.loads(T.read_text(encoding='utf-8-sig')); fm={str(q):int(x['fold']) for x in json.loads(F.read_text(encoding='utf-8'))['folds'] for q in x['validation_ids']}; ids=[q for q in fm if fm[q] in (1,2,3,4)]
 base={str(x['query_id']):[str(z) for z in x['top5']] for x in jl(B)}; short={str(x['query_id']):x for x in jl(S)}; k20={str(x['query_id']):{str(z['doc_id']) for z in x['candidates'] if int(z['union_rank'])<=20} for x in jl(C)}
 # Fixed label-free aggregation: maximum existing BGE chunk score per q-doc.
 bge=defaultdict(dict)
 for row in jl(BG):
  q=str(row['query_id'])
  if q not in ids or int(row['fold'])!=fm[q]: raise RuntimeError('BLOCKED_NEURAL_SIGNAL_PROVENANCE: BGE fold/query mismatch')
  for hit in row['hits']:
   d=str(hit['doc_id']); v=float(hit['bge_score']); bge[q][d]=max(v,bge[q].get(d,-float('inf')))
 qwen={}
 for row in jl(Q):
  q=str(row['query_id']);
  if q in qwen: raise RuntimeError('BLOCKED_NEURAL_SIGNAL_PROVENANCE: duplicate Qwen query')
  qwen[q]={str(z['doc_id']):float(z['score']) for z in row['document_scores']}
 if set(qwen)!=set(ids) or len(qwen)!=5600 or set(bge)!=set(ids): raise RuntimeError('BLOCKED_NEURAL_SIGNAL_PROVENANCE: neural coverage query mismatch')
 prev=json.loads(PREV.read_text(encoding='utf-8'))
 if prev['result']['recall']!=.924157738095238 or prev['result']['query_count']!=5600: raise RuntimeError('BLOCKED_NEURAL_SIGNAL_PROVENANCE: previous-direct parity mismatch')
 docs={}; coverage=Counter()
 for q in ids:
  row=short[q]; actual={str(x['doc_id']) for x in row['docs']}; expected=k20[q]|set(base[q])
  if actual!=expected or len(actual)>25 or int(row['fold'])!=fm[q]: raise RuntimeError('BLOCKED_NEURAL_SIGNAL_PROVENANCE: K20 membership mismatch')
  out=[]
  for d in row['docs']:
   did=str(d['doc_id']); br=base[q].index(did)+1 if did in base[q] else 0; bv=bge[q].get(did); qv=qwen[q].get(did)
   coverage['candidate_rows']+=1; coverage['bge_present']+=bv is not None; coverage['qwen_present']+=qv is not None
   out.append({'q':q,'fold':fm[q],'doc_id':did,'base':base[q],'v':{'bge_document_max':0. if bv is None else bv,'bge_missing':float(bv is None),'qwen_document_score':0. if qv is None else qv,'qwen_missing':float(qv is None),'is_baseline_top5':float(bool(br)),'baseline_rank':float(br)}})
  docs[q]=out
 bm=met({q:base[q] for q in ids},train); bf={f:met({q:base[q] for q in ids if fm[q]==f},train) for f in (1,2,3,4)}
 if not(math.isclose(bm['recall'],CAN[0],abs_tol=1e-15) and math.isclose(bm['precision'],CAN[1],abs_tol=1e-15) and all(math.isclose(bf[f]['recall'],CAN[2][f],abs_tol=1e-15) for f in bf)): raise RuntimeError('BLOCKED_BASELINE_MISMATCH')
 outs=[]; model_info={}; pred={}
 for held in (1,2,3,4):
  tr=[d for q in ids if fm[q]!=held for d in docs[q]]; te=[d for q in ids if fm[q]==held for d in docs[q]]
  X=np.asarray([[d['v'][n] for n in FEATURES] for d in tr]); y=np.asarray([int(d['doc_id'] in {str(z) for z in train[d['q']]['answer']}) for d in tr]); cc=np.bincount(y,minlength=2); w=np.asarray([len(y)/(2*cc[z]) for z in y])
  m=HistGradientBoostingClassifier(**HP); m.fit(X,y,sample_weight=w); pp=m.predict_proba(np.asarray([[d['v'][n] for n in FEATURES] for d in te]))[:,1]; group=defaultdict(list)
  for d,p in zip(te,pp): group[d['q']].append({**d,'p':float(p)})
  for q,x in group.items():
   final=top(x)
   pred[q]=final
   for d in x: outs.append({'query_id':q,'fold':held,'doc_id':d['doc_id'],'probability':d['p'],'bge_document_max':d['v']['bge_document_max'],'bge_missing':d['v']['bge_missing'],'qwen_document_score':d['v']['qwen_document_score'],'qwen_missing':d['v']['qwen_missing'],'relevant':int(d['doc_id'] in {str(z) for z in train[q]['answer']}),'final_top5':final,'baseline_top5':d['base']})
  model_info[str(held)]={'training_queries':4200,'training_rows':len(tr),'class_counts':{'0':int(cc[0]),'1':int(cc[1])},'weights':{'0':float(len(y)/(2*cc[0])),'1':float(len(y)/(2*cc[1]))}}
 rm=met(pred,train); rf={f:met({q:pred[q] for q in ids if fm[q]==f},train) for f in (1,2,3,4)}; dr=rm['recall']-bm['recall']; dp=rm['precision']-bm['precision']; fd={f:rf[f]['recall']-bf[f]['recall'] for f in rf}
 ch=Counter(); near={'known_candidate_occurrences':0,'recovered':0}
 old={}; oldgroups=defaultdict(list)
 for x in jl(R/'reports/task1/direct_document_relevance_top5/direct_document_relevance_oof_predictions.jsonl'):
  old.setdefault(x['query_id'],x); oldgroups[x['query_id']].append(x)
 for q in ids:
  truth={str(z) for z in train[q]['answer']}; before=set(base[q]); after=set(pred[q]); dh=len(truth&after)-len(truth&before); repl=len(before-after); ch['changed']+=bool(repl); ch['improved']+=repl and dh>0; ch['harmed']+=repl and dh<0; ch['neutral']+=repl and dh==0; ch['zero_hit_rescues']+=repl and not(truth&before) and bool(truth&after); ch['previously_correct_broken']+=repl and bool(truth&before) and not(truth&after); ch['net_relevant_document_gain']+=dh
  oldrows=oldgroups[q]
  ranks={z['doc_id']:i+1 for i,z in enumerate(sorted(oldrows,key=lambda z:-z['probability']))}
  for d in truth-set(old[q]['final_top5']):
   if 6<=ranks.get(d,999)<=22: near['known_candidate_occurrences']+=1; near['recovered']+=d in after
 sep={}
 for key in ('bge_document_max','qwen_document_score','probability'):
  z=defaultdict(list)
  for x in outs: z[x['relevant']].append(x[key])
  sep[key]={str(k):{'count':len(v),'mean':float(np.mean(v)),'median':float(np.median(v))} for k,v in z.items()}
 gate={'pooled_positive':dr>0,'every_fold_nonnegative':all(x>=0 for x in fd.values()),'precision_guard':dp>=-.001,'complete_oof':len(pred)==5600,'provenance_clean':True}; passed=all(gate.values())
 if not passed: status,mat='DIRECT_NEURAL_FAIL','NEGATIVE_OR_ZERO'
 elif dr>=.003: status,mat='DIRECT_NEURAL_TARGET_LEVEL_PASS','TARGET_LEVEL'
 elif dr>=.002: status,mat='DIRECT_NEURAL_STRONG_PASS','STRONG_POSITIVE'
 elif dr>=.001: status,mat='DIRECT_NEURAL_MATERIAL_PASS','MATERIAL_POSITIVE'
 else: status,mat='DIRECT_NEURAL_WEAK_PASS','WEAK_POSITIVE'
 OUT.mkdir(parents=True,exist_ok=True)
 with (OUT/'direct_neural_oof_predictions.jsonl').open('w',encoding='utf-8') as h:
  for x in sorted(outs,key=lambda z:(z['fold'],z['query_id'],z['doc_id'])): h.write(json.dumps(x,ensure_ascii=False,sort_keys=True)+'\n')
 prov={'BGE':{'artifact':str(BG.relative_to(R)),'sha256':sh(BG),'aggregation':'MAX(bge_score) per query/document','coverage':{'queries':len(bge),'candidate_rows':coverage['candidate_rows'],'present':coverage['bge_present'],'missing':coverage['candidate_rows']-coverage['bge_present']},'label_free':True,'heldout_safe':True},'Qwen':{'artifact':str(Q.relative_to(R)),'sha256':sh(Q),'coverage':{'queries':len(qwen),'candidate_rows':coverage['candidate_rows'],'present':coverage['qwen_present'],'missing':coverage['candidate_rows']-coverage['qwen_present']},'label_free':True,'heldout_safe':True},'feature_list':list(FEATURES),'excluded':['union_rank','source_support','adaptive-k500 fields','V3 action features','BENEFIT/HARM features','label-derived statistics','word/char-KNN'],'fold0_used':False,'public_labels_used':False}
 ev={'status':status,'baseline':bm,'baseline_by_fold':bf,'previous_direct_parity':{'stored_recall':prev['result']['recall'],'expected':.924157738095238,'population_parity':True},'result':rm,'result_by_fold':rf,'delta_vs_baseline':{'recall':dr,'precision':dp},'delta_vs_previous_direct':rm['recall']-prev['result']['recall'],'fold_recall_deltas':fd,'query_outcomes':dict(ch),'near_top_missed_gold':{**near,'still_missed':near['known_candidate_occurrences']-near['recovered'],'recovery_rate':near['recovered']/near['known_candidate_occurrences'] if near['known_candidate_occurrences'] else 0},'signal_separation':sep,'materiality':mat,'scientific_gate':{**gate,'overall':passed}}
 man={'experiment':'DIRECT_DOCUMENT_NEURAL_SIGNAL_TOP5','status':status,'estimator':'HistGradientBoostingClassifier','sklearn_version':SKVER,'parameters':HP,'weighting':'deterministic inverse-frequency outer-training sample_weight','seed':2026,'outer_folds':{str(f):[x for x in (1,2,3,4) if x!=f] for f in (1,2,3,4)},'models':model_info,'inputs':{str(p.relative_to(R)):sh(p) for p in (S,C,CR,B,BG,Q,F,T,PREV)}}
 for n,v in [('direct_neural_evaluation.json',ev),('direct_neural_provenance.json',prov),('direct_neural_training_manifest.json',man)]: (OUT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({'status':status,'recall':rm['recall'],'delta':dr,'precision':rm['precision'],'coverage':dict(coverage)},indent=2))
if __name__=='__main__': main()
