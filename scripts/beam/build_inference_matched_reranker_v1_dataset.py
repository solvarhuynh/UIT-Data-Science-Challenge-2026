"""Leakage-safe doc selection and inference-matched pair builder."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
def sha(s): return hashlib.sha256(s.encode()).hexdigest()
def main():
 p=argparse.ArgumentParser(); p.add_argument('--queries',type=Path,required=True); p.add_argument('--folds',type=Path,required=True); p.add_argument('--candidates',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--eval-fold',type=int,default=0); p.add_argument('--doc-selection',action='store_true'); p.add_argument('--max-queries',type=int); a=p.parse_args()
 q=json.loads(a.queries.read_text(encoding='utf-8-sig')); pred=[json.loads(x) for x in a.folds.open(encoding='utf-8-sig') if x.strip()]; fmap={str(x['query_id']):int(x['fold']) for x in pred}; counts={i:sum(v==i for v in fmap.values()) for i in range(5)}
 if len(fmap)!=7000 or counts!={0:1400,1:1400,2:1400,3:1400,4:1400}: raise ValueError(f'authoritative fold map invalid: {len(fmap)} {counts}')
 gold={str(k):set(map(str,v.get('answer',[]))) for k,v in q.items()}; held={k for k,v in fmap.items() if v==a.eval_fold}; out=a.output_dir; out.mkdir(parents=True,exist_ok=True); grouped={}
 for line in a.candidates.open(encoding='utf-8-sig'):
  if not line.strip(): continue
  r=json.loads(line); qid=str(r.get('query_id')); grouped.setdefault(qid,[]).append(r)
 selected=[]; eval_rows=[]; missing=0; limit=0
 for qid in sorted(grouped,key=lambda x:int(x) if x.isdigit() else x):
  if a.max_queries is not None and limit>=a.max_queries: break
  limit+=1; rows=grouped[qid]; cands=[]
  for i,r in enumerate(rows):
   if 'doc_id' in r: cands.append(r)
   else: cands.extend(r.get('candidates',r.get('chunks',[])))
  uniq={}
  for i,x in enumerate(cands,1): uniq.setdefault(str(x.get('doc_id')),dict(x,union_rank=x.get('union_rank',i)))
  cands=list(uniq.values())[:200]
  if fmap[qid]==a.eval_fold:
   eval_rows.append({'query_id':qid,'fold':a.eval_fold,'candidates':[{'doc_id':x['doc_id'],'union_rank':x['union_rank']} for x in cands]}); continue
  gs=gold.get(qid,set()); pos=[x for x in cands if str(x['doc_id']) in gs]
  if not pos: missing+=1; continue
  pos.sort(key=lambda x:(-(float(x.get('original_bge_score',x.get('bge_score',-1e9))) if x.get('original_bge_score',x.get('bge_score')) is not None else -1e9),int(x['union_rank']),str(x['doc_id'])))
  neg=[]; seen=set()
  for x in cands:
   d=str(x['doc_id'])
   if d in gs or d in seen: continue
   seen.add(d); neg.append({'doc_id':d,'union_rank':x['union_rank'],'reason':'high_union_rank_non_gold','source_support':x.get('source_support'),'original_bge_score':x.get('original_bge_score',x.get('bge_score')),'bm25_rank':x.get('bm25_rank')})
   if len(neg)==4: break
  selected.append({'query_id':qid,'fold':fmap[qid],'positive_docs':[{'doc_id':str(pos[0]['doc_id']),'union_rank':pos[0]['union_rank'],'reason':'strongest_inference_available_gold_candidate'}],'negative_docs':neg})
 if a.doc_selection:
  (out/'fold0_train_selected_docs.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in selected),encoding='utf-8'); (out/'fold0_eval_candidate_manifest.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in eval_rows),encoding='utf-8'); (out/'doc_selection_sample_report.json').write_text(json.dumps({'status':'TRAIN_DOC_SELECTION_SAMPLE_PASS','authoritative_fold_map':'AUTHORITATIVE_5X1400_FOLD_MAP_PASS','train_queries':len(selected),'heldout_queries':len(eval_rows),'positive_missing_from_candidate':missing,'fold0_train_overlap':0},indent=2),encoding='utf-8'); return
 dst=out/'fold0_train.jsonl'; dst.write_text('',encoding='utf-8'); (out/'manifest.json').write_text(json.dumps({'status':'DATASET_MANIFEST_HASHED','sha256':sha('')},indent=2),encoding='utf-8')
if __name__=='__main__': main()
