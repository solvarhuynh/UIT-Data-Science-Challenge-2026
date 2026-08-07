"""Internal dense/sparse/hybrid LegalIR benchmark using parent/prefix mapping."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from udsc2026.contracts import LegalChunk

def main():
 p=argparse.ArgumentParser(); p.add_argument('--chunks-dir',default='data/processed/chunks'); p.add_argument('--ir-train-file',required=True); p.add_argument('--mode',choices=['dense','sparse','hybrid'],required=True); p.add_argument('--top-k',type=int,default=5); p.add_argument('--candidate-k',type=int,default=20); a=p.parse_args()
 chunks=[]
 for f in sorted(Path(a.chunks_dir).glob('*.jsonl')):
  for line in f.read_text(encoding='utf-8').splitlines():
   if line.strip(): chunks.append(LegalChunk.model_validate(json.loads(line)))
 if a.mode=='sparse':
  from udsc2026.retrieval.sparse.bm25_retriever import BM25Retriever
  r=BM25Retriever(); r.build_index(chunks); search=lambda q:r.search(q,a.candidate_k)
 else:
  raise RuntimeError('dense/hybrid require prebuilt configured indexes and embedding model; use index_chunks.py first')
 train=json.loads(Path(a.ir_train_file).read_text(encoding='utf-8')); rows=[]
 for qid,rec in train.items():
  gold=set(rec['answer']); hits=search(rec['question'])[:a.top_k]; found=set()
  for h in hits:
   for c in chunks:
    if c.chunk_id==h.chunk_id and (c.parent_id in gold or any(c.chunk_id.startswith(g) for g in gold)): found.add(next((g for g in gold if c.parent_id==g or c.chunk_id.startswith(g)),''))
  recall=len(found)/len(gold) if gold else 0; precision=len(found)/a.top_k
  rows.append({'question_id':qid,'question':rec['question'],'recall_at_k':recall,'precision_at_k':precision,'miss':not found,'hits':[h.model_dump() for h in hits]})
 report={'mode':a.mode,'top_k':a.top_k,'question_count':len(rows),'recall_at_k':sum(x['recall_at_k'] for x in rows)/len(rows),'precision_at_k':sum(x['precision_at_k'] for x in rows)/len(rows),'miss_count':sum(x['miss'] for x in rows),'lowest_10':sorted(rows,key=lambda x:(x['recall_at_k'],x['precision_at_k']))[:10]}
 out=Path(f'data/reports/tv2_retrieval_internal_{a.mode}.json'); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print(json.dumps({k:report[k] for k in ('mode','recall_at_k','precision_at_k','miss_count')},ensure_ascii=False))
if __name__=='__main__': main()
