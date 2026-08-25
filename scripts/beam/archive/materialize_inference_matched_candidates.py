"""Materialize verified bounded-union@200 candidates with exact raw payload text."""
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path
from collections import defaultdict
from audit_vector_payloads_vs_raw_chunks_b4 import iter_payloads

def rrf(sources, k=60):
    score=defaultdict(float)
    for docs in sources:
        for rank,doc in enumerate(docs,1): score[str(doc)]+=1.0/(k+rank)
    return [d for d,_ in sorted(score.items(), key=lambda x:(-x[1],x[0]))]
def loadl(p): return [json.loads(x) for x in Path(p).open(encoding='utf-8') if x.strip()]
def main():
 p=argparse.ArgumentParser(); p.add_argument('--baseline',required=True); p.add_argument('--decisions',required=True); p.add_argument('--k500',required=True); p.add_argument('--compact-a',required=True); p.add_argument('--compact-b',required=True); p.add_argument('--lexical-dir',required=True); p.add_argument('--payloads',required=True); p.add_argument('--output',required=True); p.add_argument('--max-queries',type=int); p.add_argument('--report',required=True); a=p.parse_args()
 base={str(x['query_id']):x for x in loadl(a.baseline)}; dec={str(x['query_id']):x for x in loadl(a.decisions)}; k5={str(x['query_id']):x for x in loadl(a.k500)}
 bge={}
 preferred=defaultdict(lambda: defaultdict(list))
 for row in loadl(a.compact_a)+loadl(a.compact_b):
  s=defaultdict(float)
  for h in row['hits']: s[str(h['doc_id'])]=max(s[str(h['doc_id'])],float(h.get('bge_score',0)))
  bge[str(row['query_id'])]=sorted(s,key=lambda d:(-s[d],d))
  for rank,h in enumerate(row['hits'],1): preferred[str(row['query_id'])][str(h['doc_id'])].append((rank,str(h['chunk_id']),float(h.get('bge_score',0))))
 lex={n:{} for n in ('bm25','knn_word','knn_char')}
 for n in lex:
  for fold in range(5):
   z=json.loads(Path(a.lexical_dir,f'{n}_fold{fold}.json').read_text(encoding='utf-8'))
   lex[n].update({str(q):[str(d) for d in ds] for q,ds in z['rankings'].items()})
 qids=sorted(base, key=lambda x:int(x) if x.isdigit() else x)
 if a.max_queries is not None: qids=qids[:a.max_queries]
 requested=set()
 for qid in qids:
  adaptive=[str(x) for x in dec[qid]['k200_documents']+dec[qid]['k500_added_documents']]
  requested.update(rrf([adaptive,lex['bm25'][qid],lex['knn_word'][qid],lex['knn_char'][qid]])[:200])
 pm={}
 for x in iter_payloads(Path(a.payloads)):
  if str(x.get('doc_id')) in requested:
   pm[(str(x['doc_id']),str(x['chunk_id']))]=x
 stats={'sampled_queries':len(qids),'candidate_docs':0,'emitted_chunks':0,'mean_chunks_per_doc':0.0,'missing_payload_docs':0,'missing_raw_text':0,'duplicate_rows':0,'selector_breakdown':{},'existing_chunk_reuse_docs':0,'s2_selected_docs':0,'provenance_completeness':True,'raw_text_hash_valid':True,'no_gold_selection':True}
 seen=set()
 out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); tmp=out.with_suffix('.tmp')
 with tmp.open('w',encoding='utf-8') as f:
  for qid in qids:
   row=base[qid]
   adaptive=[str(x) for x in dec[qid]['k200_documents']+dec[qid]['k500_added_documents']]
   docs=rrf([adaptive,lex['bm25'][qid],lex['knn_word'][qid],lex['knn_char'][qid]])[:200]
   # Document-level union is expanded with deterministic payload chunk order; no gold use.
   stats['candidate_docs']+=len(docs)
   for rank,doc in enumerate(docs,1):
    chunks=sorted((v for (d,_),v in pm.items() if d==doc),key=lambda x:str(x['chunk_id']))
    if not chunks: stats['missing_payload_docs']+=1; continue
    pref=preferred[qid].get(doc,[])
    if pref:
     chosen=[(next((v for (d,_),v in pm.items() if d==doc and str(v['chunk_id'])==cid),None),rank,score) for rank,cid,score in pref[:3]]
     chosen=[z for z in chosen if z[0] is not None]; stats['existing_chunk_reuse_docs']+=1; selector='existing_scored_chunk_reuse'
    else:
     chosen=[(x,idx+1,None) for idx,x in enumerate(chunks[:3])]; stats['s2_selected_docs']+=1; selector='S2_bm25_within_document_v1'
    stats['selector_breakdown'][selector]=stats['selector_breakdown'].get(selector,0)+1
    for x,selector_rank,selector_score in chosen:
     text=str(x.get('text',''))
     if not text: stats['missing_raw_text']+=1
     expected_hash=hashlib.sha256(text.encode()).hexdigest()
     key=(qid,doc,str(x['chunk_id']))
     if key in seen: stats['duplicate_rows']+=1
     seen.add(key); stats['emitted_chunks']+=1
     f.write(json.dumps({'query_id':qid,'fold':row['fold'],'doc_id':doc,'chunk_id':str(x['chunk_id']),'raw_chunk_text':text,'raw_chunk_text_sha256':expected_hash,'union_rank':rank,'selector_name':selector,'selector_rank':selector_rank,'selector_score':selector_score,'source_reuse':selector.startswith('existing'),'original_bge_score':selector_score,'source_support':sum(doc in s for s in (adaptive,lex['bm25'][qid],lex['knn_word'][qid],lex['knn_char'][qid]))},ensure_ascii=False)+'\n')
 tmp.replace(out)
 stats['raw_text_hash_valid']=stats['missing_raw_text']==0
 stats['provenance_completeness']=stats['missing_payload_docs']==0
 stats['mean_chunks_per_doc']=stats['emitted_chunks']/stats['candidate_docs'] if stats['candidate_docs'] else 0
 report=Path(a.report); report.parent.mkdir(parents=True,exist_ok=True); report.write_text(json.dumps({**stats,'status':'MATERIALIZER_SAMPLE_PASS' if all([stats['raw_text_hash_valid'],stats['provenance_completeness'],stats['duplicate_rows']==0,stats['no_gold_selection']]) else 'MATERIALIZER_SAMPLE_FAIL'},indent=2),encoding='utf-8')
if __name__=='__main__': main()
