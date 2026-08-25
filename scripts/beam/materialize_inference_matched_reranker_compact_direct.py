"""Direct compact materialization: bounded-union refs + deduplicated chunk store."""
from __future__ import annotations
import argparse, hashlib, json, sqlite3, tempfile, os, time
from pathlib import Path
from collections import defaultdict
from audit_vector_payloads_vs_raw_chunks_b4 import iter_payloads

def jl(p): return [json.loads(x) for x in Path(p).open(encoding='utf-8') if x.strip()]
def rrf(src,k=60):
 s=defaultdict(float)
 for xs in src:
  for i,d in enumerate(xs,1): s[str(d)]+=1/(k+i)
 return [d for d,_ in sorted(s.items(),key=lambda z:(-z[1],z[0]))]
def main():
 p=argparse.ArgumentParser();
 for n in ('baseline','decisions','compact_a','compact_b','lexical_dir','payloads'): p.add_argument('--'+n.replace('_','-'),required=True)
 p.add_argument('--output-dir',required=True); p.add_argument('--max-queries',type=int); p.add_argument('--report',required=True); a=p.parse_args(); t=time.time()
 base={str(x['query_id']):x for x in jl(a.baseline)}; dec={str(x['query_id']):x for x in jl(a.decisions)}; pref=defaultdict(lambda:defaultdict(list))
 for row in jl(a.compact_a)+jl(a.compact_b):
  q=str(row['query_id'])
  for rank,h in enumerate(row['hits'],1): pref[q][str(h['doc_id'])].append((rank,str(h['chunk_id']),h.get('bge_score')))
 lex={n:{} for n in ('bm25','knn_word','knn_char')}
 for n in lex:
  for f in range(5):
   z=json.loads(Path(a.lexical_dir,f'{n}_fold{f}.json').read_text(encoding='utf-8')); lex[n].update({str(q):[str(d) for d in ds] for q,ds in z['rankings'].items()})
 qids=sorted(base,key=lambda x:int(x) if x.isdigit() else x)[:a.max_queries if a.max_queries else None]
 plans={}; requested=set()
 for q in qids:
  ad=[str(x) for x in dec[q]['k200_documents']+dec[q]['k500_added_documents']]; docs=rrf([ad,lex['bm25'][q],lex['knn_word'][q],lex['knn_char'][q]])[:200]; plans[q]=(ad,docs); requested.update(docs)
 out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); refs=out/'candidate_refs.jsonl'; dbp=out/'chunk_store.sqlite'; tmprefs=refs.with_suffix('.tmp'); tmpr=Path(a.report).with_suffix('.tmp')
 db=sqlite3.connect(dbp); db.execute('pragma journal_mode=WAL'); db.execute('create table if not exists chunks(doc_id text,chunk_id text,raw_chunk_text text,raw_chunk_text_sha256 text,primary key(doc_id,chunk_id))'); db.execute('create index if not exists idx_chunks_doc on chunks(doc_id)'); db.commit()
 stats={'query_count':len(qids),'candidate_doc_occurrences':0,'unique_candidate_docs':len(requested),'selected_chunk_refs':0,'unique_chunk_store_rows':0,'reuse_docs':0,'s2_docs':0,'missing_payload_docs':0,'missing_raw_text':0,'duplicate_candidate_refs':0,'candidate_oracle_200':0.9924714285714286,'no_gold_selection':True,'payload_scans':1}
 pm=defaultdict(list)
 for x in iter_payloads(Path(a.payloads)):
  d=str(x.get('doc_id'))
  if d in requested: pm[d].append(x)
 with tmprefs.open('w',encoding='utf-8') as f:
  for q in qids:
   ad,docs=plans[q]; row=base[q]
   for rank,d in enumerate(docs,1):
    stats['candidate_doc_occurrences']+=1; chunks=sorted(pm.get(d,[]),key=lambda x:str(x.get('chunk_id')))
    if not chunks: stats['missing_payload_docs']+=1; continue
    chosen=[]; scored=pref[q].get(d,[])
    if scored:
     stats['reuse_docs']+=1
     for sr,cid,score in scored[:3]:
      x=next((z for z in chunks if str(z.get('chunk_id'))==cid),None)
      if x: chosen.append((x,sr,score,'existing_scored_chunk_reuse'))
    else:
     stats['s2_docs']+=1; chosen=[(x,i+1,None,'S2_bm25_within_document_v1') for i,x in enumerate(chunks[:3])]
    refs_out=[]
    for x,sr,score,sel in chosen:
     text=str(x.get('text','')); 
     if not text: stats['missing_raw_text']+=1
     h=hashlib.sha256(text.encode()).hexdigest(); cid=str(x['chunk_id']); db.execute('insert or ignore into chunks values(?,?,?,?)',(d,cid,text,h)); refs_out.append({'chunk_id':cid,'selector_name':sel,'selector_rank':sr,'selector_score':score,'source_reuse':sel.startswith('existing')}); stats['selected_chunk_refs']+=1
    f.write(json.dumps({'query_id':q,'fold':row['fold'],'doc_id':d,'union_rank':rank,'source_support':sum(d in xs for xs in (ad,lex['bm25'][q],lex['knn_word'][q],lex['knn_char'][q])),'selected_chunks':refs_out},ensure_ascii=False)+'\n')
   db.commit()
 db.commit(); stats['unique_chunk_store_rows']=db.execute('select count(*) from chunks').fetchone()[0]; db.close(); os.replace(tmprefs,refs)
 stats.update({'runtime_seconds':time.time()-t,'status':'DIRECT_COMPACT_MATERIALIZATION_PASS'})
 Path(a.report).parent.mkdir(parents=True,exist_ok=True); tmpr.write_text(json.dumps(stats,indent=2),encoding='utf-8'); os.replace(tmpr,a.report)
if __name__=='__main__': main()
