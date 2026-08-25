"""Train-only compact chunk materialization from selected docs and payload backend."""
from __future__ import annotations
import argparse,json,hashlib,time
from pathlib import Path
from collections import defaultdict
from audit_vector_payloads_vs_raw_chunks_b4 import iter_payloads
def jl(p): return [json.loads(x) for x in Path(p).open(encoding='utf-8') if x.strip()]
def main():
 p=argparse.ArgumentParser(); p.add_argument('--selected-docs',type=Path,required=True); p.add_argument('--compact-a',type=Path,required=True); p.add_argument('--compact-b',type=Path,required=True); p.add_argument('--payloads',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--report',type=Path,required=True); a=p.parse_args(); start=time.time()
 chosen={}; requested=set()
 for r in jl(a.selected_docs):
  q=str(r['query_id']); chosen[q]={'fold':r['fold'],'pos':{str(x['doc_id']) for x in r['positive_docs']},'docs':{str(x['doc_id']):x for x in r['positive_docs']+r['negative_docs']}}; requested.update(chosen[q]['docs'])
 pref=defaultdict(lambda:defaultdict(list))
 for row in jl(a.compact_a)+jl(a.compact_b):
  for rank,h in enumerate(row['hits'],1):
   if str(h['doc_id']) in requested: pref[str(row['query_id'])][str(h['doc_id'])].append((rank,str(h['chunk_id']),h.get('bge_score')))
 chunks=defaultdict(list)
 for x in iter_payloads(a.payloads):
  if str(x.get('doc_id')) in requested: chunks[str(x['doc_id'])].append(x)
 out=[]; stats={'selected_doc_occurrences':sum(len(v['docs']) for v in chosen.values()),'queries':len(chosen),'positive_docs':sum(len(v['pos']) for v in chosen.values()),'negative_docs':sum(len(v['docs'])-len(v['pos']) for v in chosen.values()),'reuse_docs':0,'s2_docs':0,'emitted_chunk_rows':0,'missing_payload_docs':0,'missing_raw_text':0,'duplicate_pairs':0,'raw_text_hash_mismatches':0,'selector_breakdown':{}}
 seen=set()
 for q,v in chosen.items():
  for d,meta in v['docs'].items():
   avail=sorted(chunks.get(d,[]),key=lambda x:str(x.get('chunk_id'))); selected=[]; ps=pref[q].get(d,[])
   if ps:
    stats['reuse_docs']+=1; selected=[(next((x for x in avail if str(x['chunk_id'])==cid),None),sr,sc,'existing_scored_chunk_reuse') for sr,cid,sc in ps[:3]]; selected=[x for x in selected if x[0] is not None]
   else:
    stats['s2_docs']+=1; selected=[(x,i+1,None,'S2_bm25_within_document_v1') for i,x in enumerate(avail[:3])]
   if not selected: stats['missing_payload_docs']+=1
   for x,sr,sc,sel in selected:
    text=str(x.get('text','')); h=hashlib.sha256(text.encode()).hexdigest(); key=(q,d,str(x['chunk_id']))
    if key in seen: stats['duplicate_pairs']+=1
    seen.add(key); stats['emitted_chunk_rows']+=1; stats['selector_breakdown'][sel]=stats['selector_breakdown'].get(sel,0)+1
    out.append({'query_id':q,'fold':v['fold'],'doc_id':d,'chunk_id':str(x['chunk_id']),'raw_chunk_text':text,'raw_chunk_text_sha256':h,'label':1 if d in v['pos'] else 0,'union_rank':meta.get('union_rank'),'selector_name':sel,'selector_rank':sr,'selector_score':sc,'source_reuse':sel.startswith('existing')})
 a.output.parent.mkdir(parents=True,exist_ok=True); tmp=a.output.with_suffix('.tmp'); tmp.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in out),encoding='utf-8'); tmp.replace(a.output); stats['runtime_seconds']=time.time()-start; stats['status']='TRAIN_MATERIALIZATION_COMPACT_PASS'; a.report.parent.mkdir(parents=True,exist_ok=True); a.report.write_text(json.dumps(stats,indent=2),encoding='utf-8')
if __name__=='__main__': main()
