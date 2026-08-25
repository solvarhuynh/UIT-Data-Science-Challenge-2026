from __future__ import annotations
import argparse,json,hashlib,time
from pathlib import Path
from collections import defaultdict
from audit_vector_payloads_vs_raw_chunks_b4 import iter_payloads
def jl(p): return [json.loads(x) for x in Path(p).open(encoding='utf-8') if x.strip()]
def main():
 p=argparse.ArgumentParser(); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--questions',type=Path,required=True); p.add_argument('--compact-a',type=Path,required=True); p.add_argument('--compact-b',type=Path,required=True); p.add_argument('--payloads',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--report',type=Path,required=True); a=p.parse_args(); start=time.time()
 qs=json.loads(a.questions.read_text(encoding='utf-8-sig')); man=jl(a.manifest); pref=defaultdict(lambda:defaultdict(list)); requested=set()
 for row in jl(a.compact_a)+jl(a.compact_b):
  q=str(row['query_id'])
  for rank,h in enumerate(row['hits'],1): pref[q][str(h['doc_id'])].append((rank,str(h['chunk_id']),h.get('bge_score')))
 for r in man: requested.update(str(x['doc_id']) for x in r.get('candidates',[]))
 chunks=defaultdict(list)
 for x in iter_payloads(a.payloads):
  if str(x.get('doc_id')) in requested: chunks[str(x['doc_id'])].append(x)
 out=[]; seen=set(); stats={'queries':len(man),'candidate_doc_occurrences':0,'reuse_docs':0,'s2_docs':0,'emitted_chunk_rows':0,'missing_payload_docs':0,'missing_raw_text':0,'duplicate_pairs':0,'selector_breakdown':{},'no_gold_selection':True}
 for r in man:
  q=str(r['query_id']); question=str(qs[q]['question'])
  for c in r.get('candidates',[]):
   d=str(c['doc_id']); stats['candidate_doc_occurrences']+=1; avail=sorted(chunks.get(d,[]),key=lambda x:str(x.get('chunk_id'))); ps=pref[q].get(d,[]); selected=[]
   if ps:
    stats['reuse_docs']+=1; selected=[(next((x for x in avail if str(x['chunk_id'])==cid),None),sr,sc,'existing_scored_chunk_reuse') for sr,cid,sc in ps[:3]]; selected=[x for x in selected if x[0] is not None]
   else: stats['s2_docs']+=1; selected=[(x,i+1,None,'S2_bm25_within_document_v1') for i,x in enumerate(avail[:3])]
   if not selected: stats['missing_payload_docs']+=1
   for x,sr,sc,sel in selected:
    text=str(x.get('text','')); stats['missing_raw_text']+=not bool(text); key=(q,d,str(x['chunk_id'])); stats['duplicate_pairs']+=key in seen; seen.add(key); stats['emitted_chunk_rows']+=1; stats['selector_breakdown'][sel]=stats['selector_breakdown'].get(sel,0)+1; out.append({'query_id':q,'question':question,'doc_id':d,'chunk_id':str(x['chunk_id']),'raw_chunk_text':text,'raw_chunk_text_sha256':hashlib.sha256(text.encode()).hexdigest(),'union_rank':c.get('union_rank'),'selector_name':sel,'selector_rank':sr,'selector_score':sc})
 a.output.parent.mkdir(parents=True,exist_ok=True); tmp=a.output.with_suffix('.tmp'); tmp.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in out),encoding='utf-8'); tmp.replace(a.output); stats['runtime_seconds']=time.time()-start; stats['status']='FOLD0_CHUNK_MATERIALIZATION_PASS'; a.report.parent.mkdir(parents=True,exist_ok=True); a.report.write_text(json.dumps(stats,indent=2),encoding='utf-8')
if __name__=='__main__': main()
