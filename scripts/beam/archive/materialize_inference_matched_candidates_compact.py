"""Reduce inference-matched rows into deduplicated chunk store + query refs."""
from __future__ import annotations
import argparse, json, hashlib, sqlite3
from pathlib import Path
def main():
 p=argparse.ArgumentParser(); p.add_argument('--input',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True); a=p.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
 db=sqlite3.connect(a.output_dir/'chunk_store.sqlite'); db.execute('create table if not exists chunks(doc_id text, chunk_id text, raw_chunk_text text, raw_chunk_text_sha256 text, selector_name text, primary key(doc_id,chunk_id))'); refs=a.output_dir/'candidate_refs.jsonl'; n=0
 with refs.open('w',encoding='utf-8') as out:
  for line in a.input.open(encoding='utf-8'):
   if not line.strip(): continue
   r=json.loads(line); text=str(r.get('raw_chunk_text','')); h=hashlib.sha256(text.encode()).hexdigest(); db.execute('insert or ignore into chunks values(?,?,?,?,?)',(str(r['doc_id']),str(r['chunk_id']),text,h,r.get('selector_name'))); ref={k:r.get(k) for k in ('query_id','fold','doc_id','chunk_id','union_rank','source_support','selector_name','selector_rank','selector_score','source_reuse','original_bge_score')}; out.write(json.dumps(ref,ensure_ascii=False)+'\n'); n+=1
 db.commit(); db.close(); (a.output_dir/'materializer_report.json').write_text(json.dumps({'status':'COMPACT_CHUNK_STORE_DEDUP_PASS','candidate_ref_rows':n},indent=2),encoding='utf-8')
if __name__=='__main__': main()
