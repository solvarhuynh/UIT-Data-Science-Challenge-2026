import hashlib,json,re,statistics
from collections import Counter
from pathlib import Path
R=Path(__file__).resolve().parents[2];O=R/'reports/task1/full_document_legal_field_retrieval';C=O/'full_document_legal_field_retrieval_candidates.jsonl';K=R/'artifacts/task1/recovery_096/inference_matched_reranker_v1/compact/candidate_refs_full.jsonl';QP=R/'artifacts/task1/workflow_b/tv2/b2a/qwen3-vl-reranker-2b/predictions.jsonl';QS=R/'artifacts/task1/workflow_b/tv2/b2a/qwen3-vl-reranker-2b/run_state.json';CH=R/'data/processed_v3/chunks';T=R/'data/raw/btc/LegalIR/train.json'
def jl(p):
 with open(p,encoding='utf8') as f:
  for l in f:
   if l.strip():yield json.loads(l)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for x in iter(lambda:f.read(1<<20),b''):h.update(x)
 return h.hexdigest()
def qids_with_question(p):
 out=set();q=None
 for line in open(p,encoding='utf8'):
  m=re.match(r'^\s*"(\d+)":\s*\{',line)
  if m:q=m.group(1);continue
  if q and re.match(r'^\s*"question":\s*"',line):out.add(q)
 return out
def main():
 assert sha(C)=='3519a417cbeed46491e270f11a2b63707856cad594cda6c239e37a4a78204ab0'
 assert sha(QP)=='65ef5e500f6f0f3e9006da0aeaa6f07bff98e7123d7a5c2e09fd87db5eaabc2f'
 pool={str(x['query_id']):{str(z['doc_id']) for z in x['candidates']} for x in jl(K)}
 target=[];bands=Counter()
 for x in jl(C):
  q,d,r=str(x['query_id']),str(x['document_id']),int(x['rank'])
  if r<=200 and d not in pool[q]:
   target.append((q,d,r));bands['1-20' if r<=20 else '21-50' if r<=50 else '51-100' if r<=100 else '101-200']+=1
 docs={d for q,d,r in target};queries={q for q,d,r in target}
 old=set()
 for x in jl(QP):
  q=str(x['query_id'])
  for z in x['document_scores']:old.add((q,str(z['doc_id'])))
 reuse=sum((q,d) in old for q,d,r in target)
 chunk_n={};missing_docs=[]
 for d in docs:
  p=CH/f'{d}.jsonl'
  if not p.is_file():chunk_n[d]=0;missing_docs.append(d);continue
  chunk_n[d]=sum(1 for x in open(p,encoding='utf8') if x.strip())
 vals=[chunk_n[d] for d in docs];valid=sum(chunk_n[d]>=3 for d in docs)
 invalid_docs={d for d in docs if chunk_n[d]<3};invalid_qdocs=sum(d in invalid_docs for q,d,r in target)
 qtext=qids_with_question(T);valid_queries=len(queries&qtext)
 depth=[]
 for name,limit in [('TOP20',20),('TOP50',50),('TOP100',100),('TOP200',200)]:
  xs=[x for x in target if x[2]<=limit];depth.append({'depth':name,'new_q_docs':len(xs),'inference_units':len(xs)*3,'estimated_seconds_at_historical_rate':len(xs)*3/36.351306519275624,'estimated_hours_at_historical_rate':len(xs)*3/36.351306519275624/3600})
 state=json.load(open(QS,encoding='utf8'))
 status='ACTIONABLE_WITH_ONE_RESOURCE_AUTHORIZATION' if not invalid_qdocs else 'BLOCKED_QWEN_CANDIDATE_MATERIALIZATION'
 out={'status':status,'target_universe':{'q_docs':len(target),'queries':len(queries),'unique_documents':len(docs),'ranks1_20':bands['1-20'],'ranks21_50':bands['21-50'],'ranks51_100':bands['51-100'],'ranks101_200':bands['101-200'],'reusable_existing_qwen':reuse,'require_new_inference':len(target)-reuse},'contract':{'model':state['model'],'revision':state['resolved_revision'],'snapshot_hashes':state['snapshot_provenance_sha256'],'processor_tokenizer':'snapshot tokenizer.json/tokenizer_config.json; Transformers processor','script':'scripts/modal/task1_b2a_qwen3vl2b.py','input':'Vietnamese corrected query text + each of deterministic top3 within-document BM25 chunks; text-only chat template; MAX_LENGTH=8192','aggregation':'document MAX over 3 sigmoid direction scores; doc-id deterministic tie','reproducible':'RECOVERED_EXACTLY via pinned remote snapshot; local bytes unavailable'},'materialization':{'valid_q_docs':len(target)-invalid_qdocs,'valid_chunks':(len(target)-invalid_qdocs)*3,'documents_with_at_least_3_canonical_chunks':valid,'documents_without_three_chunks':len(invalid_docs),'missing_document_text':len(missing_docs),'queries_with_corrected_text':valid_queries,'materialization_failures':invalid_qdocs+(len(queries)-valid_queries),'inference_units':(len(target)-invalid_qdocs)*3,'mean_chunks_per_qdoc':3,'median_chunks_per_qdoc':3,'p95_chunks_per_qdoc':3,'max_chunks_per_qdoc':3},'historical_throughput':{'q_docs':state['full_q_doc_pairs'],'inference_units':state['full_chunks_scored'],'device':'Modal A10 (runner declaration)','runtime_seconds':state['full_runtime_seconds'],'throughput_units_per_second':state['full_throughput_chunks_per_second'],'batch_size':1,'environment':'Modal','source':'run_state.json'},'runtime_projection':depth,'execution_options':[{'environment':'local CPU','available':True,'setup':'none','runtime':'not estimated: no historical CPU throughput','risk':'impractical / unsupported contract performance'},{'environment':'local CUDA','available':False,'setup':'CUDA-capable hardware/runtime required','runtime':'unavailable','risk':'not currently executable'},{'environment':'Modal A10 path','available':False,'setup':'one explicit remote execution authorization; existing script/image/volume contract','runtime':'TOP200 projection below','risk':'remote snapshot/volume availability must be reverified at run time'}],'prospective_artifact':{'path':str(O/'full_doc_top200_corrected_qwen_scores.jsonl'),'schema':['query_id','document_id','full_doc_rank','qwen_document_score','chunk_count','execution_batch_provenance_identifier'],'freeze_rule':'complete deduplicated target universe, verify coverage and SHA256 before any labels','provenance':'target-universe SHA256, model revision and snapshot hashes, source script hash, chunk-selection worklist hash, runtime batch identifier'}}
 (O/'full_doc_top200_qwen_deep_scoring_feasibility.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8');print(json.dumps({'qdocs':len(target),'reuse':reuse,'hours':depth[-1]['estimated_hours_at_historical_rate']}))
if __name__=='__main__':main()
