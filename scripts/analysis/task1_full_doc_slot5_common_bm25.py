"""Label-free, memory-safe prediction stage for FULLDOC_TOP10_COMMON_BM25_SLOT5."""
import ctypes, gc, glob, hashlib, json, os, re, statistics, unicodedata
from collections import defaultdict
from pathlib import Path
import sys
import numpy as np
from rank_bm25 import BM25Okapi
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R/'src'))
from udsc2026.retrieval.sparse.tokenizer import tokenize_vi
O=R/'reports/task1/full_document_legal_field_retrieval';P=O/'full_document_legal_field_retrieval_candidates.jsonl';B=R/'artifacts/task1/recovery_096/baseline_093_oof/predictions.jsonl';Q=R/'data/raw/btc/LegalIR/train.json'
OUT=O/'full_doc_top10_common_bm25_slot5_predictions.jsonl';TMP=O/'full_doc_top10_common_bm25_slot5_predictions.jsonl.tmp';PARITY=O/'full_doc_top10_common_bm25_slot5_parity.json'
def token(s):
 s=unicodedata.normalize('NFC',re.sub(r'\s+',' ',s)).strip();return [x.casefold() for x in tokenize_vi(s) if any(c.isalnum() for c in x)]
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def jl(p):
 with open(p,encoding='utf8') as f:
  for line in f:
   if line.strip():yield json.loads(line)
def rss():
 class PMC(ctypes.Structure):
  _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong),('PeakWorkingSetSize',ctypes.c_size_t),('WorkingSetSize',ctypes.c_size_t),('QuotaPeakPagedPoolUsage',ctypes.c_size_t),('QuotaPagedPoolUsage',ctypes.c_size_t),('QuotaPeakNonPagedPoolUsage',ctypes.c_size_t),('QuotaNonPagedPoolUsage',ctypes.c_size_t),('PagefileUsage',ctypes.c_size_t),('PeakPagefileUsage',ctypes.c_size_t)]
 x=PMC();x.cb=ctypes.sizeof(x);ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(),ctypes.byref(x),x.cb);return int(x.WorkingSetSize)
def questions_only(p, wanted):
 """Extract only question fields; answer arrays are never JSON-decoded or accessed."""
 out={};qid=None
 with open(p,encoding='utf8') as f:
  for line in f:
   m=re.match(r'^\s*"(\d+)":\s*\{',line)
   if m:qid=m.group(1);continue
   if qid in wanted:
    m=re.match(r'^\s*"question":\s*(".*"),\s*$',line)
    if m:out[qid]=json.loads(m.group(1))
 if set(out)!=set(wanted):raise RuntimeError('BLOCKED_QUESTION_TEXT_MAPPING')
 return out
def main():
 if OUT.exists() or TMP.exists():raise RuntimeError('REFUSE_TO_OVERWRITE_PREDICTION_ARTIFACT')
 base={str(x['query_id']):[str(d) for d in x['top5']] for x in jl(B) if x['fold']!=0}
 if len(base)!=5600 or any(len(v)!=5 for v in base.values()):raise RuntimeError('BLOCKED_BASELINE_MISMATCH')
 wanted5={q:v[4] for q,v in base.items()};retained=defaultdict(dict);scan=mapping_fail=0
 for x in jl(P):
  scan+=1;q=str(x['query_id']);d=str(x['document_id'])
  if q not in base:mapping_fail+=1;continue
  r=int(x['rank'])
  if r<=10 or d==wanted5[q]:retained[q][d]=(float(x['score']),r)
 if scan!=2800000 or sha(P)!='3519a417cbeed46491e270f11a2b63707856cad594cda6c239e37a4a78204ab0':raise RuntimeError('BLOCKED_FROZEN_RETRIEVAL_ARTIFACT_MISMATCH')
 missing=[q for q in base if wanted5[q] not in retained[q]]
 workload={'frozen_retrieval_rows_streamed':scan,'top10_rows_retained':sum(sum(r<=10 for _,r in z.values()) for z in retained.values()),'baseline_rank5_total':len(base),'baseline_rank5_already_scored':len(base)-len(missing),'baseline_rank5_requiring_recompute':len(missing),'mapping_failures':mapping_fail};telemetry={'post_scan_rss':rss()};peak=telemetry['post_scan_rss']
 questions=questions_only(Q,base);paths=sorted(glob.glob(str(R/'data/raw/btc/LegalIR/selected-contexts/context_*.json')));ids=[str(json.load(open(p,encoding='utf8'))['id']) for p in paths];pos={d:i for i,d in enumerate(ids)}
 if len(ids)!=8532 or len(pos)!=8532:raise RuntimeError('BLOCKED_DOCUMENT_MAPPING')
 def corpus():
  for p in paths:
   x=json.load(open(p,encoding='utf8'));yield token(x.get('link','')+' '+x['passage'])
 telemetry['post_tokenization_rss']=rss();bm=BM25Okapi(corpus());gc.collect();telemetry['post_index_rss']=rss();peak=max(peak,telemetry['post_index_rss'])
 candidates=sorted((q,d,s,r) for q,z in retained.items() for d,(s,r) in z.items())[:256];diffs=[];byq=defaultdict(list)
 for q,d,s,r in candidates:
  score=float(bm.get_scores(token(questions[q]))[pos[d]]);diffs.append(abs(score-s));byq[q].append((d,s,r,score));peak=max(peak,rss())
 mismatches=0
 for q,xs in byq.items():
  frozen=[x[0] for x in sorted(xs,key=lambda x:(-x[1],x[2],x[0]))];rebuilt=[x[0] for x in sorted(xs,key=lambda x:(-x[3],x[2],x[0]))];mismatches+=sum(a!=b for a,b in zip(frozen,rebuilt))
 parity={'rows':len(candidates),'exact_matches':sum(x==0 for x in diffs),'max_abs_diff':max(diffs),'mean_abs_diff':statistics.fmean(diffs),'median_abs_diff':statistics.median(diffs),'ordering_mismatches':mismatches,'gate':'PASS' if max(diffs)<=1e-5 and mismatches==0 else 'FAIL'};PARITY.write_text(json.dumps(parity,indent=2)+'\n',encoding='utf8')
 if parity['gate']!='PASS':raise RuntimeError('BLOCKED_FULLDOC_BM25_SCORE_PARITY')
 for q in missing:
  retained[q][wanted5[q]]=(float(bm.get_scores(token(questions[q]))[pos[wanted5[q]]]),999);peak=max(peak,rss())
 telemetry['peak_rss_during_query_scoring']=peak
 with open(TMP,'x',encoding='utf8') as f:
  for q in sorted(base,key=lambda x:int(x)):
   b5=wanted5[q];z=retained[q];choices={b5}|{d for d,(s,r) in z.items() if r<=10 and d not in base[q][:4]};win=sorted(choices,key=lambda d:(-z[d][0],-int(d==b5),z[d][1],d))[0];top=base[q][:4]+[win]
   if len(top)!=5 or len(set(top))!=5:raise RuntimeError('BLOCKED_PREDICTION_DUPLICATE')
   f.write(json.dumps({'query_id':q,'top5':top,'slot5_incumbent':b5,'slot5_selected':win,'changed':win!=b5,'score':z[win][0],'frozen_rank':None if z[win][1]==999 else z[win][1]})+'\n');peak=max(peak,rss())
 check=list(jl(TMP))
 if len(check)!=5600 or len({x['query_id'] for x in check})!=5600 or not all(len(set(x['top5']))==5 for x in check):raise RuntimeError('BLOCKED_PREDICTION_VALIDATION')
 del check;gc.collect();os.replace(TMP,OUT);telemetry['peak_rss_during_prediction_generation']=peak;telemetry['peak_rss']=peak;telemetry['failure_phase']=None
 (O/'full_doc_top10_common_bm25_slot5_memory_telemetry.json').write_text(json.dumps({'workload':workload,'telemetry':telemetry,'previous_attempt':'BLOCKED_PREDICTION_ARTIFACT_NOT_MATERIALIZED','previous_reason':'MEMORY_PRESSURE_BEFORE_FREEZE'},indent=2)+'\n',encoding='utf8')
 print(json.dumps({'workload':workload,'parity':parity,'prediction_sha256':sha(OUT),'telemetry':telemetry,'PREDICTIONS_FROZEN_BEFORE_LABELS':True}))
if __name__=='__main__':main()
