"""Single frozen full-document BM25 retrieval representation experiment."""
import json,glob,hashlib,re,unicodedata
from pathlib import Path
from collections import defaultdict,Counter
import sys
import numpy as np
from rank_bm25 import BM25Okapi
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from udsc2026.retrieval.sparse.tokenizer import tokenize_vi
R=Path(__file__).resolve().parents[2];O=R/'reports/task1/full_document_legal_field_retrieval'
T=R/'data/raw/btc/LegalIR/train.json'; C=R/'artifacts/task1/recovery_096/inference_matched_reranker_v1/compact/candidate_refs_full.jsonl';G=R/'artifacts/task1/recovery_096/baseline_093_oof/sources/f1to4_original_bge_chunk200_compact.jsonl'
def tok(s):return [x.casefold() for x in tokenize_vi(unicodedata.normalize('NFC',re.sub(r'\s+',' ',s)).strip()) if any(c.isalnum() for c in x)]
def jl(p):
 for x in open(p,encoding='utf8'):
  if x.strip():yield json.loads(x)
def hs(p):
 h=hashlib.sha256();h.update(open(p,'rb').read());return h.hexdigest()
def main():
 train=json.load(open(T,encoding='utf8')); cand={str(x['query_id']):x for x in jl(C) if x['fold']!=0}; ids=set(cand); docs=[]
 for p in sorted(glob.glob(str(R/'data/raw/btc/LegalIR/selected-contexts/context_*.json'))):
  x=json.load(open(p,encoding='utf8'));docs.append((str(x['id']),x.get('link','')+' '+x['passage']))
 if len(docs)!=8532 or len({x[0] for x in docs})!=8532:raise RuntimeError('raw corpus identity')
 bm=BM25Okapi([tok(x[1]) for x in docs]); did=[x[0] for x in docs]; existing={q:{str(z['doc_id']) for z in r['candidates']} for q,r in cand.items()}
 dense=defaultdict(set)
 for r in jl(G):
  for z in r['hits']:dense[str(r['query_id'])].add(str(z['doc_id']))
 # labels are intentionally used only after all top500 identities are frozen in memory.
 out=[]; overall={k:[0,0] for k in (5,10,20,50,100,200,500)}; qhit={k:0 for k in overall}; pf=defaultdict(lambda:{k:[0,0] for k in overall}); hard=[]; hard60=[]; novelty=Counter()
 frozen={}
 for q in sorted(ids,key=int):
  a=np.asarray(bm.get_scores(tok(train[q]['question']))); ix=sorted(range(len(did)),key=lambda i:(-a[i],did[i]))[:500]; frozen[q]=[(did[i],float(a[i])) for i in ix]
 for q,rs in frozen.items():
  gold=set(map(str,train[q]['answer'])); ranks={d:i+1 for i,(d,s) in enumerate(rs)}; f=cand[q]['fold']
  for k in overall:
   h=len(gold&set(d for d,s in rs[:k]));overall[k][0]+=h;overall[k][1]+=len(gold);qhit[k]+=h>0;pf[f][k][0]+=h;pf[f][k][1]+=len(gold)
  missing=gold-existing[q]
  for d in missing:
   hard.append((q,d,ranks.get(d))); 
   if d not in dense[q]:hard60.append((q,d,ranks.get(d)))
  for i,(d,s) in enumerate(rs,1):out.append({'query_id':q,'fold':f,'document_id':d,'rank':i,'score':s})
 for q,d,r in hard60:
  if r: novelty['other_full_document_lexical_overlap']+=1
 def rec(a):return {str(k):sum(r is not None and r<=k for q,d,r in a) for k in (20,50,100,200,500)}
 status='FULLDOC_RETRIEVAL_STRONG_SIGNAL' if rec(hard60)['200']>=10 and len({cand[q]['fold'] for q,d,r in hard60 if r and r<=200})>=3 else 'FULLDOC_RETRIEVAL_WEAK_SIGNAL' if rec(hard60)['200']>=4 else 'FULLDOC_RETRIEVAL_FAIL'
 O.mkdir(parents=True,exist_ok=True)
 with open(O/'full_document_legal_field_retrieval_candidates.jsonl','w',encoding='utf8') as h:
  for x in out:h.write(json.dumps(x,ensure_ascii=False)+'\n')
 result={'status':status,'overall':{str(k):{'gold_recall':v[0]/v[1],'query_hit_rate':qhit[k]/5600,'recovered':v[0],'gold_total':v[1]} for k,v in overall.items()},'per_fold':{str(f):{str(k):v[0]/v[1] for k,v in z.items()} for f,z in pf.items()},'known_66':{'count':len(hard),'recovered':rec(hard)},'hard_60':{'count':len(hard60),'recovered':rec(hard60),'folds_at200':sorted({cand[q]['fold'] for q,d,r in hard60 if r and r<=200}),'rank_bands':{'1-20':sum(r and r<=20 for q,d,r in hard60),'21-50':sum(r and 21<=r<=50 for q,d,r in hard60),'51-100':sum(r and 51<=r<=100 for q,d,r in hard60),'101-200':sum(r and 101<=r<=200 for q,d,r in hard60),'201-500':sum(r and 201<=r<=500 for q,d,r in hard60),'not_recovered':sum(r is None for q,d,r in hard60)}}}
 prov={'representation':'raw link verbatim + raw full passage; no unavailable title/structured fields invented','normalization':'NFC, whitespace collapse, existing legal tokenizer/casefold; digits/codes preserved','engine':'rank_bm25.BM25Okapi default parameters via existing repository dependency','existing_audit':{'bm25':'chunk text','dense':'chunk vectors','word_knn':'chunk representation','char_knn':'chunk representation'},'inputs':{str(x.relative_to(R)):hs(x) for x in (T,C,G)}}
 attr={'hard_gold_rows':[{'query_id':q,'document_id':d,'rank':r,'pattern':'other_full_document_lexical_overlap' if r else 'not_recovered'} for q,d,r in hard60],'patterns':dict(novelty)}
 for n,x in [('full_document_legal_field_retrieval_results.json',result),('full_document_legal_field_retrieval_provenance.json',prov),('full_document_legal_field_retrieval_hard_gold_attribution.json',attr)]:open(O/n,'w',encoding='utf8').write(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'status':status,'hard':result['hard_60'],'overall200':result['overall']['200']},indent=2))
if __name__=='__main__':main()
