"""Post-freeze evaluation only; consumes the immutable 2.8M-row top-500 artifact."""
import json,hashlib
from collections import defaultdict
from pathlib import Path
R=Path(__file__).resolve().parents[2];O=R/'reports/task1/full_document_legal_field_retrieval';P=O/'full_document_legal_field_retrieval_candidates.jsonl';T=R/'data/raw/btc/LegalIR/train.json';C=R/'artifacts/task1/recovery_096/inference_matched_reranker_v1/compact/candidate_refs_full.jsonl';G=R/'artifacts/task1/recovery_096/baseline_093_oof/sources/f1to4_original_bge_chunk200_compact.jsonl'
def jl(p):
 for x in open(p,encoding='utf8'):
  if x.strip():yield json.loads(x)
def hs(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def main():
 train=json.load(open(T,encoding='utf8')); cand={str(x['query_id']):x for x in jl(C) if x['fold']!=0}; old={q:{str(z['doc_id']) for z in x['candidates']} for q,x in cand.items()}; dense=defaultdict(set)
 for x in jl(G):
  for z in x['hits']:dense[str(x['query_id'])].add(str(z['doc_id']))
 ranks=defaultdict(dict)
 for x in jl(P):ranks[x['query_id']][x['document_id']]=x['rank']
 ks=(5,10,20,50,100,200,500);total={k:[0,0,0] for k in ks};pf=defaultdict(lambda:{k:[0,0] for k in ks});known=[];hard=[];novel=[]
 for q,x in cand.items():
  gold=set(map(str,train[q]['answer']));f=x['fold'];rr=ranks[q]
  for k in ks:
   h=sum(rr.get(d,999)>0 and rr.get(d,999)<=k for d in gold);total[k][0]+=h;total[k][1]+=len(gold);total[k][2]+=h>0;pf[f][k][0]+=h;pf[f][k][1]+=len(gold)
  for d in gold-old[q]:known.append((q,d,rr.get(d)))
  for d in gold-old[q]:
   if d not in dense[q]:hard.append((q,d,rr.get(d)));novel.append((q,d,rr.get(d)))
 def rec(a):return {str(k):sum(bool(r and r<=k) for q,d,r in a) for k in (20,50,100,200,500)}
 h200=[x for x in hard if x[2] and x[2]<=200];status='FULLDOC_RETRIEVAL_STRONG_SIGNAL' if len(h200)>=10 and len({cand[q]['fold'] for q,d,r in h200})>=3 else 'FULLDOC_RETRIEVAL_WEAK_SIGNAL' if len(h200)>=4 else 'FULLDOC_RETRIEVAL_FAIL'
 result={'status':status,'retrieval_frozen_before_labels':True,'candidate_rows':sum(map(len,ranks.values())),'overall':{str(k):{'gold_recall':a/b,'query_hit_rate':c/5600,'gold_recovered':a} for k,(a,b,c) in total.items()},'per_fold':{str(f):{str(k):a/b for k,(a,b) in z.items()} for f,z in pf.items()},'known_66':{'count':len(known),'recovered':rec(known)},'hard_60':{'count':len(hard),'recovered':rec(hard),'folds_represented':sorted({cand[q]['fold'] for q,d,r in h200}),'rank_distribution':{'1-20':sum(bool(r and r<=20) for q,d,r in hard),'21-50':sum(bool(r and 21<=r<=50) for q,d,r in hard),'51-100':sum(bool(r and 51<=r<=100) for q,d,r in hard),'101-200':sum(bool(r and 101<=r<=200) for q,d,r in hard),'201-500':sum(bool(r and 201<=r<=500) for q,d,r in hard),'not_recovered':sum(not r for q,d,r in hard)}},'novel_information':{'unique_hard_golds':len(h200),'new_relevant_vs_existing_union':len(novel)}}
 attr={'recovered_hard_golds':[{'query_id':q,'document_id':d,'rank':r,'pattern':'full_document_lexical_visibility'} for q,d,r in h200],'structural_attribution':{'full_document_lexical_visibility':len(h200)}}
 prov={'post_freeze_evaluation':True,'candidate_sha256':hs(P),'candidate_rows':2800000,'script_sha256':hs(Path(__file__)),'inputs':{'train':hs(T),'candidate_union':hs(C),'bge':hs(G)}}
 for n,x in [('full_document_legal_field_retrieval_results.json',result),('full_document_legal_field_retrieval_hard_gold_attribution.json',attr),('full_document_legal_field_retrieval_provenance.json',prov)]:open(O/n,'w',encoding='utf8').write(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
