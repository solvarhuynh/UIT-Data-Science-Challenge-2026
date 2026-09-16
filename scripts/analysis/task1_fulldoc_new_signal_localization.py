import hashlib,json,re
from collections import Counter,defaultdict
from pathlib import Path
R=Path(__file__).resolve().parents[2];O=R/'reports/task1/full_document_legal_field_retrieval'
C=O/'full_document_legal_field_retrieval_candidates.jsonl';K=R/'artifacts/task1/recovery_096/inference_matched_reranker_v1/compact/candidate_refs_full.jsonl';B=R/'artifacts/task1/recovery_096/baseline_093_oof/predictions.jsonl';P=O/'full_doc_top10_common_bm25_slot5_predictions.jsonl';T=R/'data/raw/btc/LegalIR/train.json'
bands=[('1-5',1,5),('6-10',6,10),('11-20',11,20),('21-50',21,50),('51-100',51,100),('101-200',101,200),('201-500',201,500)]
def jl(p):
 with open(p,encoding='utf8') as f:
  for l in f:
   if l.strip():yield json.loads(l)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for x in iter(lambda:f.read(1<<20),b''):h.update(x)
 return h.hexdigest()
def band(r):
 for n,a,b in bands:
  if a<=r<=b:return n
def terms(s):return {x.casefold() for x in re.findall(r'[\wÀ-ỹ]+',s) if len(x)>2}
def main():
 assert sha(C)=='3519a417cbeed46491e270f11a2b63707856cad594cda6c239e37a4a78204ab0'
 pool={str(x['query_id']):{str(z['doc_id']) for z in x['candidates']} for x in jl(K)}
 base={str(x['query_id']):x for x in jl(B) if x['fold']!=0};pred={str(x['query_id']):x for x in jl(P)};train=json.load(open(T,encoding='utf8'))
 gold={q:{str(d) for d in train[q]['answer']} for q in base};known={(q,d) for q in base for d in gold[q] if d not in pool[q]};assert len(known)==66
 novel=Counter();new=[];known_rows={};
 for x in jl(C):
  q,d,r=str(x['query_id']),str(x['document_id']),int(x['rank']);
  if d in pool[q]:continue
  novel[band(r)]+=1
  if d in gold[q]:
   z={'query_id':q,'document_id':d,'fold':int(base[q]['fold']),'rank':r,'score':x['score'],'gold_n':len(gold[q])};new.append(z)
   if (q,d) in known:known_rows[q,d]=z
 assert len(known_rows)<=66
 # Add observed document text only for new relevant documents, for descriptive classification.
 needed={x['document_id'] for x in new};doc={}
 for p in sorted((R/'data/raw/btc/LegalIR/selected-contexts').glob('context_*.json')):
  x=json.load(open(p,encoding='utf8'));d=str(x['id'])
  if d in needed:doc[d]=(x.get('link',''),x.get('passage',''))
 def pattern(x):
  q=train[x['query_id']]['question'];title,body=doc[x['document_id']];qt=terms(q);tt=terms(title);bt=terms(body)
  if re.search(r'\b(?:TT|NĐ|QĐ|LUẬT)\s*[-/]?\s*\d',q,re.I) and re.search(r'\b(?:TT|NĐ|QĐ|LUẬT)\s*[-/]?\s*\d',title,re.I):return 'legal instrument code/name'
  if re.search(r'\b(?:điều|khoản|mục|chương)\b',q,re.I) and re.search(r'\b(?:điều|khoản|mục|chương)\b',body,re.I):return 'article/section terminology'
  if len(qt&tt)>=2:return 'title overlap'
  if len(qt&bt)>=3:return 'distributed full-document lexical evidence'
  if len(qt&bt)>0:return 'generic lexical overlap'
  return 'other directly observed pattern'
 for x in new:x['pattern']=pattern(x)
 cumulative={k:sum(x['rank']<=k for x in new) for k in [5,10,20,50,100,200,500]}
 byband={n:[x for x in new if a<=x['rank']<=b] for n,a,b in bands}
 density=[];mix=[]
 for n,a,b in bands:
  xs=byband[n];u=sum(1/x['gold_n'] for x in xs);density.append({'band':n,'novel_q_docs':novel[n],'new_relevant':len(xs),'relevance_density':len(xs)/novel[n] if novel[n] else 0,'weighted_utility':u,'folds':sorted({x['fold'] for x in xs})});mix.append({'band':n,'1-gold':sum(x['gold_n']==1 for x in xs),'2-gold':sum(x['gold_n']==2 for x in xs),'3+':sum(x['gold_n']>=3 for x in xs),'utility_mass':u})
 rows12=[]
 for q,d in sorted(known_rows):
  x=known_rows[q,d]
  if x['rank']>200:continue
  btop=[str(z) for z in base[q]['top5']];p=pred[q];top10=x['rank']<=10;slot_candidate=top10 and d not in btop[:4];won=str(p['slot5_selected'])==d
  why='' if won else ('outside TOP10' if not top10 else 'baseline rank1-4 conflict' if d in btop[:4] else 'lost BM25 score comparison against baseline rank5')
  rows12.append({**x,'in_top10':top10,'slot5_candidate':slot_candidate,'won_slot5':won,'why_not':why})
 assert len(rows12)==12
 at200=[x for x in new if x['rank']<=200];top10=[x for x in at200 if x['rank']<=10]
 deeper=[]
 for k in [20,50,100]:
  xs=[x for x in at200 if x['rank']<=k];deeper.append({'budget':f'TOP{k}','new_relevant_coverage':len(xs),'utility_coverage':sum(1/x['gold_n'] for x in xs),'novel_candidate_volume':sum(novel[n] for n,a,b in bands if a<=k)})
 patt=Counter(x['pattern'] for x in new)
 diagnosis='USEFUL_NEW_SIGNAL_IS_PRIMARILY_DEEPER_THAN_TOP10' if len(at200) and len(top10)/len(at200)<.5 else 'USEFUL_NEW_SIGNAL_IS_CONCENTRATED_NEAR_TOP'
 out={'frozen_source':{'path':str(C),'sha256':sha(C),'queries':5600,'rows':2800000},'known66_rank_localization':{n:sum(x['rank'] and band(x['rank'])==n for x in known_rows.values()) for n,_,__ in bands}|{'not_recovered':66-len(known_rows)},'exact_12_recovered_at200':rows12,'all_new_relevant_cumulative':cumulative,'incremental_rank_bands':density,'gold_cardinality_mix':mix,'top10_exposure':{'new_relevant_at200':len(at200),'included_by_top10':len(top10),'percentage':len(top10)/len(at200)*100,'utility_at200':sum(1/x['gold_n'] for x in at200),'utility_exposed_top10':sum(1/x['gold_n'] for x in top10),'utility_percentage':sum(1/x['gold_n'] for x in top10)/sum(1/x['gold_n'] for x in at200)*100},'deeper_budget_diagnostic':deeper,'structural_patterns':dict(patt),'diagnosis':diagnosis,'safe_next_action':'DESIGN_COMPUTE_FEASIBLE_DEEP_CANDIDATE_SCORING_STRATEGY' if diagnosis.endswith('DEEPER_THAN_TOP10') else 'AUDIT_ONE_PROSPECTIVE_COMMON_SCORER_FOR_NEW_CANDIDATES','all_new_relevant_rows':new}
 (O/'full_doc_new_signal_localization_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print(json.dumps({'diagnosis':diagnosis,'new_at200':len(at200),'top10':len(top10),'sha256':sha(O/'full_doc_new_signal_localization_audit.json')}))
if __name__=='__main__':main()
