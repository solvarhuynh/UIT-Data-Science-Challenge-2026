"""CPU-only private-matched action model and blind semantic-review package."""
from __future__ import annotations
import hashlib,json,math,re,sys,zipfile
from collections import Counter,defaultdict
from pathlib import Path
from typing import Any
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.analysis.guarded_direct_k20_residual import TARGET_FOLDS,build_private_inputs,candidate_metadata,load_and_merge_scores,load_fold_map,load_gold,load_pv1_metadata,load_pv1_worklist,metric,qkey
from private_task1.scripts.analysis.guarded_vs_direct_top5_meta_selector import load_zip_submission,validate_submission,load_guarded,rows,write_json,write_jsonl,sha256,summary_vs_a

OUT=ROOT/'private_task1/experiments/09205_final_semantic_microsurgery'; TEAM=ROOT/'private_task1/submissions/team_09205'
I_ZIP=TEAM/'submission_private_constrained_dual_anchor_rrf_09205.zip'; S_ZIP=TEAM/'submission_private_dual_anchor_guarded_09198.zip'; G_ZIP=ROOT/'private_task1/submissions/sprint48_guarded_direct/submission_private_guarded_direct.zip'; DIRECT_OOF=ROOT/'private_task1/experiments/sprint48_guarded_direct/direct_oof_scores.jsonl'; SWAPS=ROOT/'private_task1/experiments/sprint48_guarded_direct/guarded_direct_nested_swaps.jsonl'; PRIVATE_OFFICIAL=ROOT/'private_task1/input/private-official.json'; CHUNKS=ROOT/'data/processed_pv1/chunks'
EXPECTED='aa8ef30147500164ce23eb2ecfa6aaa57efd393d0fdc953371315f14460f4dae'; SEED=20260923; THRESHOLDS=(.5,.6,.7,.8)
LABELS={'HARM':0,'NEUTRAL':1,'BENEFIT':2}; UTILITY_WEIGHT={0:2.5,1:.25,2:1.0}

FEATURES=('query_words','low_margin','high_disagreement','strong_bge_reorder','single_source_dominance','top5_top6_rrf_margin','top4_top5_rrf_margin','top5_mean_support','top5_single_source_count','delta_union_rrf','delta_candidate_rank','delta_bge_ft_raw','delta_bge_ft_minmax','delta_bge_base_raw','delta_bge_base_minmax','delta_source_support','delta_source_support_count','delta_rr_dense','delta_rr_bm25','delta_rr_knn_word','incoming_dense_rank','incoming_bm25_rank','incoming_knn_rank','incoming_bge_rank','dropped_dense_rank','dropped_bm25_rank','dropped_knn_rank','dropped_bge_rank','incoming_dense_missing','incoming_bm25_missing','incoming_knn_missing','dropped_dense_missing','dropped_bm25_missing','dropped_knn_missing','incoming_current_membership','dropped_current_membership','incoming_guarded_membership','dropped_guarded_membership')

def mm(vals:dict[str,float])->dict[str,float]:
 lo,hi=min(vals.values()),max(vals.values());s=max(1e-6,hi-lo);return{k:(v-lo)/s for k,v in vals.items()}
def rr(meta:dict[str,Any],name:str)->float:
 v=meta['source_ranks'].get(name);return 0.0 if v is None else 1/(60+int(v))
def rankv(meta:dict[str,Any],name:str)->float:return float(meta['source_ranks'].get(name,1000))
def question_text(row:dict[str,Any])->str:return str(row.get('question') or row.get('query') or row.get('text') or '')

def build_actions(q:str,current:list[str],guarded:list[str],docs:dict[str,dict[str,Any]],meta:dict[str,dict[str,Any]],scores:dict[tuple[str,str],dict[str,Any]],question:str)->list[dict[str,Any]]:
 ft=mm({d:float(scores[(q,d)]['bge_ft_score']) for d in docs});base=mm({d:float(scores[(q,d)]['bge_base_score']) for d in docs}); support=[int(meta[d]['source_support']) for d in current]
 rrf={d:float(docs[d].get('rrf_score',sum(rr(meta[d],x) for x in ('dense','bm25','knn_word')))) for d in docs}; ordered=sorted(docs,key=lambda d:(-rrf[d],int(meta[d]['candidate_rank']),d)); outside=[d for d in ordered if d not in set(current)]; sixth=outside[0] if outside else current[-1]
 bge_order=sorted(docs,key=lambda d:(-float(scores[(q,d)]['bge_ft_score']),-float(scores[(q,d)]['bge_base_score']),int(meta[d]['candidate_rank']),d));brank={d:i+1 for i,d in enumerate(bge_order)}
 low=int(abs(rrf[current[4]]-rrf[sixth])<=.00119048);single=sum(v<=1 for v in support);high=int(sum(support)/5<=1.6 or single>=2);strong=int(max(abs(brank[d]-int(meta[d]['candidate_rank'])) for d in current)>=17)
 qbase=[len(question.split()),low,high,strong,int(single>=2),rrf[current[4]]-rrf[sixth],rrf[current[3]]-rrf[current[4]],sum(support)/5,single]
 result=[]
 for pos in (3,4):
  drop=current[pos]
  for inc in outside:
   im,dm=meta[inc],meta[drop]
   def delta(fn):return float(fn(inc)-fn(drop))
   f=qbase+[delta(lambda d:rrf[d]),float(dm['candidate_rank'])-float(im['candidate_rank']),delta(lambda d:float(scores[(q,d)]['bge_ft_score'])),delta(lambda d:ft[d]),delta(lambda d:float(scores[(q,d)]['bge_base_score'])),delta(lambda d:base[d]),delta(lambda d:float(meta[d]['source_support'])),delta(lambda d:float(meta[d]['source_support'])),delta(lambda d:rr(meta[d],'dense')),delta(lambda d:rr(meta[d],'bm25')),delta(lambda d:rr(meta[d],'knn_word')),rankv(im,'dense'),rankv(im,'bm25'),rankv(im,'knn_word'),float(brank[inc]),rankv(dm,'dense'),rankv(dm,'bm25'),rankv(dm,'knn_word'),float(brank[drop]),int('dense' not in im['source_ranks']),int('bm25' not in im['source_ranks']),int('knn_word' not in im['source_ranks']),int('dense' not in dm['source_ranks']),int('bm25' not in dm['source_ranks']),int('knn_word' not in dm['source_ranks']),int(inc in current),int(drop in current),int(inc in guarded),int(drop in guarded)]
   result.append({'query_id':q,'drop_position':pos+1,'dropped_doc':drop,'incoming_doc':inc,'features':f,'flags':{'LOW_MARGIN':bool(low),'HIGH_DISAGREEMENT':bool(high),'STRONG_BGE_REORDER':bool(strong),'SINGLE_SOURCE_DOMINANCE':bool(single>=2)},'retrieval':{'incoming':{'candidate_rank':int(im['candidate_rank']),'dense_rank':im['source_ranks'].get('dense'),'bm25_rank':im['source_ranks'].get('bm25'),'knn_rank':im['source_ranks'].get('knn_word'),'bge_rank':brank[inc],'bge_ft':float(scores[(q,inc)]['bge_ft_score']),'bge_base':float(scores[(q,inc)]['bge_base_score']),'support':int(im['source_support'])},'dropped':{'candidate_rank':int(dm['candidate_rank']),'dense_rank':dm['source_ranks'].get('dense'),'bm25_rank':dm['source_ranks'].get('bm25'),'knn_rank':dm['source_ranks'].get('knn_word'),'bge_rank':brank[drop],'bge_ft':float(scores[(q,drop)]['bge_ft_score']),'bge_base':float(scores[(q,drop)]['bge_base_score']),'support':int(dm['source_support'])}}})
 return result

def hgb():return HistGradientBoostingClassifier(max_depth=3,learning_rate=.05,max_iter=150,l2_regularization=1.0,random_state=SEED)
def action_scores(model,actions):
 p=model.predict_proba(np.asarray([x['features'] for x in actions],dtype=float));cols={int(v):i for i,v in enumerate(model.classes_)}
 for x,row in zip(actions,p):x['p_benefit']=float(row[cols[2]]);x['p_neutral']=float(row[cols[1]]);x['p_harm']=float(row[cols[0]]);x['action_score']=x['p_benefit']-2.5*x['p_harm']
def best_by_query(actions,threshold):
 best={}
 for x in actions:
  if x['action_score']<threshold or x['p_benefit']<2*x['p_harm']:continue
  q=x['query_id'];key=(x['action_score'],-x['p_harm'],x['p_benefit'],-int(x['drop_position']),x['incoming_doc'])
  if q not in best or key>best[q][0]:best[q]=(key,x)
 return {q:v[1] for q,v in best.items()}
def predictions(base,chosen):
 out={q:list(v) for q,v in base.items()}
 for q,x in chosen.items():
  out[q][int(x['drop_position'])-1]=x['incoming_doc']
  if len(set(out[q]))!=5 or out[q][:3]!=base[q][:3]:raise RuntimeError(f'action contract {q}')
 return out
def weighted_delta(chosen,gold,qweights):
 num=den=0.0
 for q,w in qweights.items():
  answer=set(map(str,gold[q].get('answer',[])));d=0 if q not in chosen else (int(chosen[q]['incoming_doc'] in answer)-int(chosen[q]['dropped_doc'] in answer))/max(1,len(answer));num+=w*d;den+=w
 return num/den
def evaluate(base,chosen,gold,folds,qweights):
 pred=predictions(base,chosen);r=summary_vs_a(pred,base,gold,folds);r['weighted_recall_delta']=weighted_delta(chosen,gold,qweights);r['net_relevant_doc_gain']=r['improved']-r['harmed'];r['actions_by_flag']={name:sum(x['flags'][name] for x in chosen.values()) for name in ('LOW_MARGIN','HIGH_DISAGREEMENT','STRONG_BGE_REORDER','SINGLE_SOURCE_DOMINANCE')};return r
def choose_threshold(train_actions,base,gold,folds,qweights):
 candidates=[]
 for t in THRESHOLDS:
  c=best_by_query(train_actions,t);r=evaluate({q:base[q] for q in qweights},c,{q:gold[q] for q in qweights},{q:folds[q] for q in qweights},qweights);ok=r['weighted_recall_delta']>=0 and r['harmed']<=.35*r['improved'] and r['recall_delta']>=0;candidates.append({'threshold':t,'valid':ok,**r})
 valid=[x for x in candidates if x['valid']];return (sorted(valid,key=lambda x:(-x['weighted_recall_delta'],-x['threshold']))[0] if valid else None),candidates

def excerpt(doc:str,selected:list[str],question:str)->dict[str,Any]:
 path=CHUNKS/f'{doc}.jsonl';chunks=list(rows(path)) if path.is_file() else []
 chosen=[x for x in chunks if str(x.get('chunk_id')) in set(selected)]
 if not chosen:
  terms={w.lower() for w in re.findall(r'\w+',question,re.UNICODE) if len(w)>3};chosen=sorted(chunks,key=lambda x:-len(terms&{w.lower() for w in re.findall(r'\w+',str(x.get('text','')),re.UNICODE)}))[:1]
 x=chosen[0] if chosen else {};text=str(x.get('text','')).strip();return {'document_id':doc,'title':x.get('law_name') or (chunks[0].get('law_name') if chunks else None),'chunk_id':x.get('chunk_id'),'excerpt':text[:2500]}

def main():
 OUT.mkdir(parents=True,exist_ok=True);private_raw=json.loads(PRIVATE_OFFICIAL.read_text(encoding='utf-8'));ids=set(map(str,private_raw));I,S,G=load_zip_submission(I_ZIP),load_zip_submission(S_ZIP),load_zip_submission(G_ZIP);anchor={'sha':sha256(I_ZIP),'validator':validate_submission(I,ids)}
 if anchor['sha']!=EXPECTED or not anchor['validator']['pass']:raise RuntimeError('incumbent freeze failed')
 mutable={q for q in ids if set(I[q])==set(S[q])==set(G[q])};immutable=ids-mutable
 folds_all=load_fold_map();folds={q:f for q,f in folds_all.items() if f in TARGET_FOLDS};goldall=load_gold();gold={q:goldall[q] for q in folds};saved={str(x['query_id']):x for x in rows(DIRECT_OOF)};A=load_guarded(saved)
 _,_,vdocslist=load_pv1_metadata();_,work=load_pv1_worklist();vrecon,_=load_and_merge_scores(work);vscores={(str(x['query_id']),str(x['document_id'])):x for x in vrecon};vmeta=candidate_metadata(vdocslist,vscores);vdocs={q:{str(x['document_id']):x for x in vals} for q,vals in vdocslist.items()}
 pdocslist,pscores=build_private_inputs();pmeta=candidate_metadata(pdocslist,pscores);pdocs={q:{str(x['document_id']):x for x in vals} for q,vals in pdocslist.items()}
 vactions=[]
 for q in sorted(folds,key=qkey):vactions.extend(build_actions(q,A[q],A[q],vdocs[q],vmeta[q],vscores,question_text(gold[q])))
 pactions=[]
 for q in sorted(mutable,key=qkey):pactions.extend(build_actions(q,I[q],G[q],pdocs[q],pmeta[q],pscores,question_text(private_raw[q])))
 X=np.asarray([x['features'] for x in vactions+pactions],float);y=np.asarray([0]*len(vactions)+[1]*len(pactions));xt,xv,yt,yv=train_test_split(X,y,test_size=.2,random_state=SEED,stratify=y);domain=make_pipeline(StandardScaler(),LogisticRegression(C=1.0,class_weight='balanced',solver='liblinear',max_iter=500,random_state=SEED));domain.fit(xt,yt);auc=float(roc_auc_score(yv,domain.predict_proba(xv)[:,1]));domain.fit(X,y);pv=domain.predict_proba(np.asarray([x['features'] for x in vactions]))[:,1];weights=np.clip(pv/np.maximum(1e-9,1-pv),.25,4.0);weights/=weights.mean()
 for x,w in zip(vactions,weights):x['domain_weight']=float(w);ans=set(map(str,gold[x['query_id']].get('answer',[])));d=int(x['incoming_doc'] in ans)-int(x['dropped_doc'] in ans);x['label']='BENEFIT' if d>0 else 'HARM' if d<0 else 'NEUTRAL';x['label_id']=LABELS[x['label']]
 qw_sum=defaultdict(float);qw_count=Counter()
 for x in vactions:qw_sum[x['query_id']]+=x['domain_weight'];qw_count[x['query_id']]+=1
 qweights={q:float(qw_sum[q]/qw_count[q]) for q in folds}
 write_json(OUT/'domain_classifier_report.json',{'domain_auc':auc,'validation_actions':len(vactions),'private_actions':len(pactions),'clip':[.25,4.0],'validation_weight_mean':float(weights.mean()),'validation_weight_min':float(weights.min()),'validation_weight_max':float(weights.max()),'feature_names':list(FEATURES),'private_labels_used':False})
 counts=Counter(x['label'] for x in vactions);weighted={k:sum(x['domain_weight'] for x in vactions if x['label']==k) for k in LABELS}
 oof_chosen={};outer={}
 for f in sorted(set(folds.values())):
  train=[x for x in vactions if folds[x['query_id']]!=f];test=[x for x in vactions if folds[x['query_id']]==f];model=hgb();model.fit(np.asarray([x['features'] for x in train]),np.asarray([x['label_id'] for x in train]),sample_weight=np.asarray([x['domain_weight']*UTILITY_WEIGHT[x['label_id']] for x in train]));action_scores(model,train);choice,candidates=choose_threshold(train,A,gold,folds,{q:qweights[q] for q in folds if folds[q]!=f});threshold=None if choice is None else choice['threshold'];action_scores(model,test);selected={} if threshold is None else best_by_query(test,threshold);oof_chosen.update(selected);outer[f'F{f}']={'selected_threshold':threshold,'candidates':candidates,'heldout_actions':len(test),'selected_actions':len(selected)}
 oof=evaluate(A,oof_chosen,gold,folds,qweights);gate=oof['recall_delta']>0 and oof['weighted_recall_delta']>0 and oof['improved']>oof['harmed'] and oof['harmed']<=max(2,oof['improved']//2) and all(x['recall_delta']>=-.001 for x in oof['folds'].values()) and oof['top1_3_changed']==0
 write_json(OUT/'oof_private_weighted_report.json',{'action_counts':dict(counts),'weighted_action_counts':weighted,'outer_folds':outer,'oof':oof,'gate':'PASS' if gate else 'FAIL','fold0_used':False,'public_labels_used':False,'private_labels_used':False})
 final=hgb();final.fit(np.asarray([x['features'] for x in vactions]),np.asarray([x['label_id'] for x in vactions]),sample_weight=np.asarray([x['domain_weight']*UTILITY_WEIGHT[x['label_id']] for x in vactions]));action_scores(final,vactions);choice,cands=choose_threshold(vactions,A,gold,folds,qweights);threshold=.8 if choice is None else choice['threshold'];action_scores(final,pactions)
 # Semantic review is intentionally allowed even when the statistical gate
 # fails. Retain the highest-scoring legal action per mutable Private query;
 # threshold/model-consensus is enforced later by the semantic gate.
 best={}
 for x in pactions:
  q=x['query_id'];key=(x['action_score'],-x['p_harm'],x['p_benefit'],-int(x['drop_position']),x['incoming_doc'])
  if q not in best or key>best[q][0]:best[q]=(key,x)
 best={q:v[1] for q,v in best.items()};top=sorted(best.values(),key=lambda x:(-x['action_score'],x['p_harm'],-x['p_benefit'],qkey(x['query_id'])))[:30]
 for x in top:
  q=x['query_id'];x['question']=question_text(private_raw[q]);x['dropped_evidence']=excerpt(x['dropped_doc'],pscores[(q,x['dropped_doc'])].get('selected_chunk_ids',[]),x['question']);x['incoming_evidence']=excerpt(x['incoming_doc'],pscores[(q,x['incoming_doc'])].get('selected_chunk_ids',[]),x['question']);x['semantic_status']='PENDING_TWO_PASS_BLIND_REVIEW'
 write_jsonl(OUT/'private_top30_actions.jsonl',top);write_json(OUT/'pre_semantic_report.json',{'incumbent':anchor,'mutable':len(mutable),'immutable':len(immutable),'domain_auc':auc,'oof':oof,'statistical_gate':'PASS' if gate else 'FAIL','final_threshold':threshold,'private_scored_queries':len(best),'top30':len(top),'gpu_runs':0,'modal_runs':0,'private_labels_used':False})
 print(json.dumps({'status':'READY_FOR_SEMANTIC_REVIEW','mutable':len(mutable),'immutable':len(immutable),'domain_auc':auc,'oof_delta':oof['recall_delta'],'weighted_delta':oof['weighted_recall_delta'],'improved':oof['improved'],'harmed':oof['harmed'],'gate':'PASS' if gate else 'FAIL','top30':len(top)},indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
