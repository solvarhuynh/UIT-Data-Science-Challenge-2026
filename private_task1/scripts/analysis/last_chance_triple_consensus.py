"""CPU-only triple-consensus residual challengers for the immutable 09205 ZIP."""
from __future__ import annotations
import hashlib, json, math, sys, zipfile
from pathlib import Path
from typing import Any, Iterable
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.analysis.guarded_direct_k20_residual import (TARGET_FOLDS,build_direct_inputs,build_private_inputs,candidate_metadata,direct_outer_oof,fit_direct_model,load_and_merge_scores,load_fold_map,load_gold,load_pv1_metadata,load_pv1_worklist,metric,qkey,score_query_ids)
from private_task1.scripts.analysis.guarded_vs_direct_top5_meta_selector import (load_zip_submission,validate_submission,load_guarded,per_query_hits,class_of,summary_vs_a,rows,write_json,write_jsonl,sha256)

OUT=ROOT/'private_task1/experiments/09205_last_chance'; TEAM=ROOT/'private_task1/submissions/team_09205'
I_ZIP=TEAM/'submission_private_constrained_dual_anchor_rrf_09205.zip'; S_ZIP=TEAM/'submission_private_dual_anchor_guarded_09198.zip'
G_ZIP=ROOT/'private_task1/submissions/sprint48_guarded_direct/submission_private_guarded_direct.zip'
PRIVATE_OFFICIAL=ROOT/'private_task1/input/private-official.json'; DIRECT_OOF=ROOT/'private_task1/experiments/sprint48_guarded_direct/direct_oof_scores.jsonl'; SWAPS=ROOT/'private_task1/experiments/sprint48_guarded_direct/guarded_direct_nested_swaps.jsonl'
DIRECT_REPLAY=ROOT/'private_task1/experiments/sprint48_guarded_direct/direct_replay_summary.json'
EXPECTED_SHA='aa8ef30147500164ce23eb2ecfa6aaa57efd393d0fdc953371315f14460f4dae'; A_RECALL=.9300505952380952; B_RECALL=.9293809523809524
SEED=20260923
DELTA_NAMES=('delta_union_rrf','delta_bge_ft_minmax','delta_bge_ft_raw','delta_bge_base_minmax','delta_bge_base_raw','delta_source_support','delta_source_support_count','delta_rr_bm25','delta_rr_dense','delta_rr_knn_word','delta_candidate_rank_norm','set_overlap','membership_change_count','low_margin','high_disagreement','strong_bge_reorder','single_source_dominance')

def minmax(values:dict[str,float])->dict[str,float]:
 lo,hi=min(values.values()),max(values.values()); span=max(1e-6,hi-lo); return {k:(v-lo)/span for k,v in values.items()}
def one_swap_lock(a:list[str],b:list[str])->tuple[bool,bool,list[str],list[str]]:
 inc=list(set(b)-set(a)); drop=list(set(a)-set(b)); return len(inc)==len(drop)==1,a[:3]==b[:3],inc,drop
def rrank(meta:dict[str,Any],source:str)->float:
 rank=meta['source_ranks'].get(source); return 0.0 if rank is None else 1/(60+int(rank))
def build_record(q:str,a:list[str],b:list[str],scores:dict[str,float],meta:dict[str,dict[str,Any]],score_rows:dict[tuple[str,str],dict[str,Any]])->dict[str,Any]:
 one,lock,inc,drop=one_swap_lock(a,b); enter=inc[0] if inc else None; gone=drop[0] if drop else None
 ft=minmax({d:float(score_rows[(q,d)]['bge_ft_score']) for d in meta}); base=minmax({d:float(score_rows[(q,d)]['bge_base_score']) for d in meta})
 order=sorted(scores,key=lambda d:(-float(scores[d]),d)); rank={d:i+1 for i,d in enumerate(order)}
 def diff(fn): return 0.0 if not one else float(fn(enter)-fn(gone))
 b5,b6=float(scores[order[4]]),float(scores[order[5]])
 supports=[int(meta[d]['source_support']) for d in b]; single=sum(v<=1 for v in supports)
 movement=max(abs(rank[d]-int(meta[d]['candidate_rank'])) for d in b)
 return {'query_id':q,'one_swap':one,'top1_3_lock':lock,'incoming':enter,'dropped':gone,'features':[
  diff(lambda d:float(score_rows[(q,d)].get('rrf_score',0.0))),diff(lambda d:ft[d]),diff(lambda d:float(score_rows[(q,d)]['bge_ft_score'])),diff(lambda d:base[d]),diff(lambda d:float(score_rows[(q,d)]['bge_base_score'])),diff(lambda d:float(meta[d]['source_support'])),diff(lambda d:float(meta[d]['source_support'])),diff(lambda d:rrank(meta[d],'bm25')),diff(lambda d:rrank(meta[d],'dense')),diff(lambda d:rrank(meta[d],'knn_word')),diff(lambda d:(21-int(meta[d]['candidate_rank']))/20),len(set(a)&set(b))/5,len(set(a)^set(b))//2,int(b5-b6<=.00119048),int(sum(supports)/5<=1.6 or single>=2),int(movement>=17),int(single>=2)],'delta_union_rrf':diff(lambda d:float(score_rows[(q,d)].get('rrf_score',0.0))),'delta_bge_ft_minmax':diff(lambda d:ft[d]),'delta_candidate_rank_norm':diff(lambda d:(21-int(meta[d]['candidate_rank']))/20),'delta_retrieval_sum':diff(lambda d:rrank(meta[d],'bm25')+rrank(meta[d],'dense')+rrank(meta[d],'knn_word'))}

def apply(ids, a,b, take): return {q:list(b[q] if take(q) else a[q]) for q in ids}
def relaxed(report:dict[str,Any])->bool:
 return report['recall_delta']>0 and report['improved']>report['harmed'] and report['precision_delta']>=-.0005 and all(x['recall_delta']>=-.00075 for x in report['folds'].values()) and report['top1_3_changed']==0
def classifier_tree(): return DecisionTreeClassifier(max_depth=2,min_samples_leaf=8,class_weight={0:2.0,1:1.0},random_state=SEED)
def classifier_logit(): return make_pipeline(StandardScaler(),LogisticRegression(C=.1,class_weight={0:2.0,1:1.0},solver='liblinear',max_iter=500,random_state=SEED))
def probs_positive(model,x:np.ndarray)->np.ndarray:
 p=model.predict_proba(x); c=list(model.classes_); return p[:,c.index(1)] if 1 in c else np.zeros(len(x))

def outer_tree(records,a,b,gold,folds):
 out={q:list(a[q]) for q in folds}; audit=[]
 for f in sorted(set(folds.values())):
  train=[q for q in records if folds[q]!=f and records[q]['target']!='SAME']; test=[q for q in records if folds[q]==f]
  model=classifier_tree(); model.fit(np.asarray([records[q]['features'] for q in train]),np.asarray([records[q]['target']=='B_BETTER' for q in train],dtype=int))
  pred=model.predict(np.asarray([records[q]['features'] for q in test])); take={q:bool(v) and records[q]['one_swap'] and records[q]['top1_3_lock'] for q,v in zip(test,pred)}
  out.update(apply(test,a,b,lambda q:take[q])); audit.extend({'query_id':q,'fold':f,'take_b':bool(take[q])} for q in test)
 return out,audit
def select_logit_threshold(train,records,a,b,gold,folds):
 ids=[q for q in train if records[q]['target']!='SAME']; model=classifier_logit(); model.fit(np.asarray([records[q]['features'] for q in ids]),np.asarray([records[q]['target']=='B_BETTER' for q in ids],dtype=int)); p=probs_positive(model,np.asarray([records[q]['features'] for q in train])); prob=dict(zip(train,map(float,p)))
 candidates=[]
 for t in (.60,.70,.80):
  pred=apply(train,a,b,lambda q:prob[q]>=t and records[q]['one_swap'] and records[q]['top1_3_lock']); r=summary_vs_a(pred,{q:a[q] for q in train},{q:gold[q] for q in train},{q:folds[q] for q in train}); candidates.append({'threshold':t,**r})
 valid=[x for x in candidates if x['harmed']<=x['improved']]
 return model,(sorted(valid,key=lambda x:(-x['recall'], -x['threshold']))[0] if valid else {'threshold':None,'candidates':candidates}),candidates
def outer_logit(records,a,b,gold,folds):
 out={q:list(a[q]) for q in folds}; audit=[]; choices={}
 for f in sorted(set(folds.values())):
  train=[q for q in records if folds[q]!=f]; test=[q for q in records if folds[q]==f]; model,choice,candidates=select_logit_threshold(train,records,a,b,gold,folds); choices[f'F{f}']={'selected':choice,'candidates':candidates}
  p=probs_positive(model,np.asarray([records[q]['features'] for q in test])); take={q:choice['threshold'] is not None and v>=choice['threshold'] and records[q]['one_swap'] and records[q]['top1_3_lock'] for q,v in zip(test,p)}; out.update(apply(test,a,b,lambda q:take[q])); audit.extend({'query_id':q,'fold':f,'probability_b_better':float(v),'take_b':bool(take[q])} for q,v in zip(test,p))
 return out,audit,choices

def write_challenger(name:str,I:dict[str,list[str]],G:dict[str,list[str]],B:dict[str,list[str]],records:dict[str,dict[str,Any]],private_eligible:set[str],take:dict[str,bool],expected_ids:set[str])->dict[str,Any]:
 payload={q:list(docs) for q,docs in I.items()}; mutations=[]
 for q in sorted(private_eligible,key=qkey):
  r=records[q]
  if not take.get(q,False) or not r['one_swap'] or not r['top1_3_lock']: continue
  _,_,inc,drop=one_swap_lock(G[q],B[q]); old,new=drop[0],inc[0]
  if old not in payload[q] or payload[q].index(old)<3: continue
  proposed=list(payload[q]); proposed[proposed.index(old)]=new
  if proposed[:3]!=I[q][:3] or len(set(proposed))!=5: raise RuntimeError(f'private firewall failed {q}')
  payload[q]=proposed; mutations.append(q)
 val=validate_submission(payload,expected_ids)
 if not val['pass'] or any(payload[q]!=I[q] for q in payload if q not in set(mutations)): raise RuntimeError('private payload validation/firewall failed')
 json_path=TEAM/f'submission_private_09205_lastchance_{name}.json'; zip_path=TEAM/f'submission_private_09205_lastchance_{name}.zip'; json_path.write_text(json.dumps({q:{'answer':payload[q]} for q in sorted(payload,key=qkey)},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z: z.write(json_path,arcname='submission.json')
 return {'variant':name,'path':str(zip_path.relative_to(ROOT)).replace('\\','/'),'sha256':sha256(zip_path),'private_changed_queries':len(mutations),'validator':val,'top1_3_changed':sum(payload[q][:3]!=I[q][:3] for q in payload),'non_triple_equal_mutations':sum(q not in private_eligible for q in mutations)}

def main()->int:
 OUT.mkdir(parents=True,exist_ok=True); official=json.loads(PRIVATE_OFFICIAL.read_text(encoding='utf-8')); ids=set(map(str,official)); I,S,G=load_zip_submission(I_ZIP),load_zip_submission(S_ZIP),load_zip_submission(G_ZIP)
 anchors={'incumbent_sha':sha256(I_ZIP),'incumbent_sha_expected':EXPECTED_SHA,'I':validate_submission(I,ids),'S':validate_submission(S,ids),'G':validate_submission(G,ids)}
 if anchors['incumbent_sha']!=EXPECTED_SHA or not all(x['pass'] for x in (anchors['I'],anchors['S'],anchors['G'])): raise RuntimeError('anchor freeze validation failed')
 triple={q for q in ids if set(I[q])==set(S[q])==set(G[q])}; anchors.update({'triple_equal':len(triple),'dual_changed':sum(set(I[q])!=set(G[q]) for q in ids),'safeguard_changed':sum(set(S[q])!=set(G[q]) for q in ids),'i_vs_s_changed':sum(set(I[q])!=set(S[q]) for q in ids)})
 allfold=load_fold_map(); folds={q:f for q,f in allfold.items() if f in TARGET_FOLDS}; goldall=load_gold(); gold={q:goldall[q] for q in folds}; _,_,docs=load_pv1_metadata(); _,work=load_pv1_worklist(); recon,_=load_and_merge_scores(work); scores={(str(x['query_id']),str(x['document_id'])):x for x in recon}; meta=candidate_metadata(docs,scores); _,_,_,cache=build_direct_inputs(docs,scores)
 direct_scores,direct,_=direct_outer_oof(folds,gold,cache); saved={str(x['query_id']):x for x in rows(DIRECT_OOF)}
 if any(direct[q][:5]!=saved[q]['direct_top5'] for q in folds): raise RuntimeError('expert B exact OOF reproduction failed')
 A=load_guarded(saved); B={q:list(saved[q]['direct_top5']) for q in saved}; am,bm=metric(A,gold),metric(B,gold)
 if abs(am['recall']-A_RECALL)>1e-14 or abs(bm['recall']-B_RECALL)>1e-14: raise RuntimeError('expert metrics did not reproduce')
 replay=json.loads(DIRECT_REPLAY.read_text(encoding='utf-8'))
 exact_b=(abs(float(replay['proposed']['recall'])-B_RECALL)<=1e-14 and abs(float(replay['proposed']['precision'])-.1980357142857143)<=1e-14 and replay['queries_improved']==68 and replay['queries_harmed']==42 and all(abs(float(replay['delta_by_fold'][f]['proposed_recall'])-v)<=1e-14 for f,v in {'F1':.9330952380952381,'F2':.9336904761904762,'F3':.9230952380952381,'F4':.9276428571428571}.items()))
 if not exact_b: raise RuntimeError('expert B historical detail reproduction failed')
 records={q:build_record(q,A[q],B[q],direct_scores[q],meta[q],scores) for q in folds}
 for q in records: records[q]['fold']=folds[q]; records[q]['target']=class_of(A[q],B[q],set(map(str,gold[q].get('answer',[]))))
 write_jsonl(OUT/'validation_disagreement_table.jsonl',(records[q] for q in sorted(records,key=qkey)))
 one=[q for q in records if records[q]['one_swap']]; oracle={q:list(A[q]) for q in folds}
 for q in one:
  if records[q]['target']=='B_BETTER': oracle[q]=list(B[q])
 one_oracle=summary_vs_a(oracle,A,gold,folds); anatomy={'one_swap_queries':len(one),'one_swap_b_better':sum(records[q]['target']=='B_BETTER' for q in one),'one_swap_a_better':sum(records[q]['target']=='A_BETTER' for q in one),'one_swap_same':sum(records[q]['target']=='SAME' for q in one),'one_swap_oracle':one_oracle}
 if one_oracle['recall_delta']<=0:
  write_json(OUT/'last_chance_report.json',{'status':'NO_EVIDENCE_FOR_FURTHER_PRIVATE_MUTATION','anchors':anchors,'expert_a_recall':am['recall'],'expert_b_recall':bm['recall'],'anatomy':anatomy,'reason':'one-swap oracle not positive','gpu_runs':0,'modal_gpu_runs':0}); return 0
 c1=apply(folds,A,B,lambda q:records[q]['one_swap'] and records[q]['top1_3_lock'] and records[q]['delta_union_rrf']>0 and records[q]['delta_bge_ft_minmax']>0 and records[q]['delta_candidate_rank_norm']>0 and records[q]['delta_retrieval_sum']>0); c1r=summary_vs_a(c1,A,gold,folds)
 c2,c2audit=outer_tree(records,A,B,gold,folds); c2r=summary_vs_a(c2,A,gold,folds)
 c3,c3audit,c3choices=outer_logit(records,A,B,gold,folds); c3r=summary_vs_a(c3,A,gold,folds)
 variants={'C1_pareto':c1r,'C2_tree2':c2r,'C3_logit':c3r}; gates={k:relaxed(v) for k,v in variants.items()}; challengers=[]
 if any(gates.values()):
  # Final Expert B only becomes necessary after at least one label-only OOF gate passes.
  final=fit_direct_model(sorted(folds,key=qkey),cache,gold); pd,pred=build_private_inputs(); _,_,_,pcache=build_direct_inputs(pd,pred); pscores,pranks=score_query_ids(final,sorted(pd,key=qkey),pcache); pmeta=candidate_metadata(pd,pred); PB={q:pranks[q][:5] for q in pranks}; prec={q:build_record(q,G[q],PB[q],pscores[q],pmeta[q],pred) for q in PB}
  for q in prec: prec[q]['target']='UNLABELED'
  write_json(OUT/'private_expert_b_top5.json',{'queries':len(PB),'top5':PB})
  takes={}
  takes['C1_pareto']={q:prec[q]['one_swap'] and prec[q]['top1_3_lock'] and prec[q]['delta_union_rrf']>0 and prec[q]['delta_bge_ft_minmax']>0 and prec[q]['delta_candidate_rank_norm']>0 and prec[q]['delta_retrieval_sum']>0 for q in PB}
  train=[q for q in records if records[q]['target']!='SAME']; tree=classifier_tree(); tree.fit(np.asarray([records[q]['features'] for q in train]),np.asarray([records[q]['target']=='B_BETTER' for q in train],dtype=int)); tv=tree.predict(np.asarray([prec[q]['features'] for q in PB])); takes['C2_tree2']={q:bool(v) and prec[q]['one_swap'] and prec[q]['top1_3_lock'] for q,v in zip(PB,tv)}
  lm,choice,_=select_logit_threshold(list(records),records,A,B,gold,folds); pv=probs_positive(lm,np.asarray([prec[q]['features'] for q in PB])); takes['C3_logit']={q:choice['threshold'] is not None and v>=choice['threshold'] and prec[q]['one_swap'] and prec[q]['top1_3_lock'] for q,v in zip(PB,pv)}
  for name in sorted(k for k,v in gates.items() if v): challengers.append(write_challenger(name,I,G,PB,prec,triple,takes[name],ids))
 report={'status':'LAST_CHANCE_CHALLENGERS_READY' if challengers else 'NO_EVIDENCE_FOR_FURTHER_PRIVATE_MUTATION','anchors':anchors,'expert_a_recall':am['recall'],'expert_b_recall':bm['recall'],'expert_b_exact_reproduction':{'pass':exact_b,'precision':replay['proposed']['precision'],'fold_recalls':{f:replay['delta_by_fold'][f]['proposed_recall'] for f in ('F1','F2','F3','F4')},'improved':replay['queries_improved'],'harmed':replay['queries_harmed']},'anatomy':anatomy,'variants':variants,'gates':gates,'c2_audit_rows':len(c2audit),'c3_audit_rows':len(c3audit),'c3_thresholds':c3choices,'challengers':sorted(challengers,key=lambda x:-variants[x['variant']]['recall_delta']),'fold0_used':False,'public_labels_used':False,'private_labels_used':False,'gpu_runs':0,'modal_gpu_runs':0}
 write_json(OUT/'last_chance_report.json',report); write_jsonl(OUT/'c2_oof_predictions.jsonl',c2audit); write_jsonl(OUT/'c3_oof_predictions.jsonl',c3audit); return 0
if __name__=='__main__': raise SystemExit(main())
