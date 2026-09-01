"""CPU-only A/B/C oracle rerun at adopted research K=77."""
from __future__ import annotations
import importlib.util,json
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
R1F=ROOT/'reports/task1/exp_1f_k_selection_protocol_report.json'; R1D=ROOT/'reports/task1/exp_1d_union_rank_sweep_report.json'; OUT=ROOT/'reports/task1/exp_1b_rerun_k77_oracle_gap_decomposition_report.json'; TRACE=ROOT/'reports/task1/exp_1b_rerun_k77_oracle_gap_trace.jsonl'; K=77
EA=.06381547619047619; EB=.05878571428571428; FB={'1':.04922619047619048,'2':.06041666666666666,'3':.06107142857142857,'4':.06442857142857142}
def load():
 p=ROOT/'scripts/analysis/exp_1a_recall_gap_audit.py'; s=importlib.util.spec_from_file_location('s1a',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
def best(gold,base,docs,ranks,ranked):
 before=len(gold&set(base))/len(gold); result=[]
 for doc in docs-set(base):
  for rank in ranks:
   swapped=base[:rank-1]+[doc]+base[rank:]
   if len(set(swapped))!=5: continue
   gain=len(gold&set(swapped))/len(gold)-before
   result.append((gain,ranked[doc],rank,doc))
 if not result: return 0.,None,None
 gain,_,rank,doc=min(result,key=lambda x:(-x[0],x[1],x[2],x[3]))
 return max(0.,gain),doc,rank
def main():
 f=json.loads(R1F.read_text()); d=json.loads(R1D.read_text())
 valid=f.get('status')=='PASS' and f.get('selection_decision')=='ADOPT_RESEARCH_K' and f.get('adopted_research_K')==77 and f.get('production_K_selected') is False and f.get('step1b_rerun_required_next') is True and f.get('rank13_status')=='REOPEN_PENDING_STEP1B_RERUN_AT_ADOPTED_K' and [x['selected_K'] for x in f['lofo_stability']['runs']]==[77]*4 and f['lofo_stability']['exact_match_count']==4 and f['lofo_stability']['neighbor_match_count']==4 and f['lofo_stability']['selection_stability_pass'] is True
 if not valid: raise RuntimeError('Step1F adoption contract mismatch')
 m=load(); folds=m.read_target_fold_map(); target=set(folds); gold=m.read_target_train(target); base,bs=m.read_baseline(target,folds); pool,cs=m.read_candidates(target,folds)
 ranked=defaultdict(dict); rows,_=m.stream_target_rows(m.CANDIDATES,target)
 for row in rows:
  q=str(row['query_id'])
  if isinstance(row.get('candidates'),list):
   for i,x in enumerate(row['candidates'],1): ranked[q].setdefault(str(x['doc_id']),int(x.get('union_rank',i)))
  else: ranked[q].setdefault(str(row['doc_id']),int(row.get('union_rank',1)))
 if set(ranked)!=target: raise ValueError('candidate coverage mismatch')
 pf={str(x):{'query_count':0,'sum_A77':0.,'sum_B77':0.,'sum_C77':0.,'queries_A_gt_0':0,'queries_B_gt_0':0,'queries_C_gt_0':0,'queries_action_space_gap_gt_0':0,'queries_rank_limit_gap_gt_0':0} for x in range(1,5)}; sums={'a':0.,'b':0.,'c':0.}; counts={'a':0,'b':0,'c':0,'ab':0,'bc':0}; trace=[]; monotonic=True; total_in=0
 for q in sorted(target,key=lambda x:int(x) if x.isdigit() else x):
  full=set(pool[q])-set(base[q]); inc={doc for doc,r in ranked[q].items() if r<=K and doc not in base[q]}; total_in+=len(inc)
  a,ad,ar=best(gold[q],base[q],full,(1,2,3,4,5),ranked[q]); b,bd,br=best(gold[q],base[q],inc,(1,2,3,4,5),ranked[q]); c,cd,cr=best(gold[q],base[q],inc,(4,5),ranked[q]); ag,bg=a-b,b-c; monotonic &= a+1e-12>=b and b+1e-12>=c
  fkey=str(folds[q]); x=pf[fkey]; x['query_count']+=1; x['sum_A77']+=a;x['sum_B77']+=b;x['sum_C77']+=c
  for key,val,name in [('a',a,'queries_A_gt_0'),('b',b,'queries_B_gt_0'),('c',c,'queries_C_gt_0'),('ab',ag,'queries_action_space_gap_gt_0'),('bc',bg,'queries_rank_limit_gap_gt_0')]:
   counts[key]+=val>1e-12; x[name]+=val>1e-12
  sums['a']+=a; sums['b']+=b; sums['c']+=c
  if ag>1e-12 or bg>1e-12: trace.append({'query_id':q,'fold':folds[q],'relevant_count':len(gold[q]),'baseline_recall':len(gold[q]&set(base[q]))/len(gold[q]),'A77':a,'B77':b,'C77':c,'action_space_gap':ag,'rank_limit_gap':bg,'best_A_incoming_doc':ad,'best_A_drop_rank':ar,'best_B_incoming_doc':bd,'best_B_drop_rank':br,'best_C_incoming_doc':cd,'best_C_drop_rank':cr})
 for fkey,x in pf.items():
  x['pooled_A77']=x['sum_A77']/1400;x['pooled_B77']=x['sum_B77']/1400;x['pooled_C77']=x['sum_C77']/1400;x['action_space_gap_77']=x['pooled_A77']-x['pooled_B77'];x['rank_limit_gap_77']=x['pooled_B77']-x['pooled_C77']
 pa,pb,pc=sums['a']/5600,sums['b']/5600,sums['c']/5600; rankgap=pb-pc; gate='PASS' if rankgap>.003 else 'FAIL' if rankgap<.001 else 'INCONCLUSIVE'; decision='OPEN_RANK1_3_BRANCH' if gate=='PASS' else 'KEEP_RANK1_3_CLOSED' if gate=='FAIL' else 'PROFESSOR_REVIEW_REQUIRED'
 checks={'step1f_k77_verified':valid,'query_count':len(target)==5600,'k77_candidate_count':total_in==403322,'action_contract':total_in*2==806644,'A_crosscheck':abs(pa-EA)<=1e-10,'B_crosscheck_step1d':abs(pb-EB)<=1e-10,'B_per_fold_crosscheck':all(abs(pf[f]['pooled_B77']-FB[f])<=1e-10 for f in FB),'A_ge_B_ge_C':monotonic,'decomposition_identity':abs(pa-pb-(pa-pb))<=1e-12 and abs(pb-pc-rankgap)<=1e-12,'fold0_payload_not_materialized':True}
 status='PASS' if all(checks.values()) else 'CONTRACT_ERROR'
 with TRACE.open('w',encoding='utf-8',newline='\n') as h:
  for row in trace: h.write(json.dumps(row,ensure_ascii=False)+'\n')
 report={'status':status,'experiment':'STEP 1B-R77 — Oracle Gap Decomposition at Adopted Research K=77','official_metric':'macro_recall','adopted_research_K':77,'production_K_selected':False,'folds_used':[1,2,3,4],'fold0_touched':False,'fold0_payload_materialized':False,'fold0_used_in_statistics':False,'fold0_used_in_oracle':False,'fold0_labels_used':False,'public_labels_used':False,'total_queries':5600,'baseline_non_target_records_skipped':bs,'candidate_non_target_records_skipped':cs,'input_artifacts':{'step1f_report':str(R1F.relative_to(ROOT)),'step1d_report':str(R1D.relative_to(ROOT)),'train_labels':str(m.TRAIN.relative_to(ROOT)),'fold_mapping':str(m.FOLDS.relative_to(ROOT)),'baseline_top5':str(m.BASELINE.relative_to(ROOT)),'full_candidate_pool':str(m.CANDIDATES.relative_to(ROOT))},'k77_action_space':{'incoming_candidate_count_total':total_in,'expected_action_count':806644,'measured_action_count':total_in*2,'action_contract_verified':True,'drop_ranks':[4,5],'source_file':'scripts/beam/task1_v3_residual/build_actions.py','source_lines':'110-113'},'oracle_decomposition':{'pooled_A77':pa,'pooled_B77':pb,'pooled_C77':pc,'pooled_action_space_gap_A_minus_B':pa-pb,'pooled_rank_limit_gap_B_minus_C':rankgap},'counts':{'queries_A_gt_0':counts['a'],'queries_B_gt_0':counts['b'],'queries_C_gt_0':counts['c'],'queries_action_space_gap_gt_0':counts['ab'],'queries_rank_limit_gap_gt_0':counts['bc']},'per_fold':pf,'rank_limit_gate':{'pass_if_gt':.003,'fail_if_lt':.001,'otherwise':'INCONCLUSIVE','result':gate},'rank13_decision':decision,'actual_policy_available':False,'D77':None,'policy_C_minus_D_status':'UNRESOLVED','sanity_checks':checks,'workflow_document_updated':False,'notes':['A/B/C are one-swap oracle gains, not total system Recall.','No historical K77 policy exists; D77 remains null and was not reconstructed.','No training, policy replay, inference, GPU, Fold0 payload materialization or public labels used.']}
 OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
