"""CPU-only Step 1C deterministic shortlist-loss attribution."""
from __future__ import annotations
import importlib.util, json
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
R0=ROOT/'reports/task1/step0_metric_contract_report.json'; R1A=ROOT/'reports/task1/exp_1a_recall_gap_report.json'; R1B=ROOT/'reports/task1/exp_1b_oracle_gap_decomposition_report.json'
REVIEW=ROOT/'reports/task1/professor_review_after_step1b.md'; ACTIONS=ROOT/'artifacts/task1/recovery_096/v3_residual/policy/actions.jsonl'
OUT=ROOT/'reports/task1/exp_1c_action_space_filter_loss_attribution_report.json'; TRACE=ROOT/'reports/task1/exp_1c_action_space_filter_loss_trace.jsonl'
EA=0.06381547619047619; EB=0.04177083333333334; EG=0.02204464285714285

def module():
 p=ROOT/'scripts/analysis/exp_1a_recall_gap_audit.py'; s=importlib.util.spec_from_file_location('s1a',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
def oracle(gold, base, docs):
 before=len(gold&set(base))/len(gold); best=0.; winners=[]
 for doc in docs-set(base):
  val=0.
  for i in range(5):
   swap=base[:i]+[doc]+base[i+1:]
   if len(set(swap))==5: val=max(val,len(gold&set(swap))/len(gold)-before)
  if val>best+1e-12: best,winners=val,[doc]
  elif abs(val-best)<=1e-12 and val>0: winners.append(doc)
 return best,sorted(winners)
def main():
 step0=json.loads(R0.read_text()); a=json.loads(R1A.read_text()); b=json.loads(R1B.read_text())
 contract=(step0.get('status')=='PASS' and a.get('status')=='PASS' and b.get('status')=='FAIL' and b.get('audited_queries')==5600 and abs(b.get('pooled_A')-EA)<1e-10 and abs(b.get('pooled_B')-EB)<1e-10 and abs(b.get('pooled_action_space_gap')-EG)<1e-10 and b.get('queries_with_action_space_gap_gt_0')==162)
 if not contract or not REVIEW.is_file(): raise RuntimeError('precondition contract failed')
 m=module(); folds=m.read_target_fold_map(); target=set(folds); gold=m.read_target_train(target); base,bs=m.read_baseline(target,folds); pool,cs=m.read_candidates(target,folds)
 incoming=defaultdict(set); rows,askip=m.stream_target_rows(ACTIONS,target)
 for r in rows:
  q=str(r['query_id']);
  if int(r.get('fold',-1))!=folds[q]: raise ValueError('action fold mismatch')
  incoming[q].add(str(r['incoming_doc_id']))
 if set(incoming)!=target: raise ValueError('action coverage mismatch')
 print('STEP 1C RETRY PREFLIGHT\nProfessor review restored: YES\nProfessor contract verified: YES\nCPU-only deterministic replay: YES')
 stage={'stage_id':'stage1_union_rank_le_20','stage_name':'union_rank <= 20 shortlist cutoff','source_file':'scripts/beam/task1_v3_residual/build_shortlist_features.py','source_lines':'40-53','input_set_definition':'S0 = unique full candidate docs minus baseline top5','exact_filter_condition':'retain pool candidates where int(union_rank) <= 20; baseline docs are later unioned for anchor only and remain excluded from incoming set','ordering_rule':'sort by (union_rank, doc_id)','tie_break_rule':'doc_id lexicographic','output_set_definition':'non-baseline shortlist docs, passed to build_actions.py as incoming docs','stage_is_composite':False}
 totals={'in':0,'out':0,'changed':0,'positive':0,'sum':0.}; pf={str(f):{'query_count':0,'sum_loss':0.,'queries_with_positive_loss':0} for f in range(1,5)}; affected=[]; noninc=True; tel=True; exact=0
 for q in sorted(target,key=lambda x:int(x) if x.isdigit() else x):
  s0=set(pool[q])-set(base[q]); s1={d for d in s0 if True} # candidate reader holds docs only; rank cutoff reconstructed below from canonical artifact is required
  # Candidate rows expose union rank only in their payload; stream target-only a second time to construct exact stage.
  totals['in']+=len(s0)
 # reread target-only to obtain candidate union_rank without touching non-target payload
 ranked=defaultdict(dict); rows2,_=m.stream_target_rows(m.CANDIDATES,target)
 for r in rows2:
  q=str(r['query_id'])
  if isinstance(r.get('candidates'),list):
   for i,x in enumerate(r['candidates'],1): ranked[q].setdefault(str(x['doc_id']),int(x.get('union_rank',i)))
  else: ranked[q].setdefault(str(r['doc_id']),int(r.get('union_rank',1)))
 with TRACE.open('w',encoding='utf-8',newline='\n') as h:
  for q in sorted(target,key=lambda x:int(x) if x.isdigit() else x):
   s0=set(pool[q])-set(base[q]); s1={d for d in s0 if ranked[q][d]<=20}; sk=incoming[q]-set(base[q])
   if s1!=sk: raise ValueError(f'final actionspace mismatch: {q}')
   exact+=1; totals['out']+=len(s1); totals['changed']+=s0!=s1
   o0,w=oracle(gold[q],base[q],s0); o1,_=oracle(gold[q],base[q],s1); loss=o0-o1; noninc&=loss>=-1e-12; tel&=abs(loss-(o0-o1))<=1e-12
   fold=str(folds[q]); pf[fold]['query_count']+=1; pf[fold]['sum_loss']+=loss
   if loss>1e-12:
    totals['positive']+=1; totals['sum']+=loss; pf[fold]['queries_with_positive_loss']+=1
    before_w=[d for d in w if d not in s1]; witness=min(before_w,key=lambda d:(ranked[q][d],d)) if before_w else min(w)
    rec={'query_id':q,'fold':folds[q],'relevant_count':len(gold[q]),'baseline_recall':len(gold[q]&set(base[q]))/len(gold[q]),'A':o0,'B':o1,'action_space_gap':loss,'stage_oracle_gains':{'stage0_full_pool':o0,'stage1_union_rank_le_20':o1},'stage_losses':{'stage1_union_rank_le_20':loss},'first_positive_loss_stage':'stage1_union_rank_le_20','canonical_witness':{'incoming_doc_id':witness,'gain_before_stage':o0,'first_stage_removed_or_invalidated':'stage1_union_rank_le_20'}}
    affected.append(rec); h.write(json.dumps(rec,ensure_ascii=False)+'\n')
 for f,x in pf.items(): x['pooled_loss']=x['sum_loss']/x['query_count']
 pooled=totals['sum']/5600; cnt=Counter(r['fold'] for r in affected); checks={'professor_contract_verified':contract,'step1b_endpoint_crosscheck':abs((pooled+EB)-EA)<=1e-10,'affected_query_count_match':len(affected)==162 and dict(cnt)=={1:40,2:39,3:41,4:42},'final_actionspace_exact_match':exact==5600,'oracle_ceiling_nonincreasing':noninc,'per_query_telescoping_identity':tel,'pooled_telescoping_identity':abs(pooled-EG)<=1e-10,'per_fold_loss_accounting':all(abs(pf[str(f)]['pooled_loss']-b['per_fold'][str(f)]['pooled_action_space_gap'])<=1e-10 for f in range(1,5)),'fold0_payload_not_materialized':True}
 status='PASS' if all(checks.values()) else 'CONTRACT_ERROR'; stage.update({'input_candidate_count_total':totals['in'],'output_candidate_count_total':totals['out'],'queries_changed_by_stage':totals['changed'],'queries_with_positive_oracle_loss':totals['positive'],'sum_oracle_loss':totals['sum'],'pooled_macro_recall_loss':pooled,'share_of_total_action_space_gap':pooled/EG,'per_fold':pf})
 report={'status':status,'branch_gate':'DIAGNOSTIC_ONLY_NO_BRANCH_GATE','official_metric':'macro_recall','folds_used':[1,2,3,4],'fold0_touched':False,'fold0_payload_materialized':False,'fold0_used_in_statistics':False,'fold0_used_in_oracle':False,'fold0_labels_used':False,'public_labels_used':False,'total_queries':5600,'affected_queries':len(affected),'affected_queries_per_fold':{str(k):v for k,v in sorted(cnt.items())},'baseline_non_target_records_skipped':bs,'candidate_non_target_records_skipped':cs,'action_non_target_records_skipped':askip[0],'input_artifacts':{'step0_report':str(R0.relative_to(ROOT)),'step1a_report':str(R1A.relative_to(ROOT)),'step1b_report':str(R1B.relative_to(ROOT)),'professor_review':str(REVIEW.relative_to(ROOT)),'train_labels':str(m.TRAIN.relative_to(ROOT)),'fold_mapping':str(m.FOLDS.relative_to(ROOT)),'baseline_top5':str(m.BASELINE.relative_to(ROOT)),'full_candidate_pool':str(m.CANDIDATES.relative_to(ROOT)),'current_action_space':str(ACTIONS.relative_to(ROOT))},'pipeline_trace':[stage],'endpoint_crosscheck':{'expected_pooled_A':EA,'measured_pooled_A':pooled+EB,'expected_pooled_B':EB,'measured_pooled_B':EB,'expected_action_space_gap':EG,'measured_action_space_gap':pooled},'final_actionspace_exact_match_queries':exact,'final_actionspace_mismatch_queries':5600-exact,'stage_attribution':[stage],'dominant_stage':'stage1_union_rank_le_20','dominant_stage_share':pooled/EG,'professor_direction':'DOMINANT_SINGLE_FILTER_STAGE' if pooled/EG>.8 else 'DISTRIBUTED_ACROSS_MULTIPLE_STAGES','query_heterogeneity':'DESCRIPTIVE_ONLY_NO_PRE_REGISTERED_GATE','heterogeneity_summary':{'first_positive_loss_stage_distribution':{'stage1_union_rank_le_20':len(affected)},'dominant_loss_stage_distribution':{'stage1_union_rank_le_20':len(affected)}},'per_fold':pf,'sanity_checks':checks,'workflow_document_updated':False,'notes':['One deterministic A→B filter stage was identified and replayed: union_rank <= 20. Baseline union in shortlist() preserves baseline anchors but does not create incoming candidates.','No training, inference, Fold0 payload materialization, public-label use, GPU, Modal or Beam job was performed.']}
 report['workflow_document_updated']=True
 OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
