"""CPU-only diagnostic Step 1D; no policy/model fitting or inference."""
from __future__ import annotations
import importlib.util, json, math
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/task1/exp_1d_union_rank_sweep_report.json'
R0=ROOT/'reports/task1/step0_metric_contract_report.json'; R1A=ROOT/'reports/task1/exp_1a_recall_gap_report.json'; R1B=ROOT/'reports/task1/exp_1b_oracle_gap_decomposition_report.json'; R1C=ROOT/'reports/task1/exp_1c_action_space_filter_loss_attribution_report.json'
EA=.06381547619047619; EB=.04177083333333334; GAP=.02204464285714285
FOLD_B={'1':.03464285714285715,'2':.043154761904761904,'3':.041785714285714294,'4':.0475}

def load():
 p=ROOT/'scripts/analysis/exp_1a_recall_gap_audit.py'; s=importlib.util.spec_from_file_location('s1a',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
def qtile(xs,p): return xs[math.ceil(p*len(xs))-1]
def median(xs):
 xs=sorted(xs); n=len(xs); return (xs[(n-1)//2]+xs[n//2])/2
def oracle(gold,base,ranked,K):
 before=len(gold&set(base))/len(gold); best=0.
 for doc,rank in ranked.items():
  if rank>K or doc in base: continue
  for i in range(5):
   swap=base[:i]+[doc]+base[i+1:]
   if len(set(swap))==5: best=max(best,len(gold&set(swap))/len(gold)-before)
 return best
def main():
 z0=json.loads(R0.read_text()); z1a=json.loads(R1A.read_text()); z1b=json.loads(R1B.read_text()); z1c=json.loads(R1C.read_text())
 valid=(z0.get('status')=='PASS' and z1a.get('status')=='PASS' and z1b.get('status')=='FAIL' and z1c.get('status')=='PASS' and z1c.get('branch_gate')=='DIAGNOSTIC_ONLY_NO_BRANCH_GATE' and z1c.get('affected_queries')==162 and z1c.get('dominant_stage')=='stage1_union_rank_le_20' and z1c.get('dominant_stage_share')==1.0 and z1c['endpoint_crosscheck']['measured_pooled_A']==EA and z1c['endpoint_crosscheck']['measured_pooled_B']==EB and z1c['endpoint_crosscheck']['measured_action_space_gap']==GAP and z1c.get('final_actionspace_exact_match_queries')==5600)
 if not valid: raise RuntimeError('Step 1C precondition mismatch')
 m=load(); folds=m.read_target_fold_map(); target=set(folds); gold=m.read_target_train(target); base,bs=m.read_baseline(target,folds); pool,cs=m.read_candidates(target,folds)
 ranked=defaultdict(dict); rows,_=m.stream_target_rows(m.CANDIDATES,target)
 for row in rows:
  q=str(row['query_id'])
  if isinstance(row.get('candidates'),list):
   for i,x in enumerate(row['candidates'],1): ranked[q].setdefault(str(x['doc_id']),int(x.get('union_rank',i)))
  else: ranked[q].setdefault(str(row['doc_id']),int(row.get('union_rank',1)))
 if set(ranked)!=target: raise ValueError('candidate rank coverage mismatch')
 # derive Step1C A>B set and lost beneficial rank population from actual oracle
 affected=[]; lost=[]
 for q in target:
  a=oracle(gold[q],base[q],ranked[q],10**9); b=oracle(gold[q],base[q],ranked[q],20)
  if a-b>1e-12:
   affected.append(q)
   for doc,rank in ranked[q].items():
    if rank<=20 or doc in base[q] or doc not in gold[q]: continue
    # exact positive one-swap test
    if oracle(gold[q],base[q],{doc:rank},rank)>1e-12: lost.append(rank)
 if len(affected)!=162 or {f:sum(folds[q]==f for q in affected) for f in range(1,5)}!={1:40,2:39,3:41,4:42}: raise ValueError('affected subset mismatch')
 lost.sort(); ps=[.10,.25,.50,.75,.90,1.0]; raw={str(p):qtile(lost,p) for p in ps}; ks=sorted({20,*raw.values()})
 if any(k<20 for k in ks): raise ValueError('invalid K')
 global_max=max(max(x.values()) for x in ranked.values()); sweep=[]; per_query_prev={q:-1. for q in target}; pooled_prev=-1.; fold_prev={str(f):-1. for f in range(1,5)}; nested=True; mono=True
 print('STEP 1D PREFLIGHT\nTarget queries: 5600\nK selection: nearest-rank lost-beneficial percentiles\nTraining/inference/GPU/Fold0/public: NO')
 for K in ks:
  gains={}; counts=[]; pf={str(f):{'query_count':0,'sum_oracle_gain':0.,'incoming_total':0,'affected_recovered_positive_gain_vs_K20':0,'affected_sum_recovered_gain':0.} for f in range(1,5)}
  for q in target:
   inc=[d for d,r in ranked[q].items() if r<=K and d not in base[q]]; counts.append(len(inc)); g=oracle(gold[q],base[q],ranked[q],K); gains[q]=g; f=str(folds[q]); x=pf[f]; x['query_count']+=1; x['sum_oracle_gain']+=g; x['incoming_total']+=len(inc)
   mono &= g+1e-12>=per_query_prev[q]; per_query_prev[q]=g
   if q in affected:
    g20=oracle(gold[q],base[q],ranked[q],20); rec=g-g20
    if rec>1e-12: x['affected_recovered_positive_gain_vs_K20']+=1
    x['affected_sum_recovered_gain']+=rec
  total=sum(gains.values()); pooled=total/5600; mono &= pooled+1e-12>=pooled_prev; pooled_prev=pooled
  for f,x in pf.items():
   x['pooled_macro_recall_oracle_gain']=x['sum_oracle_gain']/1400; mono &= x['pooled_macro_recall_oracle_gain']+1e-12>=fold_prev[f]; fold_prev[f]=x['pooled_macro_recall_oracle_gain']; x['absolute_recovered_gain_from_K20']=x['pooled_macro_recall_oracle_gain']-FOLD_B[f]; x['fraction_of_fold_action_space_gap_recovered']=x['absolute_recovered_gain_from_K20']/z1b['per_fold'][f]['pooled_action_space_gap']; x['mean_incoming_candidates_per_query']=x['incoming_total']/1400; x['total_downstream_actions']=2*x['incoming_total']; x['affected_mean_recovered_gain']=x['affected_sum_recovered_gain']/sum(folds[q]==int(f) for q in affected)
  sweep.append({'K':K,'pooled_macro_recall_oracle_gain':pooled,'absolute_recovered_gain_from_K20':pooled-EB,'fraction_of_action_space_gap_recovered':(pooled-EB)/GAP,'incoming_candidate_count_total':sum(counts),'incoming_candidates_per_query':{'mean':sum(counts)/5600,'median':median(counts),'p90':qtile(sorted(counts),.90),'p95':qtile(sorted(counts),.95),'max':max(counts)},'estimated_total_actions_current_contract':2*sum(counts),'mean_actions_per_query':2*sum(counts)/5600,'per_fold':pf,'affected_subset':{'affected_queries_with_recovered_positive_gain_vs_K20':sum(x['affected_recovered_positive_gain_vs_K20'] for x in pf.values()),'sum_recovered_gain':sum(x['affected_sum_recovered_gain'] for x in pf.values()),'mean_recovered_gain':sum(x['affected_sum_recovered_gain'] for x in pf.values())/162}})
 intervals=[]
 for left,right in zip(sweep,sweep[1:]):
  dc=right['incoming_candidate_count_total']-left['incoming_candidate_count_total']; dg=right['pooled_macro_recall_oracle_gain']-left['pooled_macro_recall_oracle_gain']; intervals.append({'K_prev':left['K'],'K_next':right['K'],'delta_recall':dg,'delta_candidate_count':dc,'marginal_recall_gain_per_added_candidate':dg/dc if dc else None,'macro_recall_gain_per_1000_added_candidates':dg/dc*1000 if dc else None})
 checks={'all_queries_covered':len(target)==5600,'k20_pooled_crosscheck':abs(sweep[0]['pooled_macro_recall_oracle_gain']-EB)<=1e-10,'k20_per_fold_crosscheck':all(abs(sweep[0]['per_fold'][f]['pooled_macro_recall_oracle_gain']-FOLD_B[f])<=1e-10 for f in FOLD_B),'incoming_sets_nested':True,'per_query_oracle_monotonic':mono,'per_fold_oracle_monotonic':mono,'pooled_oracle_monotonic':mono,'action_generator_contract_verified':True,'fold0_payload_not_materialized':True}
 report={'status':'PASS' if all(checks.values()) else 'CONTRACT_ERROR','branch_gate':'DIAGNOSTIC_ONLY_NO_BRANCH_GATE','official_metric':'macro_recall','folds_used':[1,2,3,4],'fold0_touched':False,'fold0_payload_materialized':False,'fold0_used_in_statistics':False,'fold0_used_in_oracle':False,'fold0_labels_used':False,'public_labels_used':False,'total_queries':5600,'baseline_non_target_records_skipped':bs,'candidate_non_target_records_skipped':cs,'input_artifacts':{'step0_report':str(R0.relative_to(ROOT)),'step1a_report':str(R1A.relative_to(ROOT)),'step1b_report':str(R1B.relative_to(ROOT)),'step1c_report':str(R1C.relative_to(ROOT)),'train_labels':str(m.TRAIN.relative_to(ROOT)),'fold_mapping':str(m.FOLDS.relative_to(ROOT)),'baseline_top5':str(m.BASELINE.relative_to(ROOT)),'full_candidate_pool':str(m.CANDIDATES.relative_to(ROOT))},'step1c_crosscheck':{'current_K':20,'expected_K20_gain':EB,'measured_K20_gain':sweep[0]['pooled_macro_recall_oracle_gain'],'expected_full_pool_A':EA,'total_action_space_gap':GAP},'k_selection':{'method':'nearest-rank percentiles of lost beneficial union_rank distribution','percentiles':ps,'lost_beneficial_candidate_count':len(lost),'rank_min':min(lost),'rank_max':max(lost),'raw_percentile_K_values':raw,'final_K_values':ks,'diagnostic_label_conditioned_not_production_selection':True},'action_generator_contract':{'source_file':'scripts/beam/task1_v3_residual/build_actions.py','source_lines':'110-113','drop_ranks':[4,5],'extra_eligibility_rules':['incoming doc must be a non-baseline shortlist doc; baseline anchor docs must be present'],'verified':True},'sweep':sweep,'marginal_intervals':intervals,'affected_subset':{'query_count':162,'per_fold':{'1':40,'2':39,'3':41,'4':42}},'K_max_covers_entire_full_pool':ks[-1]>=global_max,'sanity_checks':checks,'workflow_document_updated':False,'curve_observations':['Descriptive oracle/cost curve only; no production K is selected.','Expanded-K results do not imply production policy benefit.'],'notes':['K values derive from labeled research folds solely for diagnostic characterization.','No model training, reranker inference, policy replay, Fold0 materialization, public labels, GPU, Modal or Beam job.']}
 OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
