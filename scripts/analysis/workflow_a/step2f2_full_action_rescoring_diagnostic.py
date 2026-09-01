"""STEP 2-F2: authorized exact frozen-v2 reconstruction and rescoring diagnostic."""
from __future__ import annotations
import importlib.util,json,math,runpy,statistics
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]; R=ROOT/'reports/task1'
PARENT=R/'step2_k77_model_contract_v2.json'; DEC=R/'step2_k77_v2_oof_policy_decisions.jsonl'
REPRO=R/'step2f2_reproduction_check_report.json'; SCORES=R/'step2f2_full_action_scores.jsonl'; OUT=R/'step2f2_rescoring_diagnostic_report.json'
C77=.058785714285714274; TOTAL=329.2
ns=runpy.run_path(str(ROOT/'scripts/analysis/step2_k77_oof_policy_realizability.py'),run_name='step2_parent')

def qstats(v):
 if not v:return {'count':0,'min':None,'p10':None,'p25':None,'median':None,'mean':None,'p75':None,'p90':None,'max':None}
 s=sorted(v); q=lambda p:s[math.ceil(p*len(s))-1]
 return {'count':len(s),'min':s[0],'p10':q(.1),'p25':q(.25),'median':statistics.median(s),'mean':statistics.fmean(s),'p75':q(.75),'p90':q(.9),'max':s[-1]}
def ordered_actions(rows, vals):return sorted(zip(rows,vals),key=lambda z:(-z[1],-z[0]['drop_rank'],z[0]['incoming_union_rank'],z[0]['incoming_doc_id']))
def threshold_curve(bestmap):
 pairs=list(bestmap.values()); ts=[math.inf]+sorted({s for _,s in pairs}); out=[]
 for t in ts:
  chosen=[r for r,s in pairs if s>=t]; gain=sum(r['truth_delta'] for r in chosen); pd=sum((1 if r['label']=='BENEFICIAL' else -1 if r['label']=='HARMFUL' else 0)/5 for r in chosen)
  out.append({'threshold':'+Infinity' if math.isinf(t) else t,'inner_oof_query_count':len(pairs),'queries_executed':len(chosen),'inner_recall_gain_sum':gain,'inner_macro_recall_gain':gain/len(pairs),'inner_precision_delta':pd/len(pairs)})
 return out
def main():
 contract=json.loads(PARENT.read_text()); r1=json.loads((R/'step2_k77_v2_oof_policy_realizability_report.json').read_text()); f=json.loads((R/'step2f_failure_attribution_report.json').read_text())
 cols=contract['feature_columns']; expected_thr={1:.9336471149727648,2:math.inf,3:.9009323675256211,4:.8974011769826262}
 ok=(r1['status']=='PASS' and r1['scientific_gate']['result']=='FAIL' and len(cols)==len(set(cols))==36 and r1['k77_contract']['actions']==806644 and f['status']=='PASS' and f['canonical']['C77_sum_gain']==TOTAL)
 if not ok: raise RuntimeError('BLOCKED canonical contracts')
 original=[json.loads(x) for x in DEC.open(encoding='utf-8') if x.strip()]; orig={x['query_id']:x for x in original}
 if len(orig)!=5600:raise RuntimeError('BLOCKED valid v2 decisions')
 # Rebuild exact v2 action data using the original deterministic helpers.
 folds=ns['target_folds'](); target=set(folds); records,gold=ns['load_gold'](target); base,_=ns['load_baseline'](target,folds); candidates,_=ns['load_candidates'](target,folds)
 # The fixed v2 columns are passed into the unchanged action builder.
 actions,incoming=ns['make_actions'](folds,records,gold,base,candidates,cols)
 x={f:ns['matrix'](actions[f],cols) for f in range(1,5)}; y={f:np.asarray([ns['LABELS'][z['label']] for z in actions[f]],dtype=np.int8) for f in range(1,5)}
 if incoming!=403322 or sum(len(v) for v in actions.values())!=806644 or any(a.shape[1]!=36 for a in x.values()):raise RuntimeError('CONTRACT_ERROR action/schema')
 reconstructed={}; all_probs={}; thresholds={}; f2curve=None
 for held in range(1,5):
  tr=[i for i in range(1,5) if i!=held]; inner={}
  for valid in tr:
   innertr=[i for i in tr if i!=valid]; model=ns['fit'](np.concatenate([x[i] for i in innertr]),np.concatenate([y[i] for i in innertr]))
   inner.update(ns['best'](actions[valid],ns['scores'](model,x[valid])))
  tau=ns['threshold'](inner); thresholds[held]=tau
  if held==2:f2curve=threshold_curve(inner)
  model=ns['fit'](np.concatenate([x[i] for i in tr]),np.concatenate([y[i] for i in tr]))
  prob=model.predict_proba(x[held]); index={int(c):i for i,c in enumerate(model.classes_)}; vals=[float(a-b) for a,b in zip(prob[:,index[0]],prob[:,index[2]])]
  grouped=defaultdict(list)
  for row,score,pv in zip(actions[held],vals,prob): grouped[row['query_id']].append((row,score,[float(pv[index[i]]) for i in (0,1,2)]))
  for q,items in grouped.items():
   ordered=ordered_actions([a for a,_,_ in items],[s for _,s,_ in items]); top,topscore=ordered[0]; take=topscore>=tau
   reconstructed[q]={'top':top,'score':topscore,'take':take,'threshold':tau}; all_probs[q]=items
 # Mandatory reproduction gate, before any score artifact or interpretation.
 thcheck={};
 for fnum,t in thresholds.items():thcheck[str(fnum)]={'expected':'+Infinity' if math.isinf(expected_thr[fnum]) else expected_thr[fnum],'measured':'+Infinity' if math.isinf(t) else t,'match':math.isinf(t)==math.isinf(expected_thr[fnum]) and (math.isinf(t) or abs(t-expected_thr[fnum])<=1e-9)}
 execrows=[z for z in original if z['selected_action'] is not None]; noops=[z for z in original if z['selected_action'] is None]
 exmatch=sum(reconstructed[z['query_id']]['take'] and {k:reconstructed[z['query_id']]['top'][k] for k in ('incoming_doc_id','drop_rank','dropped_doc_id')}==z['selected_action'] for z in execrows); nomatch=sum(not reconstructed[z['query_id']]['take'] for z in noops)
 abses=[]; rels=[]; smatch=0
 for z in original:
  a=reconstructed[z['query_id']]['score']; b=z['policy_score_of_selected_action']; ae=abs(a-b); re=ae/max(abs(b),1e-300);abses.append(ae);rels.append(re);smatch+=ae<=1e-9 or re<=1e-6
 comp=Counter(reconstructed[q]['top']['label'] for q in orig if reconstructed[q]['take'])
 dbyfold={}
 for fnum in range(1,5):
  rr=[q for q in orig if folds[q]==fnum]; dbyfold[str(fnum)]=sum((ns['recall'](gold[q], list(base[q][:reconstructed[q]['top']['drop_rank']-1])+[reconstructed[q]['top']['incoming_doc_id']]+list(base[q][reconstructed[q]['top']['drop_rank']:]))-ns['recall'](gold[q],base[q])) if reconstructed[q]['take'] else 0 for q in rr)/1400
 d77=sum(dbyfold.values())/4
 gate=all(x['match'] for x in thcheck.values()) and exmatch==362 and nomatch==5238 and smatch==5600 and comp==Counter({'NEUTRAL':348,'BENEFICIAL':7,'HARMFUL':7}) and abs(d77)<=1e-12 and all(abs(dbyfold[str(i)]-r1['per_fold'][str(i)]['D77_policy_gain'])<=1e-12 for i in range(1,5))
 repro={'status':'PASS' if gate else 'RECONSTRUCTION_MISMATCH','thresholds':thcheck,'executed_action_identity_match':{'matched':exmatch,'expected':362},'noop_decision_match':{'matched':nomatch,'expected':5238},'top_score_match':{'matched':smatch,'expected':5600,'max_abs_error':max(abses),'max_rel_error':max(rels)},'selected_action_count':sum(reconstructed[q]['take'] for q in orig),'noop_count':sum(not reconstructed[q]['take'] for q in orig),'class_counts':{k:comp[k] for k in ('BENEFICIAL','NEUTRAL','HARMFUL')},'D77':d77,'per_fold_D77':dbyfold,'reproduction_gate_pass':gate}
 REPRO.write_text(json.dumps(repro,indent=2)+'\n',encoding='utf-8')
 if not gate:return
 # Serialize all held-out actions only after reproduction passes.
 full=[]
 for q,items in all_probs.items():
  top=reconstructed[q]['top']; score=reconstructed[q]['score']; tau=reconstructed[q]['threshold']
  for a,s,pv in items: full.append({'query_id':q,'fold':folds[q],'incoming_doc_id':a['incoming_doc_id'],'incoming_union_rank':a['incoming_union_rank'],'drop_rank':a['drop_rank'],'dropped_doc_id':a['dropped_doc_id'],'true_action_class':a['label'],'truth_recall_delta':a['truth_delta'],'policy_score':s,'p_beneficial':pv[0],'p_neutral':pv[1],'p_harmful':pv[2],'outer_threshold':'+Infinity' if math.isinf(tau) else tau,'is_query_top_scored':a is top,'would_execute':a is top and s>=tau})
 with SCORES.open('w',encoding='utf-8',newline='\n') as h:
  for z in full:h.write(json.dumps(z)+'\n')
 # Deterministic oracle map, same structural target-only contract as F2.
 spec=importlib.util.spec_from_file_location('s1a',ROOT/'scripts/analysis/exp_1a_recall_gap_audit.py'); mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 crow,_=mod.stream_target_rows(mod.CANDIDATES,target); ranks=defaultdict(dict)
 for row in crow:
  q=str(row['query_id']); xs=row.get('candidates') if isinstance(row.get('candidates'),list) else [row]
  for i,a in enumerate(xs,1):ranks[q].setdefault(str(a['doc_id']),int(a.get('union_rank',i)))
 oracle={}
 for q in target:
  before=ns['recall'](gold[q],base[q]); best=0
  for doc,rk in ranks[q].items():
   if rk<=77 and doc not in base[q]:
    for d in (4,5):top=list(base[q]);top[d-1]=doc;best=max(best,ns['recall'](gold[q],top)-before)
  oracle[q]=best
 if sum(v>1e-12 for v in oracle.values())!=426 or abs(sum(oracle.values())-TOTAL)>1e-9:raise RuntimeError('CONTRACT_ERROR oracle')
 trs=defaultdict(list); qdiag=[]; gapsBN=[];gapsBH=[];deep=defaultdict(list)
 for q in sorted((q for q in target if oracle[q]>1e-12),key=lambda z:int(z) if z.isdigit() else z):
  items=all_probs[q]; ordered=ordered_actions([a for a,_,_ in items],[s for _,s,_ in items]); top,tscore=ordered[0]; bitems=ordered_actions([a for a,s,_ in items if a['label']=='BENEFICIAL'],[s for a,s,_ in items if a['label']=='BENEFICIAL']); bestb,bs=bitems[0]; brank=next(i for i,(a,_) in enumerate(ordered,1) if a is bestb)
  n=[(a,s) for a,s,_ in items if a['label']=='NEUTRAL'];h=[(a,s) for a,s,_ in items if a['label']=='HARMFUL']; bn=bs-ordered_actions([a for a,s in n],[s for a,s in n])[0][1] if n else None;bh=bs-ordered_actions([a for a,s in h],[s for a,s in h])[0][1] if h else None
  if bn is not None:gapsBN.append(bn)
  if bh is not None:gapsBH.append(bh)
  cls='S_SUCCESS' if top['label']=='BENEFICIAL' and tscore>=thresholds[folds[q]] else 'T_THRESHOLD_LOSS' if top['label']=='BENEFICIAL' else 'R_RANKING_DISCRIMINATION_LOSS';trs[cls].append(q)
  actual=orig[q];qdiag.append({'query_id':q,'fold':folds[q],'C77_query':oracle[q],'outer_threshold':'+Infinity' if math.isinf(thresholds[folds[q]]) else thresholds[folds[q]],'top_action_class':top['label'],'top_action_score':tscore,'top_action_union_rank':top['incoming_union_rank'],'top_action_drop_rank':top['drop_rank'],'best_beneficial_score':bs,'best_beneficial_union_rank':bestb['incoming_union_rank'],'best_beneficial_drop_rank':bestb['drop_rank'],'best_neutral_score':max((s for a,s,_ in items if a['label']=='NEUTRAL'),default=None),'best_harmful_score':max((s for a,s,_ in items if a['label']=='HARMFUL'),default=None),'beneficial_rank_among_actions':brank,'T_R_S_class':cls,'actual_step2_decision_class':actual['selected_action_class'],'actual_step2_recall_delta':actual['recall_delta']})
  bucket='1-20' if bestb['incoming_union_rank']<=20 else '21-29' if bestb['incoming_union_rank']<=29 else '30-45' if bestb['incoming_union_rank']<=45 else '46-77';deep[bucket].append((q,bs,brank,cls))
 def trsum(qs):
  head=sum(oracle[q] for q in qs); actual=sum(orig[q]['recall_delta'] for q in qs);return {'query_count':len(qs),'C77_headroom_sum':head,'C77_headroom_macro':head/5600,'share_of_total_headroom':head/TOTAL,'actual_policy_recall_gain_sum':actual,'residual_oracle_magnitude_gap':head-actual,'per_fold':{str(f):{'query_count':sum(folds[q]==f for q in qs),'headroom_sum':sum(oracle[q] for q in qs if folds[q]==f),'headroom_share':sum(oracle[q] for q in qs if folds[q]==f)/TOTAL} for f in range(1,5)}}
 trsout={k:trsum(trs[k]) for k in ('S_SUCCESS','T_THRESHOLD_LOSS','R_RANKING_DISCRIMINATION_LOSS')}
 def gapout(v):return {**qstats(v),'gt0':sum(x>1e-12 for x in v),'eq0':sum(abs(x)<=1e-12 for x in v),'lt0':sum(x<-1e-12 for x in v)}
 scoresep={}
 for cl in ('BENEFICIAL','NEUTRAL','HARMFUL'):scoresep[cl]={'pooled':qstats([z['policy_score'] for z in full if z['true_action_class']==cl]),'per_fold':{str(f):qstats([z['policy_score'] for z in full if z['true_action_class']==cl and z['fold']==f]) for f in range(1,5)}}
 deepout={b:{'query_count':len(v),'share_of_426':len(v)/426,'C77_headroom_sum':sum(oracle[q] for q,_,_,_ in v),'mean_best_beneficial_score':statistics.fmean(x[1] for x in v) if v else None,'median_best_beneficial_score':statistics.median(x[1] for x in v) if v else None,'mean_beneficial_rank':statistics.fmean(x[2] for x in v) if v else None,'median_beneficial_rank':statistics.median(x[2] for x in v) if v else None,'fraction_beneficial_rank_1':sum(x[2]==1 for x in v)/len(v) if v else None,**{k:sum(x[3]==k for x in v) for k in ('S_SUCCESS','T_THRESHOLD_LOSS','R_RANKING_DISCRIMINATION_LOSS')}} for b,v in deep.items()}
 finite=[x for x in f2curve if x['threshold']!='+Infinity'];best=max(finite,key=lambda z:(z['inner_macro_recall_gain'],z['inner_precision_delta'],z['threshold']));fold2={'inner_validation_query_count':4200,'candidate_curve':f2curve,'number_finite_threshold_candidates':len(finite),'best_finite_threshold':best['threshold'],'best_finite_inner_gain':best['inner_macro_recall_gain'],'best_finite_precision_delta':best['inner_precision_delta'],'selected_threshold':'+Infinity','finite_gain_summary':qstats([x['inner_macro_recall_gain'] for x in finite])}
 probsok=all(-1e-12<=z[k]<=1+1e-12 for z in full for k in ('p_beneficial','p_neutral','p_harmful')) and all(abs(z['p_beneficial']+z['p_neutral']+z['p_harmful']-1)<=1e-9 and abs(z['policy_score']-(z['p_beneficial']-z['p_harmful']))<=1e-12 for z in full)
 checks={'reproduction_gate_pass':gate,'full_action_rows_806644':len(full)==806644,'unique_action_keys':len({(z['query_id'],z['incoming_doc_id'],z['drop_rank']) for z in full})==806644,'probabilities_valid':probsok,'one_top_action_per_query':sum(z['is_query_top_scored'] for z in full)==5600,'would_execute_reproduces':sum(z['would_execute'] for z in full)==362,'trs_exhaustive':sum(len(v) for v in trs.values())==426,'trs_headroom_sum':abs(sum(v['C77_headroom_sum'] for v in trsout.values())-TOTAL)<=1e-9,'fold0_absent':all(z['fold'] in (1,2,3,4) for z in full),'public_labels_not_used':True}
 status='PASS' if all(checks.values()) else 'CONTRACT_ERROR'
 report={'status':status,'experiment':'STEP 2-F2 — Full-Action Rescoring Diagnostic via Deterministic Reconstruction of Frozen Step2-R1 Models','experiment_type':'DIAGNOSTIC_ONLY','branch_gate':'DIAGNOSTIC_ONLY_NO_BRANCH_GATE','adopted_research_K':77,'production_K_selected':False,'folds_used':[1,2,3,4],'fold0_touched':False,'fold0_payload_materialized':False,'public_labels_used':False,'reproduction_report':'reports/task1/step2f2_reproduction_check_report.json','reproduction_gate_pass':gate,'full_action_score_artifact':'reports/task1/step2f2_full_action_scores.jsonl','full_action_rows':len(full),'canonical':{'oracle_positive_queries':426,'C77_sum':TOTAL,'C77_macro':C77,'D77':0.0},'trs':trsout,'per_fold_trs':{str(f):{k:trsout[k]['per_fold'][str(f)] for k in trsout} for f in range(1,5)},'oracle_positive_query_diagnostics':qdiag,'score_separation':scoresep,'within_query_score_gaps':{'B_minus_N':gapout(gapsBN),'B_minus_H':gapout(gapsBH)},'deep_k_diagnostic':deepout,'fold2_inner_threshold_curve':fold2,'harmful_action_diagnostics':{'oracle_positive_top_harmful_count':sum(qdiag[i]['top_action_class']=='HARMFUL' for i in range(len(qdiag))),'oracle_positive_top_harmful_headroom_sum':sum(x['C77_query'] for x in qdiag if x['top_action_class']=='HARMFUL'),'executed_harmful_loss_sum':sum(z['truth_recall_delta'] for z in full if z['would_execute'] and z['true_action_class']=='HARMFUL')},'current_status':{'K77':'NOT_REJECTED','feature_change':'NOT_AUTHORIZED','classifier_change':'NOT_AUTHORIZED','threshold_change':'NOT_AUTHORIZED','smaller_K':'NOT_AUTHORIZED','query_conditioned_K':'NOT_AUTHORIZED','workflow_B':'NOT_AUTHORIZED'},'sanity_checks':checks,'workflow_document_updated':False,'notes':['Exact frozen-model reconstruction was authorized solely for diagnostic scoring.','No hyperparameter, feature, label, weight, K, fold, or threshold protocol change was made.']}
 OUT.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps({'status':status,'reproduction':repro,'trs':trsout,'rows':len(full)},indent=2))
if __name__=='__main__':main()
