"""Strict query-macro fold0 evaluation; safe to run only after scoring."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
def load(path): return [json.loads(line) for line in Path(path).open(encoding='utf-8') if line.strip()]
def macro_recall(rows):
    vals=[]
    for x in rows:
        gold=set(map(str,x.get('gold_documents',[]))); top=set(map(str,x.get('top5',[]))); vals.append(len(gold&top)/len(gold) if gold else 0.0)
    return sum(vals)/len(vals) if vals else 0.0
def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--predictions',required=True); p.add_argument('--baseline',required=True); p.add_argument('--report',required=True); p.add_argument('--train',required=True); p.add_argument('--model',required=True); p.add_argument('--materialization-report',required=True); a=p.parse_args()
    pred, all_base, train=load(a.predictions),load(a.baseline),load(a.train); base={str(x['query_id']):x for x in all_base if int(x['fold'])==0}; by_q={str(x['query_id']):x for x in pred}; expected,actual=set(base),set(by_q); errors=[]
    if len(pred)!=1400: errors.append(f'query_count={len(pred)}')
    if len(by_q)!=len(pred): errors.append('duplicate query IDs')
    if actual!=expected: errors.append('query-id set mismatch')
    rq=[]; pq=[]; better=worse=neutral=changed=0; multi=[]
    for q in sorted(expected):
        if q not in by_q: continue
        top=[str(x) for x in by_q[q].get('top5',[])]; gold=set(map(str,base[q].get('gold_documents',[]))); btop=set(map(str,base[q].get('top5',[])))
        if len(top)!=5: errors.append(f'{q}: prediction length != 5')
        if len(set(top))!=5: errors.append(f'{q}: duplicate top5')
        r=len(gold&set(top))/len(gold) if gold else 0.0; br=len(gold&btop)/len(gold) if gold else 0.0; rq.append(r); pq.append(len(gold&set(top))/5.0); better+=r>br; worse+=r<br; neutral+=r==br; changed+=top!=list(map(str,base[q].get('top5',[])))
        if len(gold)>1: multi.append(r)
    train_q={str(x.get('query_id')) for x in train}; train_folds={int(x['fold']) for x in train if 'fold' in x}; overlap=sorted(train_q&expected); mat=json.loads(Path(a.materialization_report).read_text(encoding='utf-8')); no_leak=not overlap and train_folds.issubset({1,2,3,4}); errors += [] if no_leak else ['training leakage/fold violation']; errors += [] if mat.get('missing_payload_docs')==0 else ['missing payload docs']; errors += [] if mat.get('missing_raw_text')==0 else ['missing raw text']
    rec=sum(rq)/len(rq) if rq else 0.0; base_rec=macro_recall(list(base.values()))
    report={'status':'PASS' if not errors else 'FAIL','errors':errors,'query_count':len(pred),'fold0_macro_recall':rec,'baseline_fold0_macro_recall_recomputed':base_rec,'delta':rec-base_rec,'macro_precision':sum(pq)/len(pq) if pq else 0.0,'changed_queries':changed,'better':better,'worse':worse,'neutral':neutral,'full_gold_count':sum(len(set(map(str,x.get('gold_documents',[]))))==len(set(map(str,x.get('gold_documents',[])))&set(map(str,by_q.get(str(x['query_id']),{}).get('top5',[])))) for x in base.values()),'multi_gold_query_count':len(multi),'multi_gold_macro_recall':sum(multi)/len(multi) if multi else 0.0,'distinct_top5_pass':not any('duplicate top5' in e for e in errors) and all(len(x.get('top5',[]))==5 and len(set(x['top5']))==5 for x in pred),'query_set_exact_match_pass':actual==expected,'training_fold0_overlap':overlap,'no_heldout_training_leakage':no_leak,'missing_payload_docs':mat.get('missing_payload_docs'),'missing_raw_text':mat.get('missing_raw_text'),'chunk_rows_scored':mat.get('emitted_chunk_rows'),'candidate_doc_occurrences':mat.get('candidate_doc_occurrences'),'model_path':str(a.model),'train_sha256':hashlib.sha256(Path(a.train).read_bytes()).hexdigest(),'max_length':512,'batch_size':16,'aggregation':'max_chunk_score_per_doc','inference_precision':'cuda_fp16'}
    Path(a.report).parent.mkdir(parents=True,exist_ok=True); Path(a.report).write_text(json.dumps(report,indent=2),encoding='utf-8')
    if errors: raise SystemExit(1)
if __name__=='__main__': main()
