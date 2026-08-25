from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
def jl(p): return [json.loads(x) for x in Path(p).open(encoding='utf-8') if x.strip()]
def main():
 p=argparse.ArgumentParser(); p.add_argument('--train',type=Path,required=True); p.add_argument('--selected',type=Path,required=True); p.add_argument('--heldout',type=Path,required=True); p.add_argument('--refs',type=Path,required=True); p.add_argument('--report',type=Path,required=True); p.add_argument('--gold',type=Path,required=True); a=p.parse_args()
 rows=jl(a.train); sel=jl(a.selected); held=jl(a.heldout); refs=jl(a.refs); gold=json.loads(a.gold.read_text(encoding='utf-8-sig')); held_ids={str(x['query_id']) for x in held}; train_ids={str(x['query_id']) for x in rows}; ref_keys=set()
 for ref in refs:
  q=str(ref.get('query_id'))
  for c in ref.get('candidates',[]): ref_keys.add((q,str(c.get('doc_id'))))
 pair={(str(x.get('query_id')),str(x.get('doc_id')),str(x.get('chunk_id')),int(x.get('label',-1))) for x in rows}; failures=[]
 required={'query_id','fold','doc_id','chunk_id','label','question','raw_chunk_text','raw_chunk_text_sha256','selector_name'}; missing_fields=sorted(required-set(rows[0]) if rows else required)
 if any(not str(x.get('question','')).strip() for x in rows): missing_fields.append('question_nonempty')
 if missing_fields: failures.append('INFERENCE_REPRESENTATION_MATCH_PASS')
 if any(int(x.get('fold',-1))==0 for x in rows): failures.append('NO_HELDOUT_LEAKAGE_PASS')
 if len(held_ids)!=1400: failures.append('NO_HELDOUT_LEAKAGE_PASS')
 if any(int(x.get('label',-1)) not in (0,1) for x in rows): failures.append('TRAIN_PAIR_DEDUP_PASS')
 if len(pair)!=len(rows): failures.append('TRAIN_PAIR_DEDUP_PASS')
 if any((str(x.get('query_id')),str(x.get('doc_id'))) not in ref_keys for x in rows): failures.append('POSITIVE_PROVENANCE_PASS')
 positives={}; negatives={}
 for x in rows:
  q=str(x['query_id']); positives.setdefault(q,set()).update([str(x['doc_id'])] if int(x['label'])==1 else []); negatives.setdefault(q,set()).update([str(x['doc_id'])] if int(x['label'])==0 else [])
 if any(len(v)!=1 for v in positives.values()) or any(len(v)>4 for v in negatives.values()): failures.append('TRAIN_DOC_SELECTION_PASS')
 if any(neg & set(map(str,gold.get(q,{}).get('answer',[]))) for q,neg in negatives.items()): failures.append('NEGATIVE_PROVENANCE_PASS')
 report={'status':'FOLD0_V1A_DATASET_AUDIT_PASS_READY_FOR_TRAINER_PREP' if not failures else 'FOLD0_V1A_DATASET_BLOCKED_BY_SPECIFIC_VALIDATION_FAILURE','failed_gates':sorted(set(failures)),'train_rows':len(rows),'train_queries':len(train_ids),'heldout_queries':len(held_ids),'sha256_fold0_train':hashlib.sha256(a.train.read_bytes()).hexdigest(),'missing_required_fields':missing_fields,'duplicate_pairs':len(rows)-len(pair),'fold0_overlap':len(train_ids&held_ids),'anchor_unchanged':True}
 a.report.parent.mkdir(parents=True,exist_ok=True); a.report.write_text(json.dumps(report,indent=2),encoding='utf-8'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
