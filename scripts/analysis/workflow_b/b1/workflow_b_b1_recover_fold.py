"""Bounded-memory canonical K77 action recovery, one development fold per process."""
from __future__ import annotations
import gc, hashlib, json, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
R=ROOT/"reports/task1"; OUT=R/"workflow_b_b1_recovery"
FEATURES=json.loads((R/"step2p1_model_contract.json").read_text(encoding="utf-8"))["feature_columns"]
ENC={"HARMFUL":0,"NEUTRAL":1,"BENEFICIAL":2}
def key(a): return (int(a["query_id"]),-int(a["drop_rank"]),int(a["incoming_union_rank"]),str(a["incoming_doc_id"]))
def serial(fold,q,inc,drop,drank,urank,x,label,delta):
    return json.dumps([fold,str(q),str(inc),str(drop),int(drank),int(urank),[float(v) for v in x],int(label),float(delta)],separators=(",",":"),ensure_ascii=True).encode()+b"\n"
def canonical_rows(fold):
    ns=__import__("runpy").run_path(str(ROOT/"scripts/analysis/workflow_a/step2_k77_oof_policy_realizability.py"),run_name=f"b1_recover_f{fold}")
    allfolds=ns["target_folds"](); fmap={q:f for q,f in allfolds.items() if f==fold}
    records,gold=ns["load_gold"](set(fmap)); base,_=ns["load_baseline"](set(fmap),fmap); cand,_=ns["load_candidates"](set(fmap),fmap)
    rows=[]
    for q in sorted(fmap,key=lambda x:int(x) if x.isdigit() else x):
        baseline=base[q]; bydoc={x["doc_id"]:x for x in cand[q]}; qtext=str(records[q].get("question","")); qt,qc=float(len(qtext.split())),float(len(qtext))
        incoming=[x for x in cand[q] if x["union_rank"]<=77 and x["doc_id"] not in set(baseline)]
        for incdoc in incoming:
            for rank in (4,5):
                dropped_id=baseline[rank-1]; dropped=bydoc.get(dropped_id,{"doc_id":dropped_id,"union_rank":1e9,"source_ranks":{},"source_support":0}); feat={}
                for name in ns["V3_BASE"]:
                    if f"incoming_{name}" in FEATURES:
                        feat[f"incoming_{name}"]=ns["doc_value"](incdoc,name,qt,qc,None); feat[f"dropped_{name}"]=ns["doc_value"](dropped,name,qt,qc,rank)
                for name in ns["V3_DIFF"]:
                    if f"diff_{name}" in FEATURES: feat[f"diff_{name}"]=feat[f"incoming_{name}"]-feat[f"dropped_{name}"]
                feat["dropped_baseline_rank"]=float(rank); feat["incoming_is_baseline_top5"]=0.0
                final=list(baseline); final[rank-1]=incdoc["doc_id"]; delta=ns["recall"](gold[q],final)-ns["recall"](gold[q],baseline); label="BENEFICIAL" if delta>1e-12 else "HARMFUL" if delta<-1e-12 else "NEUTRAL"
                rows.append({"query_id":q,"incoming_doc_id":incdoc["doc_id"],"incoming_union_rank":int(incdoc["union_rank"]),"drop_rank":rank,"dropped_doc_id":dropped_id,"features":feat,"label":label,"truth_delta":delta})
    rows=sorted(rows,key=key)
    return rows
def build(fold):
    OUT.mkdir(exist_ok=True)
    rows=canonical_rows(fold); n=len(rows)
    if not all(str(a["query_id"]).isdigit() and str(a["incoming_doc_id"]).isdigit() and str(a["dropped_doc_id"]).isdigit() for a in rows): raise RuntimeError("non-numeric canonical ID representation requires explicit contract")
    q=np.empty(n,np.int64); inc=np.empty(n,np.int64); drop=np.empty(n,np.int64); dr=np.empty(n,np.int8); ur=np.empty(n,np.int16); y=np.empty(n,np.uint8); d=np.empty(n,np.float64); x=np.empty((n,len(FEATURES)),np.float64); hasher=hashlib.sha256()
    for i,a in enumerate(rows):
        q[i]=int(a["query_id"]); inc[i]=int(a["incoming_doc_id"]); drop[i]=int(a["dropped_doc_id"]); dr[i]=a["drop_rank"]; ur[i]=a["incoming_union_rank"]; y[i]=ENC[a["label"]]; d[i]=a["truth_delta"]; x[i]=[a["features"][c] for c in FEATURES]
        hasher.update(serial(fold,q[i],inc[i],drop[i],dr[i],ur[i],x[i],y[i],d[i]))
    path=OUT/f"f{fold}_actions.npz"; np.savez(path,query_id=q,incoming_doc_id=inc,dropped_doc_id=drop,drop_rank=dr,incoming_union_rank=ur,label=y,truth_delta=d,features=x)
    man={"fold":fold,"rows":n,"queries":len(set(q.tolist())),"features":FEATURES,"feature_count":len(FEATURES),"K":77,"row_serialization_sha256":hasher.hexdigest(),"artifact_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"dtype":{"features":str(x.dtype),"truth_delta":str(d.dtype)},"source":"step2_k77_oof_policy_realizability.py::make_actions canonical builder"}
    (OUT/f"f{fold}_manifest.json").write_text(json.dumps(man,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(man))
def verify(fold):
    # The persisted row hash was emitted while serializing the NPZ from the
    # canonical builder. Rebuild independently in a fresh process and compare
    # that full canonical row stream; the artifact-byte hash detects mutation.
    path=OUT/f"f{fold}_actions.npz"; rows=canonical_rows(fold); h=hashlib.sha256()
    for a in rows:
        x=[a["features"][c] for c in FEATURES]
        h.update(serial(fold,a["query_id"],a["incoming_doc_id"],a["dropped_doc_id"],a["drop_rank"],a["incoming_union_rank"],x,ENC[a["label"]],a["truth_delta"]))
    man=json.loads((OUT/f"f{fold}_manifest.json").read_text()); mismatch=0 if h.hexdigest()==man["row_serialization_sha256"] and hashlib.sha256(path.read_bytes()).hexdigest()==man["artifact_sha256"] else 1; out={"fold":fold,"rows":len(rows),"mismatch_count":mismatch,"rebuilt_row_serialization_sha256":h.hexdigest(),"persisted_row_serialization_sha256":man["row_serialization_sha256"],"hash_match":h.hexdigest()==man["row_serialization_sha256"],"artifact_hash_match":hashlib.sha256(path.read_bytes()).hexdigest()==man["artifact_sha256"]}
    (OUT/f"f{fold}_equivalence.json").write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8"); print(json.dumps(out))
if __name__=="__main__":
    if len(sys.argv)!=3 or sys.argv[1] not in {"build","verify"}: raise SystemExit("usage: build|verify FOLD")
    (build if sys.argv[1]=="build" else verify)(int(sys.argv[2]))
