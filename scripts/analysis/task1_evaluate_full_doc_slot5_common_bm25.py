import hashlib,json
from collections import Counter,defaultdict
from pathlib import Path

R=Path(__file__).resolve().parents[2]
O=R/'reports/task1/full_document_legal_field_retrieval'
B=R/'artifacts/task1/recovery_096/baseline_093_oof/predictions.jsonl'
P=O/'full_doc_top10_common_bm25_slot5_predictions.jsonl'
T=R/'data/raw/btc/LegalIR/train.json'
C=O/'full_document_legal_field_retrieval_candidates.jsonl'
K=R/'artifacts/task1/recovery_096/inference_matched_reranker_v1/compact/candidate_refs_full.jsonl'

def rows(p):
    with open(p,encoding='utf8') as f:
        for line in f:
            if line.strip(): yield json.loads(line)
def sha(p):
    x=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): x.update(b)
    return x.hexdigest()
def metrics(pred,gold):
    hits={q:len(set(pred[q])&set(gold[q])) for q in pred}
    recall=sum(hits[q]/len(gold[q]) for q in pred)/len(pred)
    precision=sum(hits[q]/5 for q in pred)/len(pred)
    return recall,precision,hits
def main():
    baseline_rows=[x for x in rows(B) if x['fold']!=0]
    base={str(x['query_id']):[str(d) for d in x['top5']] for x in baseline_rows}
    folds={str(x['query_id']):int(x['fold']) for x in baseline_rows}
    new={str(x['query_id']):[str(d) for d in x['top5']] for x in rows(P)}
    meta={str(x['query_id']):x for x in rows(P)}
    train=json.load(open(T,encoding='utf8'))
    gold={q:{str(d) for d in train[q]['answer']} for q in base}
    assert set(base)==set(new) and len(new)==5600 and all(len(x)==5 and len(set(x))==5 for x in new.values())
    br,bp,bh=metrics(base,gold);nr,np,nh=metrics(new,gold)
    fold={}
    for f in range(1,5):
        qs=[q for q in base if folds[q]==f]
        a,b,_=metrics({q:base[q] for q in qs},{q:gold[q] for q in qs})
        c,d,_=metrics({q:new[q] for q in qs},{q:gold[q] for q in qs})
        fold[f]={'queries':len(qs),'baseline_recall':a,'new_recall':c,'recall_delta':c-a,'baseline_precision':b,'new_precision':d,'precision_delta':d-b}
    changed=[q for q in base if base[q][4]!=new[q][4]]
    delta={q:(nh[q]-bh[q])/len(gold[q]) for q in base}
    outcome=Counter('beneficial' if delta[q]>0 else 'harmful' if delta[q]<0 else 'neutral' for q in changed)
    # Frozen full-document ranks for novelty and structural attribution only.
    fr={}
    for x in rows(C):
        q,d=str(x['query_id']),str(x['document_id'])
        if q in base and x['rank']<=200: fr[q,d]=int(x['rank'])
    historical_pool={str(x['query_id']):{str(z['doc_id']) for z in x['candidates']} for x in rows(K)}
    known=[(q,d) for q in base for d in gold[q] if d not in historical_pool[q]]
    assert len(known)==66
    known_slot5=[(q,d) for q,d in known if new[q][4]==d]
    known_gain=[(q,d) for q,d in known_slot5 if delta[q]>0]
    rankbins=defaultdict(lambda:Counter())
    for q in changed:
        r=meta[q].get('frozen_rank')
        key='incumbent' if r is None else str(r) if r<=6 else '7-10'
        rankbins[key]['executed']+=1
        rankbins[key]['beneficial' if delta[q]>0 else 'harmful' if delta[q]<0 else 'neutral']+=1
    novel=[q for q in changed if (q,new[q][4]) not in fr]
    result={'experiment':'FULLDOC_TOP10_COMMON_BM25_SLOT5','coverage':len(new),'prediction_sha256':sha(P),
      'parity':{'status':'PASS','parity_rows':256,'gate':'max_abs_score_difference <= 1e-8 (enforced before prediction write)','representation':'canonical raw link + raw passage; repository-pinned PyVi tokenizer; rank_bm25==0.2.2'},
      'baseline':{'recall_at_5':br,'precision_at_5':bp},'new':{'recall_at_5':nr,'precision_at_5':np},
      'delta':{'recall_at_5':nr-br,'precision_at_5':np-bp},'folds':fold,
      'slot5':{'changed':len(changed),'beneficial':outcome['beneficial'],'harmful':outcome['harmful'],'neutral':outcome['neutral'],'zero_hit_rescued':sum(bh[q]==0 and nh[q]>0 for q in changed),'previously_correct_broken':sum(bh[q]>0 and nh[q]==0 for q in changed),'net_relevant_document_hits':sum(nh[q]-bh[q] for q in base),'weighted_recall_gain':sum(max(0,delta[q]) for q in base),'weighted_recall_loss':-sum(min(0,delta[q]) for q in base)},
      'known66':{'count':len(known),'recovered_at_200':sum((q,d) in fr for q,d in known),'present_in_top10':sum((q,d) in fr and fr[q,d]<=10 for q,d in known),'selected_as_final_slot5':len(known_slot5),'beneficial_final_slot5':len(known_gain)},
      'novel_information':{'executed_new_candidate_outside_fulldoc_top200':len(novel),'beneficial':sum(delta[q]>0 for q in novel),'harmful':sum(delta[q]<0 for q in novel),'recall_contribution':sum(delta[q] for q in novel)},
      'structural_attribution':{'winner_frozen_rank':{k:dict(v) for k,v in sorted(rankbins.items())}}
    }
    pass_gate=nr>br and all(x['recall_delta']>=0 for x in fold.values()) and np-bp>=-0.001
    d=nr-br
    if not pass_gate: status='FULLDOC_SLOT5_FAIL'; interpretation='NEW_RETRIEVAL_SIGNAL_EXISTS_BUT_BM25_ONLY_SLOT5_SELECTION_IS_INSUFFICIENT'
    elif d>=.003: status='FULLDOC_SLOT5_TARGET_LEVEL_PASS'; interpretation='NEW_RETRIEVAL_SIGNAL_PRODUCES_FINAL_TOP5_GAIN'
    elif d>=.002: status='FULLDOC_SLOT5_STRONG_PASS'; interpretation='NEW_RETRIEVAL_SIGNAL_PRODUCES_FINAL_TOP5_GAIN'
    elif d>=.001: status='FULLDOC_SLOT5_MATERIAL_PASS'; interpretation='NEW_RETRIEVAL_SIGNAL_PRODUCES_FINAL_TOP5_GAIN'
    else: status='FULLDOC_SLOT5_WEAK_PASS'; interpretation='NEW_RETRIEVAL_SIGNAL_PRODUCES_FINAL_TOP5_GAIN'
    result['status']=status;result['interpretation']=interpretation
    result['safe_next_action']='INDEPENDENTLY_REVIEW_FULLDOC_SLOT5_RESULT' if pass_gate else 'PRESERVE_NEW_RETRIEVAL_ARTIFACT_AND_DO_NOT_EXPAND_BM25_ONLY_SLOT5'
    ev=O/'full_doc_top10_common_bm25_slot5_evaluation.json';ev.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    prov=O/'full_doc_top10_common_bm25_slot5_provenance.json';prov.write_text(json.dumps({'prediction_sha256':sha(P),'baseline_sha256':sha(B),'frozen_candidates_sha256':sha(C),'train_sha256':sha(T),'labels_opened_after_prediction_freeze':True},indent=2)+'\n',encoding='utf8')
    att=O/'full_doc_top10_common_bm25_slot5_attribution.json';att.write_text(json.dumps({'known66':result['known66'],'novel_information':result['novel_information'],'structural_attribution':result['structural_attribution']},indent=2)+'\n',encoding='utf8')
    print(json.dumps({'status':status,'evaluation_sha256':sha(ev),'provenance_sha256':sha(prov),'attribution_sha256':sha(att),**result['delta']}))
if __name__=='__main__':main()
