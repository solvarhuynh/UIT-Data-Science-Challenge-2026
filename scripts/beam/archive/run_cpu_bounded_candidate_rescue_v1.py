"""Short strict nested OOF document-insertion rescue from completed rankings."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any

def lines(path: Path):
    with path.open(encoding='utf-8') as f: return [json.loads(x) for x in f if x.strip()]

def metric(rows, key):
    rs=[]; ps=[]
    for r in rows:
        gold=set(r['gold_documents']); pred=set(r[key]); hit=len(gold & pred)
        rs.append(hit/len(gold)); ps.append(hit/len(pred))
    return {'recall':sum(rs)/len(rs),'precision':sum(ps)/len(ps),'query_count':len(rows)}

def candidates(qid, adaptive, ranks):
    k200=set(adaptive[qid]['k200_documents']); base=set(adaptive[qid]['baseline_top5'])
    score={}; provenance={}
    for name, ranking in ranks.items():
        for rank, doc in enumerate(ranking[qid], 1):
            if doc in k200 or doc in base: continue
            score[doc]=score.get(doc,0)+1/(60+rank)
            provenance.setdefault(doc,[]).append((name,rank))
    return sorted(score, key=lambda d:(-score[d],d)), provenance

def propose(row, config):
    order, prov = row['candidate_order'], row['provenance']
    if config == 'baseline': return row['baseline_top5'], None
    for doc in order:
        supports=prov[doc]
        best=min(rank for _,rank in supports)
        if config == 'support2_rank50' and len(supports) < 2: continue
        if config == 'bm25_rank10' and not any(name=='bm25' and rank<=10 for name,rank in supports): continue
        if config == 'support1_rank10' and best > 10: continue
        return row['baseline_top5'][:-1]+[doc], {'document_id':doc,'sources':[x[0] for x in supports],'ranks':dict(supports)}
    return row['baseline_top5'], None

def main():
    p=argparse.ArgumentParser(); p.add_argument('--baseline',type=Path,required=True);p.add_argument('--adaptive',type=Path,required=True);p.add_argument('--checkpoint-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    base={str(x['query_id']):x for x in lines(a.baseline)}; adaptive={str(x['query_id']):x for x in lines(a.adaptive)}
    ranks={name:{} for name in ('bm25','knn_word','knn_char')}
    for name in ranks:
        for fold in range(5): ranks[name].update(json.loads((a.checkpoint_dir/f'{name}_fold{fold}.json').read_text(encoding='utf-8'))['rankings'])
    rows=[]
    for qid,b in base.items():
        order,prov=candidates(qid,adaptive,ranks)
        rows.append({'query_id':qid,'fold':b['fold'],'gold_documents':b['gold_documents'],'baseline_top5':b['top5'],'candidate_order':order,'provenance':prov,'miss_k200':adaptive[qid]['miss_k200']})
    configs=('baseline','support2_rank50','bm25_rank10','support1_rank10'); out=[]; selections=[]
    for holdout in range(5):
        train=[r for r in rows if r['fold']!=holdout]; test=[r for r in rows if r['fold']==holdout]
        scored=[]
        for order,config in enumerate(configs):
            tmp=[dict(r,choice=propose(r,config)[0]) for r in train]
            m=metric(tmp,'choice'); scored.append(((m['recall'],m['precision'],-order),config))
        _,chosen=max(scored); selections.append({'fold':holdout,'selected_config':chosen})
        for r in test:
            top,insert=propose(r,chosen); out.append({**r,'rescue_top5':top,'insertion':insert})
    report={'schema_version':'cpu-bounded-candidate-rescue-v1','baseline':metric(out,'baseline_top5'),'rescue':metric(out,'rescue_top5'),'per_fold':{},'selections':selections,'changed_query_count':sum(r['rescue_top5']!=r['baseline_top5'] for r in out),'changed_document_count':sum(r['rescue_top5']!=r['baseline_top5'] for r in out)}
    better=same=worse=0
    for r in out:
        g=set(r['gold_documents']); d=(len(g&set(r['rescue_top5']))-len(g&set(r['baseline_top5'])))/len(g)
        better+=d>0;same+=d==0;worse+=d<0
    report['outcomes']={'better':better,'same':same,'worse':worse}
    for fold in range(5):
        x=[r for r in out if r['fold']==fold]; report['per_fold'][str(fold)]={'baseline':metric(x,'baseline_top5'),'rescue':metric(x,'rescue_top5')}
    report['gate']={'pooled_positive':report['rescue']['recall']>.9244452381,'no_negative_fold':all(v['rescue']['recall']>=v['baseline']['recall'] for v in report['per_fold'].values()),'improvements_ge_harms':better>=worse}
    report['status']='PROMOTE_CPU_CANDIDATE_RESCUE' if all(report['gate'].values()) else 'REJECT_CPU_CANDIDATE_RESCUE'
    a.output_dir.mkdir(parents=True,exist_ok=True);(a.output_dir/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with (a.output_dir/'predictions.jsonl').open('w',encoding='utf-8') as f:
        for r in out:f.write(json.dumps({'query_id':r['query_id'],'fold':r['fold'],'top5':r['rescue_top5'],'insertion':r['insertion']},ensure_ascii=False)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
