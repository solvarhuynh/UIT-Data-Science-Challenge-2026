"""Batch-score materialized fold0 chunks with the trained single-logit reranker."""
from __future__ import annotations
import argparse, json
from pathlib import Path

def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument('--chunks', type=Path, required=True); p.add_argument('--model', type=Path, required=True); p.add_argument('--doc-scores', type=Path, required=True); p.add_argument('--predictions', type=Path, required=True); p.add_argument('--batch-size', type=int, default=16); a = p.parse_args()
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(a.model); model = AutoModelForSequenceClassification.from_pretrained(a.model).to('cuda').eval()
    if int(getattr(model.config, 'num_labels', -1)) != 1: raise ValueError(f'expected single-logit config, got num_labels={model.config.num_labels}')
    classifier = getattr(model, 'classifier', None)
    if classifier is not None and getattr(classifier, 'out_features', 1) != 1: raise ValueError('classifier output dimension is not 1')
    rows = [json.loads(line) for line in a.chunks.open(encoding='utf-8') if line.strip()]; agg = {}
    for start in range(0, len(rows), a.batch_size):
        batch = rows[start:start + a.batch_size]
        enc = tokenizer([str(x['question']) for x in batch], [str(x['raw_chunk_text']) for x in batch], padding=True, truncation=True, max_length=512, return_tensors='pt').to('cuda')
        with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.float16): logits = model(**enc).logits
        if logits.ndim != 2 or logits.shape[1] != 1: raise ValueError(f'expected [batch, 1] logits, got {tuple(logits.shape)}')
        for row, value in zip(batch, logits[:, 0].float().tolist()):
            key = (str(row['query_id']), str(row['doc_id'])); score = float(value)
            if key not in agg: agg[key] = {'query_id': key[0], 'doc_id': key[1], 'union_rank': int(row['union_rank']), 'doc_score': score}
            else: agg[key]['doc_score'] = max(agg[key]['doc_score'], score)
    a.doc_scores.parent.mkdir(parents=True, exist_ok=True); a.doc_scores.write_text(''.join(json.dumps(x) + '\n' for x in agg.values()), encoding='utf-8')
    out = []
    for q in sorted({key[0] for key in agg}):
        docs = [x for key, x in agg.items() if key[0] == q]; docs.sort(key=lambda x: (-x['doc_score'], x['union_rank'], x['doc_id'])); out.append({'query_id': q, 'top5': [x['doc_id'] for x in docs[:5]]})
    a.predictions.write_text(''.join(json.dumps(x) + '\n' for x in out), encoding='utf-8')
if __name__ == '__main__': main()
