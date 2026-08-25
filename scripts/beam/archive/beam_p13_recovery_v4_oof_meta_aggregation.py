from __future__ import annotations

from beam import function, Image, Volume

import json
import math
import os
from collections import defaultdict
from pathlib import Path

VOLUME_ROOT = Path('/workspace/p13')
RUNTIME = VOLUME_ROOT / 'runtime'
FOLDS = RUNTIME / 'artifacts/task1/evaluation/strict_cv_v2/folds.json'
F0_CACHE = RUNTIME / 'artifacts/task1/diagnostics/p13_recovery_v3_chunkagg_fold0/fold0_original_bge_chunk200_compact.jsonl'
F14_CACHE = RUNTIME / 'artifacts/task1/diagnostics/p13_recovery_v3_confirm_f1to4/f1to4_original_bge_chunk200_compact.jsonl'
OUT_DIR = RUNTIME / 'artifacts/task1/diagnostics/p13_recovery_v4_oof_meta_aggregation'
REPORT = OUT_DIR / 'oof_meta_aggregation_report.json'
OOF_PRED = OUT_DIR / 'oof_predictions.jsonl'

image = Image(python_version='python3.11')


def _read_json_records(path: Path) -> list[dict]:
    """Read normal JSONL and repair the malformed literal-\\n cache from V3 confirm."""
    raw = path.read_text(encoding="utf-8")
    if not raw.strip():
        return []

    # Normal JSONL first.
    try:
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        if rows:
            return rows
    except json.JSONDecodeError:
        pass

    # The first V3-confirm writer accidentally used "\\n" (two literal chars)
    # between JSON objects. Decode object-by-object without touching JSON strings.
    decoder = json.JSONDecoder()
    rows: list[dict] = []
    pos = 0
    n = len(raw)
    while pos < n:
        while pos < n:
            if raw[pos].isspace():
                pos += 1
                continue
            if raw.startswith("\\r\\n", pos):
                pos += 4
                continue
            if raw.startswith("\\n", pos):
                pos += 2
                continue
            break
        if pos >= n:
            break
        obj, end = decoder.raw_decode(raw, pos)
        if not isinstance(obj, dict):
            raise TypeError(f"Expected JSON object in {path}, got {type(obj).__name__}")
        rows.append(obj)
        pos = end

    if not rows:
        raise ValueError(f"Could not decode any records from {path}")

    # Self-heal the cache so Codex/other tools can consume standard JSONL later.
    tmp = path.with_suffix(path.suffix + ".repair.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    print(f"[REPAIR] normalized malformed cache to JSONL: {path} rows={len(rows)}", flush=True)
    return rows


def _z(vals: dict[str, float]) -> dict[str, float]:
    xs = list(vals.values())
    if not xs:
        return {}
    m = sum(xs) / len(xs)
    v = sum((x - m) ** 2 for x in xs) / len(xs)
    sd = math.sqrt(v)
    if sd < 1e-12:
        return {k: 0.0 for k in vals}
    return {k: (x - m) / sd for k, x in vals.items()}


def _rank_score(vals: dict[str, float]) -> dict[str, float]:
    ordered = sorted(vals, key=lambda d: (-vals[d], d))
    n = max(1, len(ordered))
    return {d: 1.0 - i / n for i, d in enumerate(ordered)}


def _agg(values: list[float], kind: str) -> float:
    xs = sorted(values, reverse=True)
    if not xs:
        return -1e9
    if kind == 'max':
        return xs[0]
    if kind == 'mean2':
        return sum(xs[:2]) / len(xs[:2])
    if kind == 'mean3':
        return sum(xs[:3]) / len(xs[:3])
    if kind == 'lse2':
        ys = xs[:2]
    elif kind == 'lse3':
        ys = xs[:3]
    elif kind == 'lse5':
        ys = xs[:5]
    else:
        raise ValueError(kind)
    mx = max(ys)
    return mx + math.log(sum(math.exp(x - mx) for x in ys))


def _doc_features(row: dict) -> dict[str, dict[str, float]]:
    by_doc = defaultdict(lambda: {'bge': [], 'dense': [], 'ranks': []})
    for hit in row['hits']:
        d = str(hit['doc_id'])
        by_doc[d]['bge'].append(float(hit['bge_score']))
        by_doc[d]['dense'].append(float(hit['dense_score']))
        by_doc[d]['ranks'].append(int(hit['dense_rank']))
    out = {}
    for d, g in by_doc.items():
        out[d] = {
            'bge_max': _agg(g['bge'], 'max'),
            'bge_mean2': _agg(g['bge'], 'mean2'),
            'bge_mean3': _agg(g['bge'], 'mean3'),
            'bge_lse2': _agg(g['bge'], 'lse2'),
            'bge_lse3': _agg(g['bge'], 'lse3'),
            'bge_lse5': _agg(g['bge'], 'lse5'),
            'dense_max': _agg(g['dense'], 'max'),
            'dense_mean2': _agg(g['dense'], 'mean2'),
            'dense_mean3': _agg(g['dense'], 'mean3'),
            'best_dense_rank': min(g['ranks']),
            'chunk_count': len(g['bge']),
        }
    return out


def _configs() -> list[dict]:
    cfgs = []
    for bge in ('bge_max', 'bge_mean2', 'bge_mean3', 'bge_lse2', 'bge_lse3', 'bge_lse5'):
        for dense in ('dense_max', 'dense_mean2', 'dense_mean3'):
            for bge_weight_i in range(50, 96, 5):
                cfgs.append({
                    'bge_feature': bge,
                    'dense_feature': dense,
                    'bge_weight': bge_weight_i / 100.0,
                    'dense_weight': 1.0 - bge_weight_i / 100.0,
                    'normalization': 'zscore',
                    'rank_prior_weight': 0.0,
                })
            for bge_weight_i in (60, 70, 80, 90):
                for prior in (0.05, 0.10, 0.15):
                    bw = bge_weight_i / 100.0
                    if bw + prior >= 1.0:
                        continue
                    cfgs.append({
                        'bge_feature': bge,
                        'dense_feature': dense,
                        'bge_weight': bw,
                        'dense_weight': 1.0 - bw - prior,
                        'normalization': 'rank',
                        'rank_prior_weight': prior,
                    })
    return cfgs


def _top5(row: dict, cfg: dict, feat: dict[str, dict[str, float]]) -> list[str]:
    b = {d: f[cfg['bge_feature']] for d, f in feat.items()}
    de = {d: f[cfg['dense_feature']] for d, f in feat.items()}
    prior = {d: 1.0 / math.log2(2.0 + f['best_dense_rank']) for d, f in feat.items()}
    if cfg['normalization'] == 'zscore':
        b = _z(b)
        de = _z(de)
        p = _z(prior)
    else:
        b = _rank_score(b)
        de = _rank_score(de)
        p = _rank_score(prior)
    score = {}
    for d in feat:
        score[d] = (
            cfg['bge_weight'] * b[d]
            + cfg['dense_weight'] * de[d]
            + cfg['rank_prior_weight'] * p[d]
        )
    return sorted(score, key=lambda d: (-score[d], feat[d]['best_dense_rank'], d))[:5]


def _recall(row: dict, top: list[str]) -> float:
    gold = set(row['gold_documents'])
    return len(gold & set(top[:5])) / len(gold)


def _precision(row: dict, top: list[str]) -> float:
    gold = set(row['gold_documents'])
    pred = set(top[:5])
    return len(gold & pred) / len(pred) if pred else 0.0


def _metrics(rows: list[dict], tops: list[list[str]]) -> dict:
    rs = [_recall(r, t) for r, t in zip(rows, tops)]
    ps = [_precision(r, t) for r, t in zip(rows, tops)]
    full = sum(set(r['gold_documents']).issubset(set(t[:5])) for r, t in zip(rows, tops))
    multi = [_recall(r, t) for r, t in zip(rows, tops) if len(set(r['gold_documents'])) > 1]
    return {
        'macro_recall': sum(rs) / len(rs),
        'macro_precision': sum(ps) / len(ps),
        'multi_gold_recall': sum(multi) / len(multi) if multi else None,
        'full_gold_query_count': full,
        'query_count': len(rows),
    }


@function(
    name='udsc-p13-recovery-v4-oof-meta-aggregation',
    cpu=8,
    memory='32Gi',
    image=image,
    volumes=[Volume(name='udsc-p13', mount_path=str(VOLUME_ROOT))],
    timeout=-1,
    retries=1,
    headless=True,
)
def run():
    print('=== P13 RECOVERY V4: NESTED OOF META-AGGREGATION ===', flush=True)
    print('CPU only. Reuses V3 original-BGE chunk caches. No training.', flush=True)
    for p in (FOLDS, F0_CACHE, F14_CACHE):
        print(f'[CHECK] {p} exists={p.is_file()}', flush=True)
        if not p.is_file():
            raise FileNotFoundError(p)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for p in (F0_CACHE, F14_CACHE):
        loaded = _read_json_records(p)
        print(f'[LOAD] {p.name}: rows={len(loaded)}', flush=True)
        rows.extend(loaded)

    fold_payload = json.loads(FOLDS.read_text(encoding='utf-8-sig'))
    qid_to_fold = {}
    for item in fold_payload.get('folds', []):
        fold = int(item['fold'])
        for qid in item.get('validation_ids', []):
            qid_to_fold[str(qid)] = fold
    for row in rows:
        row['fold'] = qid_to_fold[str(row['query_id'])]
    rows.sort(key=lambda r: str(r['query_id']))
    if len(rows) != 7000 or len({r['query_id'] for r in rows}) != 7000:
        raise RuntimeError(f'expected 7000 unique rows, got {len(rows)}')

    features = [_doc_features(r) for r in rows]
    configs = _configs()
    print(f'[GRID] configs={len(configs)}', flush=True)

    # Precompute predictions once.
    pred_cache = {}
    for ci, cfg in enumerate(configs):
        pred_cache[ci] = [_top5(r, cfg, feat) for r, feat in zip(rows, features)]

    # Candidate oracle among unique docs present in raw top200 chunks.
    oracle = []
    for r, feat in zip(rows, features):
        gold = set(r['gold_documents'])
        oracle.append(len(gold & set(feat)) / len(gold))
    oracle_recall = sum(oracle) / len(oracle)

    oof_tops = [None] * len(rows)
    fold_results = []
    selected_configs = []
    for holdout in range(5):
        train_idx = [i for i, r in enumerate(rows) if int(r['fold']) != holdout]
        test_idx = [i for i, r in enumerate(rows) if int(r['fold']) == holdout]

        best_ci = None
        best_key = None
        for ci, cfg in enumerate(configs):
            tops = pred_cache[ci]
            rec = sum(_recall(rows[i], tops[i]) for i in train_idx) / len(train_idx)
            pre = sum(_precision(rows[i], tops[i]) for i in train_idx) / len(train_idx)
            # Conservative tie break: favor higher BGE weight, then simpler max/mean.
            key = (rec, pre, cfg['bge_weight'], -cfg['rank_prior_weight'])
            if best_key is None or key > best_key:
                best_key = key
                best_ci = ci
        assert best_ci is not None
        cfg = configs[best_ci]
        tops = pred_cache[best_ci]
        for i in test_idx:
            oof_tops[i] = tops[i]
        fold_rows = [rows[i] for i in test_idx]
        fold_tops = [tops[i] for i in test_idx]
        fold_metric = _metrics(fold_rows, fold_tops)
        fold_results.append({'fold': holdout, 'selected_config': cfg, **fold_metric})
        selected_configs.append(cfg)
        print(f'[FOLD {holdout}] recall={fold_metric["macro_recall"]:.6f} cfg={cfg}', flush=True)

    if any(t is None for t in oof_tops):
        raise RuntimeError('incomplete OOF predictions')
    oof_metric = _metrics(rows, oof_tops)  # type: ignore[arg-type]

    # Diagnostic global best, explicitly non-OOF.
    global_rows = []
    for ci, cfg in enumerate(configs):
        m = _metrics(rows, pred_cache[ci])
        global_rows.append({'config': cfg, **m})
    global_rows.sort(key=lambda x: (x['macro_recall'], x['macro_precision']), reverse=True)

    with OOF_PRED.open('w', encoding='utf-8') as f:
        for row, top in zip(rows, oof_tops):
            f.write(json.dumps({
                'query_id': row['query_id'],
                'fold': row['fold'],
                'gold_documents': row['gold_documents'],
                'top5': top,
            }, ensure_ascii=False) + '\n')

    report = {
        'schema_version': 'p13-recovery-v4-oof-meta-aggregation-v1',
        'status': 'COMPLETE',
        'note': 'Nested 5-fold config selection. No neural training. Global-best section is diagnostic only.',
        'candidate_oracle_recall_within_raw_top200_unique_docs': oracle_recall,
        'nested_oof': oof_metric,
        'folds': fold_results,
        'selected_configs': selected_configs,
        'diagnostic_global_best': global_rows[:20],
        'oof_predictions': str(OOF_PRED),
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f'REPORT={REPORT}', flush=True)
    return report


if __name__ == '__main__':
    print('Enqueuing P13 Recovery V4 OOF meta-aggregation...', flush=True)
    result = run.remote()
    print('REMOTE RESULT:', result, flush=True)