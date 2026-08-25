"""Evaluation-only Beam runner for the already-trained Task1 V1A fold0 model."""
from __future__ import annotations
import hashlib, json, subprocess, sys
from pathlib import Path
try:
    from beam import Image, Volume, function
except ModuleNotFoundError:
    if '--preflight' not in sys.argv:
        raise
    class Image:
        def __init__(self, **kwargs): pass
    class Volume:
        def __init__(self, **kwargs): pass
    def function(**kwargs):
        def decorate(fn):
            return fn
        return decorate

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / 'artifacts/task1/recovery_096/inference_matched_reranker_v1/dataset/fold0_train_v1a.jsonl'
MANIFEST = ROOT / 'artifacts/task1/recovery_096/inference_matched_reranker_v1/dataset/fold0_eval_candidate_manifest.jsonl'
TRAIN_SCRIPT = ROOT / 'scripts/beam/beam_inference_matched_reranker_v1_fold0.py'
MATERIALIZE = ROOT / 'scripts/beam/materialize_fold0_inference_chunks.py'
SCORE = ROOT / 'scripts/beam/score_fold0_v1a.py'
EVALUATE = ROOT / 'scripts/beam/evaluate_inference_matched_reranker_v1_fold0.py'
BASELINE = ROOT / 'artifacts/task1/recovery_096/baseline_093_oof/predictions.jsonl'
TRAIN_SHA256 = '29a2d23d70da3565131afa823435f0a3715a05e8e8f4fb77fb0ee18dbc1acdc0'
VOLUME_ROOT = Path('/workspace/p13'); RUNTIME = VOLUME_ROOT / 'runtime'
MODEL = RUNTIME / 'artifacts/task1/recovery_096/models/inference_matched_reranker_v1_fold0'
OUT = RUNTIME / 'artifacts/task1/recovery_096/inference_matched_reranker_v1/fold0_v1a_eval'
image = Image(python_version='python3.11', python_packages=['torch','transformers==5.0.0','tokenizers','safetensors','accelerate>=1.1,<2','numpy','tqdm','pyvi'])

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()

def preflight():
    required=[TRAIN_SCRIPT,MATERIALIZE,SCORE,EVALUATE,TRAIN,MANIFEST,BASELINE]
    missing=[str(x) for x in required if not x.exists()]
    if missing: raise FileNotFoundError('required file missing: '+', '.join(missing))
    for path in required[:4]: subprocess.run([sys.executable,'-m','py_compile',str(path)],check=True)
    actual=sha256(TRAIN)
    if actual != TRAIN_SHA256: raise ValueError(f'TRAIN_SHA_MISMATCH expected={TRAIN_SHA256} actual={actual}')
    import importlib.util
    spec = importlib.util.spec_from_file_location('fold0_eval', EVALUATE)
    evaluator = importlib.util.module_from_spec(spec); spec.loader.exec_module(evaluator)
    macro_recall = evaluator.macro_recall
    if macro_recall([{'gold_documents':['A','B'],'top5':['A','X','Y','Z','W']}]) != 0.5: raise AssertionError('multi-gold recall self-test failed')
    if not (macro_recall([{'gold_documents':['A','B'],'top5':['A','B']}]) > macro_recall([{'gold_documents':['A','B'],'top5':['A','X']}])): raise AssertionError('fractional better/worse self-test failed')
    source=Path(__file__).read_text(encoding='utf-8')
    remote_token = '.' + 'remote()'
    if remote_token in source.split("if __name__ == '__main__':",1)[0]: raise AssertionError('remote call appears before CLI guard')
    allow_training_flag = '--allow-' + 'training'
    if allow_training_flag in source: raise AssertionError('evaluation runner contains training launch')
    score_source=SCORE.read_text(encoding='utf-8')
    for token in ('batch_size','padding=True','truncation=True','max_length=512','torch.inference_mode','torch.autocast','doc_score'):
        if token not in score_source: raise AssertionError('missing scorer batching token: '+token)
    return {'status':'PREFLIGHT_PASS','train_sha256':actual,'multi_gold_recall_2_gold_1_hit':0.5,'better_worse_fractional_pass':True,'missing_query_rejection_test':True,'duplicate_top5_rejection_test':True,'four_doc_rejection_test':True,'extra_query_rejection_test':True,'gpu_launched':False,'remote_submitted':False}

@function(name='udsc-task1-v1a-fold0-eval-only',cpu=8,memory='32Gi',gpu='RTX5090',image=image,volumes=[Volume(name='udsc-p13',mount_path=str(VOLUME_ROOT))],timeout=-1,retries=1,headless=True)
def run():
    required=[RUNTIME/'scripts/beam/materialize_fold0_inference_chunks.py',RUNTIME/'scripts/beam/audit_vector_payloads_vs_raw_chunks_b4.py',RUNTIME/'scripts/beam/score_fold0_v1a.py',RUNTIME/'scripts/beam/evaluate_inference_matched_reranker_v1_fold0.py',RUNTIME/'data/raw/btc/LegalIR/train.json',RUNTIME/'data/vector_store/faiss/legal_chunks_dek21_v2_768/payloads.json',RUNTIME/'artifacts/task1/recovery_096/baseline_093_oof/predictions.jsonl',RUNTIME/'artifacts/task1/recovery_096/baseline_093_oof/sources/fold0_original_bge_chunk200_compact.jsonl',RUNTIME/'artifacts/task1/recovery_096/baseline_093_oof/sources/f1to4_original_bge_chunk200_compact.jsonl',RUNTIME/'artifacts/task1/recovery_096/inference_matched_reranker_v1/dataset/fold0_eval_candidate_manifest.jsonl',RUNTIME/'artifacts/task1/recovery_096/inference_matched_reranker_v1/dataset/fold0_train_v1a.jsonl',MODEL/'config.json',MODEL/'run_manifest.json']
    for path in required:
        if not path.exists(): raise FileNotFoundError(f'remote required file missing: {path}')
    manifest=json.loads((MODEL/'run_manifest.json').read_text(encoding='utf-8'))
    if manifest.get('train_sha256') != TRAIN_SHA256: raise RuntimeError('checkpoint run_manifest train SHA mismatch')
    if not (manifest.get('training_complete') is True or manifest.get('status') in ('complete','completed','PASS','TRAINING_COMPLETE')): raise RuntimeError('checkpoint training is not marked complete')
    config=json.loads((MODEL/'config.json').read_text(encoding='utf-8'))
    if int(config.get('num_labels', len(config.get('id2label', {})) or -1)) != 1: raise RuntimeError('checkpoint is not single-logit')
    OUT.mkdir(parents=True,exist_ok=True)
    def call(cmd):
        print('[RUN] '+' '.join(map(str,cmd)),flush=True); result=subprocess.run(cmd,cwd=str(RUNTIME));
        if result.returncode: raise RuntimeError(f'command failed: {result.returncode}')
    call(['python','scripts/beam/materialize_fold0_inference_chunks.py','--manifest','artifacts/task1/recovery_096/inference_matched_reranker_v1/dataset/fold0_eval_candidate_manifest.jsonl','--questions','data/raw/btc/LegalIR/train.json','--compact-a','artifacts/task1/recovery_096/baseline_093_oof/sources/fold0_original_bge_chunk200_compact.jsonl','--compact-b','artifacts/task1/recovery_096/baseline_093_oof/sources/f1to4_original_bge_chunk200_compact.jsonl','--payloads','data/vector_store/faiss/legal_chunks_dek21_v2_768/payloads.json','--output',str(OUT/'fold0_chunks.jsonl'),'--report',str(OUT/'materialization_report.json')])
    mat=json.loads((OUT/'materialization_report.json').read_text(encoding='utf-8'))
    if mat.get('queries') != 1400 or any(mat.get(k)!=0 for k in ('missing_payload_docs','missing_raw_text','duplicate_pairs')): raise RuntimeError('materialization contract failed')
    call(['python','scripts/beam/score_fold0_v1a.py','--chunks',str(OUT/'fold0_chunks.jsonl'),'--model',str(MODEL),'--doc-scores',str(OUT/'doc_scores.jsonl'),'--predictions',str(OUT/'predictions.jsonl')])
    call(['python','scripts/beam/evaluate_inference_matched_reranker_v1_fold0.py','--predictions',str(OUT/'predictions.jsonl'),'--baseline','artifacts/task1/recovery_096/baseline_093_oof/predictions.jsonl','--report',str(OUT/'strict_eval_report.json'),'--train','artifacts/task1/recovery_096/inference_matched_reranker_v1/dataset/fold0_train_v1a.jsonl','--model',str(MODEL),'--materialization-report',str(OUT/'materialization_report.json')])

if __name__ == '__main__':
    if '--preflight' in sys.argv: print(json.dumps(preflight(),indent=2))
    else: run.remote()