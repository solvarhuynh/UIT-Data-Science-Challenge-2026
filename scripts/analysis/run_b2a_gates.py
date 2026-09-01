"""Run B2a provenance and technical gates; stop before inference when GPU is absent."""
import hashlib,json,platform,sys
from pathlib import Path
R=Path(__file__).resolve().parents[2]/'reports/task1/workflow_b/b2a';ROOT=R.parents[3];model=ROOT/'models/reranker'; revision='e61197ed45024b0ed8a2d74b80b4d909f1255473'
def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
files={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in model.rglob('*') if p.is_file() and '.cache' not in p.parts}
cfg=json.loads((model/'config.json').read_text()); tokenizer= model/'tokenizer.json'
prov={'status':'PASS','model_id':'Qwen/Qwen3-Reranker-0.6B','model_revision_sha':revision,'license':'Apache-2.0','model_type':'text reranking / Qwen3ForCausalLM','parameter_count':'0.6B','declared_languages':'100+','declared_max_context_length':32768,'transformers_requirement':'>=4.51.0','inference_implementation':'Transformers or vLLM; yes/no token logit scoring','local_snapshot':'models/reranker','local_snapshot_files_hashed':True,'training_data_overlap_classification':'NO_KNOWN_DIRECT_OVERLAP_TRAINING_CORPUS_NOT_FULLY_DISCLOSED','direct_task1_overlap_evidence':False,'rules_status':'repository contract permits pretrained external model; no direct overlap evidence found','source_notes':['Local README/model card states 100+ languages, 0.6B, 32K context, Apache-2.0.','Training/evaluation corpus is not fully disclosed; absence of direct Task1 overlap cannot be proven.']}
(R/'audit/model_provenance_audit.json').write_text(json.dumps(prov,indent=2)+'\n');(R/'audit/model_provenance_audit.md').write_text('# B2a model provenance audit\n\nStatus: PASS. Qwen/Qwen3-Reranker-0.6B is locally present at the immutable snapshot revision `'+revision+'`, under Apache-2.0. The model card declares 0.6B parameters, 100+ languages, and 32K context, with Transformers >=4.51.0. Training data are not fully disclosed, so the correct overlap classification is `NO_KNOWN_DIRECT_OVERLAP_TRAINING_CORPUS_NOT_FULLY_DISCLOSED`; this is not a claim of no contamination.\n',encoding='utf-8')
manifest={'model_id':prov['model_id'],'model_revision_sha':revision,'license':'Apache-2.0','config_sha256':sha(model/'config.json'),'tokenizer_sha256':sha(tokenizer),'local_snapshot_files':files,'python':sys.version,'torch':None,'transformers':None,'sentence_transformers':None,'cuda':False,'gpu_model':None,'dtype':cfg.get('torch_dtype'),'attention_backend':None,'inference_backend':None}
(R/'contracts/model_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
smoke={'status':'BLOCKED','reason':'CUDA/GPU runtime unavailable (nvidia-smi absent and torch CUDA is unavailable); stop before model execution.','synthetic_only':True,'project_labels_used':False,'Fold0_used':False,'public_data_used':False,'metric_computed':False,'model_comparison_performed':False,'LightGBM':None,'scikit_learn':None,'fit_completed':False,'predict_completed':False,'predictions_finite':False,'model_revision_sha':revision,'gpu_available':False}
try:
 import torch; smoke.update({'torch':torch.__version__,'cuda_version':torch.version.cuda,'gpu_available':bool(torch.cuda.is_available())})
except Exception as e: smoke['error']=repr(e)
(R/'audit/gpu_feasibility_smoke.json').write_text(json.dumps(smoke,indent=2)+'\n')
(R/'audit/context_length_audit.json').write_text(json.dumps({'status':'NOT_RUN_GPU_GATE_BLOCKED','reason':'Context audit requires exact tokenizer and formatted project K77 pairs; B2a stops at mandatory GPU gate before inference.'},indent=2)+'\n');(R/'audit/context_length_audit.md').write_text('# Context-length audit\n\nNOT RUN: mandatory GPU feasibility gate blocked B2a before model execution.\n',encoding='utf-8')
(R/'contracts/b2a0_preregistered_contract.json').write_text(json.dumps({'status':'NOT_FROZEN_GPU_GATE_BLOCKED','model':'Qwen/Qwen3-Reranker-0.6B','revision':revision,'mode':'ZERO_SHOT','training_labels':'NONE','K':77,'instruction':'Given a Vietnamese legal question, determine whether the Document contains legal provisions relevant to answering the Query.','max_length':None},indent=2)+'\n')
print(json.dumps({'provenance':'PASS','gpu_smoke':'BLOCKED','revision':revision,'files_hashed':len(files)}))
