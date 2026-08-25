"""Real Fold0 V1A train/infer pipeline (GPU execution is opt-in)."""
from __future__ import annotations
import argparse,json,hashlib,time,random
from pathlib import Path
def main():
 p=argparse.ArgumentParser(); p.add_argument('--train',type=Path,required=True); p.add_argument('--base-model',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--max-length',type=int,default=512); p.add_argument('--learning-rate',type=float,default=5e-7); p.add_argument('--epochs',type=int,default=1); p.add_argument('--batch-size',type=int,default=4); p.add_argument('--seed',type=int,default=1337); p.add_argument('--allow-training',action='store_true'); a=p.parse_args()
 rows=[json.loads(x) for x in a.train.open(encoding='utf-8') if x.strip()]
 if any(int(r['fold'])==0 for r in rows): raise ValueError('fold0 leakage')
 if a.max_length!=512 or a.learning_rate!=5e-7 or a.epochs>1: raise ValueError('V1A contract mismatch')
 a.output_dir.mkdir(parents=True,exist_ok=True); manifest={'train_rows':len(rows),'train_queries':len({r['query_id'] for r in rows}),'train_sha256':hashlib.sha256(a.train.read_bytes()).hexdigest(),'base_model':str(a.base_model),'max_length':512,'learning_rate':a.learning_rate,'epochs':a.epochs,'batch_size':a.batch_size,'seed':a.seed,'representation':'question + raw_chunk_text','aggregation':'max_chunk_score_per_doc','status':'PREFLIGHT_PASS'}
 if not a.allow_training: (a.output_dir/'run_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8'); return
 import torch
 from torch.utils.data import DataLoader
 from transformers import AutoTokenizer,AutoModelForSequenceClassification
 random.seed(a.seed); torch.manual_seed(a.seed); tok=AutoTokenizer.from_pretrained(str(a.base_model)); model=AutoModelForSequenceClassification.from_pretrained(str(a.base_model)).to('cuda');
 if int(getattr(model.config,'num_labels',-1))!=1 or getattr(model.classifier,'out_proj',model.classifier).weight.shape[0]!=1: raise ValueError('BASE_RERANKER_SINGLE_LOGIT_FAIL')
 print('BASE_RERANKER_SINGLE_LOGIT_PASS',flush=True); opt=torch.optim.AdamW(model.parameters(),lr=a.learning_rate); scaler=torch.cuda.amp.GradScaler(enabled=True); loss_fn=torch.nn.BCEWithLogitsLoss(); model.train(); start=time.time()
 for _ in range(a.epochs):
  random.shuffle(rows)
  for i in range(0,len(rows),a.batch_size):
   b=rows[i:i+a.batch_size]; enc=tok([str(x['question']) for x in b],[str(x['raw_chunk_text']) for x in b],padding=True,truncation=True,max_length=512,return_tensors='pt').to('cuda'); y=torch.tensor([int(x['label']) for x in b],device='cuda'); opt.zero_grad(set_to_none=True)
   with torch.cuda.amp.autocast(): loss=loss_fn(model(**enc).logits.reshape(-1),y.float())
   scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
 model.save_pretrained(a.output_dir); tok.save_pretrained(a.output_dir); manifest.update({'status':'TRAINING_COMPLETE','train_runtime_seconds':time.time()-start}); (a.output_dir/'run_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
if __name__=='__main__': main()
