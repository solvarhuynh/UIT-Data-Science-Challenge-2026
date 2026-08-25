from __future__ import annotations
import argparse,json,os,hashlib
from pathlib import Path
def main():
 p=argparse.ArgumentParser(); p.add_argument('--train-rows',type=Path,required=True); p.add_argument('--questions',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args(); q=json.loads(a.questions.read_text(encoding='utf-8-sig')); tmp=a.output.with_suffix('.tmp'); n=0; miss=0
 with tmp.open('w',encoding='utf-8') as o:
  for l in a.train_rows.open(encoding='utf-8'):
   if not l.strip(): continue
   r=json.loads(l); text=str(q[str(r['query_id'])]['question']) if str(r['query_id']) in q else ''; miss+=not bool(text); r['question']=text; o.write(json.dumps(r,ensure_ascii=False)+'\n'); n+=1
 os.replace(tmp,a.output); report={'train_rows':n,'missing_question':miss,'sha256':hashlib.sha256(a.output.read_bytes()).hexdigest()}; (a.output.with_suffix('.report.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
if __name__=='__main__': main()
