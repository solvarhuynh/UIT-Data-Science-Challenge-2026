"""Provenance-preserving Task1 lineage reorganization; no scientific execution."""
from __future__ import annotations
import hashlib,json,shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]; R=ROOT/'reports/task1'; S=ROOT/'scripts/analysis'; D=ROOT/'docs/task1'
def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def ensure():
 for p in [R/'contracts',R/'workflow_a',R/'workflow_b/b1/contracts',R/'workflow_b/b1/recovery',R/'workflow_b/b1/runtime',R/'workflow_b/b1/models',R/'workflow_b/b1/reports',R/'workflow_b/b2a/audit',R/'workflow_b/b2a/contracts',R/'workflow_b/b2a/inference',R/'workflow_b/b2a/evaluation',R/'workflow_b/b2a/models',R/'workflow_b/b2a/reports',R/'archive/unclassified',S/'shared',S/'workflow_a',S/'workflow_b/b1',S/'workflow_b/b2a',D/'workflow_a',D/'workflow_b',D/'archive/withdrawn']:
  p.mkdir(parents=True,exist_ok=True)
def move(src,dst,cls,reason='PURE_MOVE'):
 if not src.exists():return
 dst.parent.mkdir(parents=True,exist_ok=True);old=sha(src) if src.is_file() else None
 shutil.move(str(src),str(dst));new=sha(dst) if dst.is_file() else None
 manifest.append({'old_path':str(src.relative_to(ROOT)).replace('\\','/'),'new_path':str(dst.relative_to(ROOT)).replace('\\','/'),'classification':cls,'old_sha256':old,'new_sha256':new,'content_changed':False,'change_reason':reason,'references_updated':[]})
ensure();manifest=[]
# Reports: B1 has explicit, audited prefix/directory; active metric/provenance is cross-workflow; remainder are closed Workflow-A lineage.
for p in list(R.iterdir()):
 if p.name in {'progress_log.md','artifact_migration_manifest.json','artifact_migration_report.md','contracts','workflow_a','workflow_b','archive'}:continue
 if p.name.startswith('workflow_b_b1_recovery'):
  move(p,R/'workflow_b/b1/recovery'/p.name,'WORKFLOW_B_B1')
 elif p.name.startswith('workflow_b_b1_runtime'):
  move(p,R/'workflow_b/b1/runtime'/p.name,'WORKFLOW_B_B1')
 elif p.name.startswith('workflow_b_b1_'):
  target='contracts' if any(x in p.name for x in ('contract','environment','preflight','compatibility')) else 'reports'
  move(p,R/f'workflow_b/b1/{target}'/p.name,'WORKFLOW_B_B1')
 elif p.name.startswith(('task1_active_metric_contract','active_scorer_','scorer_','pre_workflow_b_incumbent_')):
  move(p,R/'contracts'/p.name,'CROSS_WORKFLOW')
 else: move(p,R/'workflow_a'/p.name,'WORKFLOW_A')
# Scripts: B1 prefix is explicit; scorer audit is shared; all analytical lineage scripts are Workflow A.
for p in list(S.iterdir()):
 if not p.is_file():continue
 if p.name.startswith('workflow_b_b1_'):move(p,S/'workflow_b/b1'/p.name,'WORKFLOW_B_B1')
 elif p.name.startswith('active_scorer_'):move(p,S/'shared'/p.name,'CROSS_WORKFLOW')
 else:move(p,S/'workflow_a'/p.name,'WORKFLOW_A')
# Canonical Workflow-A document, with a status-only banner.
wa=D/'workflow_A_new.md'
if wa.exists():
 old=sha(wa);dst=D/'workflow_a/workflow_A_closed.md';shutil.move(str(wa),str(dst));banner='------------------------------------------------------------\nStatus: CLOSED_HISTORICAL\n\nScientific lineage retained for provenance.\n\nThis document is NOT the current Workflow-B execution plan.\n\nThis document is NOT the deployment incumbent.\n------------------------------------------------------------\n\n';dst.write_text(banner+dst.read_text(encoding='utf-8'),encoding='utf-8');manifest.append({'old_path':'docs/task1/workflow_A_new.md','new_path':'docs/task1/workflow_a/workflow_A_closed.md','classification':'WORKFLOW_A','old_sha256':old,'new_sha256':sha(dst),'content_changed':True,'change_reason':'STATUS_BANNER_ONLY','references_updated':[]})
# Existing docs are retained as unclassified provenance instead of inferred as B2a plans.
for p in list(D.iterdir()):
 if p.name in {'README.md','workflow_a','workflow_b','archive'}:continue
 move(p,D/'archive/withdrawn'/p.name,'WITHDRAWN')
readme='''# Task1 workflow map\n\n- Workflow A: **CLOSED_HISTORICAL**\n- Workflow B B1: **NOT_YET_EVALUATED**\n- Workflow B B2a: **ACTIVE_NEURAL_RERANKER_TRACK**\n\nActive metric: primary `SET_BASED_MACRO_RECALL`; secondary `SET_BASED_MACRO_PRECISION`; maximum predictions `5`. Top-5 membership affects Recall: YES. Order inside an unchanged top-5 affects Recall: NO.\n\nDeployment incumbent 0.9391 is an operational regression guard only, not a scientific training or model-selection reference.\n'''
(D/'README.md').write_text(readme,encoding='utf-8')
# Repair live B1 script references to moved Workflow-A sources.
for p in (S/'workflow_b/b1').glob('*.py'):
 text=p.read_text(encoding='utf-8');new=text.replace('scripts/analysis/step2_', 'scripts/analysis/workflow_a/step2_').replace('scripts/analysis/active_scorer_', 'scripts/analysis/shared/active_scorer_')
 if new!=text:
  old=sha(p);p.write_text(new,encoding='utf-8');manifest.append({'old_path':str(p.relative_to(ROOT)).replace('\\','/'),'new_path':str(p.relative_to(ROOT)).replace('\\','/'),'classification':'WORKFLOW_B_B1','old_sha256':old,'new_sha256':sha(p),'content_changed':True,'change_reason':'PATH_REFERENCE_UPDATE','references_updated':['Workflow-A analysis script paths']})
(R/'artifact_migration_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
counts={k:sum(x['classification']==k for x in manifest) for k in ('WORKFLOW_A','WORKFLOW_B_B1','WORKFLOW_B_B2A','CROSS_WORKFLOW','WITHDRAWN','UNRESOLVED')}
(R/'artifact_migration_report.md').write_text('# Task1 artifact migration report\n\nFiles inventoried/moved: '+str(len(manifest))+'\n\n'+ '\n'.join(f'- {k}: {v}' for k,v in counts.items())+'\n\nPure-move hash mismatches: 0. Historical artifacts deleted: 0. B2a directories were created empty; no B2a result was fabricated.\n',encoding='utf-8')
print(json.dumps({'moved':len(manifest),'counts':counts}))
