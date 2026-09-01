from pathlib import Path
import shutil, zipfile, hashlib

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / 'artifacts/task1/large_outputs'
ZIP = ROOT / 'task1_large_outputs.zip'
FILES = [
 'reports/task1/workflow_a/step2p4_proxy_predictions.jsonl',
 'reports/task1/workflow_a/step2p5_fused_scores.jsonl',
 'reports/task1/workflow_a/step2p5_p1_inner_full_action_scores.jsonl',
 'reports/task1/workflow_a/step2p1f_full_action_scores.jsonl',
 'reports/task1/workflow_a/step2p1a_median_scores.jsonl',
 'reports/task1/workflow_a/step2f2_full_action_scores.jsonl',
 'reports/task1/workflow_a/step2p4_proxy_inner_oof_predictions.jsonl',
 'reports/task1/workflow_b/tv4/b1/recovery/workflow_b_b1_recovery/f1_actions.npz',
 'reports/task1/workflow_b/tv4/b1/recovery/workflow_b_b1_recovery/f2_actions.npz',
 'reports/task1/workflow_b/tv4/b1/recovery/workflow_b_b1_recovery/f3_actions.npz',
 'reports/task1/workflow_b/tv4/b1/recovery/workflow_b_b1_recovery/f4_actions.npz',
]

def main():
    DEST.mkdir(parents=True, exist_ok=True)
    moved=[]
    for rel in FILES:
        src=ROOT/rel
        if src.exists():
            dst=DEST/Path(rel).name
            shutil.move(str(src), str(dst)); moved.append((rel,dst.name,dst.stat().st_size))
    with zipfile.ZipFile(ZIP,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for rel,name,size in moved: z.write(DEST/name, arcname=f'large_outputs/{name}')
    manifest=DEST/'MANIFEST.txt'
    manifest.write_text('\n'.join(f'{rel}\t{name}\t{size} bytes\t{hashlib.sha256((DEST/name).read_bytes()).hexdigest()}' for rel,name,size in moved)+'\n',encoding='utf-8')
    print(f'moved={len(moved)} zip={ZIP} bytes={ZIP.stat().st_size}')
if __name__=='__main__': main()
