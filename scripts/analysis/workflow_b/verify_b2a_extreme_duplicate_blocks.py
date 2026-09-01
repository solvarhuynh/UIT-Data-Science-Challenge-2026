"""Read-only supplemental verification of the historical 120-character block rule."""
import json, re
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
AUDIT=ROOT/'reports/task1/workflow_b/tv2/b2a/audit/passage_provenance_corruption_audit.json'
CTX=ROOT/'data/raw/btc/LegalIR/selected-contexts'

def count_legacy(text):
    blocks=[]
    for p in re.split(r'\n\s*\n+',text):
        p=' '.join(p.split())
        if len(p)>=120: blocks.append(p)
    return len([1 for _,n in Counter(blocks).items() if n>1])

def main():
    report=json.loads(AUDIT.read_text(encoding='utf-8'))
    for did in ('68843','4644'):
        text=str(json.loads((CTX/f'context_{did}.json').read_text(encoding='utf-8')).get('passage',''))
        report['special_extreme_documents'][did]['historical_120_char_normalized_paragraph_rule']=count_legacy(text)
    AUDIT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print({d:report['special_extreme_documents'][d]['historical_120_char_normalized_paragraph_rule'] for d in ('68843','4644')})
if __name__=='__main__': main()
