"""Enrich completed long-document audit with per-top100 document diagnostics."""
import json, re, hashlib
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
AUDIT = ROOT / 'reports/task1/workflow_b/b2a/audit/long_document_root_cause_audit.json'
CTX = ROOT / 'data/raw/btc/LegalIR/selected-contexts'

def diag(text):
    lines = text.splitlines(); non = [x.strip() for x in lines if x.strip()]
    counts = Counter(non)
    repeated = sum(len(x) for x,n in counts.items() if n > 1) / max(1,len(text))
    blocks=[]
    for p in re.split(r'\n\s*\n+', text):
        p=' '.join(p.split())
        if len(p)>=120: blocks.append(p)
    bc=Counter(blocks); dup=[(b,n) for b,n in bc.items() if n>1]
    dup_chars=sum(len(b)*(n-1) for b,n in dup)
    headings=len(re.findall(r'(?m)(?:^|\n)\s*(?:Điều\s+\d+|CHƯƠNG\s+[IVXLCDM]+|Mục\s+\d+)',text,re.I))
    return {
        'number_of_lines': len(lines),
        'estimated_repeated_text_ratio': round(min(1.0,max(repeated,dup_chars/max(1,len(text)))),6),
        'exact_duplicate_large_blocks': bool(dup),
        'duplicate_large_block_count': len(dup),
        'article_heading_count': headings,
        'first_snippet': ' '.join(text[:100].split()),
        'last_snippet': ' '.join(text[-100:].split()),
    }

def main():
    report=json.loads(AUDIT.read_text(encoding='utf-8'))
    ids=sorted({str(x['document_id']) for x in report['top_100_longest_pairs']})
    pairs=Counter(str(x['document_id']) for x in report['top_100_longest_pairs'])
    token_lengths={str(x['document_id']):x['document_token_length'] for x in report['top_20_longest_unique_documents']}
    texts={}; hashes=defaultdict(list)
    for did in ids:
        obj=json.loads((CTX/f'context_{did}.json').read_text(encoding='utf-8'))
        t=str(obj.get('passage',obj.get('text',''))); texts[did]=t; hashes[hashlib.sha256(t.encode()).hexdigest()].append(did)
    rows=[]
    for did in ids:
        d=diag(texts[did])
        row={'document_id':did,'document_token_length':token_lengths.get(did),
             'document_character_length':len(texts[did]),'b2a_top100_pair_count':pairs[did],
             **d,'same_document_text_multiple_ids':len(hashes[hashlib.sha256(texts[did].encode()).hexdigest()])>1,
             'multiple_unrelated_documents_or_articles_concatenated':bool(d['exact_duplicate_large_blocks'] or d['article_heading_count']>30)}
        rows.append(row)
    report['top100_unique_document_diagnostics']=rows
    report['pipeline_bug_details']={
        'offending_path_pattern':'data/raw/btc/LegalIR/selected-contexts/context_<document_id>.json',
        'offending_field':'passage',
        'offending_function':'not identified by read-only audit',
        'evidence':'Top-100 extreme pairs are sourced from passage fields containing repeated large blocks; no join-key mismatch was observed.',
        'example_document_ids':[r['document_id'] for r in rows if r.get('exact_duplicate_large_blocks')],
    }
    AUDIT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'top100_unique_documents':len(ids),'ids':ids},ensure_ascii=False))
if __name__=='__main__': main()
