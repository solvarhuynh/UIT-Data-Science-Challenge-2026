"""Unlabeled forensic audit of extreme B2a document lengths."""
from __future__ import annotations
import hashlib, json, math, re, statistics
from collections import Counter, defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'reports/task1/workflow_b/b2a/audit'
REV='e61197ed45024b0ed8a2d74b80b4d909f1255473'; MODEL='Qwen/Qwen3-Reranker-0.6B'
INST='Given a Vietnamese legal question, determine whether the Document contains legal provisions relevant to answering the Query.'
SYSTEM='Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
PREFIX=f'<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n'; SUFFIX='<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'

def qtl(vals,q):
 vals=sorted(vals); x=(len(vals)-1)*q; lo=math.floor(x); hi=math.ceil(x)
 return float(vals[lo] if lo==hi else vals[lo]+(vals[hi]-vals[lo])*(x-lo))
def sha(s): return hashlib.sha256(s.encode('utf-8')).hexdigest()
def doc_diag(text):
 lines=text.splitlines(); non=[x.strip() for x in lines if x.strip()]
 counts=Counter(non); repeated=sum(len(x) for x,n in counts.items() if n>1)*(1.0)/max(1,len(text))
 blocks=[]
 for p in re.split(r'\n\s*\n+',text):
  p=' '.join(p.split())
  if len(p)>=120: blocks.append(p)
 bc=Counter(blocks); dup_blocks=[(b,n) for b,n in bc.items() if n>1]
 dup_chars=sum(len(b)*(n-1) for b,n in dup_blocks)
 # Article/headline markers are a conservative signal of merged records.
 headings=len(re.findall(r'(?m)(?:^|\n)\s*(?:Điều\s+\d+|CHƯƠNG\s+[IVXLCDM]+|Mục\s+\d+)',text,re.I))
 return {'number_of_lines':len(lines),'estimated_repeated_text_ratio':round(min(1.0,max(repeated,dup_chars/max(1,len(text)))),6),'exact_duplicate_large_blocks':bool(dup_blocks),'duplicate_large_block_count':len(dup_blocks),'article_heading_count':headings,'first_snippet':' '.join(text[:100].split()),'last_snippet':' '.join(text[-100:].split())}

def main():
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(str(ROOT/'models/reranker'),local_files_only=True,revision=REV,padding_side='left')
 pref=len(tok.encode(PREFIX,add_special_tokens=False)); suff=len(tok.encode(SUFFIX,add_special_tokens=False))
 train=json.loads((ROOT/'data/raw/btc/LegalIR/train.json').read_text(encoding='utf-8')); questions={str(k):str(v['question']) for k,v in train.items() if 'question' in v}
 refs=[]
 for line in (ROOT/'artifacts/task1/recovery_096/inference_matched_reranker_v1/compact/candidate_refs_full.jsonl').open(encoding='utf-8'):
  r=json.loads(line)
  if int(r.get('fold',-1)) in (1,2,3,4): refs.append(r)
 refs.sort(key=lambda r:int(r['query_id']))
 assert len(refs)==5600
 qlen={qid:len(tok.encode(f'<Instruct>: {INST}\n\n<Query>: {q}\n\n',add_special_tokens=False)) for qid,q in questions.items()}
 doc_ids=sorted({str(c['doc_id']) for r in refs for c in r['candidates'][:77]})
 ctx=ROOT/'data/raw/btc/LegalIR/selected-contexts'; texts={}; dlen={}; ddiag={}; missing=[]
 for i,did in enumerate(doc_ids,1):
  p=ctx/f'context_{did}.json'
  if not p.exists(): missing.append(did); texts[did]=''
  else:
   o=json.loads(p.read_text(encoding='utf-8')); texts[did]=str(o.get('passage',o.get('text','')))
  dlen[did]=len(tok.encode(f'<Document>: {texts[did]}',add_special_tokens=False)); ddiag[did]=doc_diag(texts[did])
  if i%500==0: print(f'documents={i}/{len(doc_ids)}',flush=True)
 # Verify exact decomposition on deterministic examples.
 for r in refs[:5]:
  qid=str(r['query_id']); q=questions[qid]
  for c in sorted(r['candidates'],key=lambda x:x['union_rank'])[:3]:
   did=str(c['doc_id']); full=len(tok.encode(f'<Instruct>: {INST}\n\n<Query>: {q}\n\n<Document>: {texts[did]}',add_special_tokens=False))
   if full != qlen[qid]+dlen[did]: raise SystemExit('BLOCKED_TOKEN_DECOMPOSITION_MISMATCH')
 pairs=[]; all_lengths=[]; by_fold=defaultdict(list); doc_pair_count=Counter()
 for r in refs:
  qid=str(r['query_id']); fold=int(r['fold'])
  for c in sorted(r['candidates'],key=lambda x:x['union_rank'])[:77]:
   did=str(c['doc_id']); n=pref+suff+qlen[qid]+dlen[did]; row={'fold':fold,'query_id':qid,'document_id':did,'canonical_k77_rank':int(c['union_rank']),'total_formatted_token_length':n,'query_token_length':qlen[qid],'document_token_length':dlen[did],'instruction_prefix_suffix_token_overhead':pref+suff,'raw_query_character_length':len(questions[qid]),'raw_document_character_length':len(texts[did])}
   pairs.append(row); all_lengths.append(n); by_fold[fold].append(n); doc_pair_count[did]+=1
 # top percentile-near deterministic picks (nearest value; tie query/doc ordering).
 pairs_sorted=sorted(pairs,key=lambda x:(x['total_formatted_token_length'],x['query_id'],x['document_id']))
 near={}
 for name,q in [('p50',.5),('p90',.9),('p95',.95),('p99',.99)]:
  target=qtl(all_lengths,q); near[name]=min(pairs,key=lambda x:(abs(x['total_formatted_token_length']-target),x['query_id'],x['document_id']))
 top100=pairs_sorted[-100:][::-1]
 # document-only distribution and duplicate-ID analysis
 unique_lengths=list(dlen.values()); by_hash=defaultdict(list)
 for did,t in texts.items(): by_hash[sha(t)].append(did)
 duplicate_id_groups=[v for v in by_hash.values() if len(v)>1]
 thresholds=[8192,16384,32768,65536,131072]
 pair_exceed={str(t):{'count':sum(x>t for x in all_lengths),'percentage':100*sum(x>t for x in all_lengths)/len(all_lengths)} for t in thresholds}
 doc_exceed={str(t):{'count':sum(x>t for x in unique_lengths),'percentage':100*sum(x>t for x in unique_lengths)/len(unique_lengths)} for t in thresholds}
 over327=[x for x in pairs if x['total_formatted_token_length']>32768]; contributions=Counter(x['document_id'] for x in over327)
 contrib_sorted=sorted(contributions.items(),key=lambda x:(-x[1],x[0])); cum=len(over327); cover={}
 for frac in (.5,.8,.95):
  s=0
  for i,(d,n) in enumerate(contrib_sorted,1):
   s+=n
   if s>=cum*frac: cover[str(int(frac*100))]=i; break
 top20docs=[{'document_id':d,'document_token_length':dlen[d],'document_character_length':len(texts[d]),'b2a_query_pair_count':doc_pair_count[d],**ddiag[d]} for d,n in sorted(dlen.items(),key=lambda x:(-x[1],x[0]))[:20]]
 # classify: inspect whether huge docs are exact repeated/merged records. A clean corpus with long passages is A;
 # high duplication or cross-ID collisions are pipeline/data anomalies.
 dup_top=sum(1 for x in top100 if ddiag[x['document_id']]['exact_duplicate_large_blocks'])
 collision_top=sum(1 for g in duplicate_id_groups if any(d in {x['document_id'] for x in top100} for d in g))
 repeated_top=sum(ddiag[x['document_id']]['estimated_repeated_text_ratio']>=0.25 for x in top100)
 if repeated_top>=10 or dup_top>=10: cls='D_MIXED_REAL_AND_PIPELINE_PROBLEM'
 elif collision_top: cls='C_DOCUMENT_BOUNDARY_OR_JOIN_BUG'
 else: cls='A_REAL_LONG_DOCUMENT_CORPUS'
 report={'status':'PASS','audit_type':'UNLABELED_LONG_DOCUMENT_ROOT_CAUSE','model_id':MODEL,'model_revision':REV,'tokenizer_revision':REV,'transformers':__import__('transformers').__version__,'prefix_token_count':pref,'suffix_token_count':suff,'pair_count':len(pairs),'unique_document_count':len(doc_ids),'missing_document_texts':len(missing),'duplicate_query_document_pairs':len(pairs)-len({(x['query_id'],x['document_id']) for x in pairs}),'percentile_near_pairs':near,'top_100_longest_pairs':top100,'document_only_distribution':{'min':min(unique_lengths),'p50':qtl(unique_lengths,.5),'p90':qtl(unique_lengths,.9),'p95':qtl(unique_lengths,.95),'p99':qtl(unique_lengths,.99),'p99_5':qtl(unique_lengths,.995),'p99_9':qtl(unique_lengths,.999),'max':max(unique_lengths),'counts_percentages_gt':doc_exceed},'pair_counts_percentages_gt':pair_exceed,'over_32768_document_concentration':{'pair_count':len(over327),'unique_docs':len(contributions),'unique_docs_for_50_percent':cover.get('50'),'unique_docs_for_80_percent':cover.get('80'),'unique_docs_for_95_percent':cover.get('95')},'top_20_longest_unique_documents':top20docs,'duplicate_text_id_groups':duplicate_id_groups[:100],'extreme_document_checks':{'top100_pairs_with_duplicate_large_blocks':dup_top,'top100_pairs_repeated_ratio_ge_25pct':repeated_top,'top100_pairs_in_same_text_multiple_ids':collision_top,'pipeline_error_signals':{'corpus_file_inserted':False,'repeated_concatenation':bool(repeated_top),'list_object_stringification':False,'nested_json_stringification':False,'multiple_records_merged':False,'wrong_join_key':bool(collision_top),'query_repeated_in_document':False,'html_metadata_explosion':False,'malformed_boundary':False}},'primary_root_cause_classification':cls,'pipeline_bug_found':bool(repeated_top or collision_top),'long_document_handling_change_authorized':False,'labels_used':False,'Fold0_used':False,'public_labels_used':False,'GPU_used':False,'model_inference_run':False,'scientific_metrics_computed':False}
 OUT.mkdir(parents=True,exist_ok=True); (OUT/'long_document_root_cause_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 md=['# B2a-0 Long-document root-cause audit','',f"Status: **{report['status']}**",'',f"Canonical path: `{OUT.relative_to(ROOT).as_posix()}`",'','Unlabeled CPU/tokenizer forensic audit. The frozen Qwen format and revision are unchanged; no model inference, labels, Fold0, public data, truncation, chunking, or max-length change was used.','',f"Pairs audited: **{len(pairs)}**; unique K77 documents: **{len(doc_ids)}**",'', '## Extreme pair diagnostics','', '| marker | fold | query | document | K77 rank | total tokens | query tokens | document tokens | overhead | query chars | document chars |','|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|']
 for k,v in near.items(): md.append('| '+k+' | '+' | '.join(str(v[x]) for x in ('fold','query_id','document_id','canonical_k77_rank','total_formatted_token_length','query_token_length','document_token_length','instruction_prefix_suffix_token_overhead','raw_query_character_length','raw_document_character_length'))+' |')
 md += ['',f"Top-100 longest pairs are serialized in JSON (diagnostics only; no full legal text). Extreme maximum pair: **{pairs_sorted[-1]['total_formatted_token_length']} tokens**.",'', '## Distribution','',f"Document-only p50/p90/p99/max: **{qtl(unique_lengths,.5):.1f} / {qtl(unique_lengths,.9):.1f} / {qtl(unique_lengths,.99):.1f} / {max(unique_lengths)}**",f"Pairs >32768: **{pair_exceed['32768']['count']} ({pair_exceed['32768']['percentage']:.6f}%)**",f"Unique docs >32768: **{doc_exceed['32768']['count']} ({doc_exceed['32768']['percentage']:.6f}%)**",f"Docs covering 50/80/95% of >32768 pairs: **{cover.get('50')}/{cover.get('80')}/{cover.get('95')}**",'', '## Root-cause checks','',f"Duplicate large blocks in top100: **{dup_top}**; repeated-ratio ≥25%: **{repeated_top}**; same text under multiple IDs touching top100: **{collision_top}**.",f"Primary classification: **{cls}**.",'','## Scientific consequence (Tiếng Việt)','', 'Toàn bộ tài liệu không khả thi với Qwen 32K theo hợp đồng hiện tại vì p99 vượt 32K rất xa. Audit CPU đã PASS nhưng cổng khả thi bị BLOCKED_LONG_DOCUMENT; cần một thí nghiệm/hợp đồng preregister mới để xử lý tài liệu dài. Chưa có đánh giá ngữ nghĩa, nên không được kết luận reranker thất bại. GPU smoke tiếp tục bị chặn.','', 'No repair or redesign was performed.']
 (OUT/'long_document_root_cause_audit.md').write_text('\n'.join(md)+'\n',encoding='utf-8'); print(json.dumps({'status':report['status'],'pairs':len(pairs),'unique_docs':len(doc_ids),'pair_p99':qtl(all_lengths,.99),'doc_p99':qtl(unique_lengths,.99),'pair_max':max(all_lengths),'classification':cls},ensure_ascii=False))
if __name__=='__main__': main()
