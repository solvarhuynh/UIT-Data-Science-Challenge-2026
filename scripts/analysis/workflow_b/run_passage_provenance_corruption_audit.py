"""Read-only B2a passage provenance and long-document corruption audit."""
from __future__ import annotations
import hashlib, json, math, re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
REV = "e61197ed45024b0ed8a2d74b80b4d909f1255473"
MODEL = "Qwen/Qwen3-Reranker-0.6B"
INST = "Given a Vietnamese legal question, determine whether the Document contains legal provisions relevant to answering the Query."

def quantile(values, q):
    values = sorted(values); x = (len(values) - 1) * q; lo, hi = math.floor(x), math.ceil(x)
    return values[lo] if lo == hi else values[lo] + (values[hi] - values[lo]) * (x - lo)

def normalize(s): return " ".join(s.split()).lower()

def diagnostics(text):
    # Large-block rule: paragraphs normalized from blank-line boundaries, >=512 chars.
    paragraphs = [normalize(x) for x in re.split(r"\n\s*\n+", text)]
    blocks = [x for x in paragraphs if len(x) >= 512]
    counts = Counter(blocks); duplicate_blocks = {b:n for b,n in counts.items() if n > 1}
    duplicate_chars = sum(len(b) * (n - 1) for b,n in duplicate_blocks.items())
    # Also inspect substantial complete lines, never short legal phrases.
    lines = [normalize(x) for x in text.splitlines() if len(normalize(x)) >= 160]
    line_counts = Counter(lines); duplicate_lines = sum(n - 1 for n in line_counts.values() if n > 1)
    first, last = normalize(text[:2048]), normalize(text[-2048:])
    marker_re = re.compile(r"(đăng nhập|đăng ký thành viên|quên mật khẩu|mật khẩu|thành viên tại đây|login|sign in)", re.I)
    headers = len(re.findall(r"(?mi)^\s*(?:BỘ |QUỐC HỘI|CHÍNH PHỦ|THỦ TƯỚNG|ỦY BAN NHÂN DÂN|CỘNG HÒA XÃ HỘI)", text))
    return {
        "normalization_rule": "lowercase, collapse whitespace; blank-line paragraphs of >=512 normalized characters; substantial lines >=160 normalized characters",
        "line_count": len(text.splitlines()),
        "normalized_blocks_total": len(blocks),
        "normalized_blocks_unique": len(counts),
        "exact_duplicate_large_block_count": len(duplicate_blocks),
        "duplicate_large_block_occurrences": sum(n - 1 for n in duplicate_blocks.values()),
        "identical_substantial_line_repetitions": duplicate_lines,
        "estimated_repeated_text_ratio": round(duplicate_chars / max(1, len(text)), 6),
        "first_last_region_repetition": bool(first and last and (first == last or first in last or last in first)),
        "web_template_footer_markers": sorted(set(m.group(0).lower() for m in marker_re.finditer(text))),
        "mechanical_multi_source_header_count": headers,
        "abrupt_unrelated_transition_mechanically_detected": bool(headers > 8 and (duplicate_blocks or marker_re.search(text))),
    }

def canonical_output_dir():
    manifest = ROOT / "reports/task1/workflow_b/artifact_migration_manifest.json"
    if manifest.exists():
        m = json.loads(manifest.read_text(encoding="utf-8"))
        for item in m.get("moves", []):
            if item.get("old_path") == "reports/task1/workflow_b/b2a/":
                p = ROOT / item["new_path"] / "audit"
                if p.exists(): return p, manifest.relative_to(ROOT).as_posix()
    return ROOT / "reports/task1/workflow_b/tv2/b2a/audit", "fallback modern namespace"

def main():
    from transformers import AutoTokenizer
    out, migration_source = canonical_output_dir(); out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(str(ROOT / "models/reranker"), local_files_only=True, revision=REV, padding_side="left")
    prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    overhead = len(tok.encode(prefix, add_special_tokens=False)) + len(tok.encode(suffix, add_special_tokens=False))
    train = json.loads((ROOT / "data/raw/btc/LegalIR/train.json").read_text(encoding="utf-8"))
    questions = {str(k):str(v["question"]) for k,v in train.items() if "question" in v}
    refs = [json.loads(x) for x in (ROOT / "artifacts/task1/recovery_096/inference_matched_reranker_v1/compact/candidate_refs_full.jsonl").read_text(encoding="utf-8").splitlines()]
    refs = [r for r in refs if int(r.get("fold", -1)) in (1,2,3,4)]
    assert len(refs) == 5600
    qlen = {qid:len(tok.encode(f"<Instruct>: {INST}\n\n<Query>: {q}\n\n", add_special_tokens=False)) for qid,q in questions.items()}
    doc_ids = sorted({str(c["doc_id"]) for r in refs for c in sorted(r["candidates"], key=lambda x:x["union_rank"])[:77]})
    ctx = ROOT / "data/raw/btc/LegalIR/selected-contexts"; texts={}; lens={}; pairs=Counter()
    for i,did in enumerate(doc_ids, 1):
        obj=json.loads((ctx / f"context_{did}.json").read_text(encoding="utf-8")); text=str(obj.get("passage", obj.get("text", "")))
        texts[did]=text; lens[did]=len(tok.encode(f"<Document>: {text}", add_special_tokens=False))
        if i % 500 == 0: print(f"documents={i}/{len(doc_ids)}", flush=True)
    pair_lengths=[]; pair_over_docs=Counter(); pair_max_by_doc=Counter()
    for r in refs:
        qid=str(r["query_id"])
        for c in sorted(r["candidates"], key=lambda x:x["union_rank"])[:77]:
            did=str(c["doc_id"]); n=overhead+qlen[qid]+lens[did]; pair_lengths.append(n); pairs[did]+=1; pair_max_by_doc[did]=max(pair_max_by_doc[did],n)
            if n > 32768: pair_over_docs[did] += 1
    strict_docs=sorted(d for d,n in lens.items() if n > 32768)
    ge_docs=sorted(d for d,n in lens.items() if n >= 32768)
    eq_docs=sorted(d for d,n in lens.items() if n == 32768)
    pair_gt=sum(n > 32768 for n in pair_lengths); pair_ge=sum(n >= 32768 for n in pair_lengths); pair_eq=sum(n == 32768 for n in pair_lengths)
    pair_only_docs=sorted(set(pair_over_docs)-set(strict_docs))
    pair_only_details=[{"document_id":d,"document_token_length":lens[d],"max_formatted_pair_token_length":pair_max_by_doc[d],"over_threshold_pair_count":pair_over_docs[d]} for d in pair_only_docs]
    records=[]; categories=Counter(); repair=Counter()
    for did in strict_docs:
        d=diagnostics(texts[did])
        suspect = d["exact_duplicate_large_block_count"] > 0 or d["estimated_repeated_text_ratio"] >= 0.10 or bool(d["web_template_footer_markers"])
        category = "D_MIXED_OR_AMBIGUOUS" if suspect else "C_LIKELY_GENUINE_LONG_DOCUMENT"
        d.update({"document_id":did,"token_length":lens[did],"character_length":len(texts[did]),"B2a_query_pair_count":pairs[did],"source_context_path":f"data/raw/btc/LegalIR/selected-contexts/context_{did}.json","category":category,"anomaly_provenance":"UNRESOLVED","repairability":"MANUAL_OR_ARCHITECTURE_REVIEW_REQUIRED" if suspect else "NO_CORRUPTION_REPAIR_NEEDED"})
        records.append(d); categories[category]+=1; repair[d["repairability"]]+=1
    special={r["document_id"]:r for r in records if r["document_id"] in {"68843","4644"}}
    source_provenance={
      "passage_producer_identified":False,
      "status":"BLOCKED_PASSAGE_PRODUCER_UNRESOLVED",
      "repository_code_search":{
        "searched_terms":["selected-contexts","context_<document_id>.json","passage"],
        "producer_script_found":None,
        "consumer_scripts":["scripts/task1/run_legal_ir_pipeline.py","scripts/analysis/workflow_b/run_b2a_context_audit.py","scripts/analysis/workflow_b/run_long_document_root_cause_audit.py"],
        "evidence":"Repository references selected-contexts as an input corpus only; no code that creates or appends context_<id>.json was found."
      },
      "upstream_source":"BTC-distributed data/raw/btc/LegalIR/selected-contexts.zip (documented source archive; archive/producer code unavailable in repository)",
      "transformation_sequence":"External BTC archive -> extracted data/raw/btc/LegalIR/selected-contexts/context_<id>.json -> consumer scripts read JSON field passage",
      "source_comparison":"UNRESOLVED: upstream pre-extraction/web source and generator code are unavailable locally."
    }
    report={
      "status":"BLOCKED", "audit_type":"READ_ONLY_DATA_PROVENANCE_CORRUPTION_MAP", "canonical_output_directory":out.relative_to(ROOT).as_posix(), "migration_manifest":migration_source,
      "model_id":MODEL,"model_revision":REV,"tokenizer_revision":REV,"tokenizer_name_or_path":str(ROOT / "models/reranker"),"data_modified":False,"GPU_used":False,"model_inference":False,"labels_used":False,"Fold0_used":False,"public_labels_used":False,"chunking_introduced":False,
      "passage_provenance":source_provenance,
      "boundary_reconciliation":{"operator":"STRICT_GT_32768","document_token_length_gt_32768":len(strict_docs),"document_token_length_ge_32768":len(ge_docs),"document_token_length_eq_32768":len(eq_docs),"document_ids_eq_32768":eq_docs,"pair_total_length_gt_32768":pair_gt,"pair_total_length_ge_32768":pair_ge,"pair_total_length_eq_32768":pair_eq,"pair_unique_documents_gt_32768":len(pair_over_docs),"pair_only_document_details":pair_only_details,"resolution":"AUDIT_BOUNDARY_SEMANTICS_MISMATCH: document-only uses <Document>-field length while pair concentration uses full formatted pair length including query/template overhead; pair-only documents are below the document-only cutoff but cross it as formatted pairs.","previous_document_only_gt":668,"previous_pair_concentration_unique_docs":669},
      "long_document_count":len(records),"long_documents":records,"category_counts":dict(categories),"repairability_counts":dict(repair),"safe_deterministic_repair_candidates":[],"dry_run_projection":{"label":"DRY_RUN_PROJECTION_ONLY","safe_candidate_count":0,"projected_unique_docs_gt_32768":len(strict_docs),"projected_pair_count_gt_32768":pair_gt,"projected_pair_percentage_gt_32768":100*pair_gt/len(pair_lengths)},
      "special_extreme_documents":{did:{**row,"first_confirmed_duplicate_enters":"The released context JSON passage field; earlier producer/source cannot be established from repository code.","identical_content_exists_upstream":"UNRESOLVED","unrelated_content_concatenated":"LIKELY_BUT_UNRESOLVED","producing_function":"UNRESOLVED"} for did,row in special.items()},
      "decision":"NEEDS_MORE_PROVENANCE","scientific_B2a_result_produced":False,"scientific_B2a_status":"NOT_YET_EVALUATED"
    }
    (out / "passage_provenance_corruption_audit.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    c=report["category_counts"]; r=report["boundary_reconciliation"]
    md=f"""# B2a-0 Passage provenance and long-document corruption map\n\nStatus: **BLOCKED** — `BLOCKED_PASSAGE_PRODUCER_UNRESOLVED`\n\nCanonical output directory: `{report['canonical_output_directory']}`\n\n## Boundary reconciliation\n\nThe canonical operator is **STRICT_GT_32768**. Document-only count is **{r['document_token_length_gt_32768']}**; `==32768` count is **{r['document_token_length_eq_32768']}**. Pair total length has **{r['pair_total_length_gt_32768']}** rows and touches **{r['pair_unique_documents_gt_32768']}** documents above 32768. The historical 668/669 difference is resolved: document-only length excludes query/template content, while pair length includes it; a document at the boundary crosses the pair threshold after overhead.\n\n## Passage provenance\n\nRepository code consumes `data/raw/btc/LegalIR/selected-contexts/context_<id>.json`, field `passage`; it does not contain a producer that generates, appends, paginates, deduplicates, or concatenates those files. The available documentation identifies `selected-contexts.zip` as BTC-distributed source data. Therefore source-vs-generator attribution is **UNRESOLVED** and no data repair is authorized.\n\n## Corruption map\n\nLong documents audited: **{len(records)}**. Likely genuine long documents: **{c.get('C_LIKELY_GENUINE_LONG_DOCUMENT',0)}**. Mixed/ambiguous: **{c.get('D_MIXED_OR_AMBIGUOUS',0)}**. Confirmed pipeline/upstream corruption: **0 / 0**.\n\nA long document is not automatically faulty: legal annexes, schedules, tables, and consolidated instruments can be legitimately long. `68843` and `4644` retain strong duplicate-block/boilerplate signals, but the available evidence only proves that they are present in the released `passage` field—not whether they were introduced upstream or by an unavailable generator.\n\nSafe deterministic repair candidates: **0**. Dry-run projection without any repair: **{pair_gt} pairs ({100*pair_gt/len(pair_lengths):.6f}%)** and **{len(strict_docs)} unique documents** remain above 32K. GPU and chunking remain out of scope; no Qwen inference or scientific metric was run.\n\nNext: **NEEDS_MORE_PROVENANCE**.\n"""
    (out / "passage_provenance_corruption_audit.md").write_text(md,encoding="utf-8")
    print(json.dumps({"status":report["status"],"canonical_output_directory":report["canonical_output_directory"],"strict_gt_docs":len(strict_docs),"eq_docs":len(eq_docs),"pair_gt":pair_gt,"pair_docs_gt":len(pair_over_docs),"categories":dict(categories)},ensure_ascii=False))

if __name__ == "__main__": main()
