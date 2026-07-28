import type { LegalDocEntry } from "./mock-data";
import type { Citation, RetrievalHit, SavedLegalSource } from "./types";

function sourceId(documentId: string, article?: string, clause?: string, point?: string) {
  return [documentId, article ?? "", clause ?? "", point ?? ""]
    .map((part) => part.trim().toLocaleLowerCase("vi"))
    .join("::");
}

export function savedSourceFromDoc(doc: LegalDocEntry): SavedLegalSource {
  return {
    id: sourceId(doc.doc_id, doc.article, doc.clause, doc.point),
    title: doc.title,
    doc_id: doc.doc_id,
    law_name: doc.law_name,
    doc_number: doc.doc_number,
    article: doc.article,
    clause: doc.clause,
    point: doc.point,
    excerpt: doc.excerpt,
    source: doc.source,
    savedAt: Date.now(),
  };
}

export function savedSourceFromCitation(citation: Citation): SavedLegalSource {
  return {
    id: sourceId(
      citation.doc_id ?? citation.doc_number ?? citation.law_name,
      citation.article,
      citation.clause,
      citation.point,
    ),
    title: citation.title ?? citation.law_name,
    doc_id: citation.doc_id,
    law_name: citation.law_name,
    doc_number: citation.doc_number,
    article: citation.article,
    clause: citation.clause,
    point: citation.point,
    excerpt: citation.excerpt,
    source: citation.source,
    savedAt: Date.now(),
  };
}

export function savedSourceFromHit(hit: RetrievalHit): SavedLegalSource {
  return {
    id: sourceId(hit.doc_id, hit.article, hit.clause, hit.point),
    title: hit.law_name,
    doc_id: hit.doc_id,
    law_name: hit.law_name,
    doc_number: hit.metadata.doc_number,
    article: hit.article,
    clause: hit.clause,
    point: hit.point,
    excerpt: hit.text,
    source: hit.source,
    savedAt: Date.now(),
  };
}
