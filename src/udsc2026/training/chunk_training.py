"""Sample query groups from the same chunks that the cross-encoder scores."""

import re
import unicodedata
from collections.abc import Collection, Sequence


def _signature(text: str) -> str:
    return re.sub(r"\W+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def build_training_record(
    query_id: str,
    question: str,
    gold_documents: Sequence[str],
    hits: list[dict],
    *,
    excluded_documents: Collection[str] = (),
    positives_per_query: int = 3,
    negatives_per_band: int = 3,
) -> dict | None:
    gold = set(gold_documents)
    positives = []
    seen_positive_docs = set()
    for hit in hits:
        if hit["doc_id"] in gold and hit["doc_id"] not in seen_positive_docs:
            positives.append(hit)
            seen_positive_docs.add(hit["doc_id"])
        if len(positives) == positives_per_query:
            break
    if not positives:
        return None

    positive_laws = {
        hit.get("law_name") or hit.get("metadata", {}).get("law_name")
        for hit in positives
    } - {None, ""}
    positive_text = [_signature(hit["text"]) for hit in positives]
    seen_text = set(positive_text)
    seen_docs = gold.union(excluded_documents)
    negatives = {"hard": [], "medium": [], "tail": [], "same_law": []}
    for rank, hit in enumerate(hits, 1):
        if hit["doc_id"] in seen_docs:
            continue
        band = "hard" if rank <= 20 else "medium" if rank <= 100 else "tail"
        law = hit.get("law_name") or hit.get("metadata", {}).get("law_name")
        if law in positive_laws and len(negatives["same_law"]) < negatives_per_band:
            band = "same_law"
        if len(negatives[band]) >= negatives_per_band:
            continue
        text = _signature(hit["text"])
        if not text or text in seen_text:
            continue
        # Exclude long verbatim subsections of positive passages.
        if len(text) >= 120 and any(
            len(pos) >= 120 and (text in pos or pos in text) for pos in positive_text
        ):
            continue
        negatives[band].append(hit)
        seen_docs.add(hit["doc_id"])
        seen_text.add(text)

    if not any(negatives.values()):
        return None
    fields = ("doc_id", "chunk_id", "text")
    return {
        "query_id": query_id,
        "question": question,
        "positive_chunks": [{key: hit[key] for key in fields} for hit in positives],
        "negative_chunks": {
            band: [{key: hit[key] for key in fields} for hit in values]
            for band, values in negatives.items()
        },
    }
