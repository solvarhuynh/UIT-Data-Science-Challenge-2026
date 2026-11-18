# Model strategy

## Recommendation

Keep the current DEk21 embedding and BGE reranker as the reproducible anchor until the official training/evaluation split is available for paired out-of-fold (OOF) evaluation. The top-team evidence makes AITeamVN models the first candidates to evaluate, while VNLegal-LAL is a promising legal-domain embedding candidate with a different inference contract.

No public model card or another team's score establishes a 70% probability of improving this repository's competition score. That estimate needs to come from paired OOF results on the same data and metric. Legal Zalo retrieval metrics are useful for screening, not a substitute for the DSC leaderboard.

## Candidates

| Role | Model | Reason to evaluate | Compatibility / caveat |
|---|---|---|---|
| Embedding | [AITeamVN/Vietnamese_Embedding_v2](https://huggingface.co/AITeamVN/Vietnamese_Embedding_v2) | Top-team use; card reports strong legal retrieval results | Candidate only; confirm tokenizer, pooling, dimension and query/document formatting before indexing |
| Reranker | [AITeamVN/Vietnamese_Reranker](https://huggingface.co/AITeamVN/Vietnamese_Reranker) | Top-team use; card reports strong legal retrieval results | Candidate only; benchmark with identical candidate lists; confirm CrossEncoder compatibility |
| Embedding | [darklethelong/vnlegal-lal](https://huggingface.co/darklethelong/vnlegal-lal) | Legal retrieval-specific model; card reports nDCG@10 0.8488 on Zalo AI Legal Text Retrieval | Not drop-in: instruction-prefixed queries and last-token pooling; requires an adapter and parity evaluation |
| Embedding | [mainguyen9/vietlegal-harrier-0.6b](https://huggingface.co/mainguyen9/vietlegal-harrier-0.6b) | Legal-domain model found in top-team materials | Evaluate its documented embedding contract and licensing before use |
| Embedding | [mainguyen9/vietlegal-e5](https://huggingface.co/mainguyen9/vietlegal-e5) | E5-style legal retrieval candidate used by a top team | Query/document prefixes and pooling must match its card |

Model cards and benchmark claims were reviewed on 2026-11-18. Recheck the current model card, license, parameter count (<4B per competition rules), and organizer rules before any competition use. Third-party legal training data may have restrictions even when weights are public.

## Evaluation before promotion

1. Pin a Hub commit and record it with all evaluation outputs.
2. Use exactly the same official training/evaluation split, corpus, chunking, filtering, and metric for each arm.
3. Compare current anchor vs each embedding alone; rebuild the index for each embedding. Compare each reranker on the same fixed candidate set.
4. Report paired query-level deltas, confidence interval/bootstrap, latency, peak VRAM, index size, and failure cases.
5. Promote only when the gain is repeatable and the full inference path meets resource and rule constraints. Keep DEk21/BGE as fallback otherwise.

Do not combine the reported legal benchmark scores with this competition score or infer a numeric improvement probability from them. The downloader exposes candidate checkpoints for evaluation; it does not alter production configuration.

## Download

Install the optional Hub dependency, then request only the role/model needed:

```powershell
python -m pip install -e ".[gpu]"
python scripts/download_models.py --list
python scripts/download_models.py --only aiteam_embedding
```

See `model_registry.md` for the active baseline and model keys. The script records resolved commit SHAs in `models/download_manifest.json`.
