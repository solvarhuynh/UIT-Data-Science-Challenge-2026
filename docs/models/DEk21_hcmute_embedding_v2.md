# DEk21_hcmute_embedding_v2

DEk21_hcmute_embedding_v2 is a Vietnamese sentence embedding model fine-tuned for semantic retrieval.

This model is the v2 successor of huyydangg/DEk21_hcmute_embedding. It keeps the Vietnamese retrieval focus while improving production usability through cached contrastive learning and Matryoshka embedding truncation.

## Model Highlights

- **Task**: Vietnamese semantic search and sentence similarity
- **Domain**: Vietnamese retrieval data
- **Base model**: bkai-foundation-models/vietnamese-bi-encoder
- **Training loss**: MatryoshkaLoss(CachedMultipleNegativesRankingLoss)
- **Matryoshka dimensions**: 768, 512, 256
- **Similarity**: Cosine similarity
- **License**: Apache-2.0

## Training Data

The model was trained on an in-house Vietnamese question-context retrieval dataset with approximately 90,000 rows.

## Training Overview

The model was fine-tuned with Sentence Transformers from bkai-foundation-models/vietnamese-bi-encoder using a retrieval-focused contrastive objective.

The training objective combines:

- CachedMultipleNegativesRankingLoss for memory-efficient contrastive learning;
- MatryoshkaLoss for truncation-friendly embeddings at 768, 512, and 256 dimensions.

## Model Details

- **Model Type**: Sentence Transformer
- **Language**: Vietnamese
- **Domain**: Vietnamese question-context retrieval
- **Output Dimensionality**: 768 dimensions
- **Truncation-Friendly Dimensions**: 512 and 256 dimensions
- **Similarity Function**: Cosine Similarity
- **License**: Apache-2.0

## Why V2?

Compared with huyydangg/DEk21_hcmute_embedding, this version is tuned with a more retrieval-focused, memory-efficient, and production-friendly training setup.

The Matryoshka setup allows downstream systems to use shorter vectors such as 512 or 256 dimensions, reducing storage and speeding up vector search while keeping useful semantic information.

## Usage

### Direct Usage With Sentence Transformers + Word Segmentation

```bash
pip install -U sentence-transformers pyvi
```

```python
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from pyvi import ViTokenizer

model = SentenceTransformer("huyydangg/DEk21_hcmute_embedding_v2")

query = "Điều kiện để kết hôn hợp pháp là gì?"
documents = [
    "Điều 18 Luật Hôn nhân và gia đình 2014 quy định về độ tuổi kết hôn của nam và nữ.",
    "Điều kiện doanh nghiệp được hoạt động tư vấn giám sát.",
    "Quy định về trợ cấp với cán bộ xã già yếu nghỉ việc.",
]

segmented_query = ViTokenizer.tokenize(query)
segmented_documents = [ViTokenizer.tokenize(document) for document in documents]

query_embedding = model.encode(segmented_query, convert_to_tensor=True)
document_embeddings = model.encode(segmented_documents, convert_to_tensor=True)

scores = cos_sim(query_embedding, document_embeddings)[0]
ranked_results = sorted(zip(documents, scores.tolist()), key=lambda item: item[1], reverse=True)

for document, score in ranked_results:
    print(f"{score:.4f} - {document}")
```

### Optional Truncated Embeddings

Because the model was trained with Matryoshka dimensions, retrieval systems can evaluate or deploy shorter vectors such as 512 or 256 dimensions when embedding truncation is supported.

## Intended Use

This model is intended for:

- Vietnamese semantic retrieval;
- production vector search systems that need a balance between quality, vector size, and search speed.

## Evaluation

The models were evaluated on the same Vietnamese question-context retrieval benchmark using cosine similarity.

The comparison includes:

- DEk21_hcmute_embedding_v2 (ours)
- DEk21_hcmute_embedding (previous version)
- bkai-vietnamese-bi-encoder (base model)

### Retrieval Performance (768 Dimensions)

| Metric | DEk21_v2 | DEk21_v1 | BKAI Base |
|--------|----------|----------|-----------|
| Accuracy@1 | 0.231 | 0.239 | 0.167 |
| Accuracy@3 | 0.367 | 0.342 | 0.269 |
| Accuracy@5 | 0.437 | 0.380 | 0.312 |
| Accuracy@10 | 0.519 | 0.436 | 0.374 |
| Accuracy@20 | 0.589 | 0.502 | 0.441 |
| Accuracy@25 | 0.612 | 0.521 | 0.467 |
| MAP@10 | 0.317 | 0.299 | 0.230 |
| MAP@100 | 0.326 | 0.307 | 0.239 |
| MRR@10 | 0.317 | 0.299 | 0.230 |
| MRR@25 | 0.323 | 0.304 | 0.236 |
| nDCG@10 | 0.365 | 0.332 | 0.264 |
| nDCG@25 | 0.388 | 0.353 | 0.287 |

### Matryoshka Evaluation

| Dimension | Accuracy@10 | MAP@10 | nDCG@10 |
|-----------|-------------|--------|---------|
| 768 | 0.519 | 0.317 | 0.365 |
| 512 | 0.514 | 0.317 | 0.364 |
| 256 | 0.497 | 0.306 | 0.352 |
| 128 | 0.460 | 0.283 | 0.325 |
| 64 | 0.395 | 0.230 | 0.269 |

### Relative Improvement

| Metric | v2 vs v1 | v2 vs BKAI |
|--------|----------|------------|
| Accuracy@10 | +19.0% | +38.8% |
| Accuracy@20 | +17.3% | +33.6% |
| Accuracy@25 | +17.5% | +31.0% |
| MAP@10 | +6.1% | +38.0% |
| MAP@100 | +6.1% | +36.7% |
| MRR@10 | +6.1% | +38.0% |
| nDCG@10 | +10.2% | +38.4% |
| nDCG@25 | +10.1% | +35.4% |

### Key Observations

- DEk21_hcmute_embedding_v2 achieves the best overall retrieval performance across almost all retrieval metrics.
- Compared with DEk21_hcmute_embedding, the new model substantially improves retrieval quality at larger candidate pools (Top-10 to Top-25).
- Compared with the original bkai-vietnamese-bi-encoder, the model consistently improves retrieval performance by more than 30% on most ranking metrics.
- Thanks to MatryoshkaLoss, embeddings can be truncated to 512 or 256 dimensions with only a small reduction in retrieval quality, enabling faster vector search and lower storage cost.

## Citation

You can cite this model as below:

```bibtex
@misc{DEk21_hcmute_embedding_v2,
  title={DEk21_hcmute_embedding_v2: Vietnamese Text Embedding for Retrieval},
  author={QUANG HUY},
  year={2026},
  publisher={Hugging Face},
}
```

### BibTeX

**Sentence Transformers**

```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

**MatryoshkaLoss**

```bibtex
@misc{kusupati2024matryoshka,
    title={Matryoshka Representation Learning},
    author={Aditya Kusupati and Gantavya Bhatt and Aniket Rege and Matthew Wallingford and Aditya Sinha and Vivek Ramanujan and William Howard-Snyder and Kaifeng Chen and Sham Kakade and Prateek Jain and Ali Farhadi},
    year={2024},
    eprint={2205.13147},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

**CachedMultipleNegativesRankingLoss / GradCache**

```bibtex
@misc{gao2021scaling,
    title={Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup},
    author={Luyu Gao and Yunyi Zhang and Jiawei Han and Jamie Callan},
    year={2021},
    eprint={2101.06983},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

**MultipleNegativesRankingLoss**

```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
