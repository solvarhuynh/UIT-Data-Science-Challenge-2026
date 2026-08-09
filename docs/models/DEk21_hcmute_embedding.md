# DEk21_hcmute_embedding

DEk21_hcmute_embedding is a Vietnamese text embedding focused on RAG and production efficiency:

**Trained Dataset**: The model was trained on an in-house dataset consisting of approximately 100,000 examples of legal questions and their related contexts.

**Efficiency**: Trained with a Matryoshka loss, allowing embeddings to be truncated with minimal performance loss. This ensures that smaller embeddings are faster to compare, making the model efficient for real-world production use.

## Model Details

### Model Description

- **Model Type**: Sentence Transformer
- **Maximum Sequence Length**: 256 tokens
- **Output Dimensionality**: 768 dimensions
- **Similarity Function**: Cosine Similarity
- **Language**: Vietnamese
- **License**: Apache-2.0

### Model Sources

- **Documentation**: [Sentence Transformers Documentation](https://www.sbert.net/)
- **Repository**: [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face**: [Sentence Transformers on Hugging Face](https://huggingface.co/sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False}) with Transformer model: RobertaModel 
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.

```python
from sentence_transformers import SentenceTransformer
import torch
from pyvi import ViTokenizer

# Load the embedding model from Hugging Face.
# This model is based on PhoBERT, so Vietnamese word segmentation
# is recommended before encoding for optimal performance.
model = SentenceTransformer("huyydangg/DEk21_hcmute_embedding")

# Define a legal question (query) and a set of legal provisions (documents)
query = "Điều kiện để kết hôn hợp pháp là gì?"
docs = [
    "Điều 8 Bộ luật Dân sự 2015 quy định về quyền và nghĩa vụ của công dân trong quan hệ gia đình.",
    "Điều 18 Luật Hôn nhân và gia đình 2014 quy định về độ tuổi kết hôn của nam và nữ.",
    "Điều 14 Bộ luật Dân sự 2015 quy định về quyền và nghĩa vụ của cá nhân khi tham gia hợp đồng.",
    "Điều 27 Luật Hôn nhân và gia đình 2014 quy định về các trường hợp không được kết hôn.",
    "Điều 51 Luật Hôn nhân và gia đình 2014 quy định về việc kết hôn giữa công dân Việt Nam và người nước ngoài."
]

# Apply Vietnamese word segmentation to the query.
# PhoBERT was pretrained on segmented Vietnamese text where
# multi-word expressions are joined by underscores.
segmented_query = ViTokenizer.tokenize(query)

# Apply Vietnamese word segmentation to all documents.
segmented_docs = [ViTokenizer.tokenize(doc) for doc in docs]

# Generate embeddings for the query and documents.
query_embedding = model.encode([segmented_query])
doc_embeddings = model.encode(segmented_docs)

# Compute cosine similarity between the query embedding
# and each document embedding.
similarities = torch.nn.functional.cosine_similarity(
    torch.tensor(query_embedding),
    torch.tensor(doc_embeddings)
).flatten()

# Rank documents by similarity score in descending order.
sorted_indices = torch.argsort(similarities, descending=True)
sorted_docs = [docs[idx] for idx in sorted_indices]
sorted_scores = [similarities[idx].item() for idx in sorted_indices]

# Display the ranked documents and their similarity scores.
for doc, score in zip(sorted_docs, sorted_scores):
    print(f"Document: {doc} - Cosine Similarity: {score:.4f}")
```

## Evaluation

### Metrics

**Information Retrieval**

**Datasets**: [another-symato/VMTEB-Zalo-legel-retrieval-wseg](https://huggingface.co/datasets/another-symato/VMTEB-Zalo-legel-retrieval-wseg)

Evaluated with InformationRetrievalEvaluator

| Model | Type | ndcg@3 | ndcg@5 | ndcg@10 | mrr@3 | mrr@5 | mrr@10 |
|-------|------|--------|--------|---------|-------|-------|--------|
| huyydangg/DEk21_hcmute_embedding_wseg | dense | 0.908405 | 0.914792 | 0.917742 | 0.889583 | 0.893099 | 0.894266 |
| AITeamVN/Vietnamese_Embedding | dense | 0.842687 | 0.854993 | 0.865006 | 0.822135 | 0.82901 | 0.833389 |
| bkai-foundation-models/vietnamese-bi-encoder | hybrid | 0.827247 | 0.844781 | 0.846937 | 0.799219 | 0.809505 | 0.806771 |
| bkai-foundation-models/vietnamese-bi-encoder | dense | 0.814116 | 0.82965 | 0.839567 | 0.796615 | 0.805286 | 0.809572 |
| AITeamVN/Vietnamese_Embedding | hybrid | 0.788724 | 0.810062 | 0.820797 | 0.758333 | 0.77224 | 0.776461 |
| BAAI/bge-m3 | dense | 0.784056 | 0.80665 | 0.817016 | 0.763281 | 0.775859 | 0.780293 |
| BAAI/bge-m3 | hybrid | 0.775239 | 0.797382 | 0.811962 | 0.747656 | 0.763333 | 0.77128 |
| huyydangg/DEk21_hcmute_embedding | dense | 0.752173 | 0.769259 | 0.785101 | 0.72474 | 0.734427 | 0.741076 |
| hiieu/halong_embedding | hybrid | 0.73627 | 0.757183 | 0.779169 | 0.710417 | 0.721901 | 0.731976 |
| bm25 | bm25 | 0.728122 | 0.74974 | 0.761612 | 0.699479 | 0.711198 | 0.715738 |
| dangvantuan/vietnamese-embedding | dense | 0.718971 | 0.746521 | 0.763416 | 0.696354 | 0.711953 | 0.718854 |
| dangvantuan/vietnamese-embedding | hybrid | 0.71711 | 0.743537 | 0.758315 | 0.690104 | 0.704792 | 0.712261 |
| VoVanPhuc/sup-SimCSE-VietNamese-phobert-base | hybrid | 0.688483 | 0.713829 | 0.733894 | 0.660156 | 0.671198 | 0.676961 |
| hiieu/halong_embedding | dense | 0.656377 | 0.675881 | 0.701368 | 0.630469 | 0.641406 | 0.652057 |
| VoVanPhuc/sup-SimCSE-VietNamese-phobert-base | dense | 0.558852 | 0.584799 | 0.611329 | 0.536979 | 0.55112 | 0.562218 |

## Citation

You can cite our work as below:

```bibtex
@misc{DEk21_hcmute_embedding,
  title={DEk21_hcmute_embedding: A Vietnamese Text Embedding},
  author={QUANG HUY},
  year={2025},
  publisher={Huggingface},
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
