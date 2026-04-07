# 04 – Semantic & Vector Search

This chapter marks the transition from classical lexical search to semantic retrieval — where the goal shifts from matching keywords to understanding meaning. While BM25 and inverted indexes remain essential, modern search systems increasingly rely on neural embeddings to capture semantic similarity, handle vocabulary mismatch, and support cross-lingual retrieval. Here you'll learn how text becomes dense vectors, how to search billions of vectors efficiently, when to use sparse vs. dense methods, how to combine them effectively, and how to apply neural re-ranking for maximum precision.

## Topics

1. [Word Embeddings](./word-embeddings.md) — Word2Vec (CBOW, Skip-gram), GloVe (global co-occurrence factorization), FastText (subword representations), training objectives, properties of embedding spaces, and limitations for search.
2. [Sentence and Document Embeddings](./sentence-embeddings.md) — BERT pooling strategies, Sentence-BERT (Siamese networks, contrastive learning), multi-lingual models, asymmetric embeddings for queries vs. documents, and practical considerations for production.
3. [Approximate Nearest Neighbor Search](./ann-search.md) — The ANN problem definition, Locality-Sensitive Hashing (LSH), tree-based methods (Annoy, KD-trees), graph-based methods (HNSW, NSG), IVF with quantization (IVF-PQ, IVF-HNSW), complexity analysis, and index-building trade-offs.
4. [Dense vs. Sparse Retrieval](./dense-vs-sparse.md) — Sparse retrieval (BM25, learned sparse models like SPLADE), dense retrieval (bi-encoders, dual encoders), when each method wins, vocabulary mismatch problems, and evaluation on BEIR/MS MARCO benchmarks.
5. [Hybrid Search](./hybrid-search.md) — Combining lexical + semantic signals, score normalization strategies, fusion algorithms (linear combination, Reciprocal Rank Fusion), late interaction models (ColBERT), and practical implementation patterns in production systems.
6. [Cross-Encoder Re-Ranking](./cross-encoder-reranking.md) — Bi-encoder vs. cross-encoder architectures, two-stage retrieval pipelines, training cross-encoders with pairwise/listwise losses, latency vs. accuracy trade-offs, distillation for faster re-rankers, and when re-ranking is worth the cost.

## Recommended Study Order

Follow the order above. Each file builds on the previous:

- **Word Embeddings** establishes the foundation — you need to understand how static embeddings work before moving to contextualized models.
- **Sentence and Document Embeddings** shows how to move from word-level to passage-level representations, which is what you actually index in search.
- **Approximate Nearest Neighbor Search** is critical infrastructure — without efficient ANN, vector search doesn't scale.
- **Dense vs. Sparse Retrieval** compares the two paradigms and clarifies when each is appropriate.
- **Hybrid Search** shows how to combine both approaches to get the best of both worlds.
- **Cross-Encoder Re-Ranking** completes the picture with a two-stage architecture that balances recall and precision.

## Prerequisites

- Chapter 02 (Core Concepts) — especially [Vector Space Model](../02-core-concepts/retrieval-models.md) and [TF-IDF and BM25](../02-core-concepts/tf-idf-and-bm25.md)
- Linear algebra fundamentals — dot products, cosine similarity, matrix factorization
- Basic neural network concepts — embeddings, loss functions, backpropagation
- Familiarity with the recall/precision trade-off and evaluation metrics (MAP, NDCG)

## How This Relates to Later Chapters

| Concept from this chapter | Where it shows up later |
|---|---|
| Bi-encoders, Sentence-BERT | Ch 06 (vector databases), Ch 07 (neural ranking), Ch 08 (RAG) |
| HNSW, IVF indexes | Ch 06 (Pinecone, Weaviate, Qdrant internals) |
| Hybrid search (BM25 + vectors) | Ch 06 (Elasticsearch vector search), Ch 09 (production patterns) |
| Cross-encoder re-ranking | Ch 07 (learning to rank, ColBERT), Ch 09 (relevance tuning) |
| Dense retrieval | Ch 08 (RAG retrieval stage, multi-modal search) |
| Quantization (PQ, scalar) | Ch 06 (vector DB compression), Ch 09 (scaling to billions) |

## Why Vector Search Matters Now

A few years ago, vector search was a research curiosity. In 2026, it's production infrastructure:

1. **LLMs shifted the landscape** — RAG systems require dense retrieval as a primitive. Every LLM application has a retrieval component.
2. **Quality gains are real** — On out-of-domain queries, dense retrieval can improve recall@10 by 30-50% over BM25 alone. Hybrid search is now the default recommendation for new systems.
3. **Infrastructure matured** — HNSW is production-ready. Vector databases handle billions of vectors with <10ms p99 latency. Embedding models are cheap to run.
4. **But classical methods still matter** — BM25 wins on entity search, rare terms, and exact match scenarios. The best systems use both.

Understanding both paradigms deeply — and knowing when to apply each — is the mark of a strong search engineer in 2026. This chapter gives you that foundation.
