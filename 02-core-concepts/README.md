# 02 – Core Concepts

This chapter bridges the mathematical foundations from Chapter 01 with the practical systems you'll build in later chapters. Here you'll learn how to think about relevance, how to measure whether your search system is any good, how the classical scoring functions (TF-IDF, BM25) actually work in practice, what the major retrieval models look like as implemented systems, and how the query processing pipeline transforms raw user input into something a search engine can act on.

## Topics

1. [Relevance and Ranking](./relevance-and-ranking.md) — What relevance means, how human judgments are collected, ranking frameworks, and multi-stage ranking architectures.
2. [Evaluation Metrics](./evaluation-metrics.md) — Precision, recall, F1, MAP, NDCG, MRR — with worked examples, statistical significance testing, and online vs. offline evaluation.
3. [TF-IDF and BM25](./tf-idf-and-bm25.md) — From raw term counts to production scoring functions. TF-IDF weighting schemes, BM25 parameter tuning, BM25F for multi-field documents, and practical worked examples.
4. [Retrieval Models](./retrieval-models.md) — Boolean retrieval, the vector space model, and language models as practical retrieval systems, with implementation considerations and trade-offs.
5. [Query Processing](./query-processing.md) — The full query processing pipeline: parsing, normalization, expansion, rewriting, spell correction, and intent classification.

## Recommended Study Order

Follow the order above. Each file builds on the previous:

- **Relevance and Ranking** defines the problem you're trying to solve — without understanding relevance, metrics are just numbers.
- **Evaluation Metrics** gives you the tools to measure whether your system is improving — you need these before comparing scoring functions.
- **TF-IDF and BM25** is the practical core — these are the scoring functions you'll use in every system from Elasticsearch to custom engines.
- **Retrieval Models** zooms out to the model-level view — Boolean, vector space, and language models as complete retrieval frameworks.
- **Query Processing** covers the online path — everything that happens between the user pressing Enter and the retrieval engine receiving a processed query.

## Prerequisites

- Chapter 01 (Fundamentals) — especially [Mathematical Foundations](../01-fundamentals/mathematical-foundations.md) for the probabilistic and linear algebra background
- Familiarity with the inverted index from [Data Structures](../01-fundamentals/data-structures.md)
- Comfort with logarithms, conditional probability, and vector operations

## How This Relates to Later Chapters

| Concept from this chapter | Where it shows up later |
|---|---|
| Relevance judgments | Ch 07 (training data for learning to rank), Ch 09 (evaluation pipelines) |
| NDCG, MAP | Ch 07 (LTR loss functions), Ch 09 (A/B testing) |
| BM25 | Ch 03 (full-text search), Ch 06 (Lucene/ES default similarity) |
| BM25F | Ch 06 (multi-field search in Elasticsearch) |
| Vector space model | Ch 04 (dense retrieval, hybrid search) |
| Query expansion | Ch 03 (fuzzy search), Ch 04 (semantic expansion) |
| Intent classification | Ch 07 (query understanding with ML), Ch 08 (conversational search) |
| Multi-stage ranking | Ch 05 (query routing), Ch 07 (re-ranking with neural models) |
