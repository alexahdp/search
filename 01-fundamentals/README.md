# 01 – Fundamentals

This chapter builds the foundation for everything that follows. Before diving into ranking algorithms, distributed architectures, or neural retrieval models, you need a solid grasp of the primitives: what search actually is as a problem, the math that underlies retrieval models, how raw text becomes something a machine can work with, and which data structures make sub-second query latency possible over billions of documents.

## Topics

1. [What Is Search](./what-is-search.md) — Problem definition, taxonomy of search types, the information retrieval pipeline, and how search differs from databases.
2. [Mathematical Foundations](./mathematical-foundations.md) — Set theory (Boolean retrieval), probability (language models, BM25 derivation), and linear algebra (vector space model, cosine similarity, LSA).
3. [Text Representation](./text-representation.md) — Tokenization strategies, stemming vs. lemmatization, normalization (Unicode, case folding, stop words), and how these choices directly affect recall and precision.
4. [Data Structures for Search](./data-structures.md) — Inverted index internals (posting lists, skip pointers, compression), tries, suffix arrays, B-trees, and skip lists — with complexity analysis and real-world trade-offs.

## Recommended Study Order

Follow the order above. Each file is self-contained, but later sections assume familiarity with earlier ones:

- **What Is Search** frames the problem space and introduces terminology used everywhere else.
- **Mathematical Foundations** gives you the formal models (Boolean, probabilistic, vector space) that motivate design decisions in data structures and text processing.
- **Text Representation** is where theory meets practice — you'll see why tokenization choices can make or break recall.
- **Data Structures** ties everything together: these are the concrete implementations that turn the theory into systems serving thousands of queries per second.

## Prerequisites

- Comfort with Big-O notation and basic algorithm analysis
- Working knowledge of hash maps, trees, and linked lists
- Familiarity with basic probability (Bayes' theorem, conditional probability)
- Some exposure to linear algebra (vectors, matrices, dot products)

## How This Relates to Later Chapters

| Concept from this chapter | Where it shows up later |
|---|---|
| Inverted index | Ch 03 (full-text search), Ch 05 (indexing strategies), Ch 06 (Lucene internals) |
| Boolean retrieval | Ch 02 (retrieval models), Ch 03 (faceted search) |
| Vector space model | Ch 02 (TF-IDF), Ch 04 (dense retrieval, hybrid search) |
| Tokenization & stemming | Ch 03 (fuzzy search), Ch 06 (analyzer chains in ES/Solr) |
| Probabilistic models | Ch 02 (BM25), Ch 07 (learning to rank) |
| Skip lists & B-trees | Ch 05 (sharding), Ch 06 (Lucene segment internals) |
