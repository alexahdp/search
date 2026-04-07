# 03 – Classical Search Algorithms

This chapter covers the practical algorithms and techniques that power production search systems. These are the "classical" approaches that dominated search for decades and remain essential building blocks today, even as neural and semantic methods gain adoption. Here you'll learn how to implement full-text search, handle typos and spelling errors, build faceted navigation, create autocomplete systems, handle geospatial queries, and design efficient sorting and pagination.

## Topics

1. [Full-Text Search](./full-text-search.md) — Inverted index construction in depth, posting list intersection algorithms (TAAT vs DAAT), skip pointers, phrase queries, proximity scoring, and query optimization strategies.
2. [Fuzzy Search](./fuzzy-search.md) — Edit distance algorithms (Levenshtein, Damerau-Levenshtein), n-gram indexing for fuzzy matching, phonetic algorithms (Soundex, Metaphone, Double Metaphone), and spelling correction techniques.
3. [Faceted Search and Filtering](./faceted-search.md) — Facet computation algorithms, filter strategies (post-filtering vs pre-filtering), combining filters efficiently, facet counting, and drill-down/drill-up navigation.
4. [Autocomplete and Typeahead](./autocomplete.md) — Prefix trees (tries) for completion, Finite State Transducers (FSTs), frequency-based ranking, personalized suggestions, and handling misspellings in autocomplete.
5. [Geospatial Search](./geospatial-search.md) — R-trees and variants (R*-tree, R+ tree), geohashing and quadtrees, distance calculations (Haversine, Vincenty), bounding box queries, and radius search optimization.
6. [Sorting and Pagination](./sorting-pagination.md) — Top-K algorithms (heap-based, quickselect), distributed sorting, deep pagination challenges, cursor-based pagination, and search-after patterns.

## Recommended Study Order

Follow the order above. Each file is self-contained but builds on concepts from earlier chapters:

- **Full-Text Search** is the foundation — you need to understand posting list operations before the other topics make sense.
- **Fuzzy Search** extends full-text search to handle user errors — essential for real-world systems.
- **Faceted Search** shows how to layer structured filtering on top of full-text retrieval.
- **Autocomplete** is a specialized retrieval problem with unique performance constraints.
- **Geospatial Search** introduces spatial data structures and how they integrate with text search.
- **Sorting and Pagination** covers the final stage of the query pipeline — getting results to users efficiently.

## Prerequisites

- Chapter 01 (Fundamentals) — especially [Data Structures](../01-fundamentals/data-structures.md) for inverted index and trie internals
- Chapter 02 (Core Concepts) — especially [TF-IDF and BM25](../02-core-concepts/tf-idf-and-bm25.md) for scoring and [Query Processing](../02-core-concepts/query-processing.md) for the pipeline context
- Comfort with algorithm analysis, complexity notation, and data structure trade-offs

## How This Relates to Later Chapters

| Concept from this chapter | Where it shows up later |
|---|---|
| Posting list operations (TAAT/DAAT) | Ch 05 (query routing), Ch 06 (Lucene internals) |
| Edit distance, n-grams | Ch 06 (Elasticsearch analyzers), Ch 07 (query understanding) |
| Faceted search | Ch 06 (Solr/ES facet APIs), Ch 09 (e-commerce search patterns) |
| Prefix trees, FSTs | Ch 06 (Lucene suggest, term dictionary) |
| R-trees, geohashing | Ch 06 (geo queries in ES/Solr), Ch 09 (location-based search) |
| Top-K algorithms | Ch 07 (re-ranking, multi-stage retrieval) |
| Deep pagination issues | Ch 05 (sharding strategies), Ch 09 (common failure modes) |

## Why These Are Still "Classical"

With the rise of transformer models and dense retrieval, you might wonder if these techniques are obsolete. They're not:

1. **BM25 + filters + facets** still power the vast majority of production e-commerce, enterprise, and SaaS search systems.
2. **Hybrid search** (combining neural + classical) is the current best practice — you need both.
3. **Fuzzy matching** and **autocomplete** have no adequate neural replacements for interactive, low-latency use cases.
4. **Geospatial** is inherently a classical problem (spatial data structures).
5. **Efficiency** — classical methods scale to billions of documents with predictable latency. Neural methods are improving but still more expensive.

Understanding these algorithms deeply makes you a better search engineer, even when using high-level APIs. When your Elasticsearch query is slow, you need to know what's happening under the hood to fix it.
