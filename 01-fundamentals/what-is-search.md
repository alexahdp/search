# What Is Search

## Table of Contents

1. [Problem Definition](#problem-definition)
2. [Search vs. Database Queries](#search-vs-database-queries)
3. [The Information Retrieval Pipeline](#the-information-retrieval-pipeline)
4. [Types of Search](#types-of-search)
5. [Anatomy of a Search Query](#anatomy-of-a-search-query)
6. [Key Dimensions of Search Quality](#key-dimensions-of-search-quality)
7. [Search at Scale: Industry Examples](#search-at-scale-industry-examples)
8. [References](#references)

---

## Problem Definition

At its core, search is a **ranking problem**: given a user's information need (expressed as a query) and a collection of documents, return the most relevant documents in order of decreasing relevance.

More formally, let:

- $Q$ = a query (the user's expressed information need)
- $D = \{d_1, d_2, \ldots, d_N\}$ = a document collection (the corpus)
- $rel(q, d) \in \mathbb{R}$ = a relevance function that scores how well document $d$ satisfies query $q$

The search problem is: compute $rel(q, d_i)$ for all $d_i \in D$, then return the top-$k$ documents sorted by decreasing relevance score.

The challenge is threefold:

1. **Defining relevance** — relevance is subjective, context-dependent, and multidimensional. A document can be topically relevant but poorly written, timely but untrustworthy, exactly matching the query but not what the user actually wanted.

2. **Computing relevance efficiently** — you can't score every document in a billion-document corpus for every query. You need index structures that prune the search space from $O(N)$ to $O(k \log N)$ or better.

3. **Understanding the query** — users type 2-3 words on average. Those words are ambiguous, misspelled, underspecified, and rarely express the actual information need.

### The Vocabulary Gap

A fundamental challenge in search is the **vocabulary mismatch problem** (also called the lexical gap). Users and document authors use different words to describe the same concepts:

- User searches: "cheap flights to NYC"
- Document says: "affordable airfare to New York City"

No keywords overlap, yet the document is perfectly relevant. Closing this gap drives much of search engineering — from simple techniques like stemming and synonyms to advanced approaches like semantic embeddings.

## Search vs. Database Queries

If you come from a backend engineering background, it's tempting to think of search as just SQL `WHERE` + `ORDER BY`. The differences are fundamental:

| Dimension | Database Query | Search |
|---|---|---|
| **Query language** | Structured (SQL) | Unstructured (natural language) |
| **Result set** | Exact match — rows either match or don't | Ranked — every document has a relevance score |
| **Schema** | Fixed, known upfront | Loose or absent; documents are heterogeneous |
| **Recall expectation** | 100% — return all matching rows | Partial — return the best $k$ results |
| **Ambiguity** | None — predicates are precise | High — "apple" could be fruit or company |
| **Error tolerance** | Zero — typo in WHERE = no results | High — should handle misspellings, synonyms |
| **Optimization target** | Minimize latency / maximize throughput | Maximize relevance (precision@k, NDCG) |

A database answers: "Give me exactly the rows that match these predicates."
A search engine answers: "Here are the documents that best satisfy this vague information need, ranked by how well they satisfy it."

That said, modern systems blur the line. Elasticsearch is used as both a search engine and an analytics datastore. PostgreSQL has `ts_vector` full-text search. The mental model still holds though: the core problem is different.

## The Information Retrieval Pipeline

Every search system, from a 100-line prototype to Google, follows the same high-level pipeline:

```
┌─────────────┐    ┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│  Document    │───▶│   Text      │───▶│   Indexing   │───▶│   Index     │
│  Ingestion   │    │  Processing │    │              │    │   Storage   │
└─────────────┘    └─────────────┘    └──────────────┘    └──────┬──────┘
                                                                  │
                                                                  ▼
┌─────────────┐    ┌─────────────┐    ┌──────────────┐    ┌──────────────┐
│   Results    │◀───│  Ranking /  │◀───│   Retrieval  │◀───│    Query     │
│  Rendering  │    │  Re-ranking │    │  (Candidate   │    │  Processing  │
└─────────────┘    └─────────────┘    │   Selection)  │    └──────────────┘
                                      └──────────────┘
```

### Offline path (indexing)

1. **Document ingestion** — Crawl, fetch, or receive documents. Handle deduplication, format detection (HTML, PDF, JSON), encoding detection.

2. **Text processing** — Tokenize, normalize, stem/lemmatize (covered in detail in [Text Representation](./text-representation.md)). This is where raw text becomes a sequence of indexable terms.

3. **Indexing** — Build data structures (inverted index, forward index, columnar stores) that enable fast retrieval. Covered in [Data Structures](./data-structures.md).

4. **Index storage** — Persist to disk with efficient compression. Handle segment merges, updates, deletes.

### Online path (query time)

1. **Query processing** — Parse the query, apply the same text processing as indexing (same tokenizer, same stemmer), expand with synonyms, rewrite for spelling correction, classify intent.

2. **Retrieval (candidate selection)** — Use the index to efficiently find a candidate set. In a typical system, this takes billions of documents down to thousands. This is the "recall" stage.

3. **Ranking / re-ranking** — Score candidates with increasingly expensive models. A common pattern is multi-stage ranking:
   - **L0**: Index-time pruning (e.g., BM25 top-1000)
   - **L1**: Lightweight ML model (e.g., GBDT with a few features)
   - **L2**: Heavy model (e.g., cross-encoder neural model on top-50)

4. **Results rendering** — Snippet generation, highlighting, facet aggregation, spell-check suggestions, related searches.

### Latency budget

Users expect results in under 200ms. A rough budget for web search:

| Stage | Budget |
|---|---|
| Query processing | 5-10ms |
| Retrieval | 20-50ms |
| L1 ranking | 30-50ms |
| L2 re-ranking | 50-100ms |
| Rendering + network | 20-50ms |
| **Total** | **~200ms** |

## Types of Search

Search is not a monolith. Different domains have fundamentally different characteristics:

### By domain

**Web search** — Billions of heterogeneous documents. Dealing with spam, link analysis, freshness, authority. Query distribution follows a power law: a small fraction of queries account for most volume.

**E-commerce search** (Amazon, eBay) — Structured product catalogs. Key signals: purchase history, price, availability, reviews, click-through rate. The relevance function is often: "what will the user buy?" not just "what matches the query?" Result diversity matters — show different brands, price points.

**Enterprise search** (Confluence, SharePoint, Slack) — Heterogeneous document types with access control. Recency is critical. Corpus is smaller but queries are more specific. People search and expertise finding.

**Log/observability search** (Splunk, ELK) — Time-series text data. Queries are often structured (field:value). Aggregations matter as much as individual results. Volume is enormous (terabytes/day), but data has a natural expiry.

**Code search** (GitHub, Sourcegraph) — Exact-match matters more than fuzzy retrieval. Must respect programming language syntax. Symbol resolution, call-graph navigation. Regex support is expected.

**Multimedia search** — Image search (visual similarity, text-to-image), video search (temporal, transcript-based), audio search. Often requires embedding-based retrieval.

### By query type

**Navigational** — User wants a specific page. "facebook login", "amazon". Success metric: did the user click the first result?

**Informational** — User wants to learn something. "how does HTTPS work", "best practices for React performance". Success: did the user's information need get satisfied?

**Transactional** — User wants to perform an action. "buy iPhone 15", "book flight to Berlin". Success: did the user convert?

Understanding query intent is crucial because the same ranking function shouldn't be applied uniformly. A navigational query should heavily weight domain authority; an informational query should weight content depth; a transactional query should weight product availability and price.

### By retrieval method

**Keyword (lexical) search** — Match query terms against document terms. Built on inverted indices. Fast, interpretable, well-understood. Fails on vocabulary mismatch.

**Semantic (dense) search** — Encode queries and documents as dense vectors in a shared embedding space. Retrieve by nearest-neighbor search. Handles vocabulary mismatch well. Harder to debug, more expensive.

**Hybrid search** — Combine lexical and semantic retrieval. This is the state of the art for most production systems. Typically: run both retrievers, merge candidate sets, re-rank.

**Structured search** — Filter by structured fields (price, date, category) combined with text matching. Standard in e-commerce and enterprise search.

## Anatomy of a Search Query

When a user types "red running shoes under $100" into an e-commerce search box, here's what a mature search system extracts:

```
Query: "red running shoes under $100"

├── Keywords: ["red", "running", "shoes"]
├── Intent: transactional (purchase)
├── Entity extraction:
│   ├── Color: red
│   ├── Product type: running shoes
│   └── Price constraint: < $100
├── Filters (implicit):
│   ├── category = footwear
│   ├── subcategory = running
│   └── price < 100
├── Spelling: OK
├── Synonyms: shoes → footwear, sneakers, trainers
└── Personalization:
    ├── preferred_brands: [Nike, Adidas] (from history)
    ├── preferred_size: 10.5 (from profile)
    └── gender: men's (from profile)
```

Each of these extraction steps is its own engineering challenge. The query processing pipeline for a system like Amazon or Google is often 50+ components running in sequence or parallel.

## Key Dimensions of Search Quality

When evaluating a search system, these are the dimensions that matter:

### Relevance
The core metric. Are the results what the user wanted? Measured by precision@k, NDCG, MAP (covered in Chapter 02).

### Recall
Did the system find all the relevant documents, or did good results get missed? A user will never know about results that didn't make it to the candidate set.

### Latency
P50, P95, P99 response times. A 100ms increase in latency can reduce revenue by 1% in e-commerce. Tail latency matters because it compounds in distributed systems (the slowest shard determines response time).

### Freshness
How quickly do new or updated documents become searchable? News search needs seconds; product search can tolerate minutes; web search operates in hours to days.

### Coverage
What fraction of the corpus is actually indexed and searchable? Crawl coverage, language coverage, format coverage.

### Robustness
How does the system handle:
- Typos: "runing shoes" → "running shoes"
- Zero results: show alternatives instead of an empty page
- Ambiguity: "jaguar" — animal, car, or macOS?
- Adversarial content: SEO spam, keyword stuffing

### Scalability
Query throughput (QPS) and index size. Most systems need to handle at least:
- Small: 10K docs, 10 QPS
- Medium: 10M docs, 1K QPS
- Large: 1B+ docs, 100K+ QPS

Each jump in scale changes the architecture fundamentally.

## Search at Scale: Industry Examples

To build intuition for what "search" means in practice, consider these systems:

**Google Web Search** — 100B+ pages indexed. Serves 8.5B queries/day. Multi-stage ranking with ML at every level. Index is sharded across thousands of machines. Query latency target: <500ms globally.

**Amazon Product Search** — ~350M products. Query understanding is paramount (mapping "laptop for college student" to category filters + feature preferences). Revenue-optimized: ranking balances relevance, conversion likelihood, and margin. A/B tests everything.

**LinkedIn Search** — People, jobs, posts, companies — each is a different search vertical. Heavy personalization (your network graph affects ranking). Typeahead is critical because job titles and company names have specific patterns.

**Spotify Search** — Searches across tracks, artists, albums, playlists, podcasts. Must handle multilingual content. Audio fingerprinting for duplicate detection. Combines text matching with collaborative filtering signals.

**Elasticsearch / OpenSearch** — Used as the search backend for thousands of companies. Provides the building blocks (inverted index, analyzers, query DSL) but relevance tuning is the customer's problem. Understanding Lucene internals (Chapter 06) is key to using it well.

## References

- Manning, C. D., Raghavan, P., & Schütze, H. — *Introduction to Information Retrieval* (Cambridge University Press, 2008)
- Büttcher, S., Clarke, C., & Cormack, G. — *Information Retrieval: Implementing and Evaluating Search Engines* (MIT Press, 2010)
- Croft, W. B., Metzler, D., & Strohman, T. — *Search Engines: Information Retrieval in Practice* (Pearson, 2015)
- Baeza-Yates, R. & Ribeiro-Neto, B. — *Modern Information Retrieval* (Addison-Wesley, 2011)
