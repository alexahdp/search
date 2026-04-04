# Retrieval Models

A retrieval model defines how documents are represented, how queries are represented, and how relevance is computed. This section covers the three classical models — Boolean, vector space, and language models — not from a theoretical perspective (that's in [Mathematical Foundations](../01-fundamentals/mathematical-foundations.md)), but from a **systems implementation** perspective: how they work in practice, where they're used, and what trade-offs they make.

## Table of Contents

1. [Boolean Retrieval in Practice](#boolean-retrieval-in-practice)
2. [Vector Space Model Implementation](#vector-space-model-implementation)
3. [Language Models for Retrieval](#language-models-for-retrieval)
4. [Extended Boolean Models](#extended-boolean-models)
5. [Model Comparison and When to Use Each](#model-comparison-and-when-to-use-each)
6. [Combining Models](#combining-models)
7. [References](#references)

---

## Boolean Retrieval in Practice

Boolean retrieval is the simplest model: documents either match the query or they don't. There's no ranking — results are unordered (or ordered by some secondary criterion like date).

### Core Operations

Queries are Boolean expressions using `AND`, `OR`, `NOT`:

```
"search" AND "engine"
→ Retrieve documents containing both terms

"python" OR "rust" OR "golang"
→ Retrieve documents containing at least one term

"database" AND NOT "nosql"
→ Retrieve documents containing "database" but not "nosql"
```

Implementation (see [Data Structures](../01-fundamentals/data-structures.md#the-inverted-index)):
- `AND` → posting list intersection
- `OR` → posting list union
- `NOT` → set difference (all doc IDs minus the NOT-term's posting list)

### Query Planning

For multi-term queries, the order of operations matters enormously:

**Query**: `A AND B AND C`

If $|posting(A)| = 1M$, $|posting(B)| = 10K$, $|posting(C)| = 500$:

**Bad plan**: Intersect A and B (1M vs 10K comparisons), then intersect with C
**Good plan**: Start with C (smallest), intersect with B, then with A

**Optimization rule**: Process terms in order of increasing document frequency. This is the **term-at-a-time (TAAT)** query processing strategy.

### Where Boolean Retrieval Is Used

**Legal search** (LexisNexis, Westlaw): Lawyers need precise boolean queries. "Patent" AND "infringement" AND NOT "expired" — they want control, not an opaque ranking algorithm.

**Patent search**: Similar precision requirements. Examiners search with complex Boolean queries combining classifications, dates, and keywords.

**Database-style search**: When the user specifies exact filters. E-commerce: "laptop" AND category:electronics AND price:[500 TO 1500]

**PubMed**: Medical literature search. Boolean logic is the primary interface.

### Limitations

1. **No ranking**: All matching documents are equally relevant. Users must sort by date, author, or some other criterion.
2. **Keyword matching only**: "laptop" won't match "notebook computer" unless you add OR clauses manually.
3. **Cognitive burden**: Users must construct precise Boolean expressions. Most users can't or won't do this (average Google query: 2-3 words, rarely uses operators).
4. **All-or-nothing**: A document missing one required term is completely excluded, even if it's otherwise perfect.

### Extended Boolean: Adding Ranking

Pure Boolean retrieval returns an unordered set. **Extended Boolean** models add ranking while preserving Boolean semantics.

**Approach 1: Threshold-based ranking**
- Documents match if they satisfy the Boolean constraint
- Then rank by TF-IDF or BM25 score
- This is what Elasticsearch does with `filter` context (Boolean) + `must` context (scored)

**Approach 2: Fuzzy Boolean (p-norm model)**

Replace Boolean AND/OR with "soft" versions using vector space norms:

$$\text{score}(\text{A AND B}) = \sqrt[p]{w_A^p + w_B^p}$$

where $w_A$, $w_B$ are the term weights (e.g., TF-IDF). As $p \to \infty$, this approaches Boolean AND (minimum). As $p \to 1$, it approaches Boolean OR (sum).

This allows "partial matches" — a document with only one query term can still score non-zero, but lower than a document with both.

## Vector Space Model Implementation

The vector space model (VSM) represents documents and queries as vectors in a high-dimensional term space. Relevance is the **cosine similarity** between query and document vectors.

### Document Representation

Each document is a vector $\vec{d} \in \mathbb{R}^{|V|}$ where $V$ is the vocabulary:

$$\vec{d} = [w_{t_1}, w_{t_2}, \ldots, w_{t_{|V|}}]$$

$w_{t_i}$ is the weight of term $t_i$ in the document, typically TF-IDF:

$$w_{t,d} = (1 + \log \text{TF}(t, d)) \times \log \frac{N}{df(t)}$$

**Query representation**: Same structure. A query "search engine" becomes:

$$\vec{q} = [0, 0, \ldots, w_{\text{search}}, 0, \ldots, w_{\text{engine}}, \ldots, 0]$$

Most elements are zero (sparse vector).

### Scoring: Cosine Similarity

$$\text{score}(q, d) = \cos(\vec{q}, \vec{d}) = \frac{\vec{q} \cdot \vec{d}}{|\vec{q}| \times |\vec{d}|}$$

Expanded:

$$\cos(\vec{q}, \vec{d}) = \frac{\sum_{t \in q \cap d} w_{t,q} \times w_{t,d}}{\sqrt{\sum_{t \in q} w_{t,q}^2} \times \sqrt{\sum_{t \in d} w_{t,d}^2}}$$

**Key insight**: Only terms appearing in **both** the query and document contribute. This is why inverted indices work — you only need to consider documents in the posting lists of query terms.

### Normalization

The denominator normalizes vectors to unit length, making the score independent of document length. Two equivalent approaches:

**At scoring time**: Compute the full formula above.

**At index time** (more efficient):
1. Store normalized term weights in the index: $w'_{t,d} = \frac{w_{t,d}}{|\vec{d}|}$
2. At query time, normalize the query: $w'_{t,q} = \frac{w_{t,q}}{|\vec{q}|}$
3. Score = $\sum_{t \in q \cap d} w'_{t,q} \times w'_{t,d}$ (no division needed)

This is what Lucene does when you select "ClassicSimilarity" (TF-IDF with cosine normalization).

### Implementation Sketch

```python
def cosine_similarity(query_terms, doc_id, index):
    score = 0.0

    for term in query_terms:
        if term in index:
            # Get TF-IDF weight for this term in this document
            doc_weight = index[term].get_weight(doc_id)
            query_weight = query_tfidf[term]  # Precomputed

            score += query_weight * doc_weight

    # Normalization factors (precomputed)
    query_norm = precomputed_query_norm
    doc_norm = doc_norms[doc_id]  # Stored at index time

    return score / (query_norm * doc_norm)
```

### Where VSM Is Used

**Academic IR systems**: The original SMART system (Salton, 1971) was pure VSM.

**Lucene (pre-v6.0)**: Default similarity was TF-IDF with cosine normalization (VSM).

**Document similarity / clustering**: VSM is the foundation for document clustering (k-means on TF-IDF vectors) and near-duplicate detection.

**Semantic search**: Dense retrieval (Chapter 04) is VSM with learned embeddings instead of TF-IDF. The model is the same — only the representation changes.

### Limitations

1. **Independence assumption**: Terms are treated as independent dimensions. "New York" is orthogonal to "NYC," even though they're the same entity.
2. **Vocabulary mismatch**: No synonym handling. "laptop" and "notebook" are unrelated dimensions.
3. **Parameter tuning**: Requires choosing IDF variant, normalization scheme, etc. BM25 handles this better with interpretable parameters.

## Language Models for Retrieval

Language models (LMs) for retrieval take a probabilistic approach: treat each document as a language model (a probability distribution over words), then score by how likely the query would be generated by that model.

### Query Likelihood

The simplest LM retrieval approach: **query likelihood**. Rank documents by $P(q \mid d)$, the probability that the document's language model would generate the query:

$$P(q \mid d) = \prod_{t \in q} P(t \mid d)$$

Taking logs (to avoid underflow and make addition):

$$\log P(q \mid d) = \sum_{t \in q} \log P(t \mid d)$$

### Estimating $P(t \mid d)$

The naive estimate is maximum likelihood:

$$P(t \mid d) = \frac{\text{TF}(t, d)}{|d|}$$

**Problem**: Zero probabilities. If a query term doesn't appear in the document, $P(t \mid d) = 0$, so $P(q \mid d) = 0$. The document is completely excluded, even if it's otherwise highly relevant.

**Solution: Smoothing**. Mix the document model with a background collection model.

### Jelinek-Mercer Smoothing

$$P(t \mid d) = (1 - \lambda) \cdot \frac{\text{TF}(t, d)}{|d|} + \lambda \cdot \frac{cf(t)}{|C|}$$

where:
- $cf(t)$ = collection frequency (total occurrences of $t$ across all documents)
- $|C|$ = total number of tokens in the corpus
- $\lambda \in [0, 1]$ controls the mix (typically $0.1$ to $0.4$)

**Intuition**: If a term doesn't appear in the document, fall back to its corpus-wide frequency. Common terms get higher fallback probability than rare terms.

### Dirichlet Smoothing

$$P(t \mid d) = \frac{\text{TF}(t, d) + \mu \cdot \frac{cf(t)}{|C|}}{|d| + \mu}$$

where $\mu$ is a smoothing parameter (typically $1000$ to $2500$).

**Effect**: Longer documents receive less smoothing (the denominator $|d| + \mu$ is larger). Short documents receive more smoothing. This is a form of length normalization.

### LM Retrieval Score

With Dirichlet smoothing, the ranking formula becomes:

$$\text{score}(q, d) = \sum_{t \in q} \log \frac{\text{TF}(t, d) + \mu \cdot \frac{cf(t)}{|C|}}{|d| + \mu}$$

This can be expanded and rearranged to separate document-dependent and document-independent terms (constant for a given query). The result looks surprisingly similar to BM25 — both are probabilistic models with length normalization and term saturation.

### Where LM Retrieval Is Used

**Academic systems**: Lemur/Indri toolkit (CMU/UMass) is built on language models.

**Elasticsearch/Solr**: Offer LM similarity as an alternative to BM25 (`LMDirichletSimilarity`, `LMJelinekMercerSimilarity`).

**Speech recognition and NLP pipelines**: LM scoring fits naturally when you're already using n-gram models.

### LM vs. BM25

They're conceptually different (generative probabilistic model vs. discriminative retrieval function) but produce **remarkably similar rankings** in practice. Empirically:

- BM25 is slightly better on web search benchmarks
- LM (Dirichlet) is slightly better on verbose queries (questions)
- The difference is typically < 1% NDCG

**In practice**: Use BM25. It's the default everywhere, parameters are well-understood, and it's slightly faster (no log operations in the inner loop).

## Extended Boolean Models

### Ranked Boolean Queries

Many systems combine Boolean filters with ranked scoring:

```json
{
  "query": {
    "bool": {
      "must": [
        {"match": {"title": "search engine"}}  // Scored (BM25)
      ],
      "filter": [
        {"range": {"publish_date": {"gte": "2020-01-01"}}},  // Not scored
        {"term": {"category": "technology"}}  // Not scored
      ]
    }
  }
}
```

Documents must satisfy the `filter` clauses (Boolean AND), then the results are ranked by the `must` clause scores.

**Performance advantage**: Filters are cached (bitsets), enabling very fast intersection. Only matching documents are scored.

### p-Norm Model

The **p-norm model** (Salton et al., 1983) generalizes Boolean AND/OR using vector norms:

$$\text{score}_{\text{AND}}(q, d) = \left( \sum_{t \in q} w_{t,d}^p \right)^{1/p}$$

$$\text{score}_{\text{OR}}(q, d) = \left( \sum_{t \in q} w_{t,d}^p \right)^{1/p}$$

For AND, as $p \to \infty$, the score approaches $\min(w_{t,d})$ (documents must have ALL terms with high weight).

For OR, as $p = 1$, the score is the sum (documents with ANY term with high weight score well).

**Practical use**: Limited. It's conceptually interesting but doesn't perform better than BM25 in most evaluations. Not implemented in major search engines.

## Model Comparison and When to Use Each

| Model | Ranking | Semantic gap | Length norm | Best for |
|---|---|---|---|---|
| **Boolean** | No (or secondary) | Large (keyword matching only) | N/A | Legal, patent, database-style search |
| **Vector space (TF-IDF)** | Yes (cosine similarity) | Large | Built-in (cosine normalization) | Document similarity, clustering |
| **BM25** | Yes (probabilistic scoring) | Large | Built-in (parameter $b$) | **General-purpose text search (default choice)** |
| **Language Model** | Yes (query likelihood) | Large | Built-in (smoothing param $\mu$) | Verbose queries, question answering |
| **Dense retrieval** (Ch 04) | Yes (embedding similarity) | Small (semantic matching) | N/A (learned) | Semantic search, cross-lingual |

### Recommendations by Scenario

**Starting a new search system**: Use BM25. It's the most widely deployed, best understood, and performs well out-of-the-box.

**E-commerce**: BM25F (multi-field) with field boosts for title, category, brand. Combine with filters (price, availability).

**Question answering**: Language model (Dirichlet) or BM25 with query expansion (synonyms, stemming).

**Code search**: Boolean or BM25 with low length normalization ($b = 0.3$). Exact matches matter more than statistical scoring.

**Semantic search across domains**: Dense retrieval (learned embeddings). BM25 as a fallback for out-of-vocabulary terms.

**Legal/patent/medical**: Boolean retrieval with advanced query syntax. Users in these domains expect precise control.

## Combining Models

The best production systems don't use a single model — they **combine** multiple scoring signals.

### Hybrid Retrieval: BM25 + Dense Embeddings

A common pattern (Chapter 04 goes deeper):

1. **Retrieve candidates** with BM25 (fast, high recall for keyword matches)
2. **Retrieve candidates** with dense retrieval / ANN search (high recall for semantic matches)
3. **Merge** candidate sets using reciprocal rank fusion or learned weights
4. **Re-rank** with a cross-encoder or learning-to-rank model

### Multi-Signal Ranking

Combine multiple models as features in a learning-to-rank system:

| Feature | Source model |
|---|---|
| BM25 score | Probabilistic model |
| TF-IDF cosine | Vector space model |
| LM score (Dirichlet) | Language model |
| Dense retrieval score | Embedding-based VSM |
| Query term coverage | Boolean-style |
| Phrase match | N/A (structural) |
| PageRank | Link analysis |
| CTR | User behavior |

Train a gradient-boosted tree (XGBoost, LightGBM) to predict relevance from these features. This is the state-of-the-art for most large-scale systems (Google, Bing, Amazon).

### When to Combine

**Small systems (< 1M docs)**: BM25 alone is sufficient. Don't overcomplicate.

**Medium systems (1M - 100M docs)**: BM25F + field boosts + basic filters. Maybe add a lightweight re-ranker (GBDT with 10-20 features).

**Large systems (100M+ docs)**: Multi-stage ranking with BM25 + dense retrieval + GBDT L1 + cross-encoder L2. This is Google/Bing territory.

## References

- Salton, G., Fox, E. A., & Wu, H. — "Extended Boolean Information Retrieval" (CACM, 1983)
- Ponte, J. M. & Croft, W. B. — "A Language Modeling Approach to Information Retrieval" (SIGIR, 1998)
- Zhai, C. & Lafferty, J. — "A Study of Smoothing Methods for Language Models Applied to Information Retrieval" (ACM TOIS, 2004)
- Robertson, S. & Zaragoza, H. — "The Probabilistic Relevance Framework: BM25 and Beyond" (Foundations and Trends in IR, 2009)
- Singhal, A. — "Modern Information Retrieval: A Brief Overview" (IEEE Data Engineering Bulletin, 2001)
- Manning, C. D., Raghavan, P., & Schütze, H. — *Introduction to Information Retrieval*, Chapters 6-11 (Cambridge, 2008)
