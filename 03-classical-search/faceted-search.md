# Faceted Search and Filtering

Faceted search allows users to navigate large result sets by drilling down through multiple dimensions (facets). Think of e-commerce filters: category, brand, price range, color, size. This section covers the algorithms that power faceted navigation, efficient filter combination, and facet count computation.

## Table of Contents

1. [Facets vs. Filters](#facets-vs-filters)
2. [Facet Computation Algorithms](#facet-computation-algorithms)
3. [Filter Strategies](#filter-strategies)
4. [Combining Multiple Filters](#combining-multiple-filters)
5. [Distributed Faceting](#distributed-faceting)
6. [Advanced Faceting Patterns](#advanced-faceting-patterns)
7. [Performance Optimization](#performance-optimization)
8. [References](#references)

---

## Facets vs. Filters

**Filter:** Restrict results to documents matching a criterion.

```
Filter: category="Electronics"
→ Returns only documents in Electronics category
```

**Facet:** Aggregate and count documents by dimension values, showing what filters are available.

```
Facet: category
→ Returns:
  Electronics: 1,234 documents
  Books: 5,678 documents
  Clothing: 2,345 documents
  ...
```

**Faceted search = Filters + Facets together:**

```
Query: "laptop"
Filter: price:[500 TO 1000], brand="Dell"
Facets: category, color, screen_size

Results:
  - 42 matching laptops
  - Category: Laptops (42), Accessories (8)
  - Color: Black (25), Silver (15), White (2)
  - Screen Size: 13" (10), 15" (28), 17" (4)
```

Users can drill down by clicking facet values, which add filters, which update facet counts.

---

## Facet Computation Algorithms

### Field Value Faceting

The simplest facet type: count documents by discrete field values.

**Data structure:** Inverted index on the facet field.

```
Field: "brand"
Inverted index:
  "Dell"    → [doc1, doc5, doc12, doc18, ...]
  "HP"      → [doc3, doc7, doc15, ...]
  "Lenovo"  → [doc2, doc9, doc20, ...]
```

**Algorithm:**

```
COMPUTE_FACET(field, matching_docs):
    facet_counts = {}
    for value in field_values(field):
        docs_with_value = posting_list(field, value)
        matching_with_value = intersect(matching_docs, docs_with_value)
        facet_counts[value] = len(matching_with_value)
    return facet_counts
```

**Complexity:** $O(V \cdot (M + L))$ where:
- $V$ = number of unique values (cardinality of the field)
- $M$ = size of matching docs set
- $L$ = average posting list length

**Optimization 1:** If matching_docs is small (e.g., <1000), iterate through it instead:

```
COMPUTE_FACET_SMALL_MATCH(field, matching_docs):
    facet_counts = {}
    for doc in matching_docs:
        value = get_field_value(doc, field)
        facet_counts[value] = facet_counts.get(value, 0) + 1
    return facet_counts
```

**Complexity:** $O(M)$ — much better when $M \ll V \cdot L$.

**Optimization 2:** Use doc values (columnar storage). Lucene stores facet fields in a columnar format for efficient iteration:

```
Doc values (columnar):
docID → brand
1     → "Dell"
2     → "Lenovo"
3     → "HP"
5     → "Dell"
...

Iterate matching_docs, lookup brand for each → O(M)
```

### Range Faceting

Count documents in numeric or date ranges.

```
Facet: price
Ranges: [0-100], [100-500], [500-1000], [1000+]

Results:
  0-100: 123 docs
  100-500: 456 docs
  500-1000: 234 docs
  1000+: 89 docs
```

**Algorithm (using range trees):**

```
RANGE_FACET(field, ranges, matching_docs):
    facet_counts = {}
    for (min, max) in ranges:
        docs_in_range = range_query(field, min, max)
        matching_in_range = intersect(matching_docs, docs_in_range)
        facet_counts[(min, max)] = len(matching_in_range)
    return facet_counts
```

**Data structure:** BKD-tree (Block K-D tree) in Lucene for numeric/geo fields.

**BKD-tree:** A space-partitioning tree that indexes numeric values, supporting efficient range queries.

```
For field "price":

            [0 - 10000]
           /           \
      [0-500]        [500-10000]
      /     \          /       \
  [0-100] [100-500] [500-1000] [1000-10000]
```

Range query $[100, 1000]$ visits only relevant leaf nodes → $O(\log N + k)$ where $k$ = results.

### Histogram Faceting

Auto-generate ranges (buckets) based on data distribution.

```
Facet: date (histogram with interval=1 month)

Results:
  2024-01: 234 docs
  2024-02: 567 docs
  2024-03: 890 docs
  ...
```

**Algorithm:**

```
HISTOGRAM_FACET(field, interval, matching_docs):
    facet_counts = {}
    for doc in matching_docs:
        value = get_field_value(doc, field)
        bucket = floor(value / interval) * interval
        facet_counts[bucket] = facet_counts.get(bucket, 0) + 1
    return facet_counts
```

**Elasticsearch implementation:**

```json
{
  "aggs": {
    "prices": {
      "histogram": {
        "field": "price",
        "interval": 100
      }
    }
  }
}
```

---

## Filter Strategies

### Post-Filtering

Evaluate query first, then apply filters to results.

```
SEARCH_POST_FILTER(query, filters):
    results = evaluate_query(query)  // e.g., BM25 scoring
    filtered = apply_filters(results, filters)
    return top_k(filtered)
```

**Pros:**
- Simple
- Facet counts reflect unfiltered results (useful for "drill-down" UX)

**Cons:**
- Wasteful if filters are highly selective (score many docs that get filtered out)

**Use case:** Filters are not very selective, or you need facets on unfiltered data.

### Pre-Filtering

Apply filters before scoring (using cached filter bitmaps).

```
SEARCH_PRE_FILTER(query, filters):
    candidate_bitmap = apply_filters_as_bitmaps(filters)
    results = evaluate_query_on_candidates(query, candidate_bitmap)
    return top_k(results)
```

**Pros:**
- Much faster when filters are highly selective (e.g., category="Electronics" reduces 1M docs to 10K)
- Scoring only happens on candidates

**Cons:**
- Facet counts must be computed differently (see below)

**Use case:** Filters are very selective (e.g., e-commerce, enterprise search).

### Elasticsearch Approach

Elasticsearch uses **filter context** (pre-filtering) vs **query context** (scoring):

```json
{
  "query": {
    "bool": {
      "must": [
        { "match": { "title": "laptop" } }  // Query context (scored)
      ],
      "filter": [
        { "term": { "category": "Electronics" } },  // Filter context (not scored)
        { "range": { "price": { "gte": 500, "lte": 1000 } } }
      ]
    }
  }
}
```

**Execution:**
1. Apply `filter` clauses → bitmap of candidates (cached)
2. Evaluate `must` clauses on candidates → BM25 scores
3. Return top-K scored docs

**Filter caching:** Elasticsearch automatically caches filter results (as Roaring bitmaps) at the segment level. Frequently used filters (e.g., category filters) are essentially free after the first query.

---

## Combining Multiple Filters

### Boolean Filter Combination

Filters can be combined with AND, OR, NOT.

```
Filters:
  - category="Electronics"
  - brand="Dell"
  - price:[500 TO 1000]

Combined: category AND brand AND price
```

**Execution (using bitmaps):**

```
COMBINE_FILTERS(filters):
    result_bitmap = all_ones()  // Start with all docs
    for filter in filters:
        filter_bitmap = evaluate_filter(filter)  // Cached or computed
        result_bitmap = result_bitmap & filter_bitmap  // Bitwise AND
    return result_bitmap
```

**Complexity:** $O(N/64)$ for bitmap AND operations (64 bits per word), essentially $O(1)$ per doc.

### Filter Order Optimization

Process the most selective filter first to minimize work:

```
Filters:
  - category="Electronics" (selects 10% of docs)
  - brand="Dell" (selects 5% of docs)
  - inStock=true (selects 80% of docs)

Naive order: category & brand & inStock → 10% → 0.5% → 0.4%
Optimized:   brand & category & inStock → 5% → 0.5% → 0.4%

Even better: brand & category (then stop if result is small enough)
```

**Heuristic:** Use filter cardinality (estimated result size) to order filters.

Elasticsearch estimates filter cardinality using Lucene's index statistics and reorders filters automatically.

### Multi-Value Field Filters

Some fields have multiple values per document (e.g., tags, categories).

```
Document:
  id: 1
  title: "Laptop"
  tags: ["electronics", "computers", "dell"]

Filter: tags="electronics"
→ Document 1 matches (one of its tags matches)
```

**Inverted index for multi-value fields:**

```
tags field:
  "electronics" → [doc1, doc5, doc12, ...]
  "computers"   → [doc1, doc7, doc15, ...]
  "dell"        → [doc1, doc3, doc18, ...]
```

Filtering is the same as single-value fields (lookup posting list for filter value).

**Faceting on multi-value fields:**

Each document can contribute to multiple facet counts.

```
Matching docs: [doc1, doc5, doc7]
Doc1 tags: ["electronics", "computers", "dell"]
Doc5 tags: ["electronics", "laptops"]
Doc7 tags: ["computers", "desktops"]

Facet counts:
  electronics: 2 (doc1, doc5)
  computers: 2 (doc1, doc7)
  dell: 1 (doc1)
  laptops: 1 (doc5)
  desktops: 1 (doc7)
```

---

## Distributed Faceting

In distributed search systems (e.g., Elasticsearch cluster with shards), faceting must aggregate results across shards.

### Naive Approach

Each shard computes local facets, then merge:

```
Shard 1 facets (brand):
  Dell: 100
  HP: 50
  Lenovo: 30

Shard 2 facets (brand):
  Dell: 80
  HP: 60
  Lenovo: 20

Merged facets:
  Dell: 180
  HP: 110
  Lenovo: 50
```

**Complexity:** $O(S \cdot V)$ where $S$ = number of shards, $V$ = facet cardinality.

**Problem:** If $V$ is large (e.g., unique user IDs), merging is expensive and doesn't scale.

### Top-K Facet Aggregation

Only return top-K facet values, not all values.

```
Query: Get top 10 brands by doc count

Each shard returns top 10:
Shard 1: [Dell:100, HP:50, Lenovo:30, ...]
Shard 2: [Dell:80, HP:60, Apple:40, ...]

Coordinator merges and re-ranks:
Global top 10: [Dell:180, HP:110, Lenovo:50, Apple:40, ...]
```

**Problem:** This can be inaccurate. If Shard 3 has `Sony:70` but it's not in its local top-10, it won't appear in the global result even though it should.

**Solution:** Over-fetch from each shard (e.g., top-50 from each shard to compute global top-10). Trade accuracy vs. network cost.

Elasticsearch parameter: `shard_size` (fetch more from each shard than `size`).

```json
{
  "aggs": {
    "brands": {
      "terms": {
        "field": "brand",
        "size": 10,          // Return top 10 globally
        "shard_size": 50     // Fetch top 50 from each shard
      }
    }
  }
}
```

### Cardinality Aggregation (HyperLogLog)

Count distinct values (e.g., "How many unique users viewed this product?").

**Exact counting:** Requires collecting all unique values → expensive and doesn't scale.

**Approximate counting (HyperLogLog):**

Sketch data structure that estimates cardinality with $< 2\%$ error using only ~10KB memory.

```
CARDINALITY_FACET(field):
    hll_sketch = HyperLogLog()
    for doc in matching_docs:
        value = get_field_value(doc, field)
        hll_sketch.add(value)
    return hll_sketch.estimate()
```

**Distributed aggregation:** HyperLogLog sketches can be merged:

```
Shard 1 sketch: hll1
Shard 2 sketch: hll2
Merged sketch: hll1.merge(hll2)
```

Elasticsearch's `cardinality` aggregation uses HyperLogLog++.

```json
{
  "aggs": {
    "unique_users": {
      "cardinality": {
        "field": "user_id",
        "precision_threshold": 40000
      }
    }
  }
}
```

---

## Advanced Faceting Patterns

### Hierarchical Facets

Category drill-down with parent-child relationships.

```
Category hierarchy:
  Electronics
    ├─ Computers
    │   ├─ Laptops
    │   └─ Desktops
    └─ Phones

Facets at depth 0 (top-level):
  Electronics: 1000
  Clothing: 500
  Books: 300

User selects "Electronics" → drill down to depth 1:
  Computers: 600
  Phones: 400

User selects "Computers" → drill down to depth 2:
  Laptops: 400
  Desktops: 200
```

**Implementation:** Store full path in a multi-value field.

```
Document:
  category_path: ["Electronics", "Electronics/Computers", "Electronics/Computers/Laptops"]

Facet query:
  - Depth 0: Filter by prefix "Electronics" → count
  - Depth 1: Filter by prefix "Electronics/Computers" → count
```

### Pivot Faceting (Nested Facets)

Facet by one dimension, then sub-facet within each value.

```
Facet: category, then brand within each category

Results:
  Electronics (1000)
    └─ Dell: 300
    └─ HP: 200
    └─ Lenovo: 100
  Clothing (500)
    └─ Nike: 150
    └─ Adidas: 100
```

**Solr's pivot facets:**

```
facet.pivot=category,brand
```

**Elasticsearch nested aggregations:**

```json
{
  "aggs": {
    "categories": {
      "terms": { "field": "category" },
      "aggs": {
        "brands": {
          "terms": { "field": "brand" }
        }
      }
    }
  }
}
```

### Stats Faceting

Compute statistics within facet buckets.

```
Facet: category
Stats: price (min, max, avg, sum)

Results:
  Electronics: 1000 docs, price: min=50, max=2000, avg=500
  Clothing: 500 docs, price: min=10, max=200, avg=50
```

**Algorithm:**

```
STATS_FACET(grouping_field, stats_field):
    facet_stats = {}
    for value in field_values(grouping_field):
        docs = posting_list(grouping_field, value)
        prices = [get_field_value(doc, stats_field) for doc in docs]
        facet_stats[value] = {
            'count': len(prices),
            'min': min(prices),
            'max': max(prices),
            'avg': sum(prices) / len(prices),
            'sum': sum(prices)
        }
    return facet_stats
```

**Elasticsearch stats aggregation:**

```json
{
  "aggs": {
    "categories": {
      "terms": { "field": "category" },
      "aggs": {
        "price_stats": {
          "stats": { "field": "price" }
        }
      }
    }
  }
}
```

---

## Performance Optimization

### Facet Caching

Cache facet results for frequent queries.

**Challenge:** Cache invalidation. Facets change as the index changes.

**Solution:** Segment-level caching (Lucene/ES approach). Cache facets per index segment (which are immutable). When segments merge, recompute.

**Trade-off:** More segments → more cache misses. Balance merge frequency vs. cache efficiency.

### Field Data vs. Doc Values

**Field data (legacy):** Load all values for a field into heap memory (inverted index → forward index conversion at query time).

**Problems:**
- Heap memory explosion (GBs of RAM for high-cardinality fields)
- GC pressure
- Slow cold start

**Doc values (modern):** Pre-built columnar storage on disk, memory-mapped.

```
Doc values (on disk, mmap'd):
docID → brand
1     → "Dell"
2     → "Lenovo"
3     → "HP"
...

OS page cache → effective caching without heap usage
```

**Always use doc values for facets.** This is the default in Elasticsearch (since 2.x).

### Global Ordinals

For high-cardinality fields, map values to integer ordinals for fast aggregation.

```
Term dictionary (string):
  "Dell"    → ordinal 0
  "HP"      → ordinal 1
  "Lenovo"  → ordinal 2

Doc values (integer ordinals):
docID → ordinal
1     → 0  (Dell)
2     → 2  (Lenovo)
3     → 1  (HP)
```

Aggregation counts ordinals (integers) instead of strings → faster.

**Challenge:** Global ordinals must be rebuilt when new segments are added (lazy computation on first query).

**Eager loading:**

```json
{
  "mappings": {
    "properties": {
      "brand": {
        "type": "keyword",
        "eager_global_ordinals": true
      }
    }
  }
}
```

Pre-builds global ordinals on index refresh (faster first query, but higher refresh cost).

### Limiting Facet Cardinality

High-cardinality fields (e.g., user IDs, unique titles) can make faceting expensive.

**Strategies:**
1. **Limit facet size:** Only return top-K values (e.g., top 100 brands).
2. **Sampling:** Compute facets on a sample of documents (Elasticsearch `sampler` aggregation).
3. **Use cardinality approximation:** HyperLogLog for distinct counts.
4. **Don't facet on high-cardinality fields:** Use search/filtering instead.

**Example:** Don't facet on `user_id` (millions of values). Instead, use cardinality aggregation or filtering.

---

## References

- Tunkelang, D. — *Faceted Search* (Morgan & Claypool, 2009) — Comprehensive overview of faceted search systems
- Hearst, M. A. — "Design recommendations for hierarchical faceted search interfaces" (ACM SIGIR Workshop, 2006)
- Rosenfeld, L. & Morville, P. — *Information Architecture for the Web and Beyond*, 4th ed. (O'Reilly, 2015) — Chapter on faceted classification
- Flajolet, P., et al. — "HyperLogLog: the analysis of a near-optimal cardinality estimation algorithm" (AOFA, 2007)
- Heuer, A., et al. — "HyperLogLog in practice: algorithmic engineering of a state of the art cardinality estimation algorithm" (EDBT, 2013) — HyperLogLog++
- Elasticsearch documentation — "Aggregations" — https://www.elastic.co/guide/en/elasticsearch/reference/current/search-aggregations.html
- Solr documentation — "Faceting" — https://solr.apache.org/guide/faceting.html
