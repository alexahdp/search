# Sorting and Pagination

After retrieving and scoring candidate documents, you need to sort them and deliver paginated results to users. This section covers top-K selection algorithms, distributed sorting challenges, pagination strategies, and the deep pagination problem.

## Table of Contents

1. [Top-K Selection Algorithms](#top-k-selection-algorithms)
2. [Distributed Sorting](#distributed-sorting)
3. [Pagination Strategies](#pagination-strategies)
4. [Deep Pagination Problem](#deep-pagination-problem)
5. [Cursor-Based Pagination](#cursor-based-pagination)
6. [Search After Pattern](#search-after-pattern)
7. [Ranking Ties and Stability](#ranking-ties-and-stability)
8. [References](#references)

---

## Top-K Selection Algorithms

When you have millions of scored documents but only need the top 10, sorting everything is wasteful. Top-K algorithms find the K highest-scoring elements without full sorting.

### Heap-Based Top-K (Min-Heap)

Maintain a min-heap of size K. As you scan scores, keep only the K largest.

**Algorithm:**

```python
import heapq

def top_k_heap(scores, k):
    """
    scores: list of (docID, score) tuples
    k: number of top results to return
    """
    if k >= len(scores):
        return sorted(scores, key=lambda x: -x[1])

    # Min-heap: smallest score is at the root
    heap = []

    for doc_id, score in scores:
        if len(heap) < k:
            heapq.heappush(heap, (score, doc_id))
        elif score > heap[0][0]:  # score > min score in heap
            heapq.heapreplace(heap, (score, doc_id))

    # Extract and sort
    result = sorted(heap, key=lambda x: -x[0])
    return [(doc_id, score) for score, doc_id in result]
```

**Complexity:**
- **Time:** $O(N \log K)$ where $N$ = number of scores, $K$ = top-K
- **Space:** $O(K)$

**When K << N** (e.g., K=10, N=1M), this is much better than $O(N \log N)$ full sort.

**Comparison to quickselect:** Heap approach is online (can process scores one at a time, useful for streaming). Quickselect requires the full array upfront.

### Quickselect

Finds the K-th largest element in $O(N)$ average time using a partitioning strategy similar to quicksort.

**Algorithm:**

```python
import random

def partition(arr, low, high):
    pivot = arr[high]
    i = low - 1
    for j in range(low, high):
        if arr[j] >= pivot:  # Descending order
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

def quickselect(arr, low, high, k):
    """Find the k-th largest element (0-indexed)"""
    if low == high:
        return low

    pivot_index = partition(arr, low, high)

    if k == pivot_index:
        return k
    elif k < pivot_index:
        return quickselect(arr, low, pivot_index - 1, k)
    else:
        return quickselect(arr, pivot_index + 1, high, k)

def top_k_quickselect(scores, k):
    arr = [(score, doc_id) for doc_id, score in scores]
    quickselect(arr, 0, len(arr) - 1, k - 1)
    # Top-K are now in arr[0:k], but not fully sorted
    result = sorted(arr[:k], key=lambda x: -x[0])
    return [(doc_id, score) for score, doc_id in result]
```

**Complexity:**
- **Time:** $O(N)$ average, $O(N^2)$ worst case
- **Space:** $O(1)$ (in-place)

**Advantage:** Faster than heap for offline batches (when you have all scores at once).

**Disadvantage:** Requires random access to the full array (not suitable for streaming).

### Partial Sort

Standard library partial sorts (C++ `std::partial_sort`, Python `heapq.nsmallest`):

```python
import heapq

def top_k_nlargest(scores, k):
    return heapq.nlargest(k, scores, key=lambda x: x[1])
```

**Complexity:** $O(N \log K)$ (heap-based internally).

**Best practice:** Use this for simplicity. Optimized implementations are as fast as hand-rolled heaps.

### Threshold Algorithms (TA, NRA)

For multi-field scoring where each field has a sorted list, you can find top-K without scanning all lists.

**Scenario:** Combine scores from multiple sources:

```
Query: "laptop" AND price < 1000

Source 1 (text relevance): sorted list of (docID, BM25_score)
Source 2 (price boost): sorted list of (docID, price_score)

Combined score: BM25_score × 0.7 + price_score × 0.3
```

**Threshold Algorithm (TA):**

```
1. Scan source lists in parallel (sorted by score, descending)
2. For each document seen, compute its combined score
3. Track the best-K candidates
4. Compute a threshold: max possible score of unseen documents
   threshold = current_score_in_source1 × 0.7 + current_score_in_source2 × 0.3
5. If the K-th best candidate score > threshold, stop (no unseen doc can beat top-K)
```

**Complexity:** $O(K \log K)$ to $O(N)$ depending on score distributions.

**Use case:** Multi-stage ranking, distributed search (see below).

---

## Distributed Sorting

In distributed search systems (Elasticsearch, Solr), documents are partitioned across shards. Sorting requires aggregating results from all shards.

### Naive Approach: Gather and Sort

```
Query: "laptop" (sort by BM25 score, return top-10)

Shard 1: Top-10 locally → [doc1:10.5, doc2:9.3, ..., doc10:5.1]
Shard 2: Top-10 locally → [doc11:12.3, doc12:8.7, ..., doc20:4.9]
Shard 3: Top-10 locally → [doc21:11.1, doc22:9.9, ..., doc30:6.0]

Coordinator:
  - Gather 30 results (10 per shard)
  - Merge-sort → global top-10
```

**Complexity (per shard):** $O(N_s \log K)$ where $N_s$ = docs per shard, $K$ = top-K per shard

**Coordinator complexity:** $O(S \cdot K \log(S \cdot K))$ where $S$ = number of shards

**Problem:** If you need top-10 globally but each shard returns top-10, you might miss results.

**Example:**

```
Shard 1 top-10: scores [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
Shard 2 top-10: scores [9.5, 9.4, 9.3, 9.2, 9.1, 9.0, 8.9, 8.8, 8.7, 8.6]

Global top-10: [10, 9.5, 9.4, 9.3, 9.2, 9.1, 9, 9.0, 8.9, 8.8]

Shard 1's doc with score 8 is not in global top-10 even though it was in local top-10.
```

**Solution:** Over-fetch from each shard.

### Over-Fetching Strategy

Request more than K results from each shard to ensure accuracy.

**Heuristic:** Fetch $K + S \times O$ from each shard where $O$ is an offset buffer (e.g., 10-50).

```
Query: top-10 globally, 5 shards

Request from each shard: top-20 (over-fetch by 10)
Coordinator merges: 100 results → global top-10
```

**Trade-off:**
- **Higher over-fetch** → more accurate, but higher network and CPU cost
- **Lower over-fetch** → faster, but potential ranking inaccuracies

**Elasticsearch default:** Fetch top `size` from each shard. For `size=10`, fetches 10 per shard (no over-fetch). Acceptable for most use cases but can be inaccurate for deep pagination.

### Distributed Top-K with Thresholds

Share score thresholds between coordinator and shards to prune early.

```
Round 1: Coordinator asks each shard for top-K candidates
Round 2: Coordinator computes global K-th score (threshold)
Round 3: Coordinator asks shards: "Re-score only docs with score > threshold"

Saves work on shards (skip scoring docs that can't make top-K).
```

**Protocol (simplified):**

```
Coordinator → Shards: "Find top-10 candidates"
Shards → Coordinator: [top-10 per shard]
Coordinator: Merge, compute global K-th score = 8.5
Coordinator → Shards: "Re-score docs with score > 8.5"
Shards → Coordinator: [refined scores]
Coordinator: Final merge → return top-10
```

**Complexity:** 2-3 rounds, but each shard does less work.

**Use case:** Google's Maglev load balancer, some large-scale search systems.

---

## Pagination Strategies

### Offset-Based Pagination (Traditional)

```
Page 1: offset=0, size=10   → docs [0-9]
Page 2: offset=10, size=10  → docs [10-19]
Page 3: offset=20, size=10  → docs [20-29]
```

**Implementation (Elasticsearch):**

```json
{
  "query": { "match_all": {} },
  "from": 20,
  "size": 10
}
```

**How it works:**
1. Retrieve top (offset + size) results
2. Discard first `offset` results
3. Return `size` results

**Complexity:** $O((offset + size) \log(offset + size))$ — must compute all results up to offset.

**Problem:** Deep pagination is expensive (see below).

### Cursor-Based Pagination

Instead of offset, use a cursor (stateful token) to remember where you left off.

```
Page 1: cursor=null → return docs [0-9] + cursor="abc123"
Page 2: cursor="abc123" → return docs [10-19] + cursor="def456"
Page 3: cursor="def456" → return docs [20-29] + cursor="ghi789"
```

**Cursor content:** Encodes the last document's sort values (e.g., score + docID).

**Advantages:**
- **Efficient:** Only compute results for current page, not all previous pages
- **Consistent:** Even if index changes between pages, cursor maintains position

**Disadvantages:**
- **Stateful:** Cursor must be stored (client or server)
- **No random access:** Can't jump to page 100 directly

**Implementation patterns:**
- **search_after** in Elasticsearch (see below)
- **Keyset pagination** in SQL databases

---

## Deep Pagination Problem

### The Problem

Requesting page 1000 (offset=10000, size=10):

```
Elasticsearch with 5 shards:

Each shard must:
  1. Score all matching documents
  2. Sort and return top 10,010 results

Coordinator:
  1. Receive 50,050 results (10,010 × 5 shards)
  2. Merge-sort 50,050 results
  3. Discard first 10,000
  4. Return 10 results to user
```

**Cost:** Compute and transfer 50,000 results to return 10. Wasteful!

**Memory:** Coordinator must hold 50,050 docs in memory during merge.

**Latency:** Slow (hundreds of ms to seconds for large offsets).

### Why Deep Pagination is Rare

**User behavior:** 99% of users don't go past page 3. Deep pagination is an edge case.

**Google's approach:** Doesn't allow pagination beyond ~1000 results. Encourages query refinement instead.

**Elasticsearch limit:** Default `index.max_result_window = 10,000`. Queries with `from + size > 10,000` are rejected.

### Solutions

**1. Limit pagination depth**

```json
// Elasticsearch config
{
  "index.max_result_window": 1000
}
```

Reject requests for pages beyond 100 (offset > 1000).

**2. Use search_after (cursor-based)**

Avoids offset entirely. See next section.

**3. Scroll API (batch export)**

For exporting all results (e.g., data dump, ETL), use scroll:

```json
// Initial request
POST /index/_search?scroll=1m
{
  "size": 100,
  "query": { "match_all": {} }
}

// Response includes scroll_id
{
  "_scroll_id": "DXF1ZXJ5QW5kRmV0Y2g...",
  "hits": { ... }
}

// Subsequent requests
POST /_search/scroll
{
  "scroll": "1m",
  "scroll_id": "DXF1ZXJ5QW5kRmV0Y2g..."
}
```

**Scroll maintains a consistent snapshot** of the index (search context is frozen). Efficient for batch processing but not for user-facing pagination.

**Deprecated in favor of:** Point-in-Time (PIT) API + search_after (Elasticsearch 7.10+).

**4. Encourage query refinement**

"Did you mean to search for X instead?" → help users narrow results rather than paginate endlessly.

---

## Search After Pattern

Elasticsearch's recommended approach for deep pagination.

### How It Works

Each page request includes the sort values of the last document from the previous page.

**Example:**

```
Query: "laptop" sorted by [score, _doc]

Page 1:
{
  "size": 10,
  "sort": [
    { "_score": "desc" },
    { "_doc": "asc" }
  ]
}

Response (last doc):
{
  "_score": 8.5,
  "_doc": 42
}

Page 2:
{
  "size": 10,
  "sort": [
    { "_score": "desc" },
    { "_doc": "asc" }
  ],
  "search_after": [8.5, 42]  ← Sort values from last doc
}
```

**Execution:**

Each shard returns docs where `(score, _doc) < (8.5, 42)` in sort order. No need to compute the first 10 docs.

**Complexity:** $O(K \log N)$ per page (K = page size, N = total docs). Independent of page depth!

### Advantages

- **Efficient:** No offset, no wasted computation
- **Scalable:** Deep pagination is as fast as shallow pagination
- **Consistent ordering:** If index changes between pages, results remain consistent within the sort key

### Disadvantages

- **No random access:** Can't jump to page 50 directly (must paginate sequentially)
- **Stateless:** Client must track `search_after` values
- **Sort key required:** Must sort by at least one unique field (e.g., `_doc` or `_id`)

### Implementation

```python
def search_with_pagination(query, page_size=10, search_after=None):
    body = {
        "size": page_size,
        "query": query,
        "sort": [
            {"_score": "desc"},
            {"_doc": "asc"}  # Tie-breaker for stable sort
        ]
    }

    if search_after:
        body["search_after"] = search_after

    response = es.search(index="products", body=body)

    hits = response["hits"]["hits"]
    next_search_after = hits[-1]["sort"] if hits else None

    return {
        "results": hits,
        "next_search_after": next_search_after
    }

# Usage
page1 = search_with_pagination({"match": {"title": "laptop"}})
page2 = search_with_pagination(
    {"match": {"title": "laptop"}},
    search_after=page1["next_search_after"]
)
```

### Point-in-Time (PIT) with search_after

Elasticsearch 7.10+ adds PIT API for consistent pagination even when the index is being updated.

**Problem with search_after alone:** If docs are added/deleted between pages, results can shift.

**Solution: PIT**

```json
// Create a PIT
POST /products/_pit?keep_alive=1m

// Response
{
  "id": "46ToAwMDaWR5BXV1a..."
}

// Search with PIT
POST /_search
{
  "pit": {
    "id": "46ToAwMDaWR5BXV1a...",
    "keep_alive": "1m"
  },
  "size": 10,
  "sort": [{"_score": "desc"}, {"_doc": "asc"}],
  "search_after": [8.5, 42]
}

// Close PIT when done
DELETE /_pit
{
  "id": "46ToAwMDaWR5BXV1a..."
}
```

**PIT freezes the index state** for the duration of pagination. Guarantees consistent results across pages.

**Keep-alive:** PIT expires after inactivity (default 1m). Extend with each request.

---

## Ranking Ties and Stability

### The Problem

Two documents with identical scores: how to order them?

```
Doc A: score = 10.0
Doc B: score = 10.0

Sort by score alone → order is undefined (may change between queries)
```

**Impact:**
- **Unstable pagination:** Doc A on page 1 in first query, page 2 in second query
- **User confusion:** "I just saw this result, now it's gone"

### Solution: Tie-Breaker Fields

Always include a unique, stable field in the sort:

```json
{
  "sort": [
    { "_score": "desc" },
    { "_doc": "asc" }  ← Tie-breaker (Lucene internal doc ID)
  ]
}
```

**`_doc`**: Lucene's internal document ID. Unique within a shard, but not globally unique.

For **global uniqueness** across shards, use a unique field:

```json
{
  "sort": [
    { "_score": "desc" },
    { "product_id": "asc" }  ← Unique field
  ]
}
```

### Deterministic Scoring

Even with tie-breakers, scores themselves can be non-deterministic:

**Causes:**
- **Floating-point precision:** Score 10.000001 vs 10.000002 may differ across shards
- **Lucene scoring details:** Segment-level statistics can cause slight variations
- **Index updates:** Doc frequencies change, affecting BM25 scores

**Mitigation:**
- **Round scores** to fewer decimal places (e.g., 2 decimals)
- **Use deterministic tie-breakers** (e.g., `_id`, timestamp)
- **Freeze index state** with PIT during pagination

---

## References

- Cormen, T. H., et al. — *Introduction to Algorithms*, 3rd ed. (MIT Press, 2009) — Chapters on quickselect, heaps
- Fagin, R., Lotem, A., & Naor, M. — "Optimal aggregation algorithms for middleware" (PODS, 2001) — Threshold Algorithm (TA)
- Ilyas, I. F., et al. — "A survey of top-k query processing techniques in relational database systems" (ACM Computing Surveys, 2008)
- Dean, J. & Ghemawat, S. — "MapReduce: Simplified data processing on large clusters" (OSDI, 2004) — Distributed sorting patterns
- Elasticsearch documentation — "Paginate search results" — https://www.elastic.co/guide/en/elasticsearch/reference/current/paginate-search-results.html
- Elasticsearch documentation — "Point in time API" — https://www.elastic.co/guide/en/elasticsearch/reference/current/point-in-time-api.html
- Solr documentation — "Pagination of results" — https://solr.apache.org/guide/pagination-of-results.html
- Brooker, M. — "Hash-based pagination" (AWS blog, 2018) — https://aws.amazon.com/blogs/database/amazon-dynamodb-deep-dive-advanced-design-patterns/
