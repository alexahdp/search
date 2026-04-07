# Full-Text Search

Full-text search is the core retrieval mechanism in information retrieval systems. This section goes deep into the algorithms that power query evaluation: how posting lists are intersected, how phrase queries work, how proximity affects scoring, and how query optimizers make these operations fast enough to handle thousands of queries per second.

## Table of Contents

1. [Query Evaluation Strategies](#query-evaluation-strategies)
2. [Posting List Intersection](#posting-list-intersection)
3. [Phrase Queries](#phrase-queries)
4. [Proximity Scoring](#proximity-scoring)
5. [Query Optimization](#query-optimization)
6. [Boolean Queries](#boolean-queries)
7. [Wildcard and Regular Expression Queries](#wildcard-and-regular-expression-queries)
8. [Implementation Patterns](#implementation-patterns)
9. [References](#references)

---

## Query Evaluation Strategies

There are two fundamental approaches to evaluating a query against an inverted index:

### Document-at-a-Time (DAAT)

Process one document at a time, computing its full score before moving to the next:

```
Query: "search engine"

DAAT execution:
1. Initialize: scores = {}
2. For each document d in posting_list("search"):
     scores[d] += IDF("search") × TF_component("search", d)
3. For each document d in posting_list("engine"):
     scores[d] += IDF("engine") × TF_component("engine", d)
4. Sort scores, return top-K
```

**Properties:**
- Simple to implement
- Natural for OR queries (union of posting lists)
- Can use a min-heap to track top-K candidates dynamically
- Requires materializing all candidates before scoring completes
- Less cache-friendly (jumping between posting lists)

**Optimizations:**
- **Max-score pruning**: Precompute upper bounds on term contributions. If current doc + remaining terms can't beat the K-th best score, skip it.
- **WAND (Weak AND)**: Sort posting lists by docID, use skip pointers to jump to candidates that might beat the threshold.

### Term-at-a-Time (TAAT)

Process one term at a time, accumulating scores across all its documents:

```
Query: "search engine"

TAAT execution:
1. scores = array[num_docs] initialized to 0
2. Process "search":
     For each (docID, TF) in posting_list("search"):
       scores[docID] += IDF("search") × TF_component("search", docID)
3. Process "engine":
     For each (docID, TF) in posting_list("engine"):
       scores[docID] += IDF("engine") × TF_component("engine", docID)
4. Sort scores, return top-K
```

**Properties:**
- Better cache locality (sequential scan through each posting list)
- Easier to parallelize (each term can be processed independently, then merge)
- Requires a score accumulator array (can be large for billion-doc corpora)
- Natural for AND queries (intersection before scoring)

**Optimizations:**
- **Early termination**: If posting lists are sorted by impact (score contribution), process high-impact docs first and terminate early.
- **Accumulator pruning**: Dynamically discard low-scoring accumulators during processing to reduce memory.

### Comparison

| Aspect | DAAT | TAAT |
|---|---|---|
| **Scoring order** | Document-by-document | Term-by-term |
| **Memory** | O(K) for top-K heap | O(num_candidates) for accumulators |
| **Cache behavior** | Poor (random access to posting lists) | Good (sequential scans) |
| **Best for** | OR queries, dynamic pruning | AND queries, parallel processing |
| **Used in** | Lucene (for some queries) | Wumpus, Terrier |

Modern systems like Lucene use **hybrid approaches**: TAAT for AND/filter portions, DAAT for final scoring and top-K selection.

---

## Posting List Intersection

AND queries require intersecting posting lists. This is one of the most performance-critical operations in a search engine.

### Naive Merge (Zipper Algorithm)

For sorted posting lists $L_1$ and $L_2$:

```
INTERSECT(L1, L2):
    result = []
    i = 0, j = 0
    while i < len(L1) and j < len(L2):
        if L1[i] == L2[j]:
            result.append(L1[i])
            i++, j++
        else if L1[i] < L2[j]:
            i++
        else:
            j++
    return result
```

**Complexity:** $O(|L_1| + |L_2|)$

This is optimal when lists are similar in size. But for queries like "the algorithm" where "the" has 100M postings and "algorithm" has 1000, we scan 100M entries even though the result size is ≤1000.

### Intersection with Skip Pointers

Skip pointers (see [Data Structures](../01-fundamentals/data-structures.md#skip-pointers-in-posting-lists)) allow jumping ahead in the longer list:

```
INTERSECT_SKIP(short_list, long_list):
    result = []
    long_iter = long_list.iterator()
    for doc in short_list:
        // Use skip pointers to jump to doc or next larger docID
        long_iter.skipTo(doc)
        if long_iter.current() == doc:
            result.append(doc)
    return result
```

**Complexity:** $O(|short| \cdot \sqrt{|long|})$ with skip interval $\sqrt{|long|}$

In practice, for $|short| \ll |long|$, this is $O(|short| \cdot \log |long|)$ with multi-level skip lists.

### Galloping Intersection

Combines binary search with linear merge. When one list is much shorter, gallop through the longer list:

```
GALLOPING_INTERSECT(short_list, long_list):
    result = []
    long_idx = 0
    for doc in short_list:
        // Exponential search (gallop) to find doc in long_list
        long_idx = galloping_search(long_list, doc, long_idx)
        if long_idx < len(long_list) and long_list[long_idx] == doc:
            result.append(doc)
            long_idx++
    return result

GALLOPING_SEARCH(list, target, start):
    // Jump in powers of 2: check start, start+1, start+2, start+4, start+8, ...
    step = 1
    while start + step < len(list) and list[start + step] < target:
        start += step
        step *= 2
    // Binary search in [start, start+step]
    return binary_search(list, target, start, start + step)
```

**Complexity:** $O(|short| \cdot \log |long|)$

Used in Lucene's `ConjunctionScorer`. Empirically faster than skip pointers when ratio $|long| / |short| > 100$.

### Adaptive Intersection

Switch strategies based on list sizes:

```
ADAPTIVE_INTERSECT(lists):
    lists = sorted_by_length(lists)
    if len(lists) == 2:
        ratio = len(lists[1]) / len(lists[0])
        if ratio < 10:
            return merge_intersect(lists[0], lists[1])
        else if ratio < 1000:
            return skip_intersect(lists[0], lists[1])
        else:
            return galloping_intersect(lists[0], lists[1])
    else:
        // Multi-way intersection: start with shortest two
        result = ADAPTIVE_INTERSECT([lists[0], lists[1]])
        for i in 2..len(lists):
            result = ADAPTIVE_INTERSECT([result, lists[i]])
        return result
```

This is the Lucene approach: heuristics choose the best algorithm at runtime based on statistics.

### SIMD-Accelerated Intersection

Modern CPUs can compare 4-8 integers in parallel using SIMD instructions (AVX2, AVX-512). Libraries like [SIMD-BP128](https://github.com/lemire/SIMDCompressionAndIntersection) achieve 4-8× speedup on posting list intersection.

**Key technique:** Decompress small blocks of postings (128-256 integers) using SIMD, then intersect blocks.

```
SIMD_INTERSECT(L1, L2):
    while L1.has_blocks() and L2.has_blocks():
        block1 = L1.decompress_block_simd()  // 128 docIDs
        block2 = L2.decompress_block_simd()
        matches = simd_intersect_sorted(block1, block2)  // AVX2 intrinsics
        result.extend(matches)
        // Advance to next blocks based on max docID in each
```

Used in cutting-edge systems (e.g., PISA, Partitioned Elias-Fano indexes). Still experimental in mainstream search engines.

### Bitmap Intersection

For very dense posting lists (many postings in a small docID range), bitmaps can be faster:

```
Posting list: [1, 2, 3, 5, 8, 9, 10, 12, 13, ...]
Bitmap:       [0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, ...]
```

Intersection becomes bitwise AND:

```
result_bitmap = bitmap1 & bitmap2
```

**Roaring Bitmaps** (used in Lucene for filter caching) combine bitmaps for dense regions with sorted arrays for sparse regions, giving the best of both.

**When to use:**
- Dense posting lists (> 10% of docIDs in a range)
- Frequently used filters (cache them as bitmaps)
- NOT queries (bitmap complement is trivial)

---

## Phrase Queries

Phrase queries like `"search engine"` require terms to appear in sequence. This needs **positional indexes** (see [Data Structures](../01-fundamentals/data-structures.md#the-inverted-index)).

### Exact Phrase Matching

```
Positional posting lists:
"search" → doc1:[5, 12], doc2:[3], doc3:[1, 8, 20]
"engine" → doc1:[6, 13], doc2:[7], doc3:[9]

Phrase "search engine" (positions must be consecutive):
1. Intersect docIDs: {doc1, doc2, doc3} ∩ {doc1, doc2, doc3} = {doc1, doc2, doc3}
2. For each candidate doc, check positions:
   doc1: "search" at [5, 12], "engine" at [6, 13]
         → 5+1=6 ✓, 12+1=13 ✓  → MATCH
   doc2: "search" at [3], "engine" at [7]
         → 3+1=4 ≠ 7  → NO MATCH
   doc3: "search" at [1, 8, 20], "engine" at [9]
         → 1+1=2 ≠ 9, 8+1=9 ✓, 20+1=21 ≠ 9  → MATCH

Result: {doc1, doc3}
```

**Algorithm:**

```
PHRASE_MATCH(terms):
    candidates = intersect(posting_lists(terms))
    result = []
    for doc in candidates:
        positions_list = [get_positions(term, doc) for term in terms]
        if has_consecutive_sequence(positions_list):
            result.append(doc)
    return result

HAS_CONSECUTIVE_SEQUENCE(pos_lists):
    // For each position p of term[0], check if term[1] at p+1, term[2] at p+2, ...
    for p0 in pos_lists[0]:
        match = True
        for i in 1..len(pos_lists):
            if (p0 + i) not in pos_lists[i]:
                match = False
                break
        if match:
            return True
    return False
```

**Complexity:** $O(k \cdot p)$ where $k$ = number of candidates, $p$ = average positions per term per doc.

### Optimizations

1. **Position filtering during intersection**: While intersecting docID lists, also check if positions are compatible before adding to candidates. Requires position-aware intersection (more complex but faster).

2. **Next-word index**: Precompute common bigrams/trigrams. "search engine" gets its own posting list. Trades index size for query speed.

3. **Phrase caching**: Cache results of frequent phrase queries. Elasticsearch does this automatically for filter phrases.

### Proximity Queries

More flexible than exact phrases: `"search engine"~5` means "search" and "engine" within 5 tokens of each other.

```
SLOP_PHRASE_MATCH(terms, slop):
    candidates = intersect(posting_lists(terms))
    result = []
    for doc in candidates:
        positions_list = [get_positions(term, doc) for term in terms]
        if has_proximity_match(positions_list, slop):
            result.append(doc)
    return result

HAS_PROXIMITY_MATCH(pos_lists, slop):
    for p0 in pos_lists[0]:
        for p1 in pos_lists[1]:
            if abs(p1 - p0) - 1 <= slop:
                return True  // Found a match within slop
    return False
```

For multi-term phrases with slop, this generalizes to checking if positions can be ordered within a window of size `len(terms) + slop`.

**Complexity:** $O(k \cdot p^2)$ for pairwise comparisons. Can be optimized with position skip lists or interval merging.

---

## Proximity Scoring

Even without explicit proximity queries, position information can improve ranking. Documents where query terms appear close together are often more relevant.

### Proximity Scoring Models

**BM25 with Proximity** (BM25P):

Add a proximity term to the standard BM25 score:

$$\text{BM25P}(q, d) = \text{BM25}(q, d) + \lambda \cdot \text{ProximityScore}(q, d)$$

where:

$$\text{ProximityScore}(q, d) = \sum_{t_i, t_j \in q, i < j} \frac{1}{\text{min\_distance}(t_i, t_j, d)}$$

**Minimum distance** = smallest gap between any occurrence of $t_i$ and $t_j$ in document $d$.

**Example:**

```
Document: "search algorithms are fundamental to search engine design"
Query: "search engine"

Positions:
  "search" at [0, 5]
  "engine" at [6]

Distances:
  between search[0] and engine[6]: 6 - 0 = 6
  between search[5] and engine[6]: 6 - 5 = 1  ← minimum

ProximityScore = 1/1 = 1.0

Compare to document with "search" at [0] and "engine" at [50]:
  min_distance = 50, ProximityScore = 1/50 = 0.02 (much lower)
```

**Parameter $\lambda$**: Controls how much proximity matters (typically 0.1 - 0.5).

### Span Queries in Lucene

Lucene's `SpanQuery` API computes exact spans (intervals where terms appear):

```java
SpanNearQuery query = new SpanNearQuery(
    new SpanQuery[]{
        new SpanTermQuery(new Term("field", "search")),
        new SpanTermQuery(new Term("field", "engine"))
    },
    5,  // slop
    true  // inOrder: must appear in this order
);
```

Spans can be used for:
- Phrase queries with slop
- Ordered vs. unordered proximity
- Complex patterns: "search NEAR/10 (engine OR algorithm)"
- Highlighting (exact match regions)

---

## Query Optimization

A naive query evaluator would be too slow for production. Query optimizers rewrite and reorder operations to minimize work.

### Query Rewriting

**1. Term expansion:**

```
Before: "running"
After:  "run" OR "running" OR "runs"  (stemming/synonyms)
```

**2. Boolean simplification:**

```
Before: (A OR A) AND B
After:  A AND B

Before: (A AND B) OR (A AND C)
After:  A AND (B OR C)  (distributive law)
```

**3. Constant folding:**

```
Before: field:value AND __all:*
After:  field:value
```

### Query Execution Optimization

**1. Term ordering for intersection:**

Process shortest posting list first:

```
Query: "the algorithm ranking"
Posting list sizes: "the" (100M), "algorithm" (1K), "ranking" (10K)

Naive order: intersect("the", "algorithm"), then intersect with "ranking"
Optimized:   intersect("algorithm", "ranking"), then intersect with "the"

Saves scanning 100M postings from "the".
```

**2. Filter ordering:**

Cheap filters (cached bitmaps) before expensive ones (script filters):

```
Query: text_match AND cached_filter AND expensive_script

Optimized execution:
1. cached_filter         → 10K candidates (fast bitmap AND)
2. text_match on 10K     → 500 candidates
3. expensive_script on 500  → 50 results
```

**3. Clause caching:**

Cache results of common clauses:

```
Query: (category:electronics AND price:[100 TO 200]) AND user_query
                └────── cacheable filter ──────┘
```

Elasticsearch's query cache stores filter clause results (docID bitmaps) per segment.

### Cost-Based Optimization

Advanced query optimizers estimate cost of different execution plans:

```
Cost model:
  - Posting list scan: cost = list_length × 1.0
  - Intersection: cost = min(list1, list2) × 2.0
  - Scoring: cost = num_candidates × 10.0
  - Filter (cached): cost = 0.1

Query: A AND B AND C, with lengths |A|=100K, |B|=10K, |C|=1K

Plan 1: ((A ∩ B) ∩ C)
  Cost = 100K×1.0 + 10K×1.0 + 10K×2.0 + 1K×1.0 + 1K×2.0 = 133K

Plan 2: ((B ∩ C) ∩ A)
  Cost = 10K×1.0 + 1K×1.0 + 1K×2.0 + 100K×1.0 + 1K×2.0 = 115K  ← cheaper

Choose Plan 2.
```

Lucene's query optimizer does this implicitly via heuristics (e.g., always process shortest list first). Full cost-based optimization is more common in database query planners but is emerging in search (e.g., Vespa).

---

## Boolean Queries

Combining terms with AND, OR, NOT.

### Evaluation Strategies

**OR (disjunction):**

```
"search" OR "engine"

DAAT approach:
- Merge posting lists (union)
- Score each document with matching terms
```

**AND (conjunction):**

```
"search" AND "engine"

TAAT approach:
- Intersect posting lists
- Score only intersection
```

**NOT (negation):**

```
"search" AND NOT "engine"

Approach:
- Compute "search" candidates
- Remove docIDs from "engine" posting list (set difference)

Optimization: If NOT clause has small posting list, compute bitmap complement.
```

### Compound Queries

```
(A OR B) AND C AND NOT D

Execution plan:
1. union = A ∪ B
2. result = union ∩ C
3. result = result \ D

Optimized:
1. Process C first (if smallest)
2. Filter with A OR B
3. Remove D
```

### Minimum-Should-Match

Lucene's `BooleanQuery` supports minimum-should-match:

```
Query: "search" OR "engine" OR "ranking" OR "algorithm" (minimumShouldMatch=2)

Meaning: Document must contain at least 2 of these terms.

Algorithm:
- Intersect combinations: (A∩B) ∪ (A∩C) ∪ (A∩D) ∪ (B∩C) ∪ ...
- Optimized: Threshold intersection algorithm (count term occurrences per doc, keep if ≥ 2)
```

**Threshold algorithm:**

```
MINIMUM_SHOULD_MATCH(posting_lists, threshold):
    accumulators = {}  // docID → count
    for posting_list in posting_lists:
        for docID in posting_list:
            accumulators[docID] = accumulators.get(docID, 0) + 1
    return [docID for docID, count in accumulators if count >= threshold]
```

**Complexity:** $O(\sum |lists|)$ — scan all lists once.

---

## Wildcard and Regular Expression Queries

### Prefix Queries

`"search*"` matches "search", "searches", "searching", ...

**Implementation via term dictionary (trie/FST):**

```
PREFIX_QUERY("search"):
    term_dict = get_term_dictionary()
    matching_terms = term_dict.range("search", "searcg")  // Next prefix
    return OR_query(matching_terms)
```

Lucene's FST makes this $O(|prefix| + k)$ where $k$ = number of matching terms.

**Limit on expansion:** Too many matches (e.g., "a*" in English) can explode into thousands of terms. Lucene limits to 1024 terms by default (configurable via `BooleanQuery.maxClauseCount`).

### Suffix and Infix Queries

`"*ing"` (suffix) or `"*arch*"` (infix) require special index structures:

**N-gram indexing:**

Index all n-grams of each term:

```
Term: "searching"
3-grams: "sea", "ear", "arc", "rch", "chi", "hin", "ing"

Query "*arch*" → search for 3-gram "arc" → retrieve "searching"
```

Trade-off: Index size grows significantly (3-4× for 3-grams).

**Suffix array / reverse index:**

For suffix queries, index terms in reverse:

```
Term: "searching" → reverse: "gnihcraes"
Query "*ing" → reverse: "gni*" → prefix query on reverse index
```

### Regular Expression Queries

`/search[a-z]+ing/` requires automaton-based matching.

**Levenshtein Automaton:**

Compile the regex into a finite automaton, then intersect it with the term dictionary FST:

```
REGEX_QUERY(pattern):
    automaton = compile_regex(pattern)
    term_dict_fst = get_term_dictionary()
    matching_terms = intersect_automatons(automaton, term_dict_fst)
    return OR_query(matching_terms)
```

Lucene's `RegexpQuery` uses this approach. Works well for simple patterns but can be slow for complex regexes (especially unbounded quantifiers like `.*`).

---

## Implementation Patterns

### Lucene's Approach

Lucene uses a **DAAT** approach with top-K heaps for most queries:

```java
// Simplified Lucene query execution
IndexSearcher searcher = new IndexSearcher(reader);
Query query = new TermQuery(new Term("field", "search"));
TopDocs results = searcher.search(query, 10);  // Top 10

Internal execution:
1. Query.createWeight() → builds a scorer
2. Scorer.iterator() → returns a DocIdSetIterator (posting list)
3. While iterator.nextDoc() != NO_MORE_DOCS:
     score = scorer.score()
     topDocsCollector.collect(docID, score)  // Maintain top-K heap
4. Return sorted top-K
```

**Key classes:**
- `Weight`: Query execution plan
- `Scorer`: Produces scores for matching documents
- `DocIdSetIterator`: Iterates posting list
- `Collector`: Aggregates results (top-K, facets, etc.)

### Elasticsearch Query DSL Mapping

```json
{
  "query": {
    "bool": {
      "must": [
        { "match": { "content": "search engine" } }
      ],
      "filter": [
        { "term": { "status": "published" } },
        { "range": { "date": { "gte": "2024-01-01" } } }
      ]
    }
  }
}
```

Execution:
1. `filter` clauses → bitmap intersection (cached)
2. `must` clause → BM25 scoring on filtered candidates
3. Combine with `should` (OR) and `must_not` (NOT) clauses

---

## References

- Zobel, J. & Moffat, A. — "Inverted Files for Text Search Engines" (ACM Computing Surveys, 2006) — Comprehensive survey of posting list operations
- Strohman, T. & Croft, W. B. — "Efficient Document Retrieval in Main Memory" (SIGIR, 2007) — DAAT vs TAAT analysis
- Broder, A. Z., et al. — "Efficient Query Evaluation using a Two-Level Retrieval Process" (CIKM, 2003) — WAND algorithm
- Ding, S. & Suel, T. — "Faster Top-k Document Retrieval Using Block-Max Indexes" (SIGIR, 2011) — Block-Max WAND (BMW)
- Lemire, D. & Boytsov, L. — "Decoding billions of integers per second through vectorization" (Software: Practice and Experience, 2015) — SIMD intersection
- Chambi, S., et al. — "Better bitmap performance with Roaring bitmaps" (Software: Practice and Experience, 2016)
- McCandless, M., Hatcher, E., & Gospodnetić, O. — *Lucene in Action*, 2nd ed. (Manning, 2010) — Practical Lucene internals
- Manning, C. D., Raghavan, P., & Schütze, H. — *Introduction to Information Retrieval*, Chapters 1-7 (Cambridge, 2008)
