# Hybrid Search

Hybrid search combines sparse (lexical) and dense (semantic) retrieval to achieve higher recall and precision than either method alone. This section covers practical implementation patterns, score fusion algorithms, architectural considerations, and advanced techniques like late interaction models.

## Why Hybrid?

### The Complementarity Argument

Sparse and dense retrieval make different types of errors:

**Sparse (BM25) misses**:
- Query: "how to fix a dripping faucet"
- Relevant doc: "repairing a leaking tap" (no term overlap)
- BM25 score: 0

**Dense misses**:
- Query: "PostgreSQL error 42P01"
- Relevant doc: "Error 42P01 indicates undefined table in PostgreSQL"
- Dense might conflate with other database errors (semantic similarity too broad)

**Hybrid catches both**: BM25 gets exact matches, dense gets semantic matches.

### Empirical Evidence

**MS MARCO** (passage ranking):
- BM25: MRR@10 = 18.7
- Dense (E5-large): MRR@10 = 37.9
- **Hybrid**: MRR@10 = 42.1 (+11% over dense alone)

**BEIR** (out-of-domain):
- BM25: NDCG@10 = 42.0
- Dense (E5-large): NDCG@10 = 50.2
- **Hybrid**: NDCG@10 = 52.1 (+4% over dense alone)

**Typical improvement**: +10-25% recall@100, +5-15% NDCG@10 over best single method.

## Hybrid Architectures

### Architecture 1: Parallel Retrieval + Fusion

**Flow**:
1. Query → BM25 index → top-K candidates
2. Query → ANN index → top-K candidates
3. Merge candidates with fusion algorithm → top-N results

```python
# Parallel retrieval
bm25_results = bm25_index.search(query, k=100)  # {doc_id: score}
dense_results = ann_index.search(encode(query), k=100)  # {doc_id: score}

# Fusion (RRF)
final_scores = rrf_fusion(bm25_results, dense_results, k=60)
return top_n(final_scores, n=10)
```

**Pros**:
- Simple, independent systems
- Easy to A/B test (run both, compare quality)
- Can tune K separately for each retriever

**Cons**:
- Maintains two indexes
- Latency = max(BM25 latency, dense latency) + fusion overhead

**When to use**: Default starting point for hybrid search.

### Architecture 2: BM25 First, Dense Re-rank

**Flow**:
1. Query → BM25 → top-1000 candidates (fast)
2. Encode query + top-1000 docs → dense scores
3. Re-rank by dense score or fusion

```python
candidates = bm25_index.search(query, k=1000)
query_embedding = encode(query)
doc_embeddings = [encode(doc) for doc_id, doc in candidates]
dense_scores = [cosine(query_embedding, doc_emb) for doc_emb in doc_embeddings]
return top_n(dense_scores, n=10)
```

**Pros**:
- Encodes fewer documents (only top-1000, not all 10M)
- Lower cost than full dense retrieval

**Cons**:
- Recall limited by BM25 first stage (if BM25 misses, dense can't fix)
- Encodes documents online (slower query latency)

**When to use**: Cost-sensitive, dense compute is expensive, BM25 has good recall.

### Architecture 3: Dense First, BM25 Boost

**Flow**:
1. Query → ANN → top-100 candidates (semantic)
2. Compute BM25 scores for those 100 docs
3. Fusion or multiplicative boost

```python
dense_results = ann_index.search(encode(query), k=100)
bm25_scores = {doc_id: bm25_score(query, doc) for doc_id in dense_results}
boosted_scores = {doc_id: dense_results[doc_id] * (1 + 0.3 * bm25_scores[doc_id])
                  for doc_id in dense_results}
return top_n(boosted_scores, n=10)
```

**Pros**:
- Semantic-first (better for fuzzy queries)
- BM25 computed only for 100 docs (cheap)

**Cons**:
- If dense misses key documents, BM25 can't fix
- Requires pre-computed dense embeddings for all docs

**When to use**: Semantic understanding is primary, want to boost exact term matches.

### Architecture 4: Multi-Stage Retrieval

**Flow**:
1. Coarse retrieval: BM25 → top-1000 (fast, high recall)
2. Re-rank stage 1: Dense bi-encoder → top-100 (semantic)
3. Re-rank stage 2: Cross-encoder → top-10 (high precision)

```python
stage1 = bm25_index.search(query, k=1000)
stage2 = rerank_dense(query, stage1, k=100)
stage3 = rerank_crossencoder(query, stage2, k=10)
return stage3
```

**Pros**:
- Highest quality (cascade of increasingly expensive models)
- Each stage narrows candidates

**Cons**:
- Complex pipeline
- High latency (3 stages)

**When to use**: Quality-critical applications (search engine result pages, premium features).

## Score Fusion Algorithms

### Challenge: Score Incompatibility

BM25 scores: typically 0-20, unbounded, different distributions per query
Dense scores: 0-1 (cosine similarity), relatively uniform

**Problem**: Can't just add or average them.

### Solution 1: Normalization

**Min-Max Normalization**:
$$
s'_i = \frac{s_i - \min(s)}{\max(s) - \min(s)}
$$

**Z-Score Normalization**:
$$
s'_i = \frac{s_i - \mu}{\sigma}
$$

**Then combine**:
$$
\text{score} = \alpha \cdot s'_{\text{bm25}} + (1 - \alpha) \cdot s'_{\text{dense}}
$$

**Pros**: Intuitive, easy to understand
**Cons**: Sensitive to outliers (min-max) or distribution shifts

### Solution 2: Reciprocal Rank Fusion (RRF)

Ignore scores entirely, use ranks.

**Algorithm**:
$$
\text{RRF}(d) = \sum_{r \in \{\text{bm25}, \text{dense}\}} \frac{1}{k + \text{rank}_r(d)}
$$

Where $k$ is a constant (typically 60).

**Example**:

BM25 ranks: {doc1: 1, doc2: 2, doc3: 5}
Dense ranks: {doc1: 3, doc2: 1, doc4: 2}

RRF(doc1) = 1/(60+1) + 1/(60+3) = 0.0164 + 0.0159 = 0.0323
RRF(doc2) = 1/(60+2) + 1/(60+1) = 0.0161 + 0.0164 = 0.0325
RRF(doc3) = 1/(60+5) + 0 = 0.0154
RRF(doc4) = 0 + 1/(60+2) = 0.0161

Final ranking: doc2, doc1, doc4, doc3

**Pros**:
- No hyperparameters to tune (k=60 is robust)
- No score calibration needed
- Simple to implement
- Surprisingly effective

**Cons**:
- Ignores score magnitudes (rank 1 vs rank 2 treated same as rank 50 vs rank 51)

**Best practice (2026)**: Start with RRF. It's the most widely used fusion method.

### Solution 3: Weighted RRF

Add weights to each retrieval system:

$$
\text{RRF}(d) = w_{\text{bm25}} \cdot \frac{1}{k + \text{rank}_{\text{bm25}}(d)} + w_{\text{dense}} \cdot \frac{1}{k + \text{rank}_{\text{dense}}(d)}
$$

Where $w_{\text{bm25}} + w_{\text{dense}} = 1$.

**Tuning**: Cross-validate on held-out data to find optimal weights.

**Typical values**: $w_{\text{bm25}} = 0.4, w_{\text{dense}} = 0.6$ (dense weighted higher for semantic tasks).

### Solution 4: Learned Fusion

Train a model to predict relevance given multiple signals.

**Features**:
- BM25 score
- Dense score
- BM25 rank
- Dense rank
- Query length
- Document length
- Exact match flags
- Number of overlapping terms
- ...

**Model**: Logistic regression, LambdaMART, or neural network.

**Training data**: (query, doc, label) triples from clickstream or relevance judgments.

**Example** (logistic regression):
```python
X = [bm25_score, dense_score, query_len, doc_len, exact_match]
y = model.predict_proba(X)  # relevance probability
```

**Pros**: Can leverage domain-specific features, optimal for your data
**Cons**: Requires labeled data (1K-10K queries), model training, more complexity

**When to use**: Large-scale systems with abundant data, mature relevance tuning teams.

## Implementation Patterns

### Pattern 1: Elasticsearch Hybrid (RRF)

Elasticsearch 8.0+ supports native hybrid search with RRF.

```json
POST /my_index/_search
{
  "retriever": {
    "rrf": {
      "retrievers": [
        {
          "standard": {
            "query": { "match": { "content": "machine learning" } }
          }
        },
        {
          "knn": {
            "field": "embeddings",
            "query_vector": [0.1, 0.2, ..., 0.5],
            "k": 100,
            "num_candidates": 1000
          }
        }
      ],
      "rank_constant": 60
    }
  },
  "size": 10
}
```

**Pros**: Built-in, no custom code, handles distribution automatically
**Cons**: Locked into Elasticsearch's RRF implementation

### Pattern 2: Pinecone Hybrid (Alpha Weighting)

Pinecone supports sparse-dense hybrid with learned sparse vectors.

```python
index.query(
    vector=[0.1, 0.2, ..., 0.5],  # dense query
    sparse_vector={
        'indices': [10, 150, 3000],  # sparse term IDs
        'values': [0.8, 0.9, 0.7]    # sparse weights
    },
    top_k=10,
    alpha=0.7  # weight between dense (1.0) and sparse (0.0)
)
```

**Pros**: Single index, single query, built-in fusion
**Cons**: Requires computing sparse vectors (SPLADE or BM25-to-sparse)

### Pattern 3: Custom Pipeline (Full Control)

```python
class HybridSearcher:
    def __init__(self, bm25_index, ann_index, encoder):
        self.bm25_index = bm25_index
        self.ann_index = ann_index
        self.encoder = encoder

    def search(self, query, k=10):
        # Parallel retrieval
        bm25_results = self.bm25_index.search(query, k=100)
        query_emb = self.encoder.encode(query)
        dense_results = self.ann_index.search(query_emb, k=100)

        # RRF fusion
        fused_scores = self.rrf_fusion(bm25_results, dense_results)

        # Return top-k
        return sorted(fused_scores.items(), key=lambda x: -x[1])[:k]

    def rrf_fusion(self, bm25_results, dense_results, k=60):
        scores = {}

        # Convert to ranks
        bm25_ranks = {doc_id: rank+1 for rank, (doc_id, _) in enumerate(bm25_results)}
        dense_ranks = {doc_id: rank+1 for rank, (doc_id, _) in enumerate(dense_results)}

        # RRF
        all_docs = set(bm25_ranks.keys()) | set(dense_ranks.keys())
        for doc_id in all_docs:
            score = 0
            if doc_id in bm25_ranks:
                score += 1 / (k + bm25_ranks[doc_id])
            if doc_id in dense_ranks:
                score += 1 / (k + dense_ranks[doc_id])
            scores[doc_id] = score

        return scores
```

**Pros**: Full control, can customize every step
**Cons**: More code, must handle distribution/sharding yourself

## ColBERT: Late Interaction Models

ColBERT (Contextualized Late Interaction over BERT) is a hybrid approach within dense retrieval.

### Traditional Bi-Encoder

Query: single vector (768-dim)
Document: single vector (768-dim)
Similarity: dot product (one number)

**Problem**: Compressing entire document to one vector loses fine-grained term-level signals.

### ColBERT Architecture

**Encoding**:
- Query: $Q = [q_1, q_2, ..., q_m]$ (m = query token count, each is 128-dim)
- Document: $D = [d_1, d_2, ..., d_n]$ (n = doc token count, each is 128-dim)

**Scoring** (late interaction):
$$
\text{score}(Q, D) = \sum_{i=1}^{m} \max_{j=1}^{n} (q_i \cdot d_j)
$$

For each query term, find the most similar document term, sum over all query terms.

**Intuition**: Like soft term matching — "machine" in query matches "machine" in doc strongly, "learning" matches "ML" moderately, etc.

### Why ColBERT Works

1. **Fine-grained matching**: Captures term-level interactions, unlike single-vector methods.
2. **Contextualized**: Each token embedding depends on context (via BERT).
3. **Efficient**: MaxSim is fast (optimized with vector instructions).
4. **Best of both worlds**: Semantic understanding (BERT) + term-level precision (MaxSim).

### ColBERT Performance

**MS MARCO**:
- BM25: 18.7
- Dense (bi-encoder): 37.9
- **ColBERT**: 39.7 (+5% over bi-encoder)

**BEIR**:
- BM25: 42.0
- Dense (bi-encoder): 50.2
- **ColBERT**: 51.8 (+3% over bi-encoder)

**Takeaway**: ColBERT is the highest-quality single dense method. But it's slower and more complex than bi-encoders.

### ColBERT Trade-offs

**Pros**:
- Higher quality than bi-encoders
- Interpretable (can see which terms matched)

**Cons**:
- Larger index (store all token embeddings, not just one doc vector)
  - Document: 200 tokens × 128 dim = 25.6 KB (vs. 3 KB for bi-encoder)
- Slower query (compute MaxSim for each candidate)
  - ~10-50ms per query vs. ~1-5ms for bi-encoder ANN

**When to use**: Quality-critical applications with budget for larger indexes (e.g., web search engines).

## Filtering and Metadata

Real-world queries often combine vector similarity with filters:

**Example**: "Find similar products to iPhone 14, under $500, in Electronics category, available in California."

### Naive Approach (Post-Filtering)

1. Hybrid search retrieves top-100 semantically similar products
2. Apply filters: price < 500, category = Electronics, location = California
3. Return top-10 after filtering

**Problem**: May return <10 results if most candidates don't pass filters.

### Pre-Filtering

1. Filter first: price < 500 AND category = Electronics AND location = California (→ 50K products)
2. Hybrid search on filtered subset
3. Return top-10

**Problem**: Breaks ANN index (small subset isn't indexed efficiently).

### Filtering-Aware Hybrid Search

Modern vector databases support filtering within ANN search.

**Approaches**:

**1. HNSW with filter checks** (Qdrant, Weaviate):
- During graph traversal, skip nodes that don't pass filter
- Ensures all results satisfy filter

**2. IVF + inverted index** (Milvus):
- Combine vector clusters with keyword inverted index
- Intersect cluster candidates with filter results

**3. Partition index by filter** (Pinecone namespaces):
- Pre-partition index by common filters (category, region)
- Search only relevant partition

**Best practice**: Use a vector database with native filter support. Don't implement this yourself.

## Performance Optimization

### 1. Cache Query Embeddings

If queries repeat (common in autocomplete, trending searches):

```python
@lru_cache(maxsize=10000)
def encode_query(query):
    return model.encode(query)
```

**Impact**: 10-50ms saved per cached query.

### 2. Batch Dense Encoding

For BM25-first architectures, batch-encode candidates:

```python
# Bad: encode one at a time (100ms for 100 docs)
embeddings = [encode(doc) for doc in candidates]

# Good: batch encode (20ms for 100 docs)
embeddings = encode_batch(candidates, batch_size=32)
```

### 3. Approximate BM25

For dense-first architectures, BM25 computed on-the-fly can be slow. Approximate:

```python
# Full BM25 (slow)
bm25_score = compute_bm25(query, doc, idf_table)

# Approximate: just count term matches (fast)
overlap = len(set(query_terms) & set(doc_terms))
approx_score = overlap * avg_idf
```

**Impact**: 80% of BM25 quality, 10x faster.

### 4. Prefetch Documents

If final step fetches document content:

```python
# Bad: serial fetches (N * 5ms = 50ms for 10 docs)
results = [fetch_doc(doc_id) for doc_id in top_10]

# Good: parallel fetch (5ms total)
results = fetch_docs_parallel(top_10)
```

### 5. Tune ANN Parameters

For HNSW:
- Increase `ef_search` if recall is low
- Decrease `ef_search` if latency is too high
- Monitor recall vs. latency curve

**Typical values**: `ef_search` = 50-200 for 95-98% recall.

## Evaluation and Tuning

### Offline Metrics

**Recall@K**: What fraction of relevant docs are in top-K?
- **Target**: >95% recall@100 (stage 1), >98% recall@10 (final)

**NDCG@10**: Ranking quality accounting for position.
- **Target**: Domain-specific (compare to BM25 baseline)

**Latency P99**: 99th percentile query latency.
- **Target**: <100ms for most systems

### Online Metrics

**Click-through rate (CTR)**: Fraction of queries with a click on top-10 results.

**Success rate**: Fraction of queries where user found what they needed (measured via session signals).

**Zero-result rate**: Fraction of queries returning no results.

### A/B Testing Hybrid vs. Single Method

**Setup**:
- Control: BM25 only (or dense only)
- Treatment: Hybrid (BM25 + dense with RRF)
- Split: 50/50 random assignment
- Duration: 2-4 weeks
- Metrics: CTR, success rate, zero-result rate

**Expected gains**: +5-15% CTR, +10-25% recall (offline)

**Cost increase**: 1.5-3x (two indexes, query encoding)

**Decision**: If quality gains justify cost, roll out hybrid.

## Common Pitfalls

### 1. Using Raw Scores Without Normalization

**Mistake**: `score = bm25_score + dense_score`

**Problem**: BM25 dominates (scale 0-20) vs. dense (scale 0-1).

**Fix**: Normalize or use RRF.

### 2. Not Tuning K for Each Retriever

**Mistake**: Retrieve top-10 from BM25, top-10 from dense, fuse.

**Problem**: Only 20 candidates total (poor recall).

**Fix**: Retrieve top-100 from each, fuse, return top-10.

### 3. Ignoring Query Length Effects

**Short queries** (1-2 words): BM25 often better (less context for dense).
**Long queries** (5+ words): Dense often better (more context).

**Fix**: Weight BM25 higher for short queries, dense higher for long.

### 4. Not Handling Empty Results

**Scenario**: BM25 returns 0 results, dense returns 100. RRF only uses dense ranks.

**Problem**: Effectively dense-only for this query (not hybrid).

**Fix**: Fallback logic or ensure BM25 always returns candidates (lower thresholds).

### 5. Over-Optimizing Fusion Weights

**Mistake**: Tune α to 3 decimal places on a small eval set.

**Problem**: Overfits to eval set, doesn't generalize.

**Fix**: Use RRF (no tuning) or coarse-tune α (0.3, 0.5, 0.7), pick best on large held-out set.

## Summary

| Aspect | Recommendation |
|---|---|
| **Default architecture** | Parallel retrieval (BM25 + dense) with RRF fusion |
| **Fusion algorithm** | RRF (k=60) — simple, robust, no tuning |
| **Top-K per retriever** | 100-200 (ensures good recall before fusion) |
| **Final top-N** | 10-20 (what user sees) |
| **When to use BM25-first** | Cost-sensitive, BM25 has good recall |
| **When to use dense-first** | Semantic understanding critical, fuzzy queries |
| **When to use ColBERT** | Quality-critical, larger index budget |
| **Filtering** | Use vector DB with native filter support |
| **Evaluation** | Measure both offline (NDCG, recall) and online (CTR, success rate) |

**Key takeaway**: Hybrid search is not just a "nice-to-have" — it's the state-of-the-art in 2026. Start with parallel retrieval + RRF. It's simple, effective, and robust. Optimize from there only if you have data showing clear gains from more complexity.

## References and Further Reading

- [Khattab & Zaharia (2020): ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT](https://arxiv.org/abs/2004.12832) — ColBERT
- [Cormack et al. (2009): Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) — RRF algorithm
- [Thakur et al. (2021): BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models](https://arxiv.org/abs/2104.08663) — Hybrid evaluation
- [Lin et al. (2023): How Does Generative Retrieval Scale to Millions of Passages?](https://arxiv.org/abs/2305.11841) — Hybrid vs. generative retrieval
- [Elasticsearch: How to Use RRF](https://www.elastic.co/guide/en/elasticsearch/reference/current/rrf.html) — Production implementation
