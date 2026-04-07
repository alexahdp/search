# Approximate Nearest Neighbor Search

Dense retrieval requires finding the nearest neighbors of a query vector in a high-dimensional space with millions or billions of document vectors. Exact search (computing distance to every vector) is prohibitively expensive at scale. Approximate Nearest Neighbor (ANN) algorithms trade small accuracy losses for massive speedups — making sub-10ms queries over billion-scale datasets possible.

## The ANN Problem

**Input**:
- Database of $n$ vectors $\mathbf{D} = \{v_1, v_2, \ldots, v_n\}$ in $\mathbb{R}^d$
- Query vector $q \in \mathbb{R}^d$
- Distance metric (Euclidean $L_2$, cosine similarity, dot product)

**Output**:
- $k$ vectors from $\mathbf{D}$ closest to $q$ (approximately)

**Exact solution** (naive):
```python
distances = [distance(q, v) for v in D]
return top_k(distances, k)
```
**Complexity**: $O(nd)$ per query. For $n=10^9, d=768$: ~4 seconds per query on modern hardware.

**Goal**: Answer queries in $O(\log n)$ or $O(\sqrt{n})$ time with >95% recall.

## Quality Metrics

**Recall@k**: Fraction of true top-k neighbors found by ANN.

$$
\text{Recall@k} = \frac{|\text{ANN-topk} \cap \text{exact-topk}|}{k}
$$

**Example**: True top-10: [5, 12, 23, ...]. ANN returns [5, 12, 99, ...]. Recall@10 = 0.8 (8/10 matches).

**QPS (Queries Per Second)**: Throughput under load.

**Build time**: Time to construct the index from scratch.

**Memory**: Index size relative to raw vector data.

**Trade-off**: Higher recall requires more computation/memory. Most production systems target 95-98% recall.

## Taxonomy of ANN Algorithms

Four main families:

1. **Hashing-based** (LSH, Multi-Probe LSH) — probabilistic bucketing
2. **Tree-based** (KD-trees, Annoy, MRPT) — space partitioning
3. **Quantization-based** (PQ, OPQ, SQ) — compression and fast distance computation
4. **Graph-based** (HNSW, NSG, DiskANN) — navigable proximity graphs

## Locality-Sensitive Hashing (LSH)

LSH uses hash functions that map similar vectors to the same bucket with high probability.

### Random Projection LSH (for Cosine Similarity)

**Idea**: Project vectors onto random hyperplanes. Vectors on the same side of the hyperplane get the same bit.

**Algorithm**:
1. Generate $L$ random hyperplanes $\{r_1, \ldots, r_L\}$ (unit vectors in $\mathbb{R}^d$)
2. For each vector $v$, compute hash:
   $$
   h(v) = [\text{sign}(v \cdot r_1), \text{sign}(v \cdot r_2), \ldots, \text{sign}(v \cdot r_L)]
   $$
3. Store vectors in hash table keyed by $h(v)$

**Query**:
1. Compute $h(q)$
2. Return all vectors in bucket $h(q)$ (plus adjacent buckets for multi-probe)
3. Rank by exact distance

**Probability of collision**:
$$
P[h_i(u) = h_i(v)] = 1 - \frac{\theta}{\pi}
$$
Where $\theta = \arccos(\cos(u, v))$ is the angle between $u$ and $v$.

**Parameters**:
- $L$: Number of hash functions. More $\Rightarrow$ higher recall, more memory.
- $m$: Multi-probe parameter (check $2^m$ adjacent buckets). More $\Rightarrow$ higher recall, slower query.

### LSH for L2 Distance

Use **p-stable distributions** (Gaussian projections + bucketing):

$$
h(v) = \left\lfloor \frac{r \cdot v + b}{w} \right\rfloor
$$

Where $r \sim \mathcal{N}(0, I)$, $b \sim \text{Uniform}(0, w)$, $w$ is bucket width.

### LSH Trade-offs

**Pros**:
- Simple to implement
- Sublinear query time: $O(d \cdot n^{1/c})$ for $c > 1$
- Provable guarantees

**Cons**:
- High memory (multiple hash tables)
- Sensitive to hyperparameter tuning
- Inferior to graph methods in practice

**When to use**: Low-dimensional spaces ($d < 100$), streaming data (easy to update), theoretical guarantees needed.

## Tree-Based Methods

### KD-Trees

Recursively partition space along axis-aligned hyperplanes.

**Build**:
1. Choose dimension with largest variance
2. Split at median
3. Recurse on left and right subsets

**Query**: Traverse tree, backtrack to explore promising branches.

**Problem**: Curse of dimensionality. In high dimensions ($d > 20$), nearly all branches must be explored. Performance degrades to $O(n)$.

**Verdict**: Don't use KD-trees for $d > 50$. They're outclassed by other methods.

### Annoy (Approximate Nearest Neighbors Oh Yeah)

Spotify's library, popular for embeddings.

**Build**:
1. Pick two random points $p, q$
2. Split space with hyperplane equidistant from $p, q$
3. Recurse on each side
4. Repeat to build $T$ trees (forest)

**Query**: Search all $T$ trees, merge candidates, rank by distance.

**Why multiple trees?**: Each tree makes different random splits. Union of results improves recall.

**Parameters**:
- $T$: Number of trees (typically 10-100). More $\Rightarrow$ higher recall, more memory.
- `search_k`: Nodes to explore during query (typically $k \cdot T$). More $\Rightarrow$ slower, higher recall.

**Pros**:
- Simple, fast, memory-efficient
- Mmapped index (shares memory across processes)
- Good for $d = 100-500$

**Cons**:
- Inferior recall to HNSW at same speed
- Static index (rebuilds expensive)

**When to use**: Read-only indexes, Python environments (good bindings), don't need SOTA.

## Graph-Based Methods

Build a proximity graph where each node (vector) connects to its nearest neighbors. Query by greedy graph traversal.

### HNSW (Hierarchical Navigable Small World)

State-of-the-art for most use cases. Used by Pinecone, Weaviate, Qdrant, Milvus, and others.

**Key insight**: Combine skip-list layering with navigable small-world graphs.

**Structure**:
- **Layer 0** (bottom): All vectors, each connected to $M$ nearest neighbors
- **Layer 1, 2, ...**: Progressively sparser subsets (exponential decay)
- **Top layer**: Few vectors (entry points)

**Build**:
1. Insert vectors incrementally
2. For each vector $v$:
   - Assign layer $\ell$ with probability $\exp(-\ell)$
   - For layers $0$ to $\ell$:
     - Find $M$ nearest neighbors via greedy search
     - Add bidirectional edges
     - Prune edges to maintain max degree $M_{max}$

**Query**:
1. Start at top layer entry point
2. Greedily navigate to nearest neighbor at current layer
3. Drop to next layer
4. Repeat until layer 0
5. Explore layer 0 with beam search (width $ef$)

**Parameters**:
- $M$: Max connections per node (typically 16-64). More $\Rightarrow$ better recall, more memory, slower build.
- $ef_{construct}$: Beam width during construction (typically 100-500). More $\Rightarrow$ better graph quality, slower build.
- $ef_{search}$: Beam width during query (typically 50-500). More $\Rightarrow$ higher recall, slower query.

**Complexity**:
- **Build**: $O(n \log n \cdot M \cdot d)$ — slow, but parallelizable
- **Query**: $O(\log n \cdot M \cdot d)$ — very fast
- **Memory**: $O(n \cdot M \cdot \log n)$ — higher than IVF, but manageable

**Why HNSW dominates**:
- Best recall/QPS trade-off in practice (often 95%+ recall at >1000 QPS)
- Robust to different distance metrics and dimensionalities
- Incremental updates (add vectors without rebuild)

**Cons**:
- High memory (stores full graph)
- Slow to build (hours for 100M vectors)
- Poor performance on very high dimensions ($d > 2000$)

### NSG (Navigable Small World Graph)

Similar to HNSW but single-layer with more careful graph construction.

**Difference**: Uses a monotonic search property (guarantees convergence) and more aggressive pruning.

**Performance**: Slightly better recall than HNSW at same speed, but slower to build.

**Adoption**: Less popular in practice — HNSW is the default.

## Inverted File Index (IVF)

Cluster the vector space, build inverted index over clusters.

### IVF-Flat

**Build**:
1. Cluster vectors into $C$ clusters (k-means)
2. Assign each vector to nearest cluster centroid
3. Store cluster → vector mapping

**Query**:
1. Find $n_{probe}$ nearest cluster centroids to query
2. Exhaustively search vectors in those clusters
3. Return top-k

**Parameters**:
- $C$: Number of clusters (typically $\sqrt{n}$ to $n/1000$)
- $n_{probe}$: Clusters to search (typically 1-100). More $\Rightarrow$ higher recall, slower.

**Complexity**:
- **Cluster search**: $O(C \cdot d)$
- **Candidate search**: $O((n/C) \cdot n_{probe} \cdot d)$
- **Total**: Much faster than exhaustive $O(n \cdot d)$ when $C \gg n_{probe}$

**Pros**:
- Simple, intuitive
- Easy to update (re-cluster periodically)
- Good for large $n$, moderate $d$

**Cons**:
- Clustering is expensive ($O(n \cdot C \cdot \text{iters} \cdot d)$)
- Sensitive to cluster quality
- Slower than HNSW at high recall

## Product Quantization (PQ)

Compress vectors aggressively to fit more in memory and compute distances faster.

### Algorithm

**Idea**: Split $d$-dimensional vector into $m$ subvectors of $d/m$ dimensions, quantize each subspace independently.

**Build**:
1. Split vector space into $m$ subspaces: $v = [v^1, v^2, \ldots, v^m]$
2. For each subspace, k-means cluster into $k$ centroids (typically $k=256$ for 8-bit codes)
3. Replace each subvector with its cluster ID: $v \rightarrow [c_1, c_2, \ldots, c_m]$
4. Store codes (just $m$ bytes instead of $d \cdot 4$ bytes for float32)

**Query**:
1. Compute distance from $q$ to all $k$ centroids in each subspace (precompute lookup table)
2. For each vector code $[c_1, \ldots, c_m]$, approximate distance:
   $$
   d(q, v) \approx \sum_{i=1}^{m} d(q^i, \text{centroid}_{c_i}^i)
   $$
3. Rank by approximate distance

**Compression ratio**: From $d \cdot 4$ bytes to $m$ bytes. Example: 768-dim float32 (3072 bytes) → 96-byte code with $m=96, k=256$.

**Accuracy**: Approximation error is typically 5-10%. Most vectors' distances are within 10% of true value.

**Pros**:
- Massive compression (10-50x)
- Fast distance computation (lookup table is tiny, fits in L1 cache)
- Can index billions of vectors on single machine

**Cons**:
- Quality loss from quantization
- Requires training (k-means on subspaces)
- Slow for very high recall (need to scan many codes)

### IVF-PQ (Inverted File + Product Quantization)

Combine IVF and PQ: cluster with IVF, compress residuals with PQ.

**Why**: IVF reduces search space, PQ compresses remaining vectors.

**Build**:
1. Cluster into $C$ clusters (IVF)
2. For each vector $v$ in cluster $c$:
   - Compute residual: $r = v - \text{centroid}_c$
   - PQ-encode residual
3. Store cluster → compressed residuals

**Query**:
1. Find $n_{probe}$ nearest centroids
2. Scan compressed vectors in those clusters using PQ distance
3. (Optional) Re-rank top candidates with exact distance

**Result**: Best of both worlds. Handles billions of vectors at 95%+ recall with <100ms latency.

**Used by**: FAISS (Meta's library), Milvus, Weaviate (as an option).

### Optimized Product Quantization (OPQ)

Standard PQ quantizes axis-aligned subspaces, which may not align with data structure. OPQ learns a rotation matrix to align subspaces with principal directions.

**Improvement**: 5-15% better recall than PQ at same compression.

## Scalar Quantization (SQ)

Simpler compression: quantize each dimension independently to int8.

**Algorithm**:
```python
min_val, max_val = v.min(), v.max()
v_quantized = ((v - min_val) / (max_val - min_val) * 255).astype(uint8)
```

**Compression**: 4x (float32 → uint8)

**Pros**: Trivial to implement, no training, good for high-precision needs

**Cons**: Less compression than PQ, still requires significant memory

## Practical Comparison (2026 Benchmarks)

On 1M vectors (768-dim), 10K queries, target recall@10 = 95%:

| Method | QPS | Memory | Build Time | Recall@10 |
|---|---|---|---|---|
| Exhaustive | 2 | 3 GB | instant | 100% |
| LSH | 800 | 12 GB | 5 min | 92% |
| Annoy (50 trees) | 1200 | 4 GB | 10 min | 94% |
| IVF-Flat (4096 clusters, nprobe=32) | 2000 | 3 GB | 20 min | 95% |
| **HNSW (M=16, ef=100)** | **5000** | **5 GB** | **60 min** | **97%** |
| IVF-PQ (4096 clusters, m=96) | 3000 | 0.5 GB | 30 min | 94% |

**Takeaway**: HNSW is the default choice for most use cases. Use IVF-PQ when memory is the bottleneck.

## Choosing an ANN Algorithm

### Decision Tree

1. **Need highest recall (>98%)?** → HNSW with high $ef_{search}$
2. **Memory constrained?** → IVF-PQ or scalar quantization
3. **Billion-scale?** → IVF-HNSW hybrid (cluster with IVF, HNSW within clusters)
4. **Streaming updates?** → HNSW (supports incremental inserts)
5. **Read-only, Python?** → Annoy (simple, good bindings)
6. **Don't know?** → Start with HNSW

### Library Recommendations

| Library | Best For | Algorithms |
|---|---|---|
| **FAISS** (Meta) | Billion-scale, research | IVF-Flat, IVF-PQ, HNSW, exhaustive |
| **hnswlib** | Simple integration, fast | HNSW only |
| **Annoy** (Spotify) | Lightweight, mmapped | Random projection trees |
| **ScaNN** (Google) | High accuracy, TPU | Learned quantization |
| **NGT** (Yahoo Japan) | Balanced | ANNG, ONNG (graph methods) |

**Production databases** (built-in ANN):
- **Pinecone**: HNSW + filtering
- **Weaviate**: HNSW
- **Qdrant**: HNSW + quantization
- **Milvus**: IVF-PQ, IVF-HNSW, HNSW
- **Elasticsearch**: HNSW (since 8.0)

## Hybrid Filtering: Metadata + Vectors

Real-world queries often combine vector similarity with filters: "Find similar products under $50 in Electronics category."

### Approaches

**1. Post-filtering**: Retrieve top-K vectors, then filter.
- **Pro**: Fast retrieval
- **Con**: May return <k results after filtering (poor recall)

**2. Pre-filtering**: Filter first, then search filtered subset.
- **Pro**: Exact filtering
- **Con**: Breaks ANN index (searches small subset inefficiently)

**3. Filtering-aware indexes**: Build ANN index with filter support.
- HNSW-F (Qdrant): Augment graph edges with filter metadata
- IVF + inverted index: Combine vector clusters with keyword index

**Best practice (2026)**: Use filtering-aware vector databases (Pinecone, Weaviate, Qdrant). They handle this complexity internally.

## Building and Updating Indexes

### Offline Batch Indexing

**When**: Initial index build, full reindex

**Process**:
1. Encode documents → embeddings (GPU, batched)
2. Build ANN index (HNSW, IVF-PQ)
3. Swap old index with new (blue-green deployment)

**Time**: 1-10 hours for 10M-100M vectors on modern hardware.

### Incremental Updates

**HNSW**: Supports inserts in $O(\log n)$ time. Just add new vector, connect to neighbors.

**IVF**: Assign to nearest cluster, append to cluster's list. Periodically re-cluster (every 10-30 days).

**PQ**: Expensive — requires re-training codebook if data distribution shifts. Usually rebuild from scratch.

### Deletions

**Lazy deletion**: Mark as deleted, skip during query. Compact periodically.

**HNSW**: Can cause graph degradation (holes in graph). Rebuild if delete rate is high.

## Benchmarking Your Index

**Tools**:
- **ann-benchmarks.com**: Standard benchmark suite with 20+ datasets
- **BEIR**: Evaluate ANN on real retrieval tasks

**Metrics to track**:
- Recall@k (k=10, 100)
- QPS at target recall
- P99 latency
- Memory usage
- Build time

**How to tune**:
1. Plot recall vs. QPS for different parameter settings
2. Find the knee of the curve (diminishing returns)
3. Choose parameters just past the knee

## Summary

| Algorithm | Best Use Case | Recall | Speed | Memory |
|---|---|---|---|---|
| **HNSW** | General-purpose, high recall | Excellent | Very fast | High |
| **IVF-PQ** | Billion-scale, memory-constrained | Good | Fast | Low |
| **IVF-Flat** | Moderate scale, high recall | Excellent | Fast | Medium |
| **Annoy** | Simple integration, read-only | Good | Fast | Medium |
| **LSH** | Streaming, theoretical guarantees | Fair | Moderate | High |

**Key takeaway**: HNSW is the state-of-the-art for 95%+ of use cases. It's fast, accurate, and robust. Use IVF-PQ if memory is tight. Everything else is specialized. Understanding the trade-offs lets you make informed decisions when quality, cost, or latency constraints push you away from the default.

## References and Further Reading

- [Malkov & Yashunin (2018): Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs](https://arxiv.org/abs/1603.09320) — HNSW
- [Jégou et al. (2011): Product Quantization for Nearest Neighbor Search](https://ieeexplore.ieee.org/document/5432202) — PQ
- [Johnson et al. (2019): Billion-scale similarity search with GPUs](https://arxiv.org/abs/1702.08734) — FAISS library
- [Bernhardsson (2018): Annoy: Approximate Nearest Neighbors in C++/Python](https://github.com/spotify/annoy) — Annoy documentation
- [ANN-Benchmarks](http://ann-benchmarks.com) — Comprehensive ANN algorithm comparison
