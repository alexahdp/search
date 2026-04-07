# Cross-Encoder Re-Ranking

Cross-encoders represent the highest-accuracy approach to semantic similarity but are too slow for first-stage retrieval. The solution is two-stage retrieval: fast retrieval with bi-encoders or BM25, followed by precise re-ranking with cross-encoders. This pattern is ubiquitous in production search systems and is essential for maximizing relevance.

## Bi-Encoder vs. Cross-Encoder

### Bi-Encoder (Dual Encoder)

**Architecture**: Separate encoders for query and document.

```
Query: "how to train neural networks"
  ↓ encode_query(q)
q_vec = [0.12, -0.34, 0.56, ..., 0.21]  (768-dim)

Document: "Neural network training guide..."
  ↓ encode_doc(d)
d_vec = [0.15, -0.30, 0.60, ..., 0.18]  (768-dim)

Similarity: cos(q_vec, d_vec) = 0.87
```

**Key property**: Query and document encoded independently.

**Pros**:
- **Pre-compute**: Encode all documents offline, store in ANN index
- **Fast query**: Encode query once (~10ms), ANN search (~5ms)
- **Scales to billions**: ANN algorithms (HNSW, IVF) make this feasible

**Cons**:
- **Fixed representations**: Once encoded, document vector is frozen. Can't attend to specific query terms.
- **Information loss**: Compressing 200 words into 768 numbers loses signal.

### Cross-Encoder

**Architecture**: Joint encoder for query+document pair.

```
Input: "[CLS] how to train neural networks [SEP] Neural network training guide... [SEP]"
  ↓ BERT (cross-attention between query and document)
Hidden states: [h_CLS, h_how, h_to, ..., h_guide, ...]
  ↓ Classifier head
Relevance score: 0.92
```

**Key property**: Query and document processed together. Every query token attends to every document token.

**Pros**:
- **Highest accuracy**: Cross-attention captures fine-grained interactions.
- **No information loss**: Full context available during scoring.

**Cons**:
- **Cannot pre-compute**: Must encode (query, doc) pair together at query time.
- **Slow**: For 1000 candidates, must run BERT 1000 times (~10-50ms each = 10-50 seconds total).
- **Doesn't scale**: Cannot use ANN. Must score every candidate.

## The Two-Stage Retrieval Paradigm

**Stage 1 (Retrieval)**: Fast, high recall, moderate precision.
- **Goal**: Narrow 10M documents to top-100 candidates.
- **Methods**: BM25, bi-encoder ANN, hybrid.
- **Latency**: 5-20ms.
- **Recall**: Target 95-98%.

**Stage 2 (Re-ranking)**: Slow, high precision.
- **Goal**: Precisely rank top-100 candidates, return top-10.
- **Methods**: Cross-encoder, late interaction (ColBERT).
- **Latency**: 50-200ms.
- **Precision**: Maximize NDCG@10.

**Why this works**:
- Retrieval casts a wide net (high recall).
- Re-ranking focuses compute on likely relevant docs (high precision).
- Combined: best of both worlds.

**Analogy**: Retrieval is a coarse filter (like searching a file system by keywords). Re-ranking is a fine filter (like reading each file carefully).

## Cross-Encoder Architecture

### Input Representation

BERT-style input:
```
[CLS] <query tokens> [SEP] <document tokens> [SEP]
```

**Token limit**: 512 tokens (BERT base/large). If query + document exceeds 512, truncate document.

**Example**:
- Query: "best laptop for programming" (5 tokens)
- Document: 500-word article (truncated to 507 tokens to fit)

### Training Objective

**Pointwise**: Classify (query, doc) pair as relevant (1) or not (0).

Loss: Binary cross-entropy.
```python
score = cross_encoder(query, doc)  # sigmoid output [0, 1]
loss = -[y * log(score) + (1 - y) * log(1 - score)]
```

**Pairwise**: Given (query, doc+, doc-), ensure score(doc+) > score(doc-).

Loss: Margin ranking loss or hinge loss.
```python
score_pos = cross_encoder(query, doc_positive)
score_neg = cross_encoder(query, doc_negative)
loss = max(0, margin - (score_pos - score_neg))
```

**Listwise**: Given (query, [doc1, doc2, ..., docN]) with relevance labels, optimize ranking.

Loss: ListNet, LambdaRank (softmax cross-entropy on permutations).

**Best practice (2026)**: Pairwise or listwise with hard negatives (BM25 top-100 as negatives).

### Training Data

**Sources**:
1. **MS MARCO**: 500K queries with sparse relevance labels.
2. **NLI** (Natural Language Inference): (premise, hypothesis, label) → use as (query, doc, label).
3. **Click logs**: (query, clicked_doc) as positives.
4. **Synthetic**: Generate (question, passage) pairs from Wikipedia using QG models.

**Hard negative mining**: Sample negatives from BM25/dense retrieval top-100. Random negatives are too easy.

```python
# Bad: random negatives (model learns trivial patterns)
negatives = random.sample(all_docs, k=10)

# Good: hard negatives (model learns nuanced distinctions)
negatives = bm25_top_100 - true_positives
```

**Data requirements**: 10K-100K labeled (query, doc, relevance) triples for fine-tuning.

## Performance Gains

### MS MARCO Passage Ranking

| Model | MRR@10 | Latency (per query) |
|---|---|---|
| BM25 (stage 1) | 18.7 | 5ms |
| Bi-encoder (stage 1) | 37.9 | 15ms |
| **Cross-encoder (stage 2, top-100)** | **42.5** | **120ms** |
| **Cross-encoder (stage 2, top-1000)** | **43.8** | **800ms** |

**Takeaway**: Cross-encoder improves bi-encoder by +12% (37.9 → 42.5), but adds 105ms latency. Re-ranking top-1000 adds more quality but impractical latency.

### BEIR (Out-of-Domain)

| Model | NDCG@10 | Notes |
|---|---|---|
| BM25 (stage 1) | 42.0 | Baseline |
| Bi-encoder (stage 1) | 50.2 | E5-large |
| **Cross-encoder (stage 2)** | **54.1** | Re-rank top-100 |

**Improvement**: +8% NDCG@10 over bi-encoder alone.

### Real-World Impact (2026 Study)

MIT researchers evaluated two-stage retrieval on 8 production datasets:
- **Average quality improvement**: +33% NDCG@10
- **Average latency increase**: +120ms
- **ROI**: For most applications, the quality gain justifies the latency cost.

**Recommendation**: If NDCG@10 is critical (e.g., first search result page), use cross-encoder re-ranking.

## Practical Implementation

### Pattern 1: Sequential Re-Ranking

```python
def search_with_reranking(query, k=10):
    # Stage 1: Retrieve candidates
    candidates = hybrid_search(query, k=100)  # BM25 + dense

    # Stage 2: Re-rank with cross-encoder
    pairs = [(query, doc) for doc_id, doc in candidates]
    cross_encoder_scores = cross_encoder.predict(pairs)  # batch inference

    # Combine
    reranked = sorted(zip(candidates, cross_encoder_scores),
                      key=lambda x: -x[1])
    return reranked[:k]
```

**Latency breakdown**:
- Stage 1 (hybrid): 20ms
- Encode 100 pairs (batch size 32, GPU): 80ms
- Total: 100ms

**Optimization**: Batch inference is critical. Encoding one pair at a time takes 1-5ms each (100-500ms total).

### Pattern 2: Cascade Re-Ranking

For very large candidate sets, re-rank in multiple stages.

```python
def cascade_reranking(query, k=10):
    # Stage 1: Retrieve 10K candidates (fast, high recall)
    candidates = hybrid_search(query, k=10000)  # 30ms

    # Stage 2: Re-rank to 1000 with fast cross-encoder (50M params)
    scores_1 = fast_cross_encoder.predict(candidates[:10000])  # 200ms
    top_1000 = top_k(scores_1, k=1000)

    # Stage 3: Re-rank to 10 with accurate cross-encoder (110M params)
    scores_2 = accurate_cross_encoder.predict(top_1000)  # 100ms
    top_10 = top_k(scores_2, k=10)

    return top_10
```

**Total latency**: 330ms (high, but acceptable for quality-critical apps).

**When to use**: Search engines, recommendation systems where first page quality is paramount.

### Pattern 3: Distillation (Faster Re-Ranking)

Train a smaller, faster model to mimic the cross-encoder.

**Process**:
1. Score 100K (query, doc) pairs with large cross-encoder (offline, slow).
2. Train a smaller model (6-layer BERT or MiniLM) to match those scores.
3. Use small model for production re-ranking.

**Result**: 3-5x faster with 90-95% of quality.

**Example**: DistilBERT (66M params) vs. BERT-large (340M params) — 2x faster, 97% quality.

### Pattern 4: Hybrid Re-Ranking

Combine cross-encoder with other signals.

```python
def hybrid_rerank(query, candidates):
    cross_scores = cross_encoder.predict(candidates)
    bm25_scores = [bm25(query, doc) for doc in candidates]
    click_scores = [get_click_score(query, doc) for doc in candidates]  # historical CTR

    # Weighted combination
    final_scores = (0.6 * normalize(cross_scores) +
                    0.2 * normalize(bm25_scores) +
                    0.2 * normalize(click_scores))
    return final_scores
```

**Pros**: Incorporates multiple relevance signals (semantic, lexical, behavioral).

**Cons**: More complex, requires tuning weights.

## Latency Optimization

### 1. Batch Inference

**Bad** (serial):
```python
scores = [cross_encoder.predict([query, doc]) for doc in candidates]
# 100 docs × 5ms = 500ms
```

**Good** (batched):
```python
pairs = [[query, doc] for doc in candidates]
scores = cross_encoder.predict(pairs, batch_size=32)
# 100 docs / 32 per batch × 20ms = 60ms
```

**Speedup**: 8x faster.

### 2. ONNX or TensorRT

Export model to optimized inference engine.

**PyTorch** (default):
```python
model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
scores = model.predict(pairs)  # 80ms for 100 pairs
```

**ONNX Runtime**:
```python
model_onnx = export_to_onnx(model)
scores = onnx_runtime.run(model_onnx, pairs)  # 40ms for 100 pairs
```

**Speedup**: 2x faster (CPU), 1.3x faster (GPU).

### 3. Quantization

Reduce model precision from float32 to int8.

**Before**: 110M params × 4 bytes = 440 MB
**After (int8)**: 110M params × 1 byte = 110 MB

**Quality loss**: <1% NDCG@10 drop (negligible).

**Speedup**: 1.5-2x faster inference.

### 4. Model Pruning

Remove less important neurons/layers.

**Example**: Prune 30% of attention heads → 1.4x faster, 1% quality drop.

**Libraries**: Hugging Face Optimum, Intel Neural Compressor.

### 5. Early Exit

Stop processing after fewer layers if confidence is high.

**Idea**: For obvious matches/non-matches, don't need full 12 layers.

**Implementation**: Add classifier heads at layers 4, 8, 12. If confidence > threshold at layer 4, return early.

**Result**: 1.5-2x faster on average, <1% quality loss.

**When to use**: When candidate list has many obvious non-matches (e.g., re-ranking top-1000).

## Model Selection

### Pre-Trained Cross-Encoders (2026)

| Model | Params | Latency (100 docs) | NDCG@10 (MS MARCO) | Use Case |
|---|---|---|---|---|
| ms-marco-MiniLM-L-6-v2 | 22M | 40ms | 38.5 | Fast, general-purpose |
| ms-marco-MiniLM-L-12-v2 | 33M | 60ms | 39.8 | Balanced speed/quality |
| ms-marco-electra-base | 110M | 100ms | 40.5 | Higher quality |
| monot5-base | 220M | 200ms | 41.2 | Highest quality, slow |
| cross-encoder/stsb-roberta-large | 355M | 300ms | 42.0 | Research/offline |

**Recommendation**: Start with `ms-marco-MiniLM-L-6-v2`. If quality is insufficient, move to L-12 or electra-base.

### Domain Adaptation

Pre-trained models work well on general text. For specialized domains (medical, legal, code), fine-tuning helps.

**Recipe**:
1. Collect domain-specific (query, doc, relevance) data (10K-100K samples).
   - Example: Stack Overflow (question, answer, upvotes).
2. Fine-tune pre-trained cross-encoder for 1-3 epochs.
3. Evaluate on held-out domain data.

**Expected improvement**: +5-15% NDCG@10 on domain-specific queries.

**Data sources**:
- Click logs (query, clicked_doc) as positives.
- BM25/dense top-100 as hard negatives.
- Human relevance judgments (expensive but highest quality).

## When NOT to Use Cross-Encoders

### 1. Latency-Critical Applications

**Example**: Autocomplete (must respond in <50ms).

**Solution**: Use only bi-encoder or BM25. Cross-encoder too slow.

### 2. Very Large Candidate Sets

**Example**: Re-ranking top-10K documents.

**Cost**: 10K docs × 5ms = 50 seconds (unacceptable).

**Solution**: Use cascade (bi-encoder → 1K, cross-encoder → 100, cross-encoder-large → 10).

### 3. Streaming / Real-Time Indexing

**Problem**: Cross-encoders require query at encoding time. Can't pre-compute.

**Solution**: Use bi-encoders for real-time scenarios (encode new docs once, index immediately).

### 4. Cost-Sensitive Deployments

**Compute cost**: Cross-encoder is 10-50x more expensive than bi-encoder per query.

**Example**: 10M queries/day with cross-encoder re-ranking = ~$1000/day GPU cost vs. $50/day for bi-encoder only.

**Solution**: Use cross-encoder only for premium features or high-value queries.

## Comparing Re-Ranking Approaches

| Method | Quality (NDCG@10) | Latency (100 docs) | Memory | Interpretability |
|---|---|---|---|---|
| **Bi-encoder** | Baseline | 5ms | Low | Low |
| **Cross-encoder** | +30-40% | 100ms | Medium | Medium (attention weights) |
| **ColBERT** | +20-30% | 50ms | High (multi-vector index) | High (token-level MaxSim) |
| **monoT5** (generative) | +35-45% | 200ms | High | Low |
| **Learned ranking** (LambdaMART) | +15-25% | 10ms | Low | High (feature importance) |

**Summary**:
- **Cross-encoder**: Highest quality, moderate latency, standard choice.
- **ColBERT**: Balanced quality and speed, but complex index.
- **monoT5**: Highest quality, very slow, research use.
- **Learned ranking**: Fast, interpretable, requires feature engineering.

## Multi-Stage Ranking in Production (2026 Best Practices)

### Stage 1: Retrieval (10M → 100)
- **Method**: Hybrid (BM25 + bi-encoder dense) with RRF fusion.
- **Target**: 98% recall@100.
- **Latency**: 20ms.

### Stage 2: Re-Ranking (100 → 20)
- **Method**: Cross-encoder (MiniLM-L-6).
- **Goal**: Boost precision, filter out false positives.
- **Latency**: 40ms.

### Stage 3: Business Logic (20 → 10)
- **Method**: Diversity, freshness, personalization, business rules.
- **Goal**: Final presentation logic (don't show duplicates, boost recent docs, personalize).
- **Latency**: 10ms.

**Total**: 70ms (acceptable for most applications).

## Monitoring and Evaluation

### Offline Metrics

**NDCG@10**: Ranking quality (standard).
**MRR@10**: Mean reciprocal rank (emphasizes first relevant result).
**Recall@10**: Coverage (did we find relevant docs?).

**Measure separately**:
- Stage 1 alone (bi-encoder or BM25).
- Stage 1 + Stage 2 (cross-encoder).

**Expected**: Stage 2 improves NDCG@10 by 20-40% over Stage 1.

### Online Metrics

**Click-through rate (CTR)**: Did users click on top results?
**Success rate**: Did users complete their task (session-level signals)?
**Time to success**: How long until user found what they needed?

**A/B test**: Re-ranking ON vs. OFF.

**Expected**: +10-20% CTR, +5-10% success rate.

### Latency Monitoring

**P50, P95, P99 latency**: Track full query pipeline.

**Budget**: Allocate latency budget across stages.
- Stage 1 (retrieval): 20ms.
- Stage 2 (re-ranking): 60ms.
- Stage 3 (business logic): 10ms.
- **Total**: 90ms (stay under 100ms).

**Alert**: If P99 > 200ms, investigate bottlenecks.

## Summary

| Aspect | Recommendation |
|---|---|
| **Use cross-encoder if** | NDCG@10 is critical, latency budget >80ms, quality justifies cost |
| **Skip cross-encoder if** | Latency <50ms required, cost-sensitive, or bi-encoder already sufficient |
| **Model choice** | Start with ms-marco-MiniLM-L-6-v2 (fast, good quality) |
| **Candidate count** | Re-rank top-100 (sweet spot for quality/latency) |
| **Optimization** | Batch inference (8x speedup), ONNX (2x speedup), quantization (1.5x speedup) |
| **Evaluation** | Measure both offline (NDCG@10) and online (CTR, success rate) |
| **Architecture** | Two-stage: hybrid retrieval → cross-encoder re-ranking |

**Key takeaway**: Cross-encoder re-ranking is the highest-ROI quality improvement for modern search systems. It's standard in production (2026) for any system where ranking quality matters. The two-stage paradigm — fast retrieval + precise re-ranking — balances cost, latency, and quality. Start with a pre-trained MiniLM cross-encoder, measure the lift, and optimize from there.

## References and Further Reading

- [Nogueira & Cho (2019): Passage Re-ranking with BERT](https://arxiv.org/abs/1901.04085) — First BERT cross-encoder for IR
- [Gao et al. (2021): Rethink Training of BERT Rerankers in Multi-Stage Retrieval Pipeline](https://arxiv.org/abs/2101.08751) — Hard negative mining for cross-encoders
- [Nogueira et al. (2020): Document Ranking with a Pretrained Sequence-to-Sequence Model](https://arxiv.org/abs/2003.06713) — monoT5 cross-encoder
- [Reimers & Gurevych (2019): Sentence-BERT](https://arxiv.org/abs/1908.10084) — Bi-encoder baseline
- [Khattab & Zaharia (2020): ColBERT](https://arxiv.org/abs/2004.12832) — Late interaction alternative
- [Hofstätter et al. (2021): Efficiently Teaching an Effective Dense Retriever with Balanced Topic Aware Sampling](https://arxiv.org/abs/2104.06967) — TAS-B distillation
- [MIT Study (2026): Cross-Encoder Reranking Improves RAG Accuracy by 40%](https://app.ailog.fr/en/blog/news/reranking-cross-encoders-study) — Recent empirical validation
