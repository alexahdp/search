# Sentence and Document Embeddings

While word embeddings capture lexical semantics, search systems need to represent entire documents and queries as single vectors. Simply averaging word embeddings loses word order, syntactic structure, and long-range dependencies. This section covers how contextualized models (BERT) and specially trained sentence encoders (Sentence-BERT) solve this problem.

## The Problem with Averaging Word Embeddings

### Why Simple Averaging Fails

Given word embeddings $\vec{w}_1, \ldots, \vec{w}_n$, the simplest sentence embedding is:

$$
\vec{s} = \frac{1}{n} \sum_{i=1}^{n} \vec{w}_i
$$

**Problems**:
1. **No word order**: "The dog bit the man" and "The man bit the dog" have identical embeddings.
2. **No compositionality**: "not good" should be close to "bad", but averaging gives something near "good".
3. **Polysemy unresolved**: "bank" still has one vector regardless of context.
4. **Information loss**: A 300-dim average of 50 words compresses 15,000 numbers into 300.

Despite these flaws, averaging Word2Vec/GloVe is a surprisingly strong baseline (often beats more complex methods on small datasets). But it's nowhere near state-of-the-art.

## BERT and Contextualized Embeddings

BERT (Devlin et al., 2018) revolutionized NLP by producing context-dependent word embeddings through a deep bidirectional Transformer.

### Architecture Overview

- **Input**: Tokenized text (WordPiece tokens, ~30K vocabulary)
- **Encoder**: 12 (base) or 24 (large) Transformer layers
- **Output**: Contextualized embedding for each token (768-dim for base, 1024-dim for large)

Key insight: The embedding for "bank" in "river bank" is different from "bank" in "savings bank" because each token's representation is computed from the entire surrounding context.

### Pre-training Objectives

1. **Masked Language Modeling (MLM)**: Randomly mask 15% of tokens, predict them from context.
   - Teaches the model to use bidirectional context.

2. **Next Sentence Prediction (NSP)**: Given two sentences A and B, predict if B follows A.
   - Teaches sentence-level relationships (later dropped in RoBERTa as unnecessary).

### Pooling Strategies for Sentence Embeddings

BERT outputs one vector per token. How do we get a single sentence vector?

**1. [CLS] token pooling**
```
BERT's [CLS] token (first position) is intended as a sentence representation.
```
- **Pro**: Simple, one line of code.
- **Con**: [CLS] is optimized for classification tasks (via NSP), not semantic similarity. Poor performance for retrieval.

**2. Mean pooling** (average all token embeddings)
```python
sentence_embedding = torch.mean(token_embeddings, dim=1)
```
- **Pro**: Uses information from all tokens. Better than [CLS] empirically.
- **Con**: Still treats all tokens equally (should "the" contribute as much as "cardiovascular"?).

**3. Max pooling** (element-wise max across tokens)
```python
sentence_embedding = torch.max(token_embeddings, dim=1)[0]
```
- **Pro**: Emphasizes most important features.
- **Con**: Loses information, unstable.

**4. Weighted pooling** (attention-weighted average)
```python
attention_weights = F.softmax(attention_scores, dim=1)
sentence_embedding = torch.sum(attention_weights * token_embeddings, dim=1)
```
- **Pro**: Learns which tokens are important.
- **Con**: Adds parameters, needs training.

**Empirical result**: Mean pooling is the most robust general-purpose choice. This was confirmed in the Sentence-BERT paper.

### Why Raw BERT Is Bad for Semantic Search

If we pool BERT embeddings and compute cosine similarity, performance is **worse than GloVe averaging**. Why?

1. **Not trained for similarity**: BERT was trained with MLM and NSP, not to produce embeddings where similar sentences are close.

2. **High anisotropy**: BERT embeddings occupy a narrow cone in the embedding space. All sentences are relatively similar to each other, leading to high cosine similarities (0.6-0.9) even for unrelated sentences.

3. **Computational cost**: Encoding each query-document pair through BERT for similarity takes ~1-5 seconds per pair. Not viable for ranking 1000 candidates.

**Solution**: Fine-tune BERT specifically for producing sentence embeddings. This is Sentence-BERT.

## Sentence-BERT (SBERT)

Sentence-BERT (Reimers & Gurevych, 2019) fine-tunes BERT with a Siamese/Triplet architecture to produce semantically meaningful sentence embeddings that can be compared with cosine similarity.

### Siamese Network Architecture

**Training setup**:
1. Two sentences $s_1, s_2$ and a label (similar/dissimilar or score 0-5)
2. Encode both through BERT (shared weights) with mean pooling → $\vec{u}, \vec{v}$
3. Concatenate $[\vec{u}, \vec{v}, |\vec{u} - \vec{v}|]$ (element-wise absolute difference)
4. Feed through a softmax classifier for similarity class

**Loss**: Cross-entropy on similarity labels.

**At inference**: Encode sentences independently, compare with cosine similarity. No need for the classification head.

### Triplet Loss Architecture

**Training setup**:
1. Triplet: anchor $a$, positive $p$ (similar), negative $n$ (dissimilar)
2. Encode all three → $\vec{a}, \vec{p}, \vec{n}$
3. Minimize:

$$
\mathcal{L} = \max(0, ||vec{a} - \vec{p}||^2 - ||\vec{a} - \vec{n}||^2 + \epsilon)
$$

Where $\epsilon$ is the margin (typically 0.5-1.0).

**Goal**: Make $a$ closer to $p$ than to $n$ by at least $\epsilon$.

### Contrastive Learning (Modern Approach)

Most recent SBERT models use **contrastive loss** (InfoNCE):

$$
\mathcal{L} = -\log \frac{\exp(\text{sim}(a, p) / \tau)}{\sum_{n \in N} \exp(\text{sim}(a, n) / \tau)}
$$

Where $\tau$ is a temperature parameter and $N$ is a batch of negatives (typically 64-256).

**Key idea**: Pull positive pairs together, push negatives apart, all in one loss.

**Why this works**: With large batch sizes and in-batch negatives, the model sees many hard negatives per update, leading to better discrimination.

### Training Data

**Natural Language Inference (NLI)**: Pairs labeled as entailment/contradiction/neutral.
- **SNLI**: 570K sentence pairs
- **Multi-NLI**: 430K sentence pairs
- Provides strong signal for semantic similarity

**Semantic Textual Similarity (STS)**: Sentence pairs with continuous similarity scores (0-5).
- **STS Benchmark**: 8.6K pairs with human ratings
- Directly optimizes for the metric we care about

**Best practice** (as of 2026): Pre-train on NLI (large dataset), fine-tune on STS (task-specific).

### Performance Gains

On semantic textual similarity benchmarks:

| Model | STS (Spearman) |
|---|---|
| GloVe averaging | 58.0 |
| BERT-[CLS] | 29.2 (!) |
| BERT-mean | 54.7 |
| InferSent | 68.0 |
| Universal Sentence Encoder | 74.9 |
| **Sentence-BERT (base)** | **84.9** |
| **Sentence-BERT (large)** | **86.5** |

SBERT also reduces the computation for finding the most similar sentence in 10K candidates from **65 hours (BERT cross-encoder)** to **5 seconds (SBERT bi-encoder)**.

## Asymmetric Search: Query vs. Document Embeddings

Queries and documents have different characteristics:
- **Queries**: Short (2-10 words), conversational, questions, often underspecified
- **Documents**: Long (50-500 words), complete sentences, declarative, information-rich

Using the same encoder for both is suboptimal.

### Asymmetric Models

Train two separate encoders:
- $f_q(\cdot)$: Query encoder (smaller, faster)
- $f_d(\cdot)$: Document encoder (larger, more capacity)

**Loss**: Contrastive loss on $(q, d^+)$ positive pairs and $(q, d^-)$ negative pairs.

**Example**: Dense Passage Retrieval (DPR) uses this approach with:
- $f_q$: BERT-base
- $f_d$: BERT-base (same size, but weights diverge during training)

**When to use**: When query/document length or style differ significantly (QA, passage retrieval). For symmetric tasks (sentence similarity, duplicate detection), symmetric encoders work fine.

## Multi-Lingual Sentence Embeddings

Early sentence encoders were English-only. Modern models support 50-100 languages.

### Multilingual Sentence-BERT (mSBERT)

Fine-tune multilingual BERT (mBERT) or XLM-RoBERTa with translation pairs:
- $(s_{en}, s_{fr})$ where $s_{fr}$ is a translation of $s_{en}$
- Loss: Make $f(s_{en})$ close to $f(s_{fr})$

**Result**: Embeddings are language-agnostic. $\cos(f(\text{"cat"}_{en}), f(\text{"chat"}_{fr})) \approx 0.9$.

**Use case**: Cross-lingual search (query in English, retrieve documents in French/German/Japanese).

### Best Models (2026)

- **multilingual-e5-large**: 560M params, 94 languages, state-of-the-art on multilingual retrieval
- **paraphrase-multilingual-mpnet-base-v2**: 278M params, 50+ languages, balanced quality/speed

## Practical Considerations

### Embedding Dimension

| Model | Dimension | Use Case |
|---|---|---|
| all-MiniLM-L6-v2 | 384 | Fast, general-purpose |
| all-mpnet-base-v2 | 768 | High quality, moderate speed |
| e5-large-v2 | 1024 | Highest quality, slower |
| gte-large | 1024 | Competitive with e5 |

**Trade-off**: Higher dimensions improve quality but increase index size and query latency (more dimensions to compare in ANN).

**Recommendation**: Start with 384-dim (MiniLM). If quality is insufficient, move to 768-dim (MPNet). Only use 1024-dim if you have the infrastructure and quality demands justify it.

### Normalization

Always L2-normalize embeddings before indexing:

```python
embedding = embedding / np.linalg.norm(embedding)
```

**Why**: Makes cosine similarity equivalent to dot product ($\cos(u, v) = u \cdot v$ when $||u|| = ||v|| = 1$), which is faster to compute. Most ANN libraries (FAISS, HNSW) optimize for dot product.

### Batching for Throughput

Encoding sentences one at a time is inefficient. Batch on GPU:

```python
# Bad: 1.2 seconds for 100 sentences
for sentence in sentences:
    embedding = model.encode(sentence)

# Good: 0.15 seconds for 100 sentences
embeddings = model.encode(sentences, batch_size=32)
```

**Batch size**: 16-64 for most GPUs. Larger batches improve throughput but increase memory.

### Model Selection

| Model | Quality | Speed | When to Use |
|---|---|---|---|
| all-MiniLM-L6-v2 | Good | Very fast | High QPS, cost-sensitive |
| all-mpnet-base-v2 | Better | Fast | General-purpose production |
| e5-large-v2 | Best | Moderate | Quality-critical applications |
| multilingual-e5-large | Best (multilingual) | Moderate | Cross-lingual search |

**How to choose**:
1. Start with `all-mpnet-base-v2` (768-dim).
2. If quality is insufficient, move to `e5-large-v2`.
3. If cost/latency is prohibitive, move to `all-MiniLM-L6-v2`.

### Domain Adaptation

Pre-trained models work well on general text but may underperform on specialized domains (medical, legal, scientific code).

**Fine-tuning recipe**:
1. Collect domain-specific positive pairs: $(q, d^+)$
   - Example: (Stack Overflow question title, accepted answer)
2. Sample negatives: $d^-$ from the same corpus (random or BM25 hard negatives)
3. Fine-tune with contrastive loss for 1-3 epochs
4. Evaluate on held-out domain data

**Data requirements**: 10K-100K pairs for meaningful improvement. Below 10K, pre-trained models are often better (less overfitting risk).

## Limitations

1. **Context window**: Most models max out at 512 tokens (~400 words). Long documents must be chunked or truncated.
   - **Solution**: Split documents into passages (150-250 words), embed separately, aggregate scores.

2. **Inference cost**: Encoding large document collections is expensive.
   - **Solution**: Offline batch encoding on GPU, cache embeddings.

3. **Training data bias**: Models trained on NLI/STS may not generalize to all domains.
   - **Solution**: Fine-tune on domain data or use instruction-tuned models (e5-instruct).

4. **No keyword anchoring**: Dense models can miss exact term matches.
   - **Solution**: Use hybrid search (next section).

## Encoding Documents: Chunking Strategies

Documents longer than 512 tokens require chunking. Two approaches:

### 1. Embed Chunks Independently

Split document into passages of ~200 words:
```python
chunks = split_document(doc, chunk_size=200, overlap=50)
embeddings = [model.encode(chunk) for chunk in chunks]
```

**At query time**: Retrieve top-K chunks, then aggregate by document (max score, sum, or count).

**Pros**: Simple, parallelizable, works with standard ANN indexes.
**Cons**: Loses cross-chunk context.

### 2. Hierarchical Encoding

Embed chunks, then embed the concatenation of chunk embeddings:
```python
chunk_embeds = [model.encode(chunk) for chunk in chunks]
doc_embed = model.encode(f"Document: {' '.join(chunks[:3])}")  # or sum/pool chunk_embeds
```

**Pros**: Captures document-level context.
**Cons**: More complex, requires careful tuning.

**Practical recommendation**: Start with independent chunks. It's simpler and performs well.

## Evaluation

### Intrinsic: STS Benchmarks

- **STS-B** (English): 1.5K sentence pairs with human similarity scores
- **SICK** (English): 10K sentence pairs
- **PAWS** (English): Paraphrase detection

Metric: Spearman correlation between model similarity and human judgments.

### Extrinsic: Retrieval Benchmarks

- **MS MARCO**: 1M passages, 500K queries, sparse relevance labels
- **BEIR**: 18 diverse retrieval datasets (out-of-domain evaluation)
- **MTEB**: 58 tasks across 112 languages (comprehensive benchmark)

Metrics: NDCG@10, MAP, Recall@100.

**Key insight**: High STS correlation doesn't guarantee good retrieval. Always evaluate on retrieval tasks.

## Summary

| Approach | Quality | Speed | When to Use |
|---|---|---|---|
| GloVe averaging | Baseline | Instant | Simple baseline, debugging |
| BERT-mean (no fine-tuning) | Poor | Slow | Don't use for retrieval |
| Sentence-BERT | Strong | Fast (cached) | General-purpose semantic search |
| DPR (asymmetric) | Strong | Fast (cached) | QA, passage retrieval |
| Multilingual SBERT | Strong (multilingual) | Fast (cached) | Cross-lingual search |

**Key takeaway**: Sentence-BERT and its variants are the standard for dense retrieval in 2026. They provide a massive quality improvement over averaging word embeddings, with reasonable computational costs when embeddings are pre-computed. The next challenge is scaling similarity search to billions of vectors — that's where ANN algorithms (next section) come in.

## References and Further Reading

- [Devlin et al. (2018): BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) — Original BERT paper
- [Reimers & Gurevych (2019): Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084) — Sentence-BERT
- [Karpukhin et al. (2020): Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906) — DPR (asymmetric encoders)
- [Wang et al. (2022): Text Embeddings by Weakly-Supervised Contrastive Pre-training](https://arxiv.org/abs/2212.03533) — E5 embeddings
- [Muennighoff et al. (2022): MTEB: Massive Text Embedding Benchmark](https://arxiv.org/abs/2210.07316) — Comprehensive evaluation framework
