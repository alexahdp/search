# Word Embeddings

Word embeddings are dense vector representations of words that capture semantic relationships through their geometric properties in a continuous vector space. Unlike one-hot encodings or traditional lexical representations, embeddings place semantically similar words close together, enabling algebraic operations on meaning.

## The Core Insight

The distributional hypothesis: "words that occur in similar contexts tend to have similar meanings" (Harris, 1954). If "dog" and "cat" appear in similar sentences ("I walked my ___", "The ___ jumped"), their vectors should be close. This simple idea underpins all modern embedding methods.

## Word2Vec

Word2Vec (Mikolov et al., 2013) introduced two architectures that revolutionized NLP by making embeddings both fast to train and semantically rich.

### Skip-Gram Architecture

**Objective**: Predict context words given a center word.

For a center word $w_c$ and context window of size $k$, maximize:

$$
\mathcal{L} = \sum_{t=1}^{T} \sum_{-k \leq j \leq k, j \neq 0} \log p(w_{t+j} | w_t)
$$

Where the conditional probability is:

$$
p(w_O | w_I) = \frac{\exp(v'_{w_O}{}^T v_{w_I})}{\sum_{w=1}^{W} \exp(v'_w{}^T v_{w_I})}
$$

**Practical implementation**: The denominator requires summing over the entire vocabulary (~100K-1M terms), which is computationally prohibitive. Two solutions emerged:

1. **Hierarchical Softmax**: Uses a binary tree (often Huffman tree) to reduce complexity from $O(V)$ to $O(\log V)$. Each word is a leaf; the probability is the product of binary decisions along the path from root to leaf.

2. **Negative Sampling** (most common): Instead of computing probability across all words, sample $k$ negative examples (typically 5-20) and optimize:

$$
\log \sigma(v'_{w_O}{}^T v_{w_I}) + \sum_{i=1}^{k} \mathbb{E}_{w_i \sim P_n(w)} [\log \sigma(-v'_{w_i}{}^T v_{w_I})]
$$

Where $P_n(w) = \frac{f(w)^{3/4}}{\sum_j f(w_j)^{3/4}}$ (smoothed unigram distribution).

### CBOW (Continuous Bag of Words)

**Objective**: Predict the center word given context words.

CBOW averages context vectors and predicts the center word:

$$
\mathcal{L} = \log p(w_t | w_{t-k}, \ldots, w_{t-1}, w_{t+1}, \ldots, w_{t+k})
$$

**Trade-offs**:
- CBOW is faster to train (one prediction per position vs. $2k$ for Skip-gram)
- Skip-gram performs better on small datasets and rare words
- CBOW performs better on frequent words and larger datasets

### Hyperparameters That Matter

- **Embedding dimension** ($d$): 100-300 is typical. Higher dimensions capture more nuance but risk overfitting on small corpora. Diminishing returns beyond 300.
- **Window size** ($k$): Small windows (2-3) capture syntactic relationships (part-of-speech). Large windows (5-10) capture topical/semantic relationships.
- **Min count**: Ignore words appearing < $n$ times (typically 5). Reduces vocabulary and improves quality.
- **Subsampling**: Downsample frequent words like "the" with probability $1 - \sqrt{t/f(w)}$ where $t \approx 10^{-5}$.

### Semantic Properties

Word2Vec embeddings exhibit remarkable algebraic properties:

- **Analogy**: $\vec{king} - \vec{man} + \vec{woman} \approx \vec{queen}$
- **Similarity**: $\cos(\vec{Paris}, \vec{France})$ is high
- **Composition**: $\vec{New} + \vec{York}$ approximates $\vec{NYC}$

These properties emerge from the training objective without explicit supervision.

## GloVe (Global Vectors)

GloVe (Pennington et al., 2014) takes a different approach: instead of predicting context words, it factorizes a global word-word co-occurrence matrix.

### Algorithm

1. **Build co-occurrence matrix $X$**: $X_{ij}$ = number of times word $j$ appears in the context of word $i$ (across the entire corpus).

2. **Define objective**: Minimize the weighted least squares error:

$$
J = \sum_{i,j=1}^{V} f(X_{ij}) (w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij})^2
$$

Where:
- $w_i$ is the word vector for word $i$
- $\tilde{w}_j$ is the context vector for word $j$
- $b_i, \tilde{b}_j$ are bias terms
- $f(X_{ij})$ is a weighting function:

$$
f(x) = \begin{cases}
(x/x_{max})^\alpha & \text{if } x < x_{max} \\
1 & \text{otherwise}
\end{cases}
$$

Typically $x_{max} = 100, \alpha = 0.75$. This prevents rare co-occurrences from being overweighted.

3. **Optimize** using AdaGrad or stochastic gradient descent.

### Word2Vec vs. GloVe

| Aspect | Word2Vec | GloVe |
|---|---|---|
| **Paradigm** | Local context window (online) | Global co-occurrence matrix (batch) |
| **Complexity** | $O(C)$ per sample ($C$ = corpus size) | $O(\|X\|)$ where $\|X\|$ = non-zero entries in $X$ |
| **Memory** | Small (no global matrix) | Requires storing $X$ (can be sparse) |
| **Training** | Streaming-friendly | Requires full corpus pass to build $X$ |
| **Performance** | Slightly better on rare words | Slightly better on analogies |

**In practice**: Performance is comparable. Word2Vec is easier to train incrementally; GloVe is more principled theoretically. Most practitioners use pre-trained embeddings (GloVe 6B, Word2Vec Google News) rather than training from scratch.

## FastText

FastText (Bojanowski et al., 2017) extends Word2Vec by representing words as bags of character n-grams.

### Subword Representation

Each word $w$ is represented as:

$$
w = \{<w>, \text{ngrams}(w)\}
$$

For example, with $n=3$ to $5$:
- "where" → {&lt;where&gt;, &lt;wh, whe, her, ere, re&gt;, &lt;whe, wher, here, ere&gt;, ...}

The embedding for a word is the sum of its subword embeddings:

$$
\vec{w} = \sum_{g \in \text{ngrams}(w)} \vec{z}_g
$$

### Why This Matters

1. **Morphology**: "unhappiness" shares subwords with "happy", "happiness", "unhappy" — the model learns morphological relationships automatically.

2. **Out-of-vocabulary (OOV)**: Even if "unbelievable" wasn't in the training data, we can construct its vector from subwords that were seen. Word2Vec/GloVe fail completely on OOV words.

3. **Rare words**: Words with few occurrences still get reasonable embeddings by sharing subwords with frequent words.

4. **Languages with rich morphology**: FastText massively outperforms Word2Vec/GloVe on Turkish, Finnish, German, etc.

### Trade-offs

- **Memory**: Subword vocabulary is much larger (~2M vs. ~100K for words). Mitigated by hashing n-grams to fixed buckets.
- **Training time**: ~2x slower than Word2Vec due to summing subword vectors.
- **Quality**: Better on morphologically rich languages and OOV. Comparable to Word2Vec on English.

**When to use FastText**: If your corpus has many rare terms, proper nouns, or morphologically complex languages. For English with large vocabularies, Word2Vec/GloVe are sufficient.

## Practical Considerations for Search

### Limitations of Static Embeddings

1. **Polysemy**: "bank" (financial) and "bank" (river) have the same vector. This is catastrophic for search.
2. **No compositionality**: Multi-word phrases like "New York" aren't naturally handled. You need phrase detection (collocation mining) or just hope the context window captures it.
3. **Query-document asymmetry**: Queries are short (2-3 words), documents are long (100s of words). Averaging word vectors for both loses signal.

### How to Use Word Embeddings in Search

**1. Query expansion**: Find synonyms via nearest neighbors and expand the query.
```
query = "laptop"
expansion = ["computer", "notebook", "pc"]  # from embedding.most_similar("laptop")
expanded_query = "laptop OR computer OR notebook OR pc"
```

**2. Re-ranking**: Score candidates with both BM25 and embedding similarity:
```
score = α * BM25(q, d) + (1-α) * cos(embed(q), embed(d))
```

**3. Feature for learning-to-rank**: Use `max(cos(query_word_i, doc_word_j))` as a feature.

**4. Building blocks for sentence embeddings**: Average word vectors (simple baseline) or use them as input to LSTM/Transformer (modern approach).

### Do NOT Use Word Embeddings For

- **Direct document retrieval at scale**: Averaging word vectors loses too much signal. Sentence embeddings (Chapter 04.2) are vastly superior.
- **Real-time query understanding**: Inference is fast (~1ms), but you can't capture context-dependent meaning.

## Training Your Own Embeddings

**When to train**:
- Domain-specific vocabulary (medical, legal, code)
- Language not covered by pre-trained models
- Privacy constraints prevent using external embeddings

**When NOT to train**:
- General English text — GloVe 6B or Word2Vec Google News are nearly optimal
- Small corpus (<10M tokens) — you'll overfit. Use pre-trained + fine-tuning.

**Training corpus guidelines**:
- **Minimum**: 10M tokens for reasonable quality
- **Ideal**: 100M-1B tokens
- **Diminishing returns**: Beyond 10B tokens, gains are small

**Libraries**:
- `gensim` (Python): Word2Vec, FastText — easy API, good docs
- `fasttext` (C++/Python): Official Facebook implementation, faster training
- `glove` (C): Official GloVe implementation, less user-friendly

## Evaluation

### Intrinsic Evaluation (Not Recommended)

- **Word analogy**: "man:woman :: king:?" — measures algebraic properties
- **Word similarity**: Correlation with human judgments (WordSim-353, SimLex-999)

**Problem**: High correlation on intrinsic benchmarks doesn't predict downstream task performance. Don't optimize for these.

### Extrinsic Evaluation (Better)

- Plug embeddings into your actual search system
- Measure MAP, NDCG, P@10 on your evaluation set
- Compare to baseline (BM25, no embeddings)

If embeddings improve your search metrics, use them. If not, they're not helping.

## Summary

| Model | Pros | Cons | Use When |
|---|---|---|---|
| **Word2Vec** | Fast, well-understood, strong analogies | No OOV handling, polysemy | General-purpose, English |
| **GloVe** | Theoretically clean, good analogies | Requires full corpus, no OOV | Batch processing, reproducibility |
| **FastText** | OOV handling, morphology-aware | Larger memory, slower | Rare words, morphologically rich languages |

**Key takeaway**: Static word embeddings are foundational but insufficient for modern search. They excel at query expansion and feature engineering but fail as standalone document representations. For end-to-end semantic search, you need sentence/document embeddings (next section).

## References and Further Reading

- [Mikolov et al. (2013): Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781) — Original Word2Vec paper
- [Pennington et al. (2014): GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf) — GloVe
- [Bojanowski et al. (2017): Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606) — FastText
- [Levy & Goldberg (2014): Neural Word Embedding as Implicit Matrix Factorization](https://papers.nips.cc/paper/2014/hash/feab05aa91085b7a8012516bc3533958-Abstract.html) — Theoretical connection between Word2Vec and matrix factorization
