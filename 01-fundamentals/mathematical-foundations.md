# Mathematical Foundations for Search

The three pillars of search mathematics are set theory, probability, and linear algebra. Each corresponds to a major retrieval model: Boolean retrieval, probabilistic retrieval, and the vector space model. Understanding all three gives you the theoretical toolkit to reason about why different systems behave the way they do.

## Table of Contents

1. [Set Theory and Boolean Retrieval](#set-theory-and-boolean-retrieval)
2. [Probability for Search](#probability-for-search)
3. [Linear Algebra for Search](#linear-algebra-for-search)
4. [Connecting the Models](#connecting-the-models)
5. [References](#references)

---

## Set Theory and Boolean Retrieval

### The Boolean Model

The Boolean retrieval model is the simplest and oldest retrieval model. Documents either match a query or they don't — there's no notion of partial match or ranking.

**Setup:**

- Each document $d$ is represented as a **set of terms**: $d = \{t_1, t_2, \ldots, t_m\}$
- A query $q$ is a **Boolean expression** over terms, using operators AND, OR, NOT
- A document matches if and only if the Boolean expression evaluates to `true`

**Example:**

```
Query: "search AND engine AND NOT database"

Document 1: {"search", "engine", "ranking", "index"}     → MATCH
Document 2: {"search", "engine", "database", "sql"}      → NO MATCH (contains "database")
Document 3: {"search", "algorithm", "graph"}              → NO MATCH (missing "engine")
```

### Set Operations on Posting Lists

In an inverted index, each term maps to a **posting list** — a sorted list of document IDs containing that term. Boolean queries map directly to set operations:

| Boolean Operator | Set Operation | Posting List Operation |
|---|---|---|
| `A AND B` | $A \cap B$ | Intersection of sorted lists |
| `A OR B` | $A \cup B$ | Union of sorted lists |
| `NOT A` | $\bar{A}$ (complement) | All doc IDs minus A's list |
| `A AND NOT B` | $A \setminus B$ (difference) | Set difference |

### Intersection Algorithm

The fundamental operation. Given two sorted posting lists, find their intersection:

```
INTERSECT(p1, p2):
    result ← []
    i ← 0, j ← 0
    while i < |p1| and j < |p2|:
        if p1[i] == p2[j]:
            result.append(p1[i])
            i++, j++
        elif p1[i] < p2[j]:
            i++
        else:
            j++
    return result
```

**Time complexity:** $O(|p_1| + |p_2|)$ — linear in the sum of list lengths.

**Optimization: query planning.** When intersecting multiple lists, process them in order of increasing length. For a query `A AND B AND C` with $|posting(A)| = 100K$, $|posting(B)| = 500$, $|posting(C)| = 50K$:

1. Start with B (shortest): 500 docs
2. Intersect with C: result shrinks
3. Intersect with A: result shrinks further

This is critical in practice. Intersecting a 500-element list with a 100K-element list is effectively $O(500)$ with skip pointers, whereas starting with the two large lists would be $O(150K)$.

### Skip Pointers for Faster Intersection

For long posting lists, we add **skip pointers** — forward links that let us jump over portions of the list during intersection:

```
Posting list for "search": [3, 5, 8, 12, 15, 21, 28, 33, 40, ...]
                                ↓              ↓              ↓
Skip pointers:              [3→15]          [15→33]        [33→...]
```

During intersection, if we're looking for doc ID 30 and the current element is 15, the skip pointer tells us the next checkpoint is at 33. Since 33 > 30, we don't skip and instead walk linearly from 15. But if we're looking for 40, we can jump from 15 to 33, saving the scan of {21, 28}.

**Optimal skip distance:** For a posting list of length $L$, place skip pointers every $\sqrt{L}$ elements. This gives expected intersection time of $O(\sqrt{L})$ per list instead of $O(L)$.

**In practice:** Lucene uses multi-level skip lists with skip distances that are powers of a configurable base (typically 8). Level 0 skips every 8 docs, level 1 every 64, level 2 every 512, etc.

### Limitations of Boolean Retrieval

1. **No ranking** — results are an unordered set. Either a document matches or it doesn't.
2. **Rigid matching** — AND is too strict (misses relevant documents that contain 9 out of 10 query terms), OR is too loose (returns too many results).
3. **No term weighting** — "the" and "spectrophotometer" are treated identically.
4. **Hard to use** — end users don't think in Boolean logic. They type natural language.

Despite these limitations, Boolean retrieval remains the backbone of candidate selection. Even systems that rank by BM25 or neural scores first need to select candidate documents, and that selection is fundamentally a set operation on posting lists.

---

## Probability for Search

### Bayes' Theorem in Information Retrieval

The probabilistic approach to IR starts from a simple question: given a query $q$ and a document $d$, what is the probability that $d$ is relevant?

By Bayes' theorem:

$$P(R=1 \mid q, d) = \frac{P(q, d \mid R=1) \cdot P(R=1)}{P(q, d)}$$

Where:
- $R=1$ means "relevant"
- $P(R=1 \mid q, d)$ is the probability of relevance given the query-document pair
- $P(q, d \mid R=1)$ is the probability of observing this query-document pair given relevance

We rank documents by $P(R=1 \mid q, d)$. Since $P(q, d)$ is the same for all documents (it doesn't depend on which document we're scoring), we can drop it and rank by:

$$P(R=1 \mid q, d) \propto P(q, d \mid R=1) \cdot P(R=1)$$

### The Probability Ranking Principle (PRP)

**Robertson (1977):** If a system ranks documents in decreasing order of $P(R=1 \mid q, d)$, then the expected loss (number of non-relevant documents ranked above relevant ones) is minimized.

This is the theoretical justification for probabilistic ranking. It says: if you could perfectly estimate relevance probabilities, ranking by those probabilities is optimal. The entire field of probabilistic IR is about approximating $P(R=1 \mid q, d)$.

### The Binary Independence Model (BIM)

The BIM makes simplifying assumptions to make the probability tractable:

1. **Binary:** Each term is either present or absent in a document (no term frequencies).
2. **Independence:** Terms are conditionally independent given relevance.

Under these assumptions, the log-odds of relevance for a document $d$ given query $q = \{t_1, \ldots, t_n\}$ is:

$$\log \frac{P(R=1 \mid d)}{P(R=0 \mid d)} = \sum_{t_i \in q \cap d} \log \frac{p_i (1 - q_i)}{q_i (1 - p_i)}$$

Where:
- $p_i = P(t_i = 1 \mid R=1)$ — probability that term $i$ appears in a relevant document
- $q_i = P(t_i = 1 \mid R=0)$ — probability that term $i$ appears in a non-relevant document

Each term contributes a weight of $\log \frac{p_i(1-q_i)}{q_i(1-p_i)}$ — the **Retrieval Status Value (RSV)**. This is the foundation that BM25 builds upon.

### From BIM to BM25

BM25 (Best Matching 25) extends the BIM by adding:

1. **Term frequency saturation** — the first occurrence of a term matters most; additional occurrences have diminishing returns.
2. **Document length normalization** — longer documents naturally contain more terms; we shouldn't penalize or reward this.

The BM25 scoring function:

$$\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{tf(t, d) \cdot (k_1 + 1)}{tf(t, d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{avgdl}\right)}$$

Where:
- $tf(t, d)$ = term frequency of $t$ in $d$
- $|d|$ = document length (in tokens)
- $avgdl$ = average document length across the corpus
- $k_1$ = term frequency saturation parameter (typically 1.2)
- $b$ = document length normalization parameter (typically 0.75)

And the IDF component:

$$\text{IDF}(t) = \log \frac{N - df(t) + 0.5}{df(t) + 0.5}$$

Where:
- $N$ = total number of documents
- $df(t)$ = document frequency — number of documents containing term $t$

**Why this formula works:**

- **IDF** gives rare terms more weight. A term appearing in 10 out of 1M documents is much more informative than one appearing in 500K out of 1M.
- **TF saturation**: the $\frac{tf \cdot (k_1 + 1)}{tf + k_1 \cdot (\ldots)}$ term approaches $(k_1 + 1)$ as $tf \to \infty$. A term appearing 100 times scores only slightly higher than one appearing 10 times. This prevents keyword stuffing from dominating.
- **Length normalization** ($b$ parameter): when $b = 1$, full normalization — a term in a 1000-word document gets much less weight than in a 100-word document. When $b = 0$, no normalization. The default $b = 0.75$ is a compromise.

### Language Models for IR

An alternative probabilistic framework. Instead of modeling $P(R=1 \mid q, d)$, we model the probability that the query was generated by the document's language model.

**Intuition:** Each document is treated as if it were generated by a probabilistic language model. We rank documents by how likely they are to have generated the query.

**Unigram language model:**

$$P(q \mid d) = \prod_{t \in q} P(t \mid d)$$

The simplest estimate of $P(t \mid d)$ is maximum likelihood:

$$P_{ML}(t \mid d) = \frac{tf(t, d)}{|d|}$$

**Problem:** If a query term doesn't appear in a document, $P(t \mid d) = 0$, making the entire product zero. This is the **zero-probability problem**.

**Solution: smoothing.** Mix the document's language model with the collection's:

**Jelinek-Mercer smoothing:**

$$P(t \mid d) = (1 - \lambda) \cdot \frac{tf(t, d)}{|d|} + \lambda \cdot \frac{cf(t)}{|C|}$$

Where $\lambda$ is the smoothing parameter, $cf(t)$ is the collection frequency, and $|C|$ is the total number of tokens in the collection.

**Dirichlet smoothing:**

$$P(t \mid d) = \frac{tf(t, d) + \mu \cdot P(t \mid C)}{|d| + \mu}$$

Where $\mu$ is the Dirichlet prior (typically 2000). This has a nice property: shorter documents get more smoothing (pulled toward the collection model), while longer documents rely more on their own term statistics.

### Information Theory Concepts

**Entropy** measures the uncertainty in a distribution:

$$H(X) = -\sum_{x} P(x) \log_2 P(x)$$

In search, entropy helps quantify:
- **Query term informativeness** — high-IDF terms have low probability in the collection, hence high information content
- **Query clarity** — how different the query's language model is from the collection's (measured by KL-divergence)
- **Index compression limits** — entropy gives the theoretical minimum bits per posting

**KL-divergence** (Kullback-Leibler divergence) measures how one distribution differs from another:

$$D_{KL}(P \| Q) = \sum_{x} P(x) \log \frac{P(x)}{Q(x)}$$

Used in query clarity analysis: $D_{KL}(P_{query} \| P_{collection})$ — if the query model is very different from the collection model, the query is clear and specific; if similar, the query is ambiguous.

---

## Linear Algebra for Search

### The Vector Space Model

The vector space model (VSM) represents documents and queries as vectors in a high-dimensional space where each dimension corresponds to a term in the vocabulary.

**Setup:**

Let the vocabulary $V = \{t_1, t_2, \ldots, t_M\}$ contain $M$ unique terms. Then:

- Each document $d$ is a vector $\vec{d} \in \mathbb{R}^M$
- Each query $q$ is a vector $\vec{q} \in \mathbb{R}^M$
- Component $d_i$ is the weight of term $t_i$ in document $d$

**Example** with vocabulary {"search", "engine", "database", "query", "index"}:

```
doc1 = "search engine index"  → [1, 1, 0, 0, 1]
doc2 = "database query index" → [0, 0, 1, 1, 1]
query = "search index"        → [1, 0, 0, 0, 1]
```

Using raw term presence (0/1) is the simplest weighting. We can do much better.

### TF-IDF Weighting

TF-IDF (Term Frequency × Inverse Document Frequency) is the standard term weighting scheme in the vector space model:

$$w(t, d) = tf(t, d) \times idf(t)$$

**Term frequency** variants:

| Variant | Formula | Behavior |
|---|---|---|
| Raw count | $tf(t, d)$ | Linear in frequency |
| Log-normalized | $1 + \log(tf(t, d))$ | Sublinear — diminishing returns |
| Augmented | $0.5 + 0.5 \cdot \frac{tf(t,d)}{\max_{t'} tf(t', d)}$ | Normalized by max frequency in doc |
| Boolean | $1$ if $tf > 0$, else $0$ | Binary presence/absence |

**Inverse document frequency:**

$$idf(t) = \log \frac{N}{df(t)}$$

Where $N$ = total documents, $df(t)$ = documents containing $t$.

IDF captures the intuition that rare terms are more informative. "the" has an IDF close to 0 (appears in almost every document); "spectrophotometer" has a high IDF.

**The combined TF-IDF vector:**

$$\vec{d} = [w(t_1, d), \, w(t_2, d), \, \ldots, \, w(t_M, d)]$$

$$\text{where} \quad w(t_i, d) = (1 + \log tf(t_i, d)) \cdot \log \frac{N}{df(t_i)}$$

(using the log-normalized TF variant, which is most common)

### Cosine Similarity

Once documents and queries are vectors, we need a similarity function. The **cosine similarity** measures the angle between two vectors, ignoring magnitude:

$$\cos(\vec{q}, \vec{d}) = \frac{\vec{q} \cdot \vec{d}}{|\vec{q}| \cdot |\vec{d}|} = \frac{\sum_{i=1}^{M} q_i \cdot d_i}{\sqrt{\sum_{i=1}^{M} q_i^2} \cdot \sqrt{\sum_{i=1}^{M} d_i^2}}$$

**Properties:**
- Range: $[0, 1]$ when all components are non-negative (which they are for TF-IDF)
- $\cos = 1$: vectors point in exactly the same direction (identical term distribution)
- $\cos = 0$: vectors are orthogonal (no terms in common)
- **Length-invariant**: a document that repeats "search engine" 100 times doesn't automatically score higher than one that says it once (unlike dot product)

**Why cosine over Euclidean distance?** In high-dimensional sparse spaces (typical vocabulary is 100K-1M dimensions), Euclidean distance is dominated by the dimensionality itself, not by meaningful differences. Cosine similarity focuses on the direction of the vector (term distribution), not its magnitude.

### Worked Example

Vocabulary: {"information", "retrieval", "search", "engine", "database"}

Corpus of 3 documents:
- $d_1$: "information retrieval and search engine design"
- $d_2$: "database search and retrieval"
- $d_3$: "information database design"

Computing IDF (N=3):
- "information": $df=2$, $idf = \log(3/2) = 0.176$
- "retrieval": $df=2$, $idf = \log(3/2) = 0.176$
- "search": $df=2$, $idf = \log(3/2) = 0.176$
- "engine": $df=1$, $idf = \log(3/1) = 0.477$
- "database": $df=2$, $idf = \log(3/2) = 0.176$

(stop words "and", "design" excluded; all $tf=1$ in this example, so $\log(1+1) = 0.301$)

TF-IDF vectors ($w = (1+\log tf) \cdot idf$, with $tf=1$ for present terms):

| | information | retrieval | search | engine | database |
|---|---|---|---|---|---|
| $d_1$ | 0.053 | 0.053 | 0.053 | 0.144 | 0 |
| $d_2$ | 0 | 0.053 | 0.053 | 0 | 0.053 |
| $d_3$ | 0.053 | 0 | 0 | 0 | 0.053 |

Query "information retrieval" → $\vec{q} = [0.053, 0.053, 0, 0, 0]$

$$\cos(\vec{q}, \vec{d_1}) = \frac{0.053 \times 0.053 + 0.053 \times 0.053}{\sqrt{0.053^2 + 0.053^2} \cdot \sqrt{0.053^2 + 0.053^2 + 0.053^2 + 0.144^2}}$$

$d_1$ ranks highest because it contains both query terms and the engine term (high IDF) adds to its profile without hurting — cosine normalizes for length.

### Dimensionality Reduction: Latent Semantic Analysis (LSA)

The vector space model works with a **term-document matrix** $A \in \mathbb{R}^{M \times N}$ where $M$ = vocabulary size and $N$ = number of documents. Entry $A_{ij}$ is the TF-IDF weight of term $i$ in document $j$.

**Problem:** This matrix is extremely sparse and high-dimensional. Words with similar meanings occupy different dimensions. "car" and "automobile" are orthogonal even though they're synonyms.

**LSA** applies **Singular Value Decomposition (SVD)** to discover latent semantic structure.

**SVD:**

$$A = U \Sigma V^T$$

Where:
- $U \in \mathbb{R}^{M \times r}$ — term-to-concept matrix (left singular vectors)
- $\Sigma \in \mathbb{R}^{r \times r}$ — diagonal matrix of singular values
- $V \in \mathbb{R}^{N \times r}$ — document-to-concept matrix (right singular vectors)
- $r = \text{rank}(A) \leq \min(M, N)$

**Truncated SVD:** Keep only the top $k$ singular values (typically $k = 100\text{-}300$):

$$A_k = U_k \Sigma_k V_k^T$$

This is the rank-$k$ approximation that minimizes the Frobenius norm $\|A - A_k\|_F$.

**What this gives us:**

- Documents are now represented as $k$-dimensional dense vectors (columns of $\Sigma_k V_k^T$)
- Terms are represented as $k$-dimensional dense vectors (rows of $U_k \Sigma_k$)
- **Synonymy**: "car" and "automobile" — if they co-occur with similar terms across the corpus, they'll map to similar regions in the reduced space
- **Polysemy** (partially addressed): "bank" (river vs. financial) may be disentangled if the contexts are sufficiently different

**Computing similarity in LSA space:**

To project a new query $\vec{q}$ into the latent space:

$$\vec{q_k} = \Sigma_k^{-1} U_k^T \vec{q}$$

Then compute cosine similarity between $\vec{q_k}$ and each document vector $\vec{d_k}$ in the reduced space.

**Complexity:**
- Full SVD: $O(\min(M^2 N, M N^2))$ — impractical for large corpora
- Truncated SVD (via randomized methods or Lanczos iteration): $O(MNk)$
- Incremental SVD (folding in new documents): $O(Mk^2)$

**Limitations:**
- Expensive to compute (though incremental methods help)
- Assumes linearity — word meaning is captured by linear combinations of latent factors
- Largely superseded by word embeddings (Word2Vec, BERT) for semantic similarity, but the mathematical framework is foundational

### Matrix Operations in Search Systems

Beyond LSA, linear algebra appears throughout search:

**Dot product for scoring:**

$$score(q, d) = \vec{q} \cdot \vec{d} = \sum_i q_i \cdot d_i$$

Both BM25 and TF-IDF scoring are fundamentally dot products between sparse vectors. The inverted index is an efficient structure for computing sparse dot products.

**Matrix-vector multiplication for batch retrieval:**

$$\vec{s} = A^T \vec{q}$$

This computes the similarity of query $\vec{q}$ against all documents at once. The result $\vec{s}$ is a vector of scores, one per document.

**Eigenvalue problems:**

PageRank is an eigenvalue problem: the PageRank vector $\vec{r}$ is the dominant eigenvector of the modified adjacency matrix of the web graph. Solved via power iteration:

$$\vec{r}^{(t+1)} = M \vec{r}^{(t)}$$

Where $M$ is the column-stochastic transition matrix.

---

## Connecting the Models

The three models aren't competitors — they're complementary perspectives on the same problem:

| Aspect | Boolean | Probabilistic | Vector Space |
|---|---|---|---|
| **Core math** | Set theory | Probability theory | Linear algebra |
| **Document representation** | Set of terms | Term statistics | Vector of weights |
| **Query representation** | Boolean expression | Bag of words | Vector of weights |
| **Matching** | Exact set operations | Probability estimation | Geometric similarity |
| **Output** | Unranked set | Ranked by $P(R \mid q, d)$ | Ranked by $\cos(\vec{q}, \vec{d})$ |
| **Strengths** | Precise filtering, fast | Principled, handles term statistics | Intuitive geometry, partial matching |
| **Weaknesses** | No ranking, all-or-nothing | Strong independence assumptions | No probabilistic interpretation |
| **Modern use** | Candidate selection | BM25 scoring | Embedding-based retrieval |

A modern search system uses all three:
1. **Boolean** operations on posting lists to select candidates
2. **Probabilistic** scoring (BM25) to rank candidates
3. **Vector space** operations (embeddings) for semantic matching and re-ranking

## References

- Robertson, S. E. & Zaragoza, H. — "The Probabilistic Relevance Framework: BM25 and Beyond" (Foundations and Trends in IR, 2009)
- Salton, G., Wong, A., & Yang, C. S. — "A Vector Space Model for Automatic Indexing" (Communications of the ACM, 1975)
- Deerwester, S. et al. — "Indexing by Latent Semantic Analysis" (Journal of the American Society for Information Science, 1990)
- Ponte, J. & Croft, W. B. — "A Language Modeling Approach to Information Retrieval" (SIGIR 1998)
- Manning, C. D., Raghavan, P., & Schütze, H. — *Introduction to Information Retrieval*, Chapters 1, 6, 11, 12
