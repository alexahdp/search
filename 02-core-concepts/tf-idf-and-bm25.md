# TF-IDF and BM25

TF-IDF and BM25 are the workhorses of text-based search. They've powered billions of queries over the past three decades. Understanding them deeply — not just the formulas, but the intuition, parameter tuning, and practical variants — is essential for any search engineer.

## Table of Contents

1. [From Term Frequency to TF-IDF](#from-term-frequency-to-tf-idf)
2. [TF-IDF Variants](#tf-idf-variants)
3. [BM25: The Practical Standard](#bm25-the-practical-standard)
4. [BM25 Parameter Tuning](#bm25-parameter-tuning)
5. [BM25F: Multi-Field Scoring](#bm25f-multi-field-scoring)
6. [BM25+: Fixing the Lower Bound](#bm25-fixing-the-lower-bound)
7. [Worked Examples](#worked-examples)
8. [Implementation Notes](#implementation-notes)
9. [References](#references)

---

## From Term Frequency to TF-IDF

### The Problem with Raw Term Frequency

Consider scoring documents for the query "search engine":

**Document A**: "Search engines use inverted indices. Search algorithms rank results. Modern search systems scale to billions of documents."

**Document B**: "This article discusses search engine optimization, search engine marketing, and search engine architecture. Search engines are critical for information retrieval."

Document B mentions "search" and "engine" more often, so a naive scoring function based on **term frequency (TF)** alone would rank it higher. But look at the signal-to-noise ratio: Document B is repetitive and probably less useful.

The issue: raw term frequency rewards keyword stuffing and doesn't account for term **importance**.

### Inverse Document Frequency (IDF)

Not all terms are equally informative. "The" appears in nearly every document — it's useless for discriminating relevance. "Inverted index" appears rarely — it's highly informative.

**Inverse Document Frequency (IDF)** measures term rarity:

$$\text{IDF}(t) = \log \frac{N}{df(t)}$$

where:
- $N$ = total number of documents in the corpus
- $df(t)$ = document frequency (number of documents containing term $t$)

**Intuition**:
- If a term appears in every document: $df(t) = N$, so $\text{IDF}(t) = \log(1) = 0$ (worthless)
- If a term appears in 1 document: $df(t) = 1$, so $\text{IDF}(t) = \log(N)$ (very informative)

The logarithm dampens the effect so that a term appearing in 10 documents isn't considered 10× more important than one appearing in 1 document.

### TF-IDF: Combining Both

$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$$

Score a document for a query by summing TF-IDF scores across query terms:

$$\text{score}(q, d) = \sum_{t \in q} \text{TF}(t, d) \times \text{IDF}(t)$$

**Example**: Corpus with $N = 1,000,000$ documents.

Query: "search engine"

| Term | TF in doc A | TF in doc B | DF | IDF = log(N/DF) |
|---|---|---|---|---|
| search | 3 | 5 | 100,000 | 2.30 |
| engine | 1 | 4 | 50,000 | 3.00 |

```
Score(doc A) = 3×2.30 + 1×3.00 = 6.90 + 3.00 = 9.90
Score(doc B) = 5×2.30 + 4×3.00 = 11.50 + 12.00 = 23.50
```

With raw TF, B wins by a larger margin. With TF-IDF, B still wins, but the formula now captures that "engine" is more informative than "search."

## TF-IDF Variants

The formula $\text{TF} \times \text{IDF}$ has many variants. This is formalized in the **SMART notation**, a three-letter code specifying the scheme:

```
[TF scheme].[IDF scheme].[Normalization scheme]

Examples:
- "ntn" → natural TF, no IDF, cosine normalization
- "lnc" → log TF, no IDF, cosine normalization
- "ltc" → log TF, IDF, cosine normalization
```

### TF Schemes

| Letter | Formula | Description |
|---|---|---|
| **n** | $\text{TF}(t,d)$ | Natural: raw term frequency |
| **l** | $1 + \log(\text{TF}(t,d))$ | Logarithmic: diminishing returns for repeated terms |
| **a** | $0.5 + \frac{0.5 \cdot \text{TF}(t,d)}{\max_{t'} \text{TF}(t',d)}$ | Augmented: normalized by max TF in the doc |
| **b** | $1$ if $\text{TF}(t,d) > 0$ else $0$ | Binary: presence/absence only |
| **L** | $\frac{1 + \log(\text{TF}(t,d))}{1 + \log(\text{avg}_{t'}\text{TF}(t',d))}$ | Log average: normalized by doc's average TF |

**Why logarithmic?** Sublinear scaling reflects diminishing returns. A document with 10 occurrences isn't 10× better than one with 1 occurrence — maybe 2-3× better. The log captures this:

```
TF:   1    2    5    10   50   100
Log:  1.0  1.3  1.7  2.0  2.7  3.0
```

### IDF Schemes

| Letter | Formula | Description |
|---|---|---|
| **n** | $1$ | No IDF (all terms weighted equally) |
| **t** | $\log \frac{N}{df(t)}$ | Standard IDF |
| **p** | $\max(0, \log \frac{N - df(t)}{df(t)})$ | Probabilistic IDF (can be negative) |

The standard "t" scheme is by far the most common.

### Normalization Schemes

| Letter | Formula | Description |
|---|---|---|
| **n** | $1$ | No normalization |
| **c** | $\frac{1}{\sqrt{\sum_{t'} w_{t'}^2}}$ | Cosine normalization (unit vector) |
| **u** | $\frac{1}{\text{unique terms in } d}$ | Pivoted unique normalization |
| **b** | $\frac{1}{0.8 + 0.2 \cdot \frac{|d|}{\text{avg\_doc\_len}}}$ | Byte-size normalization |

**Why normalize?** Long documents naturally accumulate higher scores. Without normalization, a 10,000-word essay would always beat a concise 200-word answer, even if the answer is more relevant. Cosine normalization is the most common — it treats documents and queries as vectors and scores by cosine similarity.

### Classic Combinations

**"ltc.ltn"** (SMART system default):
- Documents: log TF, IDF, cosine normalization
- Queries: log TF, IDF, no normalization

**"lnc.ltc"** (alternative):
- Documents: log TF, no IDF, cosine normalization (precompute at index time)
- Queries: log TF, IDF, cosine normalization (compute at query time)

In practice, BM25 (next section) has largely replaced hand-tuned TF-IDF variants because it performs better with less manual tuning.

## BM25: The Practical Standard

BM25 (Best Match 25) is the single most widely deployed ranking function in search. It's the default in Elasticsearch, Solr (since 6.0), Lucene, Whoosh, Xapian, and Terrier. If you're using a search engine and haven't changed the similarity function, you're using BM25.

### The Formula

$$\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{\text{TF}(t, d) \cdot (k_1 + 1)}{\text{TF}(t, d) + k_1 \cdot (1 - b + b \cdot \frac{|d|}{\text{avg\_doc\_len}})}$$

where:
- $\text{IDF}(t) = \log \frac{N - df(t) + 0.5}{df(t) + 0.5}$ (BM25's IDF variant)
- $k_1$ controls term saturation (typically $1.2$ to $2.0$)
- $b$ controls length normalization (typically $0.75$)
- $|d|$ = document length in tokens
- $\text{avg\_doc\_len}$ = average document length in the corpus

### Breaking It Down

**The TF component:**

$$\frac{\text{TF}(t, d) \cdot (k_1 + 1)}{\text{TF}(t, d) + k_1 \cdot (1 - b + b \cdot \frac{|d|}{\text{avg\_doc\_len}})}$$

This is a **saturation function**. As TF increases, the score increases but with diminishing returns:

```
With k₁=1.2, b=0.75, |d|=avg_doc_len (so the denominator is TF + k₁):

TF   Score contribution
1    0.69
2    0.86
3    0.94
5    1.03
10   1.09
50   1.17
100  1.18
```

After ~10 occurrences, additional mentions barely help. This prevents keyword stuffing.

**The length normalization term:**

$$1 - b + b \cdot \frac{|d|}{\text{avg\_doc\_len}}$$

- If $|d| = \text{avg\_doc\_len}$, this equals $1$ (no penalty or boost)
- If $|d| < \text{avg\_doc\_len}$, this is $< 1$ (boosts the score — short docs are rewarded)
- If $|d| > \text{avg\_doc\_len}$, this is $> 1$ (penalizes the score — long docs are punished)

Parameter $b$ controls how much length matters:
- $b = 0$: no length normalization
- $b = 1$: full length normalization
- $b = 0.75$: balanced (the standard default)

**The IDF component:**

$$\text{IDF}(t) = \log \frac{N - df(t) + 0.5}{df(t) + 0.5}$$

This is the **Robertson-Sparck Jones weight**, derived from probabilistic information retrieval theory (see [Mathematical Foundations](../01-fundamentals/mathematical-foundations.md#probability-for-search)). The $+0.5$ terms are smoothing constants to prevent division by zero and extreme values.

### Comparison to TF-IDF

BM25 improves on TF-IDF in three key ways:

1. **Saturation**: TF-IDF scales linearly (or log-linearly) with TF. BM25 saturates, preventing keyword stuffing from dominating.
2. **Length normalization**: TF-IDF requires manual normalization (cosine, pivoted). BM25 has length normalization built in with a tunable parameter.
3. **Probabilistic foundation**: BM25 is derived from the Binary Independence Model, making it theoretically principled.

## BM25 Parameter Tuning

### Parameter $k_1$ (Term Saturation)

Controls how quickly term frequency saturates.

- **Low $k_1$ (0.5 to 1.0)**: Fast saturation. After 2-3 occurrences, additional mentions don't help much. Use for: keyword-focused search, spam-prone corpora.
- **Medium $k_1$ (1.2 to 1.5)**: Balanced. **Default: 1.2**
- **High $k_1$ (2.0 to 3.0)**: Slow saturation. Repeated mentions continue to matter. Use for: long documents, technical content where term density is a quality signal.

### Parameter $b$ (Length Normalization)

Controls how much document length affects the score.

- **Low $b$ (0.0 to 0.5)**: Weak length penalty. Long documents aren't punished much. Use for: corpora where long docs are high quality (scientific papers, legal documents).
- **Medium $b$ (0.6 to 0.8)**: Balanced. **Default: 0.75**
- **High $b$ (0.9 to 1.0)**: Strong length penalty. Short, concise answers are strongly preferred. Use for: web search (where blogs stuff keywords), Q&A forums, snippet-based results.

### Tuning Process

1. **Start with defaults**: $k_1 = 1.2$, $b = 0.75$. These work well for most domains.

2. **Gather evaluation data**: Build a test set with 500-1,000 judged queries (see [Evaluation Metrics](./evaluation-metrics.md)).

3. **Grid search**: Try combinations:

   ```
   k₁ ∈ {0.8, 1.0, 1.2, 1.5, 2.0}
   b  ∈ {0.5, 0.6, 0.7, 0.75, 0.8, 0.9}
   ```

   Compute NDCG@10 or MAP on your test set for each combination.

4. **Analyze by query type**: Parameters might differ for navigational vs. informational queries. Some systems learn per-query-class parameters.

5. **Validate online**: A/B test the best offline parameters. Offline metrics don't always correlate perfectly with user satisfaction.

### Typical Parameter Ranges by Domain

| Domain | $k_1$ | $b$ | Rationale |
|---|---|---|---|
| Web search | 1.2 | 0.75 | Balanced; penalize keyword-stuffed pages |
| E-commerce | 1.5 | 0.6 | Product descriptions vary in length; don't over-penalize detailed listings |
| Legal / patent | 2.0 | 0.5 | Long documents; term repetition is meaningful |
| News | 1.2 | 0.8 | Prefer concise articles; penalize fluff |
| Code search | 1.5 | 0.3 | Length doesn't indicate quality; verbosity is common |
| Q&A (Stack Overflow) | 1.2 | 0.85 | Prefer concise, focused answers |

## BM25F: Multi-Field Scoring

Real documents aren't flat text — they have structure: title, body, URL, headings, anchor text. A term match in the title is more valuable than in the body. **BM25F** (BM25 with field weights) handles this.

### The Formula

$$\text{BM25F}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{\widetilde{\text{TF}}(t, d) \cdot (k_1 + 1)}{\widetilde{\text{TF}}(t, d) + k_1}$$

where the **weighted term frequency** is:

$$\widetilde{\text{TF}}(t, d) = \sum_{f \in \text{fields}} w_f \cdot \frac{\text{TF}(t, d_f)}{1 - b_f + b_f \cdot \frac{|d_f|}{\text{avg\_len}_f}}$$

Each field $f$ has:
- $w_f$: field weight (boost factor)
- $b_f$: field-specific length normalization
- $|d_f|$: length of this field in this document
- $\text{avg\_len}_f$: average length of field $f$ across the corpus

### Practical Example

Document structure:

```
Title: "Inverted Index Implementation in Rust"
Body: "This tutorial covers how to build an inverted index from scratch using Rust. An inverted index is a fundamental data structure in search engines..."
URL: "example.com/inverted-index-rust"
```

Query: "inverted index rust"

Field configuration:

| Field | Weight $w_f$ | Normalization $b_f$ |
|---|---|---|
| Title | 3.0 | 0.8 |
| Body | 1.0 | 0.75 |
| URL | 2.0 | 0.5 |

"inverted" appears: 1× in title, 2× in body, 1× in URL
"index" appears: 1× in title, 2× in body, 1× in URL
"rust" appears: 1× in title, 1× in body, 1× in URL

The weighted TF computation combines these, applying higher weights to title and URL matches. The result: title matches dominate the score, as intended.

### Tuning Field Weights

Start with these heuristics:

| Field | Typical weight |
|---|---|
| Title | 2.0 - 5.0 |
| Headings | 1.5 - 3.0 |
| Body | 1.0 (reference) |
| URL path | 1.5 - 2.0 |
| Anchor text (from inlinks) | 2.0 - 4.0 |
| Metadata (keywords, description) | 0.5 - 1.0 |

Then tune via learning-to-rank (Chapter 07) — treat field weights as features and let the model learn optimal values.

### Implementation in Elasticsearch

```json
{
  "query": {
    "multi_match": {
      "query": "inverted index rust",
      "fields": ["title^3", "body", "url^2"],
      "type": "best_fields"
    }
  }
}
```

The `^3` syntax applies a boost (field weight). Elasticsearch's `multi_match` uses BM25F internally (as of v5.0+).

## BM25+: Fixing the Lower Bound

### The Problem with BM25

BM25 has a quirk: if a document is shorter than average, the length normalization term can become very small, causing the TF component to dominate. In extreme cases, a very short document with a single mention of a rare term can score higher than a comprehensive document with many relevant terms.

More technically: the BM25 score contribution for a single term occurrence is:

$$\frac{k_1 + 1}{k_1 \cdot (1 - b + b \cdot \frac{|d|}{\text{avg\_doc\_len}}) + 1}$$

For very short documents ($|d| \ll \text{avg\_doc\_len}$), this can exceed $(k_1 + 1)$, creating an unintended boost.

### The BM25+ Fix

BM25+ adds a **lower bound** to the TF component:

$$\text{BM25+}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \left( \delta + \frac{\text{TF}(t, d) \cdot (k_1 + 1)}{\text{TF}(t, d) + k_1 \cdot (1 - b + b \cdot \frac{|d|}{\text{avg\_doc\_len}})} \right)$$

where $\delta$ (typically $1.0$) is a constant lower bound ensuring each term match contributes at least $\delta \cdot \text{IDF}(t)$ to the score.

**Effect**: Prevents the pathological behavior with very short documents. In practice, BM25+ improves retrieval quality by 1-3% NDCG on web search benchmarks.

**Adoption**: BM25+ is used at Bing and some research systems but isn't yet the default in Elasticsearch/Solr. It's a relatively recent refinement (Lv & Zhai, 2011).

## Worked Examples

### Example 1: BM25 Score Computation

**Corpus**: 10,000 documents, average length = 200 tokens

**Document**: "search engine ranking algorithms search engine optimization"
- Length: 7 tokens
- Term frequencies: {search: 2, engine: 2, ranking: 1, algorithms: 1, optimization: 1}

**Query**: "search engine"

**Parameters**: $k_1 = 1.2$, $b = 0.75$

**Document frequencies**:
- "search": appears in 500 documents
- "engine": appears in 800 documents

**Step 1: Compute IDF**

$$\text{IDF}(\text{search}) = \log \frac{10000 - 500 + 0.5}{500 + 0.5} = \log \frac{9500.5}{500.5} = \log(18.98) = 2.94$$

$$\text{IDF}(\text{engine}) = \log \frac{10000 - 800 + 0.5}{800 + 0.5} = \log \frac{9200.5}{800.5} = \log(11.49) = 2.44$$

**Step 2: Compute length normalization term**

$$L = 1 - b + b \cdot \frac{|d|}{\text{avg\_doc\_len}} = 1 - 0.75 + 0.75 \cdot \frac{7}{200} = 0.25 + 0.026 = 0.276$$

**Step 3: Compute TF component for each term**

For "search" (TF = 2):

$$\frac{2 \cdot (1.2 + 1)}{2 + 1.2 \cdot 0.276} = \frac{4.4}{2 + 0.331} = \frac{4.4}{2.331} = 1.887$$

For "engine" (TF = 2):

$$\frac{2 \cdot (1.2 + 1)}{2 + 1.2 \cdot 0.276} = 1.887$$

**Step 4: Final score**

$$\text{BM25}(q, d) = 2.94 \times 1.887 + 2.44 \times 1.887 = 5.55 + 4.60 = 10.15$$

### Example 2: Field Weight Impact

**Document A** (title match):
- Title: "Search Engine Fundamentals" (3 tokens)
- Body: "This article discusses databases." (4 tokens)

**Document B** (body match only):
- Title: "Introduction" (1 token)
- Body: "Search engine systems are complex..." (20 tokens, contains "search" 1×, "engine" 1×)

**Query**: "search engine"

**Field weights**: Title = 3.0, Body = 1.0

Without field weights (treating all text equally), Document B might score higher due to having more content. With BM25F and title boosting, Document A's title match dominates, ranking it first — which is the right behavior for most queries.

### Example 3: Length Normalization Effect

**Short document** (50 tokens): Contains "search" 2×, "engine" 1×
**Long document** (500 tokens): Contains "search" 3×, "engine" 2×

With $b = 0.75$:
- Short doc gets a length boost (denominator is smaller → higher score per term)
- Long doc gets penalized

With $b = 0.0$ (no normalization):
- Long doc scores higher (more total term occurrences)

The right choice depends on your domain. For web search (where thin content is common), $b = 0.75$ is better. For scientific papers (where long = comprehensive), $b = 0.5$ or lower is better.

## Implementation Notes

### Precomputing at Index Time

BM25 can be partially precomputed:

**Index-time**:
- Document length $|d|$ (store as doc metadata)
- Term frequencies per field (in posting lists)
- Corpus statistics: $N$, $\text{avg\_doc\_len}$, $df(t)$ for each term

**Query-time**:
- IDF computation (fast: just log + division)
- BM25 scoring using stored TFs and lengths

### Handling Updates

When documents are added or deleted:
1. **$N$ changes** → all IDF values change (slightly). Most systems recompute periodically (e.g., on index segment merges) rather than per-document.
2. **$\text{avg\_doc\_len}$ changes** → affects length normalization. Also recomputed on merges.
3. **$df(t)$ changes** → affects IDF for terms in the new document.

In practice, approximate statistics are fine. Elasticsearch recomputes on segment refresh (typically every 1 second).

### Numeric Precision

Be careful with floating-point precision:
- Log computations can produce NaN or infinity if $df(t) = 0$ (shouldn't happen in a properly built index)
- Very rare terms can have extremely high IDF values (log(10^9) ≈ 20). Consider capping IDF at a maximum value (e.g., 10-15).
- Scores should be stored as `float` (32-bit), not `double` — precision beyond 6-7 decimal places doesn't affect ranking.

### Negative IDF

If a term appears in every document (or nearly every document), BM25's IDF can be zero or negative:

$$\text{IDF}(t) = \log \frac{N - df(t) + 0.5}{df(t) + 0.5}$$

If $df(t) > N/2$, this is negative. Most implementations clamp IDF to a minimum of 0 or a small positive value (e.g., 0.01).

### Phrase Queries

BM25 is a bag-of-words model — it doesn't handle phrases. For "search engine" as a phrase:
1. Retrieve candidates using BM25 on individual terms
2. Filter to documents where the terms appear adjacent
3. Apply a phrase match boost

Elasticsearch's `match_phrase` query does this. The boost is typically additive or multiplicative (e.g., multiply BM25 score by 1.5 for phrase matches).

## References

- Robertson, S. E. & Zaragoza, H. — "The Probabilistic Relevance Framework: BM25 and Beyond" (Foundations and Trends in IR, 2009)
- Robertson, S. E., Walker, S., & Beaulieu, M. — "Okapi at TREC-7" (TREC, 1998)
- Lv, Y. & Zhai, C. — "Lower-Bounding Term Frequency Normalization" (CIKM, 2011) — introduces BM25+
- Zaragoza, H., et al. — "Microsoft Cambridge at TREC 13: Web and HARD Tracks" (TREC, 2004) — describes BM25F
- Salton, G. & Buckley, C. — "Term-Weighting Approaches in Automatic Text Retrieval" (Information Processing & Management, 1988) — SMART notation reference
- Manning, C. D., Raghavan, P., & Schütze, H. — *Introduction to Information Retrieval*, Chapter 6 (Cambridge, 2008)
