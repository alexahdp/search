# Relevance and Ranking

Relevance is the central concept in information retrieval. Everything else — indexing, scoring functions, evaluation metrics — exists to serve one goal: return results that satisfy the user's information need, in the right order. This section defines relevance precisely, explains how it's measured through human judgments, and covers the architectural patterns for ranking at scale.

## Table of Contents

1. [What Is Relevance?](#what-is-relevance)
2. [Dimensions of Relevance](#dimensions-of-relevance)
3. [Relevance Judgments](#relevance-judgments)
4. [Ranking Fundamentals](#ranking-fundamentals)
5. [Multi-Stage Ranking](#multi-stage-ranking)
6. [Relevance Signals](#relevance-signals)
7. [The Cold Start Problem](#the-cold-start-problem)
8. [References](#references)

---

## What Is Relevance?

At its simplest, relevance is the degree to which a document satisfies the user's information need. But that simplicity is deceptive. Consider the query "python":

- A tutorial on Python the programming language — relevant if the user is a developer
- A Wikipedia article about Burmese pythons — relevant if the user is a herpetologist
- A review of Monty Python's Flying Circus — relevant if the user loves comedy

The same document can be relevant or irrelevant depending entirely on the user's intent. This makes relevance **subjective**, **context-dependent**, and **multi-faceted**.

### Topical vs. User Relevance

IR theory distinguishes two types:

**Topical relevance** (also called **system relevance** or **aboutness**): the document is about the same topic as the query. This is what most retrieval models compute — a similarity score between query and document content.

**User relevance** (also called **situational relevance**): the document actually satisfies the user's information need in their specific context. A topically relevant document might still be useless if it's:
- Written in a language the user can't read
- Too technical or too basic for their expertise level
- Outdated
- A duplicate of something they've already seen
- Behind a paywall

Production search systems aim for user relevance but mostly approximate it through topical relevance combined with additional signals (freshness, authority, personalization).

### The Probability Ranking Principle

The **Probability Ranking Principle (PRP)**, formulated by Robertson (1977), provides the theoretical foundation for ranking:

> If a retrieval system's response to each query is a ranking of documents in order of decreasing probability of relevance, the overall effectiveness of the system to its users will be the best that is obtainable on the basis of the available data.

Formally: rank documents by $P(R=1 \mid q, d)$, where $R$ is a binary relevance variable, $q$ is the query, and $d$ is the document.

This principle has important assumptions:
1. Relevance of each document is independent of other documents (not true — users want diversity)
2. The user examines results top-to-bottom (approximately true for web search, less so for mobile)
3. The cost of examining a non-relevant document is the same everywhere in the ranking (not true — users have less patience lower in the list)

Despite these limitations, PRP is the theoretical bedrock of ranked retrieval. BM25, language models, and most neural rankers can all be interpreted as approximating $P(R=1 \mid q, d)$.

## Dimensions of Relevance

When senior search engineers talk about "relevance," they usually mean a multi-dimensional quality concept:

### Topical match

Does the document's content match the query's topic? This is the bread-and-butter dimension that TF-IDF, BM25, and embedding models optimize for.

### Authority / trustworthiness

Is the source credible? PageRank, domain authority scores, and E-E-A-T (Experience, Expertise, Authoritativeness, Trustworthiness) signals attempt to capture this. In e-commerce, this translates to seller ratings, brand trust, and review quality.

### Freshness

How recent is the document? For news queries, a 2-hour-old article is more relevant than yesterday's. For evergreen content ("how to tie a tie"), freshness matters less. Systems typically apply a **freshness boost** that decays over time:

$$\text{freshness\_boost}(d) = e^{-\lambda \cdot \text{age}(d)}$$

where $\lambda$ controls the decay rate (different for different query types).

### Quality

Is the document well-written, comprehensive, and useful? Thin content, content farms, and auto-generated pages may be topically relevant but low quality. Features used: content length, reading level, ad density, formatting quality.

### Location / personalization

User's geographic location, language preference, search history, and demographic signals. "Pizza near me" requires location; "best laptop 2024" might vary by purchasing power.

### Diversity

At the result-set level (not individual document level): does the list cover different aspects of the query? For "jaguar," an ideal result page shows the car, the animal, and the macOS version — not ten pages about cars.

### Commercial intent

In e-commerce and ad-supported search: does the result match the user's purchase/conversion intent? This is where relevance shades into **revenue optimization** — a tension every commercial search team navigates.

## Relevance Judgments

To build and evaluate a search system, you need ground truth: for a given query, which documents are relevant and how relevant are they? This ground truth comes from **relevance judgments** (also called **relevance assessments** or **labels**).

### Judgment Scales

**Binary**: Relevant (1) or not relevant (0). Simple but lossy — a perfect 10-page answer and a marginally useful paragraph both get "1."

**Graded (ordinal)**: Multiple levels of relevance. Common scales:

| Grade | Label | Description |
|---|---|---|
| 0 | Not relevant | Document has nothing to do with the query |
| 1 | Marginally relevant | Mentions the topic but doesn't address the need |
| 2 | Fairly relevant | Partially addresses the information need |
| 3 | Highly relevant | Directly and substantially addresses the need |
| 4 | Perfect | Completely satisfies the information need |

TREC (Text REtrieval Conference) historically uses 0-2 scales; modern web search teams often use 0-4 or 0-5.

**Preference-based (pairwise)**: Given query $q$ and two documents $d_A$, $d_B$: is $d_A$ better, worse, or equal to $d_B$? This is cognitively easier for judges and naturally produces ranking data. Used heavily in learning-to-rank (Chapter 07).

### Collecting Judgments

**Editorial judgments**: Trained human raters (often called "judges" or "assessors") evaluate query-document pairs. This is the gold standard:

1. Define a **rating guideline** — a document (often 20+ pages) explaining each grade level with examples
2. Select a **query set** — representative queries covering head, torso, and tail traffic
3. For each query, select **candidate documents** — typically from your system's top-k results (this creates a bias toward your current system, called **pooling bias**)
4. Present query-document pairs to judges (typically 3+ judges per pair)
5. Aggregate judgments (majority vote, average, or adjudication for disagreements)

**Cost**: At Google scale, editorial judgments cost $1-5 per query-document pair. A typical evaluation set has 5,000-10,000 queries × 20-50 documents each = 100K-500K judgments. This is expensive, which is why click data matters.

**Crowdsourced judgments**: Use platforms like Amazon Mechanical Turk, Appen, or Toloka. Cheaper ($0.05-0.50 per judgment) but noisier. Mitigations: gold questions (known-answer questions to filter bad workers), redundancy (5+ judges per pair), statistical aggregation (Dawid-Skene model).

**Implicit feedback**: Infer relevance from user behavior:

| Signal | What it suggests | Reliability |
|---|---|---|
| Click | User thought the snippet was relevant | Medium (position bias, attractive snippets) |
| Long click (dwell time > 30s) | User found the page useful | High |
| Short click (dwell time < 5s) | User was disappointed (pogo-sticking) | High (negative signal) |
| Skip | Document was not attractive | Low (user might not have seen it) |
| Query reformulation | Results were unsatisfying | Medium |
| Last click in session | User's need was satisfied | High |

**Position bias**: Users click higher-ranked results more often regardless of relevance. A result at position 1 gets 30-40% of clicks; position 10 gets 1-3%. This must be corrected for when using click data as relevance signal. Common approaches:

- **Examination model**: $P(\text{click} \mid q, d, \text{pos}) = P(\text{examine} \mid \text{pos}) \cdot P(\text{click} \mid q, d, \text{examined})$
- **Inverse propensity weighting (IPW)**: weight each click by $1 / P(\text{examine} \mid \text{pos})$
- **Randomization**: occasionally shuffle results and compare click rates

### Inter-Annotator Agreement

How consistent are judges? Measured by **Cohen's kappa** ($\kappa$) for two raters or **Fleiss' kappa** for multiple raters:

$$\kappa = \frac{P_o - P_e}{1 - P_e}$$

where $P_o$ is observed agreement and $P_e$ is expected agreement by chance.

| $\kappa$ value | Interpretation |
|---|---|
| < 0.20 | Poor agreement |
| 0.21 – 0.40 | Fair |
| 0.41 – 0.60 | Moderate |
| 0.61 – 0.80 | Substantial |
| 0.81 – 1.00 | Almost perfect |

In practice, search relevance judgments typically achieve $\kappa = 0.4 - 0.7$, which is moderate to substantial. Binary judgments have higher agreement than graded judgments (fewer distinctions to make).

Low agreement tells you something important: if trained judges can't agree on relevance, your ranking function is trying to optimize a fuzzy target. This is fundamental — there's a ceiling on how good any system can be.

## Ranking Fundamentals

### The Scoring Function

A ranking system assigns each document a **relevance score** and sorts by decreasing score. The score is typically a function of:

$$\text{score}(q, d) = f(\text{text\_match}(q, d), \text{static\_quality}(d), \text{freshness}(d), \text{personalization}(q, u), \ldots)$$

where $u$ is the user context. The simplest instantiation is pure BM25 (text match only). Production systems combine dozens to hundreds of signals.

### Score Combination Strategies

How do you combine multiple relevance signals into a single score?

**Linear combination**: The simplest and most common starting point:

$$\text{score} = w_1 \cdot \text{BM25}(q,d) + w_2 \cdot \text{PageRank}(d) + w_3 \cdot \text{freshness}(d) + \ldots$$

Weights can be tuned manually or learned (this is the basis of pointwise learning-to-rank, Chapter 07).

**Multiplicative boosting**: Common in Elasticsearch/Solr:

$$\text{score} = \text{BM25}(q,d) \cdot \text{boost}_{\text{freshness}}(d) \cdot \text{boost}_{\text{popularity}}(d)$$

Multiplicative boosting can cause instability — a zero or very small boost in one factor zeroes out the entire score. Production systems often use $\log(1 + x)$ to dampen extreme values.

**Learning to Rank (LTR)**: Train a model (gradient-boosted trees, neural network) to predict relevance from features. This is the state of the art for any system with enough training data. Covered in depth in Chapter 07.

### Score Normalization

Different scoring functions produce scores on different scales. BM25 scores might range from 0 to 30; a popularity score might be 0 to 1,000,000. Before combining, you often need to normalize:

**Min-max normalization**: Scale to $[0, 1]$:

$$s' = \frac{s - s_{\min}}{s_{\max} - s_{\min}}$$

Problem: sensitive to outliers. One document with a very high PageRank stretches the scale.

**Z-score normalization**: Center at 0, scale by standard deviation:

$$s' = \frac{s - \mu}{\sigma}$$

Better for combining signals from different distributions.

**Rank-based normalization**: Replace scores with ranks. Loses magnitude information but is robust to outliers and different score distributions.

In practice, LTR models handle normalization implicitly — the model learns the mapping from raw feature values to relevance. This is one reason LTR dominates hand-tuned linear combinations.

## Multi-Stage Ranking

No production search system runs a single ranker over the entire corpus. Instead, ranking happens in stages, each applying a more expensive (but more accurate) model to a smaller candidate set.

### The Funnel Architecture

```
                    ┌──────────────────────────────┐
Entire corpus       │         1B+ documents         │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────▼───────────────┐
L0: Index pruning   │     Inverted index lookup      │  ~10K candidates
(BM25, boolean)     │     (postings intersection)    │  ~1-5ms
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────▼───────────────┐
L1: Lightweight     │   Feature-based scoring        │  ~1K candidates
ranking             │   (GBDT with ~50 features)     │  ~5-20ms
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────▼───────────────┐
L2: Heavy           │   Cross-encoder or large       │  ~50-100 candidates
re-ranking          │   transformer model             │  ~50-100ms
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────▼───────────────┐
L3: Business logic  │   Diversity, dedup, policies,  │  ~10-20 results
& presentation      │   snippet generation            │  ~5-10ms
                    └──────────────────────────────┘
```

### Stage 0: Candidate retrieval

The goal is **high recall, low cost**. Retrieve a large-ish candidate set that (hopefully) contains all the good results. Tools:

- **Inverted index + BM25**: The workhorse. Posting list intersection produces candidates with term-match scores.
- **Dense retrieval (ANN)**: Query and documents encoded as vectors, retrieve by approximate nearest neighbor search (HNSW, IVF). Catches semantic matches that BM25 misses.
- **Hybrid**: Run both retrievers, merge candidate sets with reciprocal rank fusion or learned merging.

Key constraint: This stage must process the entire index, so per-document cost must be extremely low — a few microseconds.

### Stage 1: Lightweight ranking

Compute a richer feature vector for each candidate and apply a fast ML model (typically gradient-boosted decision trees: XGBoost, LightGBM, CatBoost). Common features:

| Category | Features |
|---|---|
| Text match | BM25, TF-IDF, query term coverage, phrase match, proximity |
| Document quality | PageRank, domain authority, content length, reading level |
| Freshness | Document age, last update time, time decay score |
| Popularity | Historical CTR, view count, share count |
| Query-document | Query-title overlap, query-body overlap, field-level BM25 scores |

This stage typically reduces 10K candidates to ~1K, spending ~5-20μs per candidate.

### Stage 2: Heavy re-ranking

Apply expensive models to a small candidate set. This is where cross-encoders (BERT-based models that jointly encode query + document) live. The accuracy gain from cross-attention is significant but the cost is $O(|q| \cdot |d|)$ per pair, making it too expensive for more than ~100 candidates.

Examples:
- **MonoBERT / monoT5**: Cross-encoder that scores each (query, document) pair independently
- **DuoBERT / duoT5**: Pairwise model that compares two documents given a query
- **ColBERT**: Late-interaction model — cheaper than full cross-encoding, more expensive than bi-encoders

### Stage 3: Business logic and presentation

After ranking by relevance, apply business rules:

- **Diversity**: Ensure the top results cover different intents/subtopics. MMR (Maximal Marginal Relevance) is the classic algorithm:

$$\text{MMR}(d) = \lambda \cdot \text{sim}(q, d) - (1 - \lambda) \cdot \max_{d_j \in S} \text{sim}(d, d_j)$$

where $S$ is the set of already-selected documents. $\lambda$ controls the relevance-diversity trade-off.

- **Deduplication**: Near-duplicate detection using MinHash, SimHash, or embedding similarity
- **Policy filters**: Remove blocked content, apply age restrictions, enforce legal requirements
- **Snippet generation**: Extract or synthesize the best passage to display. Typically: find the passage with highest query-term density, highlight matching terms

### Why Multi-Stage?

The math is compelling. Suppose you need to rank 1B documents:

| Approach | Per-doc cost | Total cost | Latency |
|---|---|---|---|
| Cross-encoder on all docs | 10ms | 10M seconds | Impossible |
| GBDT on all docs | 5μs | 5,000s | Impossible |
| BM25 on all docs (via index) | 0.01μs* | 10ms | OK |
| BM25 → GBDT (top 1K) → cross-encoder (top 50) | — | ~60ms | OK |

*BM25 via inverted index doesn't actually score every document — it only scores documents in the posting lists of query terms, which is typically a few percent of the corpus.

The funnel architecture is a universal pattern in search engineering. You'll see it at Google, Bing, Amazon, LinkedIn, Spotify, and every other system at scale.

## Relevance Signals

A practical inventory of signals used in production search systems:

### Text-based signals

- **BM25 score** (query vs. title, body, URL, anchor text)
- **Phrase match** — does the exact query phrase appear in the document?
- **Query term coverage** — what fraction of query terms appear in the document?
- **Term proximity** — how close are the query terms to each other in the document?
- **Field-level scores** — separate BM25 for title, body, headings, URL path
- **TF-IDF cosine similarity** — vector space model score

### Link-based signals

- **PageRank** — global authority based on link graph
- **HITS** (Hyperlink-Induced Topic Search) — hub and authority scores
- **Anchor text** — text of incoming links (what others say about the page)
- **Inlink count** — number of pages linking to this page
- **Domain authority** — aggregate link-based score for the entire domain

### User behavior signals

- **Click-through rate (CTR)** — for this query, how often is this result clicked?
- **Dwell time** — how long do users stay after clicking?
- **Pogo-sticking rate** — how often do users click back quickly?
- **Long click ratio** — fraction of clicks with dwell time > 30s
- **Session success rate** — does the user stop searching after visiting this result?

### Document quality signals

- **Content length** — very short pages are often low quality
- **Spam score** — classifier output for web spam detection
- **Ad density** — ratio of ad content to main content
- **Reading level** — Flesch-Kincaid or similar metrics
- **Structured data** — presence of schema.org markup
- **Mobile friendliness** — responsive design, page speed

### Freshness signals

- **Document creation date**
- **Last modification date**
- **Crawl freshness** — how recently was this page recrawled?
- **Content change rate** — how often does this page change?

### Personalization signals

- **Query history** — user's past queries (short-term and long-term)
- **Click history** — pages the user has clicked before
- **Geographic location**
- **Language preference**
- **Device type** (mobile vs. desktop)

## The Cold Start Problem

A new search system faces a chicken-and-egg problem:

1. To rank well, you need user behavior data (clicks, dwell time)
2. To get user behavior data, you need users
3. To get users, you need to rank well

**Solutions for cold start:**

**Phase 1: Text match only.** Start with BM25 + basic document quality signals. No personalization, no ML ranking. This is surprisingly effective for most domains.

**Phase 2: Editorial judgments.** Collect 1,000-5,000 judged query-document pairs. Use these to train a simple L1 ranker (GBDT with 10-20 features). This gets you from "basic" to "decent."

**Phase 3: Implicit feedback.** Once you have users, start logging click data. After collecting enough (typically 100K+ queries with clicks), train models that incorporate behavioral features. This is where the biggest quality jumps happen.

**Phase 4: Advanced ML.** With millions of queries of behavioral data, train cross-encoders, add personalization, implement advanced query understanding. This is Google/Amazon-level search.

The key insight: **you don't need ML to launch.** BM25 with good text processing (Chapter 01) and basic field boosting is a solid starting point for any domain.

## References

- Robertson, S. E. — "The Probability Ranking Principle in IR" (Journal of Documentation, 1977)
- Saracevic, T. — "Relevance: A Review of the Literature and a Framework for Thinking on the Notion in Information Science" (JASIST, 2007)
- Chapelle, O., Metlzer, D., Zhang, Y., & Grinspan, P. — "Expected Reciprocal Rank for Graded Relevance" (CIKM, 2009)
- Carbonell, J. & Goldstein, J. — "The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries" (SIGIR, 1998)
- Wang, X., et al. — "Position Bias Estimation for Unbiased Learning to Rank in Personal Search" (WSDM, 2018)
- Liu, T.-Y. — *Learning to Rank for Information Retrieval* (Springer, 2011)
