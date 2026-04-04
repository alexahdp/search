# Evaluation Metrics

You can't improve what you can't measure. Search evaluation metrics quantify how well a ranking satisfies the user's information need. This section covers every major metric you'll encounter in practice, with worked examples to build intuition, and discusses the infrastructure around evaluation: offline vs. online, statistical significance, and building evaluation sets.

## Table of Contents

1. [Set-Based Metrics](#set-based-metrics)
2. [Rank-Based Metrics](#rank-based-metrics)
3. [Worked Example: All Metrics on One Ranking](#worked-example-all-metrics-on-one-ranking)
4. [Online Evaluation](#online-evaluation)
5. [Statistical Significance](#statistical-significance)
6. [Building an Evaluation Pipeline](#building-an-evaluation-pipeline)
7. [Metric Selection Guide](#metric-selection-guide)
8. [References](#references)

---

## Set-Based Metrics

These metrics treat the result set as an **unordered** set of documents. They don't care about ranking order — only about which documents were returned.

### Precision

**Precision at $k$ ($P@k$)**: Of the top $k$ results returned, what fraction is relevant?

$$P@k = \frac{|\{\text{relevant documents in top } k\}|}{k}$$

**Example**: Your system returns 10 results for "inverted index implementation." You judge each:

```
Position:  1  2  3  4  5  6  7  8  9  10
Relevant:  ✓  ✓  ✗  ✓  ✗  ✗  ✓  ✗  ✗  ✓

P@5  = 3/5 = 0.60
P@10 = 5/10 = 0.50
```

**When to use**: When users look at a fixed number of results (e.g., a SERP showing 10 blue links). $P@1$ is especially important for navigational queries ("did we nail the first result?").

**Limitation**: Doesn't reward ranking relevant results higher within the top $k$. The rankings `[R, R, N, N, N]` and `[N, N, N, R, R]` both have $P@5 = 0.4$.

### Recall

**Recall at $k$ ($R@k$)**: Of all relevant documents in the corpus, what fraction is in the top $k$?

$$R@k = \frac{|\{\text{relevant documents in top } k\}|}{|\{\text{all relevant documents}\}|}$$

**Example**: Same query, suppose there are 8 relevant documents total in the corpus:

```
R@5  = 3/8 = 0.375
R@10 = 5/8 = 0.625
```

**When to use**: When missing a relevant document is costly. Patent search, legal discovery, medical literature review — domains where you need high recall.

**Limitation**: Requires knowing the total number of relevant documents in the corpus. In web search, this is unknowable (you can't judge billions of documents). Typically estimated via pooling — combine the top-k results from multiple systems and judge those.

### F1 Score

The harmonic mean of precision and recall:

$$F_1 = 2 \cdot \frac{P \cdot R}{P + R}$$

More generally, $F_\beta$ weights recall $\beta$ times more than precision:

$$F_\beta = (1 + \beta^2) \cdot \frac{P \cdot R}{\beta^2 \cdot P + R}$$

- $F_1$: Equal weight to precision and recall
- $F_2$: Recall is twice as important (use for legal/medical search)
- $F_{0.5}$: Precision is twice as important (use when false positives are costly)

**Limitation for search**: F1 treats retrieval as a binary classification problem (retrieved or not). It ignores ranking, which is the whole point of search. Use F1 for evaluating retrieval stages (candidate selection), not final rankings.

### Precision-Recall Trade-off

There's an inherent tension:
- To increase recall, retrieve more documents → precision drops (you include marginal results)
- To increase precision, retrieve fewer, more confident results → recall drops (you miss good results)

The **precision-recall curve** plots this trade-off. Compute precision and recall at every rank position, then plot:

```
Precision
1.0 ┤●
    │ ╲
0.8 ┤  ╲●
    │    ╲
0.6 ┤     ●───●
    │          ╲
0.4 ┤           ╲●
    │             ╲
0.2 ┤              ●───●
    │
0.0 ┼──┬──┬──┬──┬──┬──┬──
    0  0.2 0.4 0.6 0.8 1.0
              Recall
```

The area under this curve is **Average Precision** — one of the most important metrics.

## Rank-Based Metrics

These metrics care about **where** in the ranking relevant documents appear. They reward systems that place relevant documents higher.

### Mean Reciprocal Rank (MRR)

**Reciprocal Rank (RR)**: The reciprocal of the position of the **first** relevant result:

$$RR = \frac{1}{\text{rank of first relevant result}}$$

**Mean Reciprocal Rank (MRR)**: Average RR across a set of queries:

$$MRR = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$

**Example**: Three queries:

```
Query 1: [N, R, N, N, N]  → RR = 1/2 = 0.500
Query 2: [R, N, N, N, N]  → RR = 1/1 = 1.000
Query 3: [N, N, N, R, N]  → RR = 1/4 = 0.250

MRR = (0.500 + 1.000 + 0.250) / 3 = 0.583
```

**When to use**: When only the **first** relevant result matters. Navigational queries ("facebook login"), question answering (one correct answer), autocomplete (user picks the first good suggestion).

**Limitation**: Completely ignores relevant results after the first one. If your system returns `[R, N, N, N, N]` and `[R, R, R, R, R]`, both have RR = 1.0.

### Average Precision (AP)

AP captures both precision and recall in a single number that rewards placing relevant results earlier. It's the area under the precision-recall curve, computed as:

$$AP = \frac{1}{|\text{Rel}|} \sum_{k=1}^{n} P@k \cdot \text{rel}(k)$$

where $|\text{Rel}|$ is the total number of relevant documents, $n$ is the total number of results, $P@k$ is precision at position $k$, and $\text{rel}(k)$ is 1 if the result at position $k$ is relevant, 0 otherwise.

**Worked example**: Ranking with 4 relevant documents total in the corpus:

```
Position:  1   2   3   4   5   6   7   8   9  10
Relevant:  ✓   ✗   ✓   ✗   ✗   ✓   ✗   ✗   ✗   ✗

Only compute P@k at positions where rel(k) = 1:

Position 1: P@1 = 1/1 = 1.000  (1 relevant in top 1)
Position 3: P@3 = 2/3 = 0.667  (2 relevant in top 3)
Position 6: P@6 = 3/6 = 0.500  (3 relevant in top 6)

The 4th relevant document is not in the top 10. It contributes 0 to the sum.

AP = (1.000 + 0.667 + 0.500 + 0) / 4 = 0.542
```

Note: the denominator is 4 (total relevant docs), not 3 (found docs). This penalizes systems with low recall.

### Mean Average Precision (MAP)

Average the AP across all queries in the evaluation set:

$$MAP = \frac{1}{|Q|} \sum_{q=1}^{|Q|} AP(q)$$

**When to use**: The single most widely used metric for evaluating ranked retrieval with **binary** relevance judgments. Standard in TREC evaluations.

**Limitation**: Requires binary relevance (relevant or not). Doesn't distinguish between "somewhat relevant" and "perfectly relevant." For graded relevance, use NDCG.

### Discounted Cumulative Gain (DCG)

DCG was designed for **graded** relevance judgments. It captures two intuitions:

1. **Highly relevant documents are more useful than marginally relevant ones** (via gain values)
2. **Documents lower in the ranking are less likely to be examined** (via a logarithmic discount)

$$DCG@k = \sum_{i=1}^{k} \frac{rel_i}{\log_2(i + 1)}$$

where $rel_i$ is the relevance grade of the document at position $i$.

An alternative formulation (used in most modern systems) that emphasizes highly relevant documents:

$$DCG@k = \sum_{i=1}^{k} \frac{2^{rel_i} - 1}{\log_2(i + 1)}$$

With this formulation, a document with grade 4 contributes $2^4 - 1 = 15$ gain, while grade 2 contributes only $2^2 - 1 = 3$. This heavily rewards placing the best results at the top.

**Worked example** (using the exponential formulation):

```
Position:  1    2    3    4    5
Grade:     3    2    0    1    3

DCG@5 = (2³-1)/log₂(2) + (2²-1)/log₂(3) + (2⁰-1)/log₂(4) + (2¹-1)/log₂(5) + (2³-1)/log₂(6)
      = 7/1.000 + 3/1.585 + 0/2.000 + 1/2.322 + 7/2.585
      = 7.000 + 1.893 + 0.000 + 0.431 + 2.708
      = 12.031
```

### Normalized DCG (NDCG)

DCG values are query-dependent (a query with many highly relevant documents naturally has higher DCG). To compare across queries, normalize by the **ideal DCG** (IDCG) — the DCG of the perfect ranking:

$$NDCG@k = \frac{DCG@k}{IDCG@k}$$

**Continuing the example:**

```
Ideal ranking (sort by grade descending):
Position:  1    2    3    4    5
Grade:     3    3    2    1    0

IDCG@5 = 7/1.000 + 7/1.585 + 3/2.000 + 1/2.322 + 0/2.585
       = 7.000 + 4.416 + 1.500 + 0.431 + 0.000
       = 13.347

NDCG@5 = 12.031 / 13.347 = 0.901
```

$NDCG \in [0, 1]$, where 1.0 means the ranking is perfect (matches the ideal order).

**When to use**: The de facto standard for graded relevance evaluation. Used at Google, Bing, Amazon, Netflix — any system with multi-level relevance judgments. Most learning-to-rank models optimize NDCG directly.

**Which variant of DCG?** Use $\frac{2^{rel} - 1}{\log_2(i+1)}$ (exponential gain) unless you have a specific reason not to. It's the standard in TREC, and it appropriately emphasizes the difference between highly relevant and marginally relevant results.

### Expected Reciprocal Rank (ERR)

ERR models a more realistic user behavior than NDCG: a user scans results from top to bottom and stops when they find a satisfying result. The probability of stopping at position $i$ depends on the relevance of results at positions $1$ through $i$.

$$ERR = \sum_{i=1}^{n} \frac{1}{i} \prod_{j=1}^{i-1} (1 - R_j) \cdot R_i$$

where $R_i$ is the probability that the user is satisfied by the document at position $i$, typically mapped from grades:

$$R_i = \frac{2^{g_i} - 1}{2^{g_{\max}}}$$

ERR has a nice property: unlike NDCG, it accounts for **inter-document dependencies**. A highly relevant result at position 3 is less valuable if position 1 already has a perfect result (because the user probably stopped there).

**When to use**: When the user is looking for a single answer (navigational queries, question answering). ERR was the primary metric at Yahoo and is used in TREC Web Track.

## Worked Example: All Metrics on One Ranking

Let's compute every metric on a single ranking to build intuition.

**Query**: "BM25 parameter tuning"
**Total relevant documents in corpus**: 6 (graded: two with grade 3, two with grade 2, two with grade 1)

**System's ranking** (top 10):

```
Pos   Doc    Grade   Binary (grade ≥ 1)
 1    d_14     3       R
 2    d_07     0       N
 3    d_22     2       R
 4    d_03     0       N
 5    d_41     1       R
 6    d_09     3       R
 7    d_55     0       N
 8    d_31     0       N
 9    d_17     2       R
10    d_88     0       N

(The 6th relevant document, with grade 1, is not in the top 10)
```

### Set-based metrics

```
P@5  = 3/5 = 0.600
P@10 = 5/10 = 0.500
R@5  = 3/6 = 0.500
R@10 = 5/6 = 0.833
F1@10 = 2 × 0.500 × 0.833 / (0.500 + 0.833) = 0.625
```

### MRR

```
First relevant result at position 1.
RR = 1/1 = 1.000
```

### AP

```
P@1 = 1/1 = 1.000  (relevant at pos 1)
P@3 = 2/3 = 0.667  (relevant at pos 3)
P@5 = 3/5 = 0.600  (relevant at pos 5)
P@6 = 4/6 = 0.667  (relevant at pos 6)
P@9 = 5/9 = 0.556  (relevant at pos 9)
6th relevant doc not found → contributes 0

AP = (1.000 + 0.667 + 0.600 + 0.667 + 0.556 + 0) / 6 = 0.582
```

### DCG and NDCG (exponential gain)

```
DCG@10:
Pos 1: (2³-1)/log₂(2) = 7/1.000 = 7.000
Pos 2: (2⁰-1)/log₂(3) = 0/1.585 = 0.000
Pos 3: (2²-1)/log₂(4) = 3/2.000 = 1.500
Pos 4: (2⁰-1)/log₂(5) = 0/2.322 = 0.000
Pos 5: (2¹-1)/log₂(6) = 1/2.585 = 0.387
Pos 6: (2³-1)/log₂(7) = 7/2.807 = 2.493
Pos 7: (2⁰-1)/log₂(8) = 0/3.000 = 0.000
Pos 8: (2⁰-1)/log₂(9) = 0/3.170 = 0.000
Pos 9: (2²-1)/log₂(10) = 3/3.322 = 0.903
Pos 10: (2⁰-1)/log₂(11) = 0/3.459 = 0.000

DCG@10 = 12.283

Ideal ranking (grades sorted: 3,3,2,2,1,1,0,0,0,0):
Pos 1: 7/1.000 = 7.000
Pos 2: 7/1.585 = 4.416
Pos 3: 3/2.000 = 1.500
Pos 4: 3/2.322 = 1.292
Pos 5: 1/2.585 = 0.387
Pos 6: 1/2.807 = 0.356

IDCG@10 = 14.951

NDCG@10 = 12.283 / 14.951 = 0.822
```

### Summary

| Metric | Value |
|---|---|
| P@5 | 0.600 |
| P@10 | 0.500 |
| R@10 | 0.833 |
| F1@10 | 0.625 |
| MRR | 1.000 |
| AP | 0.582 |
| DCG@10 | 12.283 |
| NDCG@10 | 0.822 |

The ranking is decent: it nails position 1 (MRR = 1.0) and has good overall relevance (NDCG = 0.822). The main weakness is recall — one relevant document is missing, and the system places irrelevant documents at positions 2, 4, 7, 8, and 10.

## Online Evaluation

Offline metrics (NDCG, MAP) are computed on static judgment sets. They tell you how your system performs on a fixed evaluation set but not how real users experience it. **Online evaluation** measures quality on live traffic.

### A/B Testing

The gold standard for online evaluation. Split live traffic randomly:

- **Control (A)**: Existing system
- **Treatment (B)**: System with proposed change

Run for a sufficient duration (typically 1-4 weeks for search), then compare metrics.

**Key metrics for online search evaluation:**

| Metric | What it measures |
|---|---|
| **Abandonment rate** | Fraction of queries with no clicks |
| **CTR@1** | Click-through rate on the first result |
| **Time to first click** | How quickly users find something useful |
| **Session success rate** | Fraction of sessions ending without query reformulation |
| **Pogo-sticking rate** | Quick back-clicks (user was disappointed) |
| **Queries per session** | Fewer is usually better (user found what they needed) |
| **Revenue per search** | For e-commerce: direct business impact |

**Common pitfalls:**

1. **Novelty effect**: Users may click more on new UI elements simply because they're novel. Wait for the effect to stabilize.
2. **Day-of-week effects**: Traffic patterns vary (Mon-Fri vs. weekends). Run experiments for full weeks.
3. **Interaction effects**: Multiple simultaneous experiments can interfere. Use orthogonal experiment designs.
4. **Simpson's paradox**: A change might help on every segment but hurt overall (or vice versa) due to segment size shifts.

### Interleaving

A more sensitive alternative to A/B testing for ranking evaluation. Instead of showing different rankings to different users, **interleave** results from systems A and B into a single ranking. Users implicitly "vote" for one system by clicking on its results.

**Team Draft interleaving** (the simplest algorithm):

```
Given rankings A = [a1, a2, a3, ...] and B = [b1, b2, b3, ...]:

1. Flip a coin to decide which team "picks first"
2. The first team adds its top-ranked unpicked document to the interleaved list
3. Alternate picks between teams
4. Track which team each result belongs to

Result: [a1, b1, a2, b2, ...] or [b1, a1, b2, a2, ...] (randomized)

After users interact:
- Count clicks on team A's results → score_A
- Count clicks on team B's results → score_B
- System with more clicks wins
```

Interleaving is 10-100x more sensitive than A/B testing (detects smaller differences with less traffic) because each user directly compares both systems.

## Statistical Significance

When you see "NDCG improved from 0.42 to 0.44," is that a real improvement or random noise?

### Paired t-test

For metrics computed per-query (AP, NDCG), use a **paired t-test** because you have paired observations (same query, two systems):

$$t = \frac{\bar{d}}{s_d / \sqrt{n}}$$

where $\bar{d}$ is the mean difference in per-query scores, $s_d$ is the standard deviation of differences, and $n$ is the number of queries.

With $p < 0.05$ (or $p < 0.01$ for higher confidence), reject the null hypothesis that the systems perform equally.

### Bootstrap Test

A non-parametric alternative (makes no distributional assumptions):

```
1. Given per-query metric values for systems A and B
2. Sample n queries WITH replacement (bootstrap sample)
3. Compute the metric difference on the bootstrap sample
4. Repeat 10,000+ times
5. If the 95% confidence interval of differences excludes 0,
   the difference is significant
```

Bootstrap is preferred when:
- The metric distribution is skewed (MAP values are often left-skewed)
- Sample sizes are small
- You want confidence intervals, not just p-values

### Effect Size

Statistical significance alone isn't enough. An improvement can be statistically significant but practically meaningless (e.g., NDCG improves by 0.001 on 100K queries — significant but nobody notices).

**Cohen's d**: Measures effect size in standard deviation units:

$$d = \frac{\bar{d}}{s_d}$$

| d value | Interpretation |
|---|---|
| 0.2 | Small effect |
| 0.5 | Medium effect |
| 0.8 | Large effect |

In search, a "meaningful" improvement is typically:
- NDCG: +0.01 to +0.02 (1-2 points) for a mature system
- MAP: similar magnitude
- MRR: +0.02 to +0.05

For a new system, improvements of +0.05 to +0.10 are common. For Google/Bing-level systems, even +0.005 is celebrated.

### Multiple Comparisons

When comparing multiple system variants, the probability of at least one false positive increases:

$$P(\text{at least one false positive}) = 1 - (1 - \alpha)^m$$

With $\alpha = 0.05$ and $m = 20$ comparisons: $P = 1 - 0.95^{20} = 0.64$. A 64% chance of at least one false positive!

**Corrections:**
- **Bonferroni**: Use $\alpha' = \alpha / m$. Conservative (too strict for large $m$).
- **Holm-Bonferroni**: Step-down procedure. Less conservative.
- **Benjamini-Hochberg (FDR)**: Controls false discovery rate rather than family-wise error rate. Standard in industry.

## Building an Evaluation Pipeline

A practical evaluation pipeline for a production search system:

### Query set construction

1. **Sample from query logs**: Take a random sample of real queries, stratified by frequency:
   - **Head queries** (top 1% by volume): Cover the most traffic
   - **Torso queries** (next 19%): Good coverage of common patterns
   - **Tail queries** (bottom 80%): Where most unique queries live

2. **Include navigational, informational, and transactional queries** in proportion to your traffic mix.

3. **Size**: 2,000-10,000 queries for a comprehensive evaluation set. 500-1,000 for a "quick eval" set.

### Judgment collection

1. **Depth**: Judge at least top 10-20 results per query. For metrics like MAP that penalize missing relevant documents, use pooling: collect top-20 from multiple system variants and judge the union.

2. **Redundancy**: At least 3 judges per (query, document) pair. Use majority vote or Dawid-Skene aggregation.

3. **Quality control**: Include 10% gold questions (known-answer pairs). Remove judges with < 80% accuracy on gold questions.

### Evaluation cadence

- **Every code change**: Run on the quick eval set (500 queries). Automated in CI.
- **Weekly**: Run on the full eval set (5,000+ queries). Report to the team.
- **Quarterly**: Refresh the query set and judgments to avoid overfitting.

### Pitfalls

1. **Evaluation set overfitting**: If you repeatedly optimize against the same query set, you'll overfit to its quirks. Refresh regularly.
2. **Judgment drift**: Rater standards change over time. Re-judge a subset periodically to calibrate.
3. **Pooling bias**: If you only judge documents returned by your systems, you'll never discover relevant documents that no system found. This systematically underestimates recall.
4. **Position bias in judgments**: Judges may rate higher-ranked documents more favorably. Randomize presentation order.

## Metric Selection Guide

| Your scenario | Primary metric | Secondary metrics |
|---|---|---|
| Web search (general) | NDCG@10 | ERR@10, P@1, Abandonment rate |
| E-commerce product search | NDCG@10, Revenue/search | CTR@1, Conversion rate, R@100 |
| Question answering | MRR | P@1, Exact match, R@10 |
| Legal / patent search | MAP, R@1000 | F2 score (recall-weighted) |
| Recommendation | NDCG@20 | MAP, Catalog coverage, Diversity |
| Autocomplete | MRR | P@1, Keystroke savings |
| Document retrieval (candidate stage) | R@1000 | R@100, P@100 |
| Known-item search (navigational) | MRR | P@1, Success@1 |

**Rules of thumb:**

1. **Binary relevance → MAP**. Graded relevance → NDCG.
2. **Only first result matters → MRR.** Whole page matters → NDCG@10 or MAP.
3. **High-recall required → R@k + F2.** High-precision required → P@k + F0.5.
4. **Always report multiple metrics.** A single number hides important behavior.
5. **Always report online metrics alongside offline metrics.** Offline NDCG can improve while user satisfaction drops (and vice versa).

## References

- Järvelin, K. & Kekäläinen, J. — "Cumulated Gain-Based Evaluation of IR Techniques" (ACM TOIS, 2002)
- Voorhees, E. M. — "The Philosophy of Information Retrieval Evaluation" (CLEF, 2001)
- Chapelle, O., et al. — "Expected Reciprocal Rank for Graded Relevance" (CIKM, 2009)
- Radlinski, F. & Craswell, N. — "Optimized Interleaving for Online Retrieval Evaluation" (WSDM, 2013)
- Sanderson, M. — "Test Collection Based Evaluation of Information Retrieval Systems" (Foundations and Trends in IR, 2010)
- Sakai, T. — "Statistical Significance, Power, and Sample Sizes: A Systematic Review of SIGIR and TOIS, 2006-2015" (SIGIR, 2016)
- Kohavi, R., Tang, D., & Xu, Y. — *Trustworthy Online Controlled Experiments: A Practical Guide to A/B Testing* (Cambridge University Press, 2020)
