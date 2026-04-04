# Query Processing

Query processing is everything that happens between the user pressing Enter and the retrieval engine receiving a structured, normalized query ready for matching against the index. This pipeline can make or break search quality — a poorly processed query can doom even the best ranking algorithm.

## Table of Contents

1. [The Query Processing Pipeline](#the-query-processing-pipeline)
2. [Query Parsing and Tokenization](#query-parsing-and-tokenization)
3. [Query Normalization](#query-normalization)
4. [Query Expansion](#query-expansion)
5. [Query Rewriting](#query-rewriting)
6. [Spell Correction](#spell-correction)
7. [Intent Classification](#intent-classification)
8. [Structured Query Generation](#structured-query-generation)
9. [Performance Considerations](#performance-considerations)
10. [References](#references)

---

## The Query Processing Pipeline

The typical flow from raw user input to structured query:

```
┌─────────────────┐
│   Raw query      │  "recmmend laptop for machin learning"
│   (user input)   │
└────────┬─────────┘
         │
         ▼
┌─────────────────┐
│  Spell check     │  "recommend laptop for machine learning"
│  & correction    │
└────────┬─────────┘
         │
         ▼
┌─────────────────┐
│  Tokenization    │  ["recommend", "laptop", "for", "machine", "learning"]
└────────┬─────────┘
         │
         ▼
┌─────────────────┐
│  Normalization   │  ["recommend", "laptop", "machine", "learn"]
│  (case, stem,    │  (lowercased, "for" removed, "learning" → "learn")
│   stopwords)     │
└────────┬─────────┘
         │
         ▼
┌─────────────────┐
│  Intent          │  Intent: TRANSACTIONAL (shopping)
│  classification  │  Entity: Product type = "laptop"
└────────┬─────────┘
         │
         ▼
┌─────────────────┐
│  Query           │  Add synonyms: ["recommend", "suggest"]
│  expansion       │  Add related: ["notebook", "laptop"]
└────────┬─────────┘
         │
         ▼
┌─────────────────┐
│  Structured      │  {
│  query           │    "text": ["recommend", "laptop", "machine", "learn"],
│  generation      │    "filters": {"category": "electronics"},
│                  │    "boosts": {"field": "title", "weight": 3.0},
│                  │    "synonyms": ["notebook"],
│                  │    "intent": "TRANSACTIONAL"
│                  │  }
└────────┬─────────┘
         │
         ▼
┌─────────────────┐
│  Retrieval       │  → Execute against index
│  (BM25/ANN)      │
└──────────────────┘
```

Not all queries go through every stage. Simple queries ("github") might only need tokenization and normalization. Complex queries ("best gaming laptop under $1000 for machine learning") benefit from the full pipeline.

## Query Parsing and Tokenization

### Tokenization

Apply the **same tokenizer** used at indexing time. If you indexed with the "standard" tokenizer (Unicode-aware whitespace + punctuation splitting), use it at query time too. Mismatches cause silent failures:

**Index time**: "high-performance" → ["high", "performance"]
**Query time** (if different tokenizer): "high-performance" → ["high-performance"]
**Result**: No matches, even though the content exists.

### Special Syntax Handling

Many search systems support advanced syntax:

**Field-specific queries**: `title:search` (match "search" in title field only)

**Phrase queries**: `"search engine"` (match exact phrase with terms adjacent)

**Boolean operators**: `search AND engine`, `python OR java`

**Wildcards**: `search*` (prefix match), `se?rch` (single-character wildcard)

**Ranges**: `price:[100 TO 500]`, `date:[2020-01-01 TO *]`

**Boosts**: `search^2 engine` (boost "search" 2×)

**Fuzzy**: `search~` (allow edit distance of 1-2)

**Example from Elasticsearch query string**:

```
title:"machine learning"^3 AND (python OR java) AND date:[2020 TO *]
```

The parser must:
1. Identify special syntax (quoted phrases, field prefixes, operators)
2. Tokenize the text parts (terms inside phrases, standalone terms)
3. Build a structured query AST (Abstract Syntax Tree)

### Query Language Options

**Simple query string** (user-friendly):
- No Boolean operators required
- `+` = must match, `-` = must not match
- `|` = OR, `"..."` = phrase
- Example: `+machine +learning -deep "neural network"`

**Full Lucene query syntax** (power users):
- Boolean operators: `AND`, `OR`, `NOT`, `()`
- Field queries: `field:term`
- All advanced features enabled
- Example: `(title:search^2 OR body:search) AND NOT spam:true`

**Natural language** (default for most web search):
- No special syntax
- Everything is treated as terms
- Operators like "AND" are treated as stopwords and ignored
- Example: "best laptop for machine learning" → ["best", "laptop", "machine", "learning"]

## Query Normalization

Apply the **same normalization** as at index time.

### Case Folding

Lowercase all terms (unless you have a case-sensitive domain like code search):

```
"Machine Learning" → "machine learning"
```

### Stemming

Reduce terms to their root form (see [Text Representation](../01-fundamentals/text-representation.md#stemming)):

```
"running shoes" → "run shoe"
"searched" → "search"
```

**Critical**: Use the **exact same stemmer** as at index time. Porter at index + Snowball at query = disaster.

### Stop Word Removal

Historically, stop words ("the", "a", "and") were removed from queries. Modern practice: **keep them**, especially for:

1. **Phrase queries**: "to be or not to be" — removing stops destroys meaning
2. **Short queries**: "the who" (band name) becomes empty
3. **Position-sensitive matching**: Stop words affect term proximity

BM25 naturally down-weights common terms via IDF. Explicit removal is no longer necessary and often harmful.

**Exception**: Remove stops from very long queries (10+ terms) to reduce query latency, but only if you're sure it won't break phrases.

### Unicode Normalization

Queries may contain accented characters, ligatures, or equivalent Unicode representations:

```
"café" (U+0063 U+0061 U+0066 U+00E9)  vs.
"café" (U+0063 U+0061 U+0066 U+0065 U+0301)  — composed vs. decomposed
```

Apply the same Unicode normalization as at index time (typically **NFKC** or **NFC**). See [Text Representation](../01-fundamentals/text-representation.md#unicode-normalization).

## Query Expansion

Query expansion adds terms to the original query to improve recall. The user typed "laptop" — you expand to include "notebook," "notebook computer," "portable computer."

### Synonym Expansion

**At query time** (most common):
```
Query: "laptop"
Expanded: "laptop OR notebook OR 'notebook computer'"
```

Elasticsearch syntax:

```json
{
  "query": {
    "match": {
      "description": {
        "query": "laptop",
        "auto_generate_synonyms_phrase_query": true
      }
    }
  }
}
```

With a synonym filter containing: `laptop, notebook, notebook computer`, the system automatically expands the query.

**At index time** (alternative):

Index each document with synonyms included:

```
Original text: "This laptop is great."
Indexed as: "This laptop notebook is great."
```

**Trade-off**:
- **Query-time expansion**: Flexible (can change synonym dictionaries without reindexing), but slower (more terms to match)
- **Index-time expansion**: Fast at query time, but inflexible (changing synonyms requires full reindex) and increases index size

### Pseudo-Relevance Feedback (PRF)

**Idea**: Assume the top-k results from an initial retrieval are relevant, extract the most characteristic terms from those results, and add them to the query.

**Algorithm (Rocchio)**:

1. Execute the original query, retrieve top-$k$ documents (typically $k = 5-10$)
2. Compute term weights for terms in those documents (TF-IDF or similar)
3. Select the top-$m$ terms with highest weights (typically $m = 5-20$)
4. Add those terms to the query with reduced weight
5. Re-execute the expanded query

**Example**:

Original query: "machine learning"

Top-5 results contain frequent terms: "algorithm", "neural", "training", "model", "supervised"

Expanded query: "machine learning algorithm neural training" (with weights)

**When it helps**: Long-tail queries, ambiguous queries where context clarification helps, exploratory search.

**When it hurts**: Short navigational queries ("facebook"), already-precise queries (adding noise), queries where top results are poor (garbage in, garbage out).

**Production use**: Rare in web search (too slow, too risky). Common in academic search, medical literature search (PubMed uses PRF variants).

### Acronym and Abbreviation Expansion

"ML" → "machine learning"
"NLP" → "natural language processing"
"JS" → "JavaScript"

Maintain a dictionary of domain-specific abbreviations. Be careful with ambiguity:

- "ML" = machine learning or maximum likelihood or major league?
- "AI" = artificial intelligence or Adobe Illustrator?

Context (query terms, user history, corpus domain) helps disambiguate.

### Morphological Expansion

Go beyond stemming to include related forms:

"run" → ["run", "ran", "running", "runner", "runs"]

This is typically handled by **lemmatization** (see [Text Representation](../01-fundamentals/text-representation.md#lemmatization)). Less aggressive than synonyms, more aggressive than stemming.

## Query Rewriting

Query rewriting transforms the query structure, not just the terms.

### Query Relaxation

When a query returns zero or very few results, relax constraints:

**Original**: `laptop AND gaming AND "under $1000"`
**Relaxed**: `laptop AND gaming` (drop the price constraint)
**Further relaxed**: `laptop OR gaming` (change AND to OR)

**Elasticsearch** automatically does this with the `minimum_should_match` parameter:

```json
{
  "query": {
    "match": {
      "description": {
        "query": "gaming laptop under 1000",
        "minimum_should_match": "75%"
      }
    }
  }
}
```

This requires 3 out of 4 terms to match (75%). If no results, Elasticsearch implicitly relaxes to 50%, then to 1 term.

### Phrase to Proximity

Transform strict phrase queries into proximity queries for better recall:

**Original**: `"machine learning"` (exact phrase)
**Rewritten**: `machine NEAR/3 learning` (within 3 words)

This matches:
- "machine learning"
- "machine and deep learning"
- "learning about machine"

BM25 doesn't natively support proximity scoring, but Elasticsearch's `match_phrase` with `slop` parameter does:

```json
{
  "match_phrase": {
    "text": {
      "query": "machine learning",
      "slop": 3
    }
  }
}
```

### Query Segmentation

For long queries (especially verbose natural language queries), segment into clauses and treat them differently:

**Query**: "What are the best programming languages for machine learning in 2024?"

**Segmentation**:
- Core terms: ["programming", "languages", "machine", "learning"]
- Temporal modifier: ["2024"] → add freshness boost
- Stopwords/question words: ["what", "are", "the", "best", "for", "in"] → discard or de-emphasize

Search engines increasingly use **query understanding models** (neural networks) to perform this segmentation automatically.

## Spell Correction

Spell correction is critical for web search (15-25% of queries contain typos) but less important for expert users in specialized domains.

### When to Correct

**Auto-correct** (Google's "Did you mean?"):
- Query is misspelled with high confidence
- Corrected query returns significantly more/better results
- Example: "speling" → "spelling"

**Suggest** (don't auto-correct, just show suggestion):
- Query is ambiguous ("their" vs "there")
- Query returns results, but a corrected version might be better
- User might have intentionally typed the "wrong" version (brand names, slang)

### Correction Approaches

**Edit distance-based**:

Use Levenshtein (edit) distance to find candidate corrections:

1. For each query term, generate candidates within edit distance 1-2
2. Filter candidates by dictionary (valid words or indexed terms)
3. Rank by frequency (common words preferred)
4. Compute probability: $P(\text{candidate} \mid \text{typo})$

**Example**: "machne" → candidates: ["machine", "machete"] → "machine" (more frequent)

**N-gram based (Peter Norvig's algorithm)**:

See https://norvig.com/spell-correct.html — simple and effective.

**Probabilistic (noisy channel model)**:

$$P(\text{correction} \mid \text{query}) \propto P(\text{query} \mid \text{correction}) \times P(\text{correction})$$

where:
- $P(\text{correction})$ = language model probability (how common is the word?)
- $P(\text{query} \mid \text{correction})$ = error model (how likely is this typo?)

This is Bayes' theorem. The error model can be trained from query logs (what corrections do users click?).

**Neural models**:

Encoder-decoder models (seq2seq) trained on pairs of (misspelled query, corrected query) from click logs. Modern systems (Google, Bing) use BERT-based models for context-aware correction:

**Context-agnostic**: "their going" → "their going" (both are valid words, no correction)

**Context-aware**: "their going" → "they're going" (model understands grammar)

### Spell Correction in Elasticsearch

Elasticsearch offers **suggester APIs**:

**Term suggester**: Edit distance-based, per-term:

```json
{
  "suggest": {
    "text": "machne lerning",
    "my_suggestion": {
      "term": {
        "field": "content"
      }
    }
  }
}
```

**Phrase suggester**: Context-aware, considers term co-occurrence:

```json
{
  "suggest": {
    "text": "machne lerning",
    "my_suggestion": {
      "phrase": {
        "field": "content",
        "gram_size": 2
      }
    }
  }
}
```

**Completion suggester**: For typeahead/autocomplete (covered in Chapter 03).

### When NOT to Correct

1. **Named entities**: "Krzyzewski" (Duke basketball coach) looks misspelled but isn't
2. **Code / technical terms**: "strcmp", "malloc", "async/await"
3. **Rare/emerging terms**: New slang, product names (iPhone was flagged as misspelled when it launched)
4. **Domain-specific jargon**: Medical terms, legal terms

Solution: Maintain a **whitelist** of domain-specific terms that should never be corrected.

## Intent Classification

Understanding what the user wants to **do** with the query.

### Intent Taxonomies

**Common taxonomy** (from Broder, 2002):

1. **Navigational**: User wants a specific website/page
   - Examples: "facebook login", "amazon", "github"
   - Success metric: Did they click the first result?

2. **Informational**: User wants to learn something
   - Examples: "how does BM25 work", "python tutorial", "weather"
   - Success metric: Did they find a satisfying answer? (dwell time)

3. **Transactional**: User wants to perform an action
   - Examples: "buy iPhone 15", "download python", "book hotel Paris"
   - Success metric: Did they convert? (purchase, download, booking)

**E-commerce-specific intents**:

- **Product search**: "laptop 16GB RAM"
- **Brand navigation**: "Dell XPS"
- **Comparison**: "iPhone vs Samsung"
- **Research**: "best laptop for video editing"

### Why Intent Matters

Different intents need different ranking functions and result layouts:

**Navigational**:
- **Ranking**: Heavily weight domain authority, exact URL matches
- **Result format**: Show the homepage as #1, maybe site links below

**Informational**:
- **Ranking**: Weight content depth, comprehensiveness, freshness
- **Result format**: Featured snippet, knowledge panel, "People also ask"

**Transactional**:
- **Ranking**: Weight conversion likelihood, product availability, price
- **Result format**: Product cards, reviews, "Add to cart" buttons

### Intent Classification Approaches

**Rule-based**:

```python
if query.startswith("buy ") or "purchase" in query:
    intent = "TRANSACTIONAL"
elif query in ["facebook", "youtube", "amazon", ...]:
    intent = "NAVIGATIONAL"
elif query.startswith("how to") or "what is" in query:
    intent = "INFORMATIONAL"
```

Simple but brittle. Fails on "facebook news" (informational, not navigational).

**ML-based** (modern approach):

Train a classifier (logistic regression, GBDT, or BERT) on labeled queries:

**Features**:
- Query length (navigational queries are typically 1-2 words)
- Presence of question words ("how", "what", "why")
- Presence of action words ("buy", "download")
- Query category (from classification model)
- Historical click patterns (do users click e-commerce sites or informational pages?)

**Training data**: Label 10K-100K queries by hand (or use click logs as weak supervision).

**Production systems**: Google, Bing, and Amazon all use BERT-based intent classifiers trained on millions of queries.

### Query Categorization

Map queries to topic categories: "Sports", "Technology", "Entertainment", etc.

**Use cases**:
- Route query to a specialized index (e.g., product search vs. content search)
- Apply category-specific ranking models
- Personalization (boost categories the user historically engages with)

**Approaches**:
- **Keyword matching**: "laptop" → "Technology"
- **Classification model**: Train on labeled queries
- **Zero-shot classification**: Use pre-trained models (BERT + NLI) to classify without training data

## Structured Query Generation

The final output of query processing is a **structured query** that the retrieval engine can execute.

### Elasticsearch Query DSL Example

Raw query: "best laptop under $1000"

After processing:

```json
{
  "query": {
    "bool": {
      "must": [
        {
          "multi_match": {
            "query": "laptop",
            "fields": ["title^3", "description", "brand^2"],
            "type": "best_fields"
          }
        }
      ],
      "should": [
        { "match": { "description": "best" }},
        { "match": { "category": "laptops" }}
      ],
      "filter": [
        { "range": { "price": { "lte": 1000 }}}
      ]
    }
  },
  "sort": [
    { "popularity": { "order": "desc" }},
    "_score"
  ]
}
```

**Key transformations**:
- "laptop" → multi-field match with field boosts
- "best" → soft requirement (should clause, not must)
- "under $1000" → structured filter on price field
- Implicit ranking by popularity + relevance score

### Query Understanding Output

Modern systems (Google, Amazon) output a **query understanding object**:

```json
{
  "original_query": "best laptop under $1000",
  "corrected_query": null,
  "intent": "TRANSACTIONAL",
  "entities": [
    {
      "text": "laptop",
      "type": "PRODUCT_TYPE",
      "category": "electronics"
    }
  ],
  "constraints": [
    {
      "field": "price",
      "operator": "<=",
      "value": 1000
    }
  ],
  "modifiers": [
    {
      "type": "QUALITY",
      "term": "best",
      "boost": 1.2
    }
  ],
  "expanded_terms": ["notebook", "laptop computer"],
  "segments": {
    "core": ["laptop"],
    "filters": ["under $1000"],
    "qualifiers": ["best"]
  }
}
```

This structured representation can be used for:
- Generating faceted search UI (show price filter based on detected constraint)
- Query logging and analysis
- Personalization (adjust based on user profile)
- Multi-stage ranking (different stages use different parts)

## Performance Considerations

Query processing happens **on the critical path** (every millisecond counts). Typical budget: 5-10ms.

### Caching

**Query result caching**: Cache entire result sets for popular queries
- Hit rate: 20-40% for head queries
- TTL: 5-60 minutes (depending on freshness requirements)

**Query rewrite caching**: Cache the output of query processing (spell correction, expansion, parsing)
- Much cheaper than full result caching
- Hit rate: 60-80%

### Fast Spell Correction

Computing edit distance for every query term against every dictionary term is $O(|Q| \cdot |D| \cdot L)$ where $L$ is average word length. For a 100K-word dictionary, this is too slow.

**Optimization**: Use a BK-tree or Symmetric Delete algorithm:
- Precompute a data structure that allows fast retrieval of words within edit distance $k$
- Query time: $O(\log |D|)$ or better

**Alternative**: Only spell-check queries that return zero results (avoid the overhead for most queries).

### Parallel Processing

Many query processing stages can run in parallel:

```
                    ┌─────────────┐
Raw query ────────▶│ Tokenization │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌─────────┐  ┌──────────┐  ┌──────────┐
        │ Spell   │  │ Synonym  │  │ Intent   │
        │ check   │  │ expansion│  │ classify │
        └────┬────┘  └────┬─────┘  └────┬─────┘
             │            │             │
             └────────────┼─────────────┘
                          ▼
                  ┌────────────────┐
                  │ Structured     │
                  │ query gen      │
                  └────────────────┘
```

Run spell correction, synonym expansion, and intent classification in parallel threads/coroutines, then merge results.

### Query Complexity Limits

Long queries (20+ terms) or complex Boolean queries can be expensive. Impose limits:

- **Max query length**: 512 characters (discard or truncate)
- **Max terms**: 50 terms (or use `minimum_should_match` to relax)
- **Max Boolean nesting depth**: 5 levels

Elasticsearch has built-in limits (`index.query.bool.max_clause_count`, default 1024) to prevent denial-of-service via overly complex queries.

## References

- Broder, A. — "A Taxonomy of Web Search" (SIGIR Forum, 2002)
- Cucerzan, S. & Brill, E. — "Spelling Correction as an Iterative Process that Exploits the Collective Knowledge of Web Users" (EMNLP, 2004)
- Jones, R., et al. — "Generating Query Substitutions" (WWW, 2006)
- Guo, J., et al. — "A Deep Relevance Matching Model for Ad-hoc Retrieval" (CIKM, 2016) — neural query understanding
- Zamani, H., et al. — "From Neural Re-Ranking to Neural Ranking: Learning a Sparse Representation for Inverted Indexing" (CIKM, 2018)
- Mitra, B. & Craswell, N. — "Query Expansion with Locally-Trained Word Embeddings" (ACL, 2016)
- Norvig, P. — "How to Write a Spelling Corrector" (http://norvig.com/spell-correct.html)
