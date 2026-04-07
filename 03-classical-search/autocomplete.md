# Autocomplete and Typeahead

Autocomplete (also called typeahead, search-as-you-type, or query completion) provides suggestions as users type, helping them formulate queries faster and discover relevant content. This section covers the data structures, ranking algorithms, and system design patterns that power fast, relevant autocomplete.

## Table of Contents

1. [Autocomplete Requirements](#autocomplete-requirements)
2. [Prefix Tree (Trie) Implementation](#prefix-tree-trie-implementation)
3. [Finite State Transducers (FSTs)](#finite-state-transducers-fsts)
4. [Ranking Suggestions](#ranking-suggestions)
5. [Handling Typos in Autocomplete](#handling-typos-in-autocomplete)
6. [Personalization](#personalization)
7. [System Design Patterns](#system-design-patterns)
8. [Performance Optimization](#performance-optimization)
9. [References](#references)

---

## Autocomplete Requirements

### Functional Requirements

1. **Prefix matching**: "searc" → ["search", "search engine", "searching"]
2. **Low latency**: <50ms p99 (users expect instant feedback)
3. **Relevance**: Most useful suggestions first (not just alphabetical)
4. **Freshness**: Recent trending queries should appear quickly
5. **Typo tolerance**: "serch" → ["search", "research"]
6. **Context awareness**: Personalized or location-based suggestions

### Scale Characteristics

**Typical stats for a medium-sized e-commerce site:**
- Dictionary size: 100K - 1M terms/phrases
- QPS: 1000 - 10,000 queries per second
- Latency budget: 20-50ms
- Suggestion count: 5-10 per query

**Large-scale (Google Search):**
- Dictionary: billions of query phrases
- QPS: millions per second
- Latency: <10ms
- Suggestions: personalized, context-aware

---

## Prefix Tree (Trie) Implementation

### Basic Trie Structure

```python
class TrieNode:
    def __init__(self):
        self.children = {}  # char → TrieNode
        self.is_end_of_word = False
        self.frequency = 0  # How many times this suggestion was selected
        self.completions = []  # Top-K completions from this node

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word, frequency=1):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True
        node.frequency += frequency

    def search_prefix(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []  # Prefix not found
            node = node.children[char]
        # Collect all words from this node
        return self._collect_words(node, prefix)

    def _collect_words(self, node, prefix, limit=10):
        results = []
        if node.is_end_of_word:
            results.append((prefix, node.frequency))

        for char, child_node in node.children.items():
            results.extend(self._collect_words(child_node, prefix + char, limit))

        # Sort by frequency, return top K
        results.sort(key=lambda x: -x[1])
        return results[:limit]
```

**Complexity:**
- **Insert**: $O(L)$ where $L$ = word length
- **Prefix search**: $O(L + K \log K)$ where $K$ = number of matching words (due to sorting)
- **Space**: $O(N \cdot L)$ where $N$ = number of words

### Optimized Trie with Precomputed Top-K

Store top-K suggestions at each node to avoid collecting and sorting at query time.

```python
class OptimizedTrieNode:
    def __init__(self):
        self.children = {}
        self.top_k = []  # Precomputed top-K suggestions: [(word, score), ...]

    def insert_and_update(self, word, score):
        # Insert word and update top-K at all prefix nodes
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = OptimizedTrieNode()
            node = node.children[char]

            # Update top-K at this node
            node.top_k.append((word, score))
            node.top_k.sort(key=lambda x: -x[1])
            node.top_k = node.top_k[:10]  # Keep only top-10

    def autocomplete(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        return node.top_k  # O(1) lookup!
```

**Complexity:**
- **Insert**: $O(L \cdot K \log K)$ (update top-K at $L$ nodes)
- **Prefix search**: $O(L)$ (just traverse to prefix node)
- **Space**: $O(N \cdot L \cdot K)$ (store top-K at each node)

**Trade-off:** Higher insert cost and memory for much faster queries. Excellent for read-heavy workloads (autocomplete is 99% reads).

### Compressed Trie (Radix Tree)

Merge chains of single-child nodes to save space:

```
Naive trie:                   Radix tree:
    s                             s
    ↓                             ↓
    e                           earch [is_end]
    ↓                             ├─ ing
    a                             └─ es
    ↓
    r
    ↓
    c
    ↓
    h [is_end]
  ↙   ↘
ing    es
```

**Space saving:** Reduces node count by ~3-5× for natural language text.

**Implementation:** Store edge labels as strings instead of single characters.

```python
class RadixNode:
    def __init__(self):
        self.edges = {}  # string → RadixNode
        self.is_end_of_word = False
```

---

## Finite State Transducers (FSTs)

FSTs are the most space-efficient structure for autocomplete, used in production systems like Lucene's suggest component.

### FST Overview

An FST maps strings to values (e.g., suggestion → weight) while sharing common prefixes AND suffixes.

```
Input → Output mapping:
"search"   → 1000
"see"      → 500
"sea"      → 300
"set"      → 200

FST (simplified):
    s ──→ e ──→ a [output: 300]
           ├──→ e [output: 500]
           └──→ t [output: 200]
         ├──→ a ──→ r ──→ c ──→ h [output: 1000]
```

The FST shares:
- Prefix "se" for {search, see, sea, set}
- Suffix "a" for {sea, search} (suffix sharing requires a DFA minimization algorithm)

### Building an FST

Lucene's FST builder:

```java
import org.apache.lucene.util.fst.*;
import org.apache.lucene.util.IntsRefBuilder;

// Build FST
PositiveIntOutputs outputs = PositiveIntOutputs.getSingleton();
FSTCompiler<Long> fstCompiler = new FSTCompiler<>(FST.INPUT_TYPE.BYTE1, outputs);

// Add sorted inputs (MUST be sorted!)
IntsRefBuilder scratchInts = new IntsRefBuilder();
fstCompiler.add(Util.toIntsRef("search", scratchInts), 1000L);
fstCompiler.add(Util.toIntsRef("see", scratchInts), 500L);
fstCompiler.add(Util.toIntsRef("sea", scratchInts), 300L);

FST<Long> fst = fstCompiler.compile();

// Query FST
IntsRef prefix = Util.toIntsRef("se", new IntsRefBuilder());
// ... use FST.BytesReader to traverse and find matching suggestions
```

**Space efficiency:** Lucene's FST is typically 3-5× smaller than a trie, and can be memory-mapped (no heap usage).

### FST for Autocomplete

Lucene's `AnalyzingSuggester` uses FST for autocomplete:

```java
AnalyzingSuggester suggester = new AnalyzingSuggester(analyzer);

// Build from input stream: (text, weight) pairs
InputIterator iterator = new InputArrayIterator(new Input[]{
    new Input("search engine", 1000),
    new Input("search algorithm", 800),
    new Input("searching", 500)
});
suggester.build(iterator);

// Query
List<LookupResult> results = suggester.lookup("searc", 10);
// Returns: ["search engine": 1000, "search algorithm": 800, "searching": 500]
```

**Latency:** <1ms for typical autocomplete queries (on mmap'd FST).

---

## Ranking Suggestions

### Frequency-Based Ranking

Rank by how often each suggestion was used/selected.

```
Suggestions with frequencies:
  "search" → 10,000 (selected 10K times)
  "search engine" → 5,000
  "searching" → 2,000

Autocomplete for "sear":
  1. "search" (10,000)
  2. "search engine" (5,000)
  3. "searching" (2,000)
```

**Data source:** Query logs (aggregate and count unique queries).

**Update frequency:** Rebuild trie/FST daily or weekly with fresh counts.

### Score-Based Ranking

Use a scoring function combining multiple signals:

$$\text{score}(suggestion) = \alpha \cdot \text{frequency} + \beta \cdot \text{recency} + \gamma \cdot \text{CTR}$$

**Signals:**
- **Frequency**: How often was this query issued (historical)
- **Recency**: Boost recent/trending queries (time-decayed weight)
- **CTR (Click-Through Rate)**: Of users who selected this suggestion, how many clicked a result?
- **Conversion rate**: Did the query lead to a purchase/action?

**Example:**

```
"iphone 15" → frequency: 5000, recency_boost: 2.0 (trending), CTR: 0.8
"iphone 14" → frequency: 8000, recency_boost: 1.0, CTR: 0.7

score("iphone 15") = 0.5×5000 + 0.3×2.0 + 0.2×0.8 = 2500 + 0.6 + 0.16 = 2500.76
score("iphone 14") = 0.5×8000 + 0.3×1.0 + 0.2×0.7 = 4000 + 0.3 + 0.14 = 4000.44

Ranking: "iphone 14" first (higher score)
```

Adjust $\alpha, \beta, \gamma$ based on your business goals (discovery vs. popularity).

### Prefix Frequency Boosting

Shorter suggestions often beat longer ones in relevance:

```
"search" vs "search engine optimization best practices"

User typing "sear" probably wants "search", not the long-tail phrase.
```

**Length penalty:**

$$\text{score} = \frac{\text{frequency}}{\log(1 + \text{length})}$$

### Contextual Ranking

Personalize based on user context:

- **User history**: Boost queries the user has issued before
- **Location**: "pizza" → boost "pizza near me" for mobile users
- **Time of day**: "breakfast" boosted in the morning
- **Device**: Mobile users prefer shorter queries

**Example: Amazon autocomplete**

```
Generic user typing "java":
  1. "java programming"
  2. "javascript"
  3. "java coffee"

User who previously bought programming books:
  1. "java programming"
  2. "java 17 features"
  3. "javascript"
```

**Implementation:** Layer personalized trie on top of global trie, merge results.

---

## Handling Typos in Autocomplete

### Fuzzy Prefix Matching

Allow edit distance in the prefix:

```
User types: "serch"
Fuzzy match (edit distance 1): "search"

Suggestions:
  1. "search"
  2. "search engine"
  3. "searching"
```

**Algorithm: Levenshtein automaton + Trie/FST**

As described in [Fuzzy Search](./fuzzy-search.md#levenshtein-automaton), intersect a Levenshtein automaton (edit distance $k$) with the trie/FST.

**Lucene's FuzzySuggester:**

```java
FuzzySuggester suggester = new FuzzySuggester(analyzer);
suggester.build(iterator);

// Fuzzy lookup (allows edit distance)
List<LookupResult> results = suggester.lookup("serch", 10);
// Returns: "search", "research", ...
```

**Latency consideration:** Fuzzy autocomplete is slower (~5-10ms vs <1ms for exact prefix). Use sparingly or as fallback.

### Prefix Correction Before Lookup

Correct the prefix first, then do exact lookup:

```
User types: "serch"

Step 1: Spell-correct "serch" → "search"
Step 2: Autocomplete for "search" → ["search", "search engine", ...]
```

**Approach:**
1. Maintain a dictionary of valid prefixes (all prefixes from the trie)
2. Use edit distance to find closest valid prefix
3. Autocomplete from corrected prefix

**Hybrid approach:**
- Try exact match first (fast path)
- If no results or few results, try fuzzy match (slow path)

### N-Gram Prefix Index

Index all 2-grams/3-grams of suggestions for more typo tolerance:

```
Suggestion: "search engine"

2-grams: "se", "ea", "ar", "rc", "ch", "en", "ng", "gi", "in", "ne"

User types: "serch" (missing 'a')
2-grams: "se", "er", "rc", "ch"

Overlap: {"se", "rc", "ch"} = 3 out of 4 → high similarity → suggest "search engine"
```

This is slower and uses more memory, but very robust to typos.

---

## Personalization

### User-Level Autocomplete History

Store per-user autocomplete selections:

```
User 123's history:
  - "laptop dell" (selected 5 times)
  - "laptop lenovo" (selected 2 times)
  - "monitor" (selected 1 time)

User types "lap":
  1. "laptop dell" (personalized boost)
  2. "laptop lenovo" (personalized boost)
  3. "laptop" (generic suggestion)
```

**Implementation:**

```python
def autocomplete_personalized(prefix, user_id):
    global_suggestions = trie.autocomplete(prefix)
    user_history = get_user_history(user_id, prefix)

    # Merge with boosting
    suggestions = {}
    for (suggestion, score) in global_suggestions:
        suggestions[suggestion] = score

    for (suggestion, user_count) in user_history:
        suggestions[suggestion] = suggestions.get(suggestion, 0) + user_count * 100

    return sorted(suggestions.items(), key=lambda x: -x[1])[:10]
```

**Privacy consideration:** Store hashed user IDs, allow opt-out, expire old history.

### Contextual Signals

**Location-aware:**

```
User in New York typing "weather":
  1. "weather new york"
  2. "weather forecast"
  3. "weather tomorrow"

User in London:
  1. "weather london"
  2. "weather forecast uk"
  3. "weather tomorrow"
```

**Category-aware (e-commerce):**

```
User browsing Electronics, types "apple":
  1. "apple macbook"
  2. "apple iphone"
  3. "apple watch"

User browsing Grocery:
  1. "apple fruit"
  2. "apple juice"
  3. "apple pie"
```

**Implementation:** Multiple tries per category/context, or context-based scoring adjustment.

---

## System Design Patterns

### Client-Side vs. Server-Side

**Client-side (static suggestions):**
- Download dictionary to client (e.g., 50KB compressed JSON)
- JavaScript trie for autocomplete
- No server round-trip → instant results
- **Use case:** Limited dictionary (<10K items), offline-first apps

**Server-side (dynamic suggestions):**
- Client sends prefix to server
- Server responds with suggestions
- Can personalize, use full dictionary, apply business logic
- **Use case:** Large dictionaries, personalization, fresh data

**Hybrid:**
- Ship top 10K suggestions to client (covers 80% of queries)
- Fall back to server for long-tail queries

### Debouncing and Throttling

Don't send a request for every keystroke:

```javascript
let debounceTimer;
function onInput(prefix) {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
        fetchSuggestions(prefix);
    }, 150);  // Wait 150ms after user stops typing
}
```

**Debounce:** Wait until user stops typing for N ms.

**Throttle:** Limit to one request per N ms (even if user is still typing).

**Optimal delay:** 100-200ms (balance responsiveness vs. server load).

### Caching

**Server-side caching:**

```
Cache key: prefix + context (e.g., "sear|category:electronics")
Cache value: [suggestions]
TTL: 5 minutes to 1 hour
```

**Hit rate:** 80-90% for popular prefixes (Zipf distribution: a few prefixes account for most traffic).

**Cache warming:** Pre-populate cache with top 1000 prefixes.

### Tiered Suggestions

Provide different tiers of suggestions:

```
User types: "java"

Tier 1: Query suggestions (from query logs)
  - "java programming"
  - "javascript tutorial"

Tier 2: Document titles
  - "Java: The Complete Reference"
  - "JavaScript: The Good Parts"

Tier 3: Categories
  - "Books > Programming > Java"

Tier 4: People/Entities
  - "James Gosling" (creator of Java)
```

**UI:** Group by tier or interleave intelligently.

**Elasticsearch multi-field suggester:**

```json
{
  "suggest": {
    "query-suggest": {
      "prefix": "java",
      "completion": { "field": "query_suggest" }
    },
    "title-suggest": {
      "prefix": "java",
      "completion": { "field": "title_suggest" }
    }
  }
}
```

---

## Performance Optimization

### In-Memory Data Structures

**Requirement:** Entire autocomplete index must fit in memory for <50ms latency.

**Sizing:**

```
1M suggestions, avg length 30 chars
Naive trie: ~500MB
Compressed trie: ~150MB
FST: ~50MB (Lucene FST with mmap)
```

For large dictionaries (10M+), use distributed architecture (see below).

### Memory-Mapped FSTs

Lucene's FST can be memory-mapped:

```java
FST<Long> fst = new FST<>(indexInput, outputs);
// indexInput is a memory-mapped file
```

**Advantage:** OS manages caching, no heap pressure, supports larger-than-RAM datasets.

### Prefix Bloom Filters

For negative queries (prefixes with no suggestions), use a Bloom filter:

```
Bloom filter: Contains all valid prefixes

User types: "zzzzz"
Bloom filter: "zzzzz" not in filter → return empty (no trie lookup needed)

User types: "sear"
Bloom filter: "sear" in filter → proceed with trie lookup
```

**Benefit:** Filter out junk/random input before hitting the trie (~10% of autocomplete queries).

### Distributed Autocomplete

For massive scale (billions of suggestions):

**Approach 1: Partition by prefix**

```
Shard 1: Prefixes "a*" to "m*"
Shard 2: Prefixes "n*" to "z*"

User types "sear" → route to Shard 2 → autocomplete
```

**Problem:** Uneven load (prefixes starting with 's' are more popular than 'z').

**Approach 2: Replicate global dictionary**

Each server has the full dictionary (FST). Load balance requests across servers.

**Memory:** If FST is 1GB and you have 10 servers → 10GB total memory (acceptable for autocomplete).

### Precomputed Suggestions

For very popular prefixes, precompute and cache suggestions in a fast KV store (Redis, Memcached):

```
Redis:
  "se" → ["search", "search engine", "security", ...]
  "sea" → ["search", "sea", "seattle", ...]
```

**Update:** Recompute daily from query logs.

**Fallback:** If prefix not in Redis, fall back to trie/FST.

---

## References

- Gog, S. & Petri, M. — "Optimized succinct data structures for massive data" (Software: Practice and Experience, 2014) — Succinct tries
- Havrylov, M. — "Fast autocomplete in Elasticsearch" (Elastic Blog, 2013)
- Lucene — "Suggester" documentation — https://lucene.apache.org/core/8_11_0/suggest/overview-summary.html
- Lucene — "FST package" — https://lucene.apache.org/core/8_11_0/core/org/apache/lucene/util/fst/package-summary.html
- Mihov, S. & Schulz, K. U. — "Fast approximate search in large dictionaries" (Computational Linguistics, 2004) — Levenshtein automata
- Broder, A. Z. & Mitzenmacher, M. — "Network applications of Bloom filters: A survey" (Internet Mathematics, 2004)
- Bast, H. & Weber, I. — "Type less, find more: fast autocompletion search with a succinct index" (SIGIR, 2006)
- Dean, J. — "Challenges in building large-scale information retrieval systems" (WSDM, 2009) — Google's autocomplete architecture
