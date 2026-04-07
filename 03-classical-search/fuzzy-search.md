# Fuzzy Search

Users make typos, misspell words, and use different terminology than what's in your documents. Fuzzy search handles this gracefully by finding approximate matches. This section covers the algorithms that power spell correction, typo tolerance, and phonetic matching.

## Table of Contents

1. [Edit Distance](#edit-distance)
2. [N-Gram Indexing for Fuzzy Matching](#n-gram-indexing-for-fuzzy-matching)
3. [Phonetic Algorithms](#phonetic-algorithms)
4. [Spell Correction](#spell-correction)
5. [Fuzzy Query Evaluation](#fuzzy-query-evaluation)
6. [Trade-offs and Performance](#trade-offs-and-performance)
7. [References](#references)

---

## Edit Distance

Edit distance measures how different two strings are by counting the minimum number of single-character edits needed to transform one into the other.

### Levenshtein Distance

The most common edit distance metric. Allowed operations:
- **Insertion**: "cat" → "cast" (insert 's')
- **Deletion**: "cast" → "cat" (delete 's')
- **Substitution**: "cat" → "cut" (replace 'a' with 'u')

**Definition:**

For strings $a$ and $b$:

$$
\text{lev}(a, b) = \begin{cases}
|a| & \text{if } |b| = 0 \\
|b| & \text{if } |a| = 0 \\
\text{lev}(\text{tail}(a), \text{tail}(b)) & \text{if } a[0] = b[0] \\
1 + \min \begin{cases}
  \text{lev}(\text{tail}(a), b) & \text{deletion} \\
  \text{lev}(a, \text{tail}(b)) & \text{insertion} \\
  \text{lev}(\text{tail}(a), \text{tail}(b)) & \text{substitution}
\end{cases} & \text{otherwise}
\end{cases}
$$

**Dynamic Programming Algorithm:**

```python
def levenshtein(a, b):
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Base cases
    for i in range(m + 1):
        dp[i][0] = i  # Delete all chars from a
    for j in range(n + 1):
        dp[0][j] = j  # Insert all chars from b

    # Fill the matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1]  # No operation needed
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],     # Deletion
                    dp[i][j-1],     # Insertion
                    dp[i-1][j-1]    # Substitution
                )

    return dp[m][n]
```

**Example:**

```
levenshtein("kitten", "sitting")

    ""  s  i  t  t  i  n  g
""   0  1  2  3  4  5  6  7
k    1  1  2  3  4  5  6  7
i    2  2  1  2  3  4  5  6
t    3  3  2  1  2  3  4  5
t    4  4  3  2  1  2  3  4
e    5  5  4  3  2  2  3  4
n    6  6  5  4  3  3  2  3

Result: 3 (k→s, e→i, insert g)
```

**Time complexity:** $O(mn)$
**Space complexity:** $O(mn)$ (can be optimized to $O(\min(m, n))$ using rolling arrays)

### Damerau-Levenshtein Distance

Extends Levenshtein with **transposition** (swap adjacent chars):

```
"search" → "serach" (transpose 'a' and 'r')
```

This is important because transpositions are very common typos (about 10% of all typos).

**Algorithm modification:**

```python
def damerau_levenshtein(a, b):
    # ... (similar to Levenshtein) ...
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                cost = 0
            else:
                cost = 1

            dp[i][j] = min(
                dp[i-1][j] + 1,       # Deletion
                dp[i][j-1] + 1,       # Insertion
                dp[i-1][j-1] + cost,  # Substitution
            )

            # Transposition
            if i > 1 and j > 1 and a[i-1] == b[j-2] and a[i-2] == b[j-1]:
                dp[i][j] = min(dp[i][j], dp[i-2][j-2] + cost)

    return dp[m][n]
```

**Example:**

```
damerau_levenshtein("search", "serach") = 1 (transposition)
levenshtein("search", "serach") = 2 (delete a, insert a)
```

### Bounded Edit Distance

For fuzzy search, you usually only care about matches within edit distance $k$ (e.g., $k \le 2$). Computing full $O(mn)$ DP is wasteful.

**Ukkonen's algorithm:** Computes edit distance in $O(k \cdot \min(m, n))$ when distance is bounded by $k$.

**Key insight:** Only compute a diagonal band of width $2k+1$ in the DP matrix:

```
For k=2, only compute cells in the band:

    ""  s  i  t  t  i  n  g
""   0  1  2  ·  ·  ·  ·  ·
k    1  1  2  3  ·  ·  ·  ·
i    2  2  1  2  3  ·  ·  ·
t    ·  3  2  1  2  3  ·  ·
t    ·  ·  3  2  1  2  3  ·
e    ·  ·  ·  3  2  2  3  4
n    ·  ·  ·  ·  3  3  2  3

Cells outside the band are > k, so we don't care.
```

This is essential for efficient fuzzy search over large term dictionaries.

### Edit Distance in Search

**Use cases:**
- **Fuzzy term matching**: `"search~2"` finds terms within edit distance 2 of "search"
- **Spell correction**: Find closest dictionary terms to a misspelled query
- **Did-you-mean suggestions**: "Did you mean 'search' instead of 'serach'?"
- **Typo-tolerant autocomplete**: Match prefixes with small errors

---

## N-Gram Indexing for Fuzzy Matching

Edit distance is too slow to compute against every term in a million-term dictionary. N-grams provide an efficient filtering step.

### N-Gram Generation

An **n-gram** is a contiguous sequence of $n$ characters. Typically use $n = 2$ (bigrams) or $n = 3$ (trigrams).

**Example:**

```
Term: "search"

Bigrams: "se", "ea", "ar", "rc", "ch"
Trigrams: "sea", "ear", "arc", "rch"

With padding: "$$search$$" ($ = boundary marker)
Bigrams: "$$s", "$se", "sea", "ear", "arc", "rch", "ch$", "h$$"
```

### N-Gram Index Structure

Instead of mapping `term → docIDs`, map `n-gram → terms`:

```
Bigram index:
"se" → ["search", "see", "set", "sense", ...]
"ea" → ["search", "each", "early", "leap", ...]
"ar" → ["search", "large", "area", ...]
"rc" → ["search", "merchant", "urch", ...]
"ch" → ["search", "change", "reach", ...]
```

### Fuzzy Search via N-Grams

**Observation:** If two strings have edit distance $d$, they share many n-grams.

**Filtering rule (for edit distance $k$):**

If $\text{edit}(a, b) \le k$, then $a$ and $b$ share at least $\max(|a|, |b|) - k \cdot (n-1) - (n-1)$ n-grams.

For trigrams ($n=3$) and $k=1$:

```
Shared trigrams >= max(|a|, |b|) - 1×2 - 2 = max(|a|, |b|) - 4
```

**Algorithm:**

```python
def fuzzy_search_ngrams(query, k, n=3):
    # Step 1: Generate n-grams of query
    query_ngrams = set(generate_ngrams(query, n))

    # Step 2: Find candidate terms from n-gram index
    candidates = defaultdict(int)
    for ngram in query_ngrams:
        for term in ngram_index[ngram]:
            candidates[term] += 1

    # Step 3: Filter by n-gram overlap threshold
    threshold = len(query) - k * (n - 1) - (n - 1)
    filtered = [term for term, count in candidates.items()
                if count >= threshold]

    # Step 4: Verify with actual edit distance
    results = []
    for term in filtered:
        if levenshtein(query, term) <= k:
            results.append(term)

    return results
```

**Example:**

```
Query: "serach" (misspelled "search"), k=1, n=3

Trigrams of "serach": {"ser", "era", "rac", "ach"}

Threshold = 6 - 1×2 - 2 = 2 (must share at least 2 trigrams)

Candidates from index:
  "search" has trigrams {"sea", "ear", "arc", "rch"}
  Overlap with query: {"era" → no, "rac" → no, ...}

  Wait, this doesn't share trigrams! Need padding:

With padding: "$$serach$$" → {"$$s", "$se", "ser", "era", "rac", "ach", "ch$", "h$$"}
"$$search$$" → {"$$s", "$se", "sea", "ear", "arc", "rch", "ch$", "h$$"}

Shared: {"$$s", "$se", "ch$", "h$$"} = 4 trigrams ✓

Then verify: levenshtein("serach", "search") = 2 (but we want k=1, so exclude)

Try "seacrh" (another misspelling):
Shared trigrams with "search": 5 (different by transposition)
levenshtein("seacrh", "search") = 1 ✓
```

### N-Gram Index Size

**Trade-off:** Larger $n$ → fewer false positives, but larger index.

| N-gram size | Index size multiplier | Precision | Recall |
|---|---|---|---|
| 2 (bigrams) | 2-3× term count | Low | High |
| 3 (trigrams) | 4-6× term count | Medium | Medium |
| 4 (4-grams) | 8-12× term count | High | Low (misses short edits) |

**Practical choice:** Trigrams ($n=3$) for most use cases. Elasticsearch uses trigrams for fuzzy queries by default.

---

## Phonetic Algorithms

Phonetic algorithms handle spelling variations that sound similar (homophones, alternative spellings).

### Soundex

The oldest phonetic algorithm (1918, originally for US Census). Maps names to a 4-character code based on pronunciation.

**Algorithm:**

```
1. Keep the first letter
2. Map remaining letters to digits:
   - b, f, p, v → 1
   - c, g, j, k, q, s, x, z → 2
   - d, t → 3
   - l → 4
   - m, n → 5
   - r → 6
3. Remove vowels (a, e, i, o, u, h, w, y)
4. Remove duplicate digits
5. Pad with zeros to length 4 (or truncate)

Example: "Robert" → R163
  R-o-b-e-r-t
  R-b-r-t  (remove vowels)
  R-1-6-3  (map to digits)
  R163     (result)

"Rupert" → R163 (same code!)
```

**Limitations:**
- Very coarse (many false matches)
- Designed for English surnames only
- Doesn't handle first-letter errors ("smith" vs "smyth" → different codes)

### Metaphone and Double Metaphone

Metaphone (1990) is a more sophisticated phonetic algorithm.

**Key improvements over Soundex:**
- Handles consonant clusters (th, ch, sh)
- Better vowel handling
- Context-sensitive rules (e.g., 'gh' is silent in "night" but not in "ghost")

**Double Metaphone** (2000) generates *two* codes per word (primary and alternate) to handle more pronunciation variants:

```
"Schmidt" → Primary: "XMT", Alternate: "SMT"
"Smith"   → Primary: "SM0", Alternate: "XMT"

Match on alternate codes: Schmidt ≈ Smith
```

**Example:**

```python
from metaphone import doublemetaphone

doublemetaphone("knight")  # → ("NT", "NT")
doublemetaphone("night")   # → ("NT", "NT")  # Same!

doublemetaphone("Catherine")  # → ("K0RN", "KTRN")
doublemetaphone("Kathryn")    # → ("K0RN", "KTRN")  # Same!
```

### Phonetic Index Structure

Store phonetic codes alongside terms:

```
Phonetic index (Double Metaphone):
"NT" → ["knight", "night", "gnat", ...]
"XMT" → ["Schmidt", "Smith", ...]
"K0RN" → ["Catherine", "Kathryn", "Katherine", ...]
```

**Fuzzy search with phonetics:**

```
Query: "Kathryn" (but user meant "Catherine")

1. Compute phonetic codes: ("K0RN", "KTRN")
2. Lookup in phonetic index: ["Catherine", "Kathryn", "Katherine"]
3. Return matches (possibly ranked by edit distance or frequency)
```

### When to Use Phonetic Matching

**Good for:**
- Name search (people, places)
- Medical/drug search (many similar-sounding drug names)
- Voice search transcription errors

**Not good for:**
- General text (too many false positives)
- Technical terms, product codes, acronyms
- Non-English languages (Metaphone is English-specific)

**Multi-lingual phonetics:**
- **Caverphone**: New Zealand English names
- **Kölner Phonetik**: German
- **Daitch-Mokotoff Soundex**: Eastern European names
- **Beider-Morse Phonetic Matching**: Generic multi-language approach

---

## Spell Correction

### Context-Free Spell Correction

**Isolated word correction:** Given a possibly misspelled query term, find the most likely correct term.

**Algorithm:**

```
SPELL_CORRECT(query_term):
    candidates = []

    # Generate candidates within edit distance k (e.g., k=2)
    for term in dictionary:
        if edit_distance(query_term, term) <= k:
            candidates.append(term)

    # Rank by likelihood
    # Option 1: Corpus frequency (more common terms are more likely)
    # Option 2: Edit distance (closer matches preferred)
    # Option 3: Combination

    candidates.sort(key=lambda t: (edit_distance(query_term, t), -frequency(t)))
    return candidates[0] if candidates else query_term
```

**Optimization:** Use n-gram index to filter candidates (as described earlier).

**Example:**

```
Query: "serach"

Candidates (edit distance ≤ 2):
  - "search" (edit dist 2, frequency 1M)
  - "seraph" (edit dist 2, frequency 100)
  - "reach" (edit dist 3, excluded)

Correction: "search" (higher frequency)
```

### Context-Aware Spell Correction

**Multi-word queries:** Use context to disambiguate.

```
Query: "new york times"
Misspelled: "new yrok times"

Candidates for "yrok":
  - "york" (edit dist 1)
  - "rock" (edit dist 2)

Bigram model:
  P("york" | "new") = 0.001 (common)
  P("rock" | "new") = 0.00001 (rare)

Correction: "york"
```

**N-gram language models:**

$$P(w_i | w_{i-1}, w_{i-2}) = \frac{\text{count}(w_{i-2}, w_{i-1}, w_i)}{\text{count}(w_{i-2}, w_{i-1})}$$

Modern systems use neural language models (e.g., BERT) for this, but n-gram models are still common in production due to speed.

### Noisy Channel Model

The probabilistic framework for spell correction (Peter Norvig's famous approach):

**Goal:** Find the most likely correction $c$ given misspelled query $q$:

$$c^* = \arg\max_{c \in \text{candidates}} P(c | q)$$

**Bayes' theorem:**

$$P(c | q) = \frac{P(q | c) \cdot P(c)}{P(q)}$$

Since $P(q)$ is constant for all candidates:

$$c^* = \arg\max_{c} P(q | c) \cdot P(c)$$

where:
- $P(c)$ = **prior probability** of word $c$ (from corpus frequency)
- $P(q | c)$ = **error model**: probability that user typed $q$ when they meant $c$

**Error model:** Empirically derived from typo corpora:

```
P(q|c) based on edit operations:
  Substitution (adjacent keys): high probability (e.g., 's' → 'a')
  Substitution (distant keys): low probability (e.g., 's' → 'p')
  Insertion/deletion: medium probability
  Transposition (adjacent chars): high probability
```

**Example:**

```
Query: "speling"

Candidates (edit distance 1):
  - "spelling" (edit dist 1, freq 10000, P(c) = 0.01)
  - "spieling" (edit dist 1, freq 10, P(c) = 0.00001)

Error model: P("speling"|"spelling") = 0.05 (common deletion)
              P("speling"|"spieling") = 0.01 (less common substitution)

Posterior:
  "spelling": 0.05 × 0.01 = 0.0005
  "spieling": 0.01 × 0.00001 = 0.0000001

Correction: "spelling"
```

### Did-You-Mean Suggestions

Don't auto-correct; offer suggestions:

```
Query: "serach engine"

Results: 150 matches
Suggestion: "Did you mean: search engine (1.5M matches)?"
```

**Heuristic:** Only suggest if:
1. Original query has few results (e.g., <500)
2. Corrected query has many more results (e.g., >10× more)
3. Correction confidence is high (edit distance 1, high frequency)

**Elasticsearch implementation:**

```json
{
  "suggest": {
    "text": "serach engine",
    "simple_phrase": {
      "phrase": {
        "field": "content",
        "size": 1,
        "gram_size": 3,
        "direct_generator": [
          {
            "field": "content",
            "suggest_mode": "always",
            "min_word_length": 1
          }
        ]
      }
    }
  }
}
```

---

## Fuzzy Query Evaluation

### Lucene Fuzzy Query

Elasticsearch/Lucene fuzzy query:

```json
{
  "query": {
    "fuzzy": {
      "field": {
        "value": "search",
        "fuzziness": "AUTO"
      }
    }
  }
}
```

**Fuzziness parameter:**
- `0`: No fuzziness (exact match)
- `1`: Edit distance ≤ 1
- `2`: Edit distance ≤ 2
- `AUTO`: Edit distance based on term length:
  - 0-2 chars: exact match
  - 3-5 chars: edit distance 1
  - >5 chars: edit distance 2

**Execution:**

```
1. Generate candidates from term dictionary (using n-gram index or Levenshtein automaton)
2. Expand to OR query: "search" OR "searcg" OR "seatrch" OR ...
3. Evaluate as normal multi-term query
```

### Levenshtein Automaton

An elegant way to find all terms within edit distance $k$ without enumerating all possible edits.

**Key idea:** Build a finite automaton that accepts exactly the strings within edit distance $k$ of the target.

```
Target: "cat", k=1

Levenshtein automaton accepts:
  - "cat" (exact match)
  - "at", "ct", "ca" (deletions)
  - "xcat", "cxat", "caxt", "catx" (insertions, x = any char)
  - "xat", "cxt", "cax" (substitutions)
```

**Intersection with FST:**

Lucene's term dictionary is an FST. The fuzzy query intersects the Levenshtein automaton with the FST:

```
FUZZY_QUERY(term, k):
    lev_automaton = build_levenshtein_automaton(term, k)
    term_fst = get_term_dictionary_fst()
    matching_terms = intersect_automata(lev_automaton, term_fst)
    return OR_query(matching_terms)
```

This is much faster than brute-force edit distance against all terms ($O(V)$ where $V$ = vocabulary size).

**Time complexity:** $O(|term| \cdot k \cdot \log V)$ with FST intersection.

---

## Trade-offs and Performance

### Memory vs. Speed

| Approach | Index Size | Query Speed | Accuracy |
|---|---|---|---|
| No fuzzy search | 1× | Fast | Low (no typo tolerance) |
| Edit distance (brute force) | 1× | Very slow | High |
| N-gram index (trigrams) | 4-6× | Medium | High |
| Levenshtein automaton + FST | 1× | Fast | High |
| Phonetic index | 1.1-1.5× | Fast | Medium (false positives) |

**Recommended:** Levenshtein automaton (Lucene's approach) for best balance.

### Fuzzy Search Limits

**Problem:** "a~2" (all terms within edit distance 2 of "a") matches thousands of terms.

**Limits in Lucene/Elasticsearch:**
- Max edit distance: 2 (Levenshtein automaton becomes expensive for $k > 2$)
- Max term expansions: 50 by default (configurable)
- Prefix length: Require exact match on first $n$ characters (e.g., prefix_length=2 means "search~2" matches "sea*" but not "*arch")

```json
{
  "query": {
    "fuzzy": {
      "field": {
        "value": "search",
        "fuzziness": 2,
        "prefix_length": 2,  // "se" must match exactly
        "max_expansions": 50
      }
    }
  }
}
```

### When Fuzzy Search Isn't Enough

**Neural spell correction:** Use sequence-to-sequence models (e.g., Transformer-based) for complex misspellings:

```
Input: "I want to by a new lap top for collage"
Output: "I want to buy a new laptop for college"
```

Classical edit distance can't handle:
- Word splitting/merging: "lap top" → "laptop"
- Multi-word errors: "for collage" → "for college" (context-dependent)
- Phonetic errors beyond single words

Modern search systems combine classical fuzzy search (for real-time queries) with neural models (for query understanding and offline analysis).

---

## References

- Levenshtein, V. I. — "Binary codes capable of correcting deletions, insertions, and reversals" (Soviet Physics Doklady, 1966) — Original edit distance paper
- Damerau, F. J. — "A technique for computer detection and correction of spelling errors" (Communications of the ACM, 1964)
- Ukkonen, E. — "Algorithms for approximate string matching" (Information and Control, 1985) — Bounded edit distance
- Zobel, J. & Dart, P. — "Phonetic string matching: Lessons from information retrieval" (SIGIR, 1996)
- Philips, L. — "The Double Metaphone Search Algorithm" (C/C++ Users Journal, 2000)
- Norvig, P. — "How to Write a Spelling Corrector" (2007) — http://norvig.com/spell-correct.html
- Schulz, K. U. & Mihov, S. — "Fast string correction with Levenshtein automata" (International Journal on Document Analysis and Recognition, 2002)
- Boytsov, L. — "Indexing methods for approximate dictionary searching: Comparative analysis" (ACM JEAL, 2011)
- Manning, C. D., Raghavan, P., & Schütze, H. — *Introduction to Information Retrieval*, Chapter 3 (Cambridge, 2008)
