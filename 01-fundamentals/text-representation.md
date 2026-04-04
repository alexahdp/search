# Text Representation

Raw text is the input to a search system, but you can't build an inverted index on raw bytes. The text processing pipeline transforms unstructured natural language into a normalized sequence of **tokens** (also called **terms**) that can be indexed and matched. Every choice in this pipeline directly impacts recall and precision.

## Table of Contents

1. [The Analysis Pipeline](#the-analysis-pipeline)
2. [Tokenization](#tokenization)
3. [Normalization](#normalization)
4. [Stemming](#stemming)
5. [Lemmatization](#lemmatization)
6. [Stemming vs. Lemmatization: When to Use Which](#stemming-vs-lemmatization-when-to-use-which)
7. [Stop Words](#stop-words)
8. [Putting It All Together: Analyzer Chains](#putting-it-all-together-analyzer-chains)
9. [Impact on Search Quality](#impact-on-search-quality)
10. [References](#references)

---

## The Analysis Pipeline

Both documents (at index time) and queries (at query time) go through the same pipeline. If they don't, you get mismatches — the most common source of "why doesn't search find this?" bugs.

```
Raw text
  │
  ├── Character filtering  (strip HTML, normalize Unicode)
  │
  ├── Tokenization         (split into tokens)
  │
  ├── Token filtering      (lowercase, remove stop words, stem/lemmatize)
  │
  └── Output: sequence of normalized terms
```

In Elasticsearch/Lucene terminology, this entire pipeline is called an **analyzer**, composed of:
1. **Character filters** — operate on the raw character stream
2. **Tokenizer** — splits characters into tokens
3. **Token filters** — transform, add, or remove tokens

The critical invariant: **the same analyzer must be used at index time and query time.** If you stem documents but not queries (or vice versa), you'll miss matches.

---

## Tokenization

Tokenization splits a character stream into discrete tokens. This is more nuanced than `text.split(" ")`.

### Whitespace and Punctuation Tokenization

The simplest approach: split on whitespace and punctuation characters.

```
Input:  "state-of-the-art, U.S.A., user@email.com, $9.99"
Output: ["state", "of", "the", "art", "U", "S", "A", "user", "email", "com", "9", "99"]
```

**Problems:**
- Destroys meaningful units: "U.S.A." → 3 meaningless single letters
- Breaks email addresses, URLs, product codes
- Loses compound terms: "state-of-the-art" loses its meaning when split
- "$9.99" → "9" and "99" — the price is lost

### Standard/Pattern Tokenizers

More sophisticated tokenizers use rules or regex patterns to handle edge cases:

**Lucene StandardTokenizer** handles:
- Email addresses (kept as single tokens)
- URLs (kept as single tokens)
- Apostrophes in contractions: "it's" → "it's" (one token)
- Hyphens: "state-of-the-art" → "state-of-the-art" (one token) or split depending on configuration
- Numbers with decimals and commas: "1,234.56" → "1234.56"
- CJK characters: each character becomes a token (since CJK has no whitespace)

### Language-Specific Challenges

**CJK (Chinese, Japanese, Korean):**

There are no spaces between words. "東京都は日本の首都です" needs to be segmented into words/morphemes:
- 東京都 (Tokyo-to) / は / 日本 (Japan) / の / 首都 (capital) / です

Approaches:
- **Dictionary-based**: Longest-match using a word dictionary. MeCab (Japanese), Jieba (Chinese), KoNLPy (Korean).
- **Statistical**: CRF or HMM models trained on segmented corpora. Better at handling unknown words.
- **Character n-grams**: Index overlapping bigrams — "東京", "京都", "都は", ... No segmentation needed, but increases index size and produces noisier matches. Lucene's CJKAnalyzer uses this approach.

**German compound words:**

"Donaudampfschifffahrtsgesellschaft" (Danube steamship company) is a single word. A user searching "Dampfschiff" (steamship) should still find it.

Solutions:
- **Decompounding**: Split compounds using a dictionary + heuristic rules. "Krankenhaus" → "Kranken" + "Haus"
- **Index both**: Index the original compound AND the parts

**Agglutinative languages (Turkish, Finnish, Hungarian):**

Words carry extensive morphological information. Turkish "evlerinizden" = "from your houses" (ev + ler + iniz + den). A single Turkish root can generate thousands of surface forms.

Stemming becomes essential (or morphological analysis).

### Subword Tokenization

Originally from NLP/ML, increasingly relevant in search:

**Byte Pair Encoding (BPE):**
1. Start with character-level tokens
2. Iteratively merge the most frequent pair of adjacent tokens
3. Repeat for $k$ merges (vocabulary size parameter)

```
"running" with BPE vocabulary might be: ["run", "ning"]
"unhappiness" might be: ["un", "happ", "iness"]
```

**WordPiece** (used by BERT):
Similar to BPE but selects merges based on likelihood of the training data rather than frequency. Uses "##" prefix for continuation tokens: ["un", "##happi", "##ness"].

**SentencePiece:**
Language-agnostic — works directly on raw text without pre-tokenization. Treats the input as a stream of Unicode characters, including whitespace. Uses either BPE or unigram language model.

**Relevance to search:**
- Subword tokenizers are essential for neural retrieval models (BERT-based re-rankers, dense retrievers)
- They handle OOV (out-of-vocabulary) words gracefully — any word can be decomposed
- They provide a shared vocabulary between query and document encoders
- For traditional inverted-index search, they're typically not used (whole-word tokens are more interpretable)

### Tokenization Affects Recall

Consider a product catalog:

```
Product name: "MacBook Pro 16-inch M3 Max"

Tokenizer A (simple split): ["MacBook", "Pro", "16", "inch", "M3", "Max"]
Tokenizer B (preserves compounds): ["MacBook", "MacBook Pro", "Pro", "16-inch", "16", "inch", "M3", "Max", "M3 Max"]
```

User searches: "macbook pro 16-inch" — Tokenizer B indexes "16-inch" as a unit, so it matches the query token "16-inch" directly. Tokenizer A requires matching "16" AND "inch" separately, which might also match "16 GB, 14-inch" incorrectly.

User searches "m3max" (no space) — neither tokenizer handles this. You'd need character n-gram indexing or fuzzy matching.

**Rule of thumb:** Index more aggressively than you query. It's better to have multiple representations of a term in the index than to miss a match.

---

## Normalization

Normalization reduces surface variation so that equivalent forms match.

### Case Folding

Convert all text to lowercase (or uppercase, but lowercase is convention):

```
"Search" → "search"
"NASA" → "nasa"
"iPhone" → "iphone"
```

**Almost always a good idea.** The exceptions:
- "US" (United States) vs. "us" (pronoun) — case carries meaning
- Proper noun disambiguation: "Apple" (company) vs. "apple" (fruit)
- Acronyms: "IT" (information technology) vs. "it"

In practice, case fold anyway and handle ambiguity through other signals (context, entity recognition, click data).

### Unicode Normalization

The same visual character can have multiple byte representations in Unicode:

**Composed vs. decomposed forms:**
- "é" can be NFC (single code point U+00E9) or NFD (two code points: U+0065 + U+0301, i.e., "e" + combining acute accent)
- These look identical but are different byte sequences — without normalization, they won't match

**The four normalization forms:**

| Form | Name | Description |
|---|---|---|
| NFC | Canonical Composition | Decomposes then recomposes by canonical equivalence. Most compact. |
| NFD | Canonical Decomposition | Fully decomposes by canonical equivalence. |
| NFKC | Compatibility Composition | Like NFC but also normalizes compatibility equivalences (e.g., "ﬁ" → "fi") |
| NFKD | Compatibility Decomposition | Like NFD but also decomposes compatibility characters |

**For search, use NFKC.** It normalizes the broadest set of equivalent forms:
- Ligatures: "ﬁ" → "fi"
- Full-width characters: "Ｈｅｌｌｏ" → "Hello" (important for CJK input)
- Superscripts/subscripts: "x²" → "x2"
- Fractions: "½" → "1/2" (depending on implementation)

### Accent Folding (Diacritics Removal)

Remove accent marks so that accented and unaccented forms match:

```
"café" → "cafe"
"naïve" → "naive"
"résumé" → "resume"
"München" → "Munchen"
```

Implementation: apply NFD normalization, then strip characters in the Unicode "Combining Diacritical Marks" block (U+0300 to U+036F).

**Trade-off:** In some languages, diacritics change meaning:
- Spanish: "año" (year) vs. "ano" (anus)
- Turkish: "i" and "ı" are different letters ("i" has a dot, "ı" doesn't); also, uppercase "i" → "İ" (not "I")

**Common approach:** Index both the accent-folded and original forms. Search against the folded form by default, but boost exact matches.

### Synonym Handling

Synonyms can be handled at index time, query time, or both:

**Index-time expansion:**
```
"US" → index as ["US", "United States", "USA", "U.S.A."]
```
Pro: Query stays simple. Con: Index is larger; updating synonyms requires reindexing.

**Query-time expansion:**
```
query "US economy" → "(US OR United States OR USA) AND economy"
```
Pro: Synonyms can be updated without reindexing. Con: More complex queries, higher query cost.

**In Elasticsearch**, synonym filters can be applied at either stage. The recommendation is query-time for synonym lists that change frequently, and index-time for stable mappings (like unit conversions: "kg" ↔ "kilogram").

---

## Stemming

Stemming reduces words to their **stem** (base form) by stripping suffixes using heuristic rules. The stem doesn't have to be a real word.

### Porter Stemmer

The most widely used stemmer for English. Published by Martin Porter in 1980. Applies a cascade of ~60 rules in 5 steps:

```
Step 1: Plurals and past participles
  "caresses" → "caress" → "caress"
  "ponies"   → "poni"
  "cats"     → "cat"
  "running"  → "run"   (after removing "-ing", check for consonant doubling)

Step 2: Derivational suffixes
  "relational"  → "relate"
  "conditional" → "condition"
  "hesitanci"   → "hesitant"  (from step 1: "hesitancy" → "hesitanci")

Step 3: More derivational suffixes
  "triplicate" → "triplic"
  "electrical" → "electric"

Step 4: Remove suffixes
  "revival"   → "reviv"
  "allowance" → "allow"
  "inference" → "infer"

Step 5: Clean up
  "probate" → "probat"
  "rate"    → "rate"
  "cease"   → "ceas"
```

**Properties:**
- Fast: $O(n)$ per word where $n$ = word length
- Aggressive: groups many word forms together (high recall)
- Error-prone: "university" and "universe" both stem to "univers" (false conflation)
- Language-specific: designed for English only

### Snowball Stemmer

Porter's improved framework (also called Porter2). Provides stemmers for 20+ languages using a domain-specific language for defining stemming rules.

```
English Snowball examples:
  "generously" → "generous"  (Porter gives "gener")
  "communism"  → "communism" (Porter gives "commun")
```

Snowball is generally preferred over the original Porter: fewer errors, more languages, similar speed.

### Krovetz Stemmer

Takes a hybrid approach:
1. First, check a dictionary of known word forms
2. If the word is in the dictionary, look up its morphological root
3. If not, apply light suffix stripping (fewer rules than Porter)

```
"walking" → look up → "walk" (dictionary hit)
"xylophoning" → not in dictionary → apply rules → "xylophon"
```

**Properties:**
- Less aggressive than Porter — fewer false conflations
- Higher precision, lower recall than Porter
- Requires a dictionary (adds memory overhead, ~50KB)
- Better for precise search (legal, medical, technical)

### Examples of Stemming Errors

**Over-stemming** (false conflation — unrelated words mapped to the same stem):

| Word 1 | Word 2 | Stem | Problem |
|---|---|---|---|
| university | universe | univers | Completely different concepts |
| operating | operation | oper | Verb vs. noun (but actually related) |
| policy | police | polic | Different meanings |
| organization | organ | organ | Different meanings |

**Under-stemming** (failure to conflate related words):

| Word 1 | Word 2 | Stem 1 | Stem 2 | Problem |
|---|---|---|---|---|
| absorb | absorption | absorb | absorpt | Same root, different stems |
| alumni | alumnus | alumni | alumnus | Irregular plural not handled |
| European | Europe | european | europ | Derivation not captured |

---

## Lemmatization

Lemmatization reduces words to their **lemma** (dictionary form) using morphological analysis. Unlike stemming, the output is always a valid word.

### How It Works

Lemmatization requires:
1. A **morphological dictionary** mapping word forms to lemmas
2. **Part-of-speech (POS) disambiguation** — "better" as adjective → "good"; "better" as verb → "better"
3. **Irregular form handling** — "went" → "go", "mice" → "mouse", "criteria" → "criterion"

```
Input:  "The mice were running quickly through better passages"

POS:    DET  NOUN  AUX  VERB     ADV     PREP  ADJ    NOUN

Lemma:  "the mouse  be   run      quickly through good   passage"
```

### Comparison with Stemming

```
Word         | Porter Stem | Lemma
-------------|-------------|-------
"running"    | "run"       | "run"
"better"     | "better"    | "good" (adj) / "better" (verb)
"corpora"    | "corpora"   | "corpus"
"studies"    | "studi"     | "study"
"was"        | "wa"        | "be"
"feet"       | "feet"      | "foot"
"leaves"     | "leav"      | "leaf" (noun) / "leave" (verb)
```

### Tools

- **WordNet-based**: NLTK's WordNetLemmatizer. Requires POS tag input.
- **SpaCy**: Built-in lemmatizer using lookup tables + rules. Fast, production-grade.
- **Stanford CoreNLP**: Morphological analysis. Best for accuracy, slower.
- **Lucene/Elasticsearch**: No built-in lemmatizer for English. You'd use a custom token filter backed by one of the above, or use the `hunspell` filter with a morphological dictionary.

### Cost

Lemmatization is significantly more expensive than stemming:

| Property | Stemming | Lemmatization |
|---|---|---|
| Speed | $O(n)$ per token, no lookup | $O(1)$ dictionary lookup + $O(n)$ rule fallback |
| Memory | ~0 (pure rules) | 10-100MB (dictionary + model) |
| Accuracy | Good for English, degrades for morphologically rich languages | Better across languages |
| POS needed | No | Yes (for disambiguation) |
| Index-time impact | Negligible | Noticeable (POS tagging is the bottleneck) |

---

## Stemming vs. Lemmatization: When to Use Which

| Factor | Stemming | Lemmatization |
|---|---|---|
| **Recall priority** | Better — more aggressive conflation | Lower — more precise grouping |
| **Precision priority** | Lower — false conflations hurt | Better — valid word forms only |
| **Speed** | Faster | Slower (POS tagging required) |
| **Web search** | Preferred (recall matters, speed matters) | Rarely used |
| **E-commerce** | Common (user queries are short and varied) | Used for specific attributes |
| **Legal/medical** | Risky (conflation can change meaning) | Preferred |
| **Multilingual** | Snowball covers 20+ languages | Requires per-language dictionaries |
| **Neural search** | Neither (subword tokenization replaces both) | Neither |

**The practical answer for most systems:** Use Snowball stemming. The recall gain outweighs the precision loss, and the speed/simplicity advantage is real. Add lemmatization only if you have a specific precision problem (e.g., "organ" matching "organization" is causing bad results).

**The modern answer:** If you're building a hybrid system with dense retrieval (BERT-based), the dense component handles the semantic matching that stemming/lemmatization approximate. You might not need either for the keyword component, relying on the dense retriever to catch vocabulary variations. But for pure keyword search systems, stemming remains essential.

---

## Stop Words

Stop words are high-frequency, low-information words: "the", "is", "at", "which", "and".

### The Case for Removing Stop Words

- They appear in almost every document, so their posting lists are enormous (the IDF is near zero)
- They consume index space disproportionately — "the" might appear in 99% of documents
- They add noise to relevance scoring — matching "the" shouldn't count toward similarity

### The Case Against Removing Stop Words

- **Phrase queries break**: "to be or not to be" becomes "" after stop word removal
- **Meaningful phrases**: "The Who", "Let It Be", "IT department"
- **Negation**: "not working" — removing "not" changes meaning entirely
- **Modern systems handle it differently**: With TF-IDF or BM25, stop words naturally get near-zero weight. You don't need to physically remove them — the scoring function effectively ignores them.

### Modern Approach

Most modern search systems **do not remove stop words at indexing**. Instead:

1. Keep stop words in the index (needed for phrase queries and positional information)
2. Let the ranking function (BM25) naturally de-weight them via IDF
3. Optionally remove them from the query's scoring terms while keeping them for phrase matching
4. Use stop word lists only for specific purposes (e.g., autocomplete suggestions, query analysis)

Elasticsearch's default analyzer does not remove stop words. Lucene's StandardAnalyzer stopped removing stop words in version 3.1 (2011).

---

## Putting It All Together: Analyzer Chains

Here's how a practical analyzer chain looks in Elasticsearch:

### Standard English Analyzer

```json
{
  "analyzer": {
    "english_search": {
      "type": "custom",
      "char_filter": ["html_strip"],
      "tokenizer": "standard",
      "filter": [
        "lowercase",
        "english_possessive_stemmer",
        "english_stop",
        "english_stemmer"
      ]
    }
  },
  "filter": {
    "english_possessive_stemmer": {
      "type": "stemmer",
      "language": "possessive_english"
    },
    "english_stop": {
      "type": "stop",
      "stopwords": "_english_"
    },
    "english_stemmer": {
      "type": "stemmer",
      "language": "english"
    }
  }
}
```

Processing trace:

```
Input:  "<p>The Runner's running shoes aren't available.</p>"

1. html_strip:     "The Runner's running shoes aren't available."
2. standard:       ["The", "Runner's", "running", "shoes", "aren't", "available"]
3. lowercase:      ["the", "runner's", "running", "shoes", "aren't", "available"]
4. possessive:     ["the", "runner", "running", "shoes", "aren't", "available"]
5. english_stop:   ["runner", "running", "shoes", "aren't", "available"]
6. english_stem:   ["runner", "run", "shoe", "aren't", "avail"]
```

### Asymmetric Analysis (Index vs. Query)

A powerful pattern: apply different analyzers at index time and query time.

**Index time:** Apply synonym expansion + stemming (maximize recall)
```
"laptop" → index as ["laptop", "notebook", "notebook computer"]
```

**Query time:** Apply minimal processing (preserve user intent)
```
"laptop" → query for "laptop" (exact match boosted, synonyms from index cover recall)
```

This avoids query-time synonym expansion blowup while still matching synonyms via the index.

---

## Impact on Search Quality

Each text processing decision has measurable effects:

| Decision | Recall Impact | Precision Impact | Index Size Impact |
|---|---|---|---|
| Aggressive stemming | ↑ +15-30% | ↓ -5-15% | ↓ Smaller (fewer unique terms) |
| Lemmatization | ↑ +5-15% | ↑ +2-5% | ↓ Slightly smaller |
| Stop word removal | ↓ -2-5% (phrases) | ↑ +1-3% | ↓ -20-30% |
| Accent folding | ↑ +5-10% (multilingual) | ↓ -1-2% | ↓ Slightly smaller |
| Synonym expansion | ↑ +10-25% | ↓ -5-10% | ↑ Larger (index-time) |
| Character n-grams | ↑ +20-40% (fuzzy) | ↓ -10-20% | ↑ Much larger (3-5x) |

These numbers are approximate and domain-dependent. Always measure on your data.

**The fundamental tension:** Every normalization step trades precision for recall. Stemming groups more words together (higher recall) but sometimes groups unrelated words (lower precision). The right balance depends on your use case:

- **E-commerce**: Lean toward recall (users are forgiving, showing extra results is OK)
- **Legal search**: Lean toward precision (false matches have real consequences)
- **Log search**: Minimal normalization (users want exact pattern matching)

## References

- Manning, C. D., Raghavan, P., & Schütze, H. — *Introduction to Information Retrieval*, Chapters 2-3 (Text processing and indexing)
- Elasticsearch Reference — "Analysis" (analysis module documentation)
- Porter, M. F. — "An algorithm for suffix stripping" (Program, 1980)
- Krovetz, R. — "Viewing morphology as an inference process" (SIGIR 1993)
- Sennrich, R. et al. — "Neural Machine Translation of Rare Words with Subword Units" (ACL 2016) — BPE
- Kudo, T. & Richardson, J. — "SentencePiece: A simple and language independent subword tokenizer" (EMNLP 2018)
