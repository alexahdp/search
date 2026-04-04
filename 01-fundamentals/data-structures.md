# Data Structures for Search

This section covers the core data structures that make search engines fast. The inverted index is the star — it's the primary structure behind every major search engine — but tries, suffix arrays, B-trees, and skip lists each play important supporting roles.

## Table of Contents

1. [The Inverted Index](#the-inverted-index)
2. [Tries](#tries)
3. [Suffix Arrays](#suffix-arrays)
4. [B-Trees](#b-trees)
5. [Skip Lists](#skip-lists)
6. [Comparison and Trade-offs](#comparison-and-trade-offs)
7. [References](#references)

---

## The Inverted Index

The inverted index is *the* fundamental data structure in information retrieval. It maps **terms → document lists**, which is the inverse of the natural document → terms relationship (hence "inverted").

### Structure

An inverted index has two main components:

**1. Term Dictionary (Lexicon)**

Maps each unique term to metadata and a pointer to its posting list:

```
Term Dictionary:
┌────────────┬───────┬──────────┬─────────────────┐
│   Term     │  DF   │ Offset   │ Posting Pointer  │
├────────────┼───────┼──────────┼─────────────────┤
│ algorithm  │  1204 │ 0x4A20   │ ──────────────► │
│ binary     │   893 │ 0x4B08   │ ──────────────► │
│ computer   │  3571 │ 0x4C10   │ ──────────────► │
│ data       │  5892 │ 0x4D00   │ ──────────────► │
│ ...        │  ...  │ ...      │                  │
└────────────┴───────┴──────────┴─────────────────┘
```

The dictionary is typically stored as:
- **Sorted array**: Binary search for lookup, $O(\log V)$ where $V$ = vocabulary size
- **Hash table**: $O(1)$ lookup, but no prefix queries, no range queries
- **Trie / FST**: $O(|term|)$ lookup, supports prefix queries, very compact (see [Tries](#tries) below)

Lucene uses a **Finite State Transducer (FST)** — a compressed trie that maps terms to their posting list offsets. The FST is memory-mapped and kept in RAM for fast access.

**2. Posting Lists**

For each term, a sorted list of document IDs (and optionally positions, frequencies, payloads):

```
"search" → [3, 5, 12, 28, 33, 41, 67, 89, 102, ...]
"engine" → [5, 12, 41, 67, 150, ...]
```

A posting list entry typically contains:

| Field | Description | Used for |
|---|---|---|
| docID | Document identifier | Candidate retrieval |
| tf | Term frequency in this doc | BM25 scoring |
| positions | List of token offsets | Phrase queries, proximity scoring |
| offsets | Character start/end offsets | Highlighting |
| payloads | Arbitrary per-position data | Custom scoring signals |

**Positional index** (stores positions):

```
"search" → doc3: [12, 45, 89],  doc5: [3, 77],  doc12: [101], ...
```

Positions enable **phrase queries** ("search engine" requires "search" and "engine" at adjacent positions) and **proximity queries** ("search" NEAR/5 "ranking").

The cost: positional indices are 2-4x larger than non-positional ones.

### Index Construction

#### In-Memory Construction (BSBI — Blocked Sort-Based Indexing)

For corpora that fit in memory:

```
1. For each document d:
     For each term t in d:
       Add (t, docID, position) to in-memory list

2. Sort by (term, docID, position)

3. Group by term → posting lists

4. Write dictionary + posting lists to disk
```

**Time:** $O(T \log T)$ where $T$ = total term occurrences in the corpus.

#### External Sorting (SPIMI — Single-Pass In-Memory Indexing)

For corpora that don't fit in memory:

```
1. Process documents in blocks that fit in memory
2. For each block:
   a. Build in-memory inverted index (hash map: term → posting list)
   b. Sort terms and write block to disk as a "run"
3. Merge all runs using k-way merge sort

   Run 1: algorithm→[1,3]  binary→[2]     computer→[1,4]  ...
   Run 2: algorithm→[5]    binary→[5,7]   computer→[6]    ...
   Run 3: algorithm→[8,9]  binary→[10]    data→[8,9,11]   ...
                    ↓ merge ↓
   Final: algorithm→[1,3,5,8,9]  binary→[2,5,7,10]  computer→[1,4,6]  ...
```

**Time:** $O(T \log(T/M))$ where $M$ = memory block size.

#### Distributed Index Construction (MapReduce)

For web-scale corpora:

```
Map phase:    (docID, document) → emit (term, (docID, tf, positions))
Shuffle:      Group by term (each term goes to one reducer)
Reduce phase: (term, [(docID, tf, positions), ...]) → sort by docID → write posting list
```

This is how Google originally built its index. Modern systems use similar but more optimized approaches (Dremel-style columnar processing, streaming architectures).

### Posting List Compression

Posting lists dominate index size. Compression is essential for both storage and performance (less I/O = faster queries, better cache utilization).

**Key observation:** Posting lists are sorted by docID. Instead of storing absolute IDs, store **gaps** (differences between consecutive IDs):

```
Absolute: [3, 5, 12, 28, 33, 41, 67, 89]
Gaps:     [3, 2,  7, 16,  5,  8, 26, 22]
```

Gaps are typically small (especially for frequent terms), so they can be encoded in fewer bits.

#### Variable Byte Encoding (VByte)

The simplest practical compression. Each byte uses 7 bits for data and 1 bit (the high bit) as a continuation flag:

```
Value 5:    [10000101]                    → 1 byte
Value 130:  [00000001] [10000010]         → 2 bytes
Value 16385: [00000001] [00000000] [10000001] → 3 bytes

High bit: 1 = last byte of this number, 0 = more bytes follow
```

**Properties:**
- Simple to implement (no lookup tables, no state)
- Decoding is fast: ~1 billion integers/second on modern hardware
- Compression ratio: ~1.5-2 bytes per gap for typical posting lists
- No random access — must decode sequentially

#### PForDelta (Patched Frame of Reference)

A block-based compression scheme used in Lucene:

```
1. Take a block of 128 gaps
2. Find the "frame" — a bit width that covers most values (e.g., 90th percentile)
3. Encode most values in that bit width (packed tightly)
4. "Patch" exceptions (values that don't fit) in a separate overflow area

Block of gaps: [2, 3, 1, 5, 2, 3, 200, 1, 4, 2, ...]
                                    ↑ exception

Frame = 4 bits (covers values 0-15, ~95% of gaps)
Patch: position 6 → value 200 (stored separately)
```

**Properties:**
- Excellent decode speed: 2-4 billion integers/second (SIMD-optimized)
- Good compression ratio: ~1-1.5 bytes per gap
- Block-oriented: supports skip-by-block for fast intersection
- Lucene uses a variant called `FOR` (Frame of Reference) with blocks of 128 or 256 values

#### Other Compression Methods

| Method | Compression | Decode Speed | Notes |
|---|---|---|---|
| Uncompressed (32-bit) | 4 bytes/gap | Fastest | Baseline |
| VByte | ~1.5 bytes/gap | Fast | Simplest variable-length encoding |
| Group Varint | ~1.5 bytes/gap | Very fast | VByte variant, SIMD-friendly. Used by Google |
| PForDelta | ~1.0 bytes/gap | Very fast | Block-based, SIMD-optimized |
| Simple9/Simple16 | ~1.0 bytes/gap | Fast | Pack multiple small ints in one 32-bit word |
| Elias-Fano | ~0.6 bytes/gap | Fast | Quasi-succinct; supports random access |
| Roaring Bitmaps | Varies | Very fast | Hybrid: arrays for sparse, bitmaps for dense |

**Lucene's approach:** Uses a combination of PForDelta for posting lists and Roaring Bitmap-like structures for document sets (filters, deleted docs).

### Skip Pointers in Posting Lists

Skip pointers allow jumping ahead in a posting list without scanning every entry. Critical for efficient intersection of posting lists.

```
Level 2: [3]─────────────────────────────────►[89]
Level 1: [3]──────────►[28]──────────────────►[89]
Level 0: [3]──►[12]──►[28]──►[41]──►[67]──►[89]
Base:    [3, 5, 12, 28, 33, 41, 67, 89, 102, ...]
```

**Multi-level skip lists** (used in Lucene):
- Level 0 skip points every $S$ entries (e.g., every 128)
- Level 1 skip points every $S^2$ entries
- Level $i$ skip points every $S^{i+1}$ entries

At each skip point, store:
- The docID at that position
- The byte offset into the posting list at that position
- The byte offset into the position data at that position

**Impact on intersection:**

Without skip pointers, intersecting lists of length $L_1$ and $L_2$: $O(L_1 + L_2)$

With skip pointers (skip interval $\sqrt{L}$): $O(\sqrt{L_1} + \sqrt{L_2})$ in the best case

In practice, the speedup depends on the selectivity of the intersection. When one list is much shorter than the other, skip pointers on the long list provide dramatic speedups.

### Index Segments and Merging

Lucene (and Elasticsearch/Solr built on it) uses a **segmented index** architecture:

```
Segment 0 (large, older)  ──┐
Segment 1 (medium)        ──┼── Logical index
Segment 2 (small, newer)  ──┘

Each segment is a complete, immutable mini-index:
  - Its own term dictionary
  - Its own posting lists
  - Its own stored fields
```

**Why segments?**
1. **Writes don't block reads.** New documents go into a new segment. Existing segments are never modified.
2. **Concurrency.** Readers access segments without locks (immutability guarantees).
3. **Transactional commit.** A new segment is atomically visible when its commit point is written.

**Merge policy:** Periodically, segments are merged into larger ones (Lucene's `TieredMergePolicy`):
- Reduces the number of segments (fewer file handles, fewer dictionary lookups per query)
- Physically deletes documents marked as deleted
- Trade-off: merging is I/O intensive and can impact query latency during peaks

**Deletion:** Since segments are immutable, deletes are handled via a **bitset** (.del file) that marks which docIDs in a segment are deleted. Queries filter out deleted docs at read time. Physical removal happens during merging.

---

## Tries

A trie (from "re**trie**val", pronounced "try") is a tree structure for storing a set of strings where each edge represents a character.

### Structure

```
Trie for {"search", "sea", "see", "set", "sell"}:

          (root)
           │
           s
           │
         ┌─e─┐
         │   │
      ┌─a─┐  ┌─e─┐  ┌─l─┐  ┌─t─┐
      │      │      │      │
   ┌─r─┐   (*)    l     (*)
   │               │
   c              (*)
   │
   h
   │
  (*)

(*) marks end of a valid string
```

### Properties

| Operation | Time Complexity | Space Complexity |
|---|---|---|
| Insert | $O(L)$ | $O(L)$ new nodes in worst case |
| Lookup | $O(L)$ | — |
| Delete | $O(L)$ | — |
| Prefix search | $O(P + K)$ | — |

Where $L$ = string length, $P$ = prefix length, $K$ = number of matching entries.

**Space:** Naive trie with pointer-per-character: $O(N \cdot L \cdot |\Sigma|)$ where $|\Sigma|$ = alphabet size. This is enormous — each node has up to 256 (or more for Unicode) child pointers.

### Compressed Tries

**Patricia trie (radix tree):** Collapse single-child chains into single edges labeled with substrings:

```
Naive trie:               Radix tree:
  s─e─a─r─c─h              s─e─┐
       │                       ├─a─r─c─h
       a                       ├─e
                               ├─l─l
                               └─t
```

This dramatically reduces node count, especially for long strings with shared prefixes.

**LOUDS/Succinct tries:** Represent the trie topology in ~2 bits per node using level-order encoding. Used when the trie must be in memory but space is critical.

### Finite State Transducers (FSTs)

An FST is a generalization of a trie that maps keys to values (like a trie + output). Lucene's term dictionary uses an FST:

```
Input: "cat" → Output: 42 (posting list offset)
Input: "car" → Output: 37
Input: "cart" → Output: 55
```

The FST is a minimal deterministic acyclic finite state transducer — it shares both prefixes AND suffixes, making it extremely compact. Lucene's FST for a term dictionary is typically 3-5x smaller than the sorted term array it replaces.

**Properties:**
- $O(L)$ lookup time
- Supports prefix iteration (for autocomplete)
- Memory-mappable (can be used without fully loading into RAM)
- Immutable (rebuilt when segments merge)
- Lucene's FST implementation is in `org.apache.lucene.util.fst`

### Applications in Search

1. **Term dictionary storage** — FSTs in Lucene, tries in Solr's suggest component
2. **Autocomplete / typeahead** — Prefix query: "sear" → return all terms starting with "sear" and their frequencies
3. **Spell correction** — Find terms within edit distance $k$ by traversing the trie with allowed substitutions/insertions/deletions (Levenshtein automaton)
4. **IP routing** — Longest prefix match on IP addresses (not search, but same structure)

---

## Suffix Arrays

A suffix array is a sorted array of all suffixes of a string (or all suffixes of all documents in a corpus). It enables fast substring search.

### Construction

Given string $S$ = "banana$":

```
Suffixes:
  0: banana$
  1: anana$
  2: nana$
  3: ana$
  4: na$
  5: a$
  6: $

Sorted suffixes:
  6: $
  5: a$
  3: ana$
  1: anana$
  0: banana$
  4: na$
  2: nana$

Suffix Array SA = [6, 5, 3, 1, 0, 4, 2]
```

### Substring Search

To find all occurrences of pattern $P$ in text $S$:
- Binary search in the suffix array for the range of suffixes that start with $P$
- Each match gives a starting position in $S$

```
Search for "ana" in "banana$":

Binary search in sorted suffixes:
  "ana" matches:
    SA[2] = 3 → suffix "ana$"    → occurrence at position 3
    SA[3] = 1 → suffix "anana$"  → occurrence at position 1

Found 2 occurrences at positions 1 and 3.
```

**Time:** $O(|P| \log |S|)$ for the binary search. With an **LCP array** (Longest Common Prefix), this improves to $O(|P| + \log |S|)$.

### Construction Algorithms

| Algorithm | Time | Space | Notes |
|---|---|---|---|
| Naive (sort all suffixes) | $O(n^2 \log n)$ | $O(n)$ | Simple but slow |
| Prefix doubling (Karp-Miller-Rosenberg) | $O(n \log^2 n)$ or $O(n \log n)$ | $O(n)$ | Practical, well-understood |
| SA-IS (Nong, Zhang, Chan 2009) | $O(n)$ | $O(n)$ | Optimal, used in practice |
| DC3/Skew (Kärkkäinen-Sanders 2003) | $O(n)$ | $O(n)$ | First practical linear-time algorithm |

SA-IS is the state-of-the-art for suffix array construction — linear time, small constant, and straightforward to implement.

### Enhanced Suffix Arrays

The suffix array alone supports substring search. Combined with auxiliary structures, it can replace a suffix tree:

- **LCP array** — Longest Common Prefix between adjacent suffixes in the sorted order. Enables $O(|P| + \log n)$ search and efficient duplicate detection.
- **Inverse suffix array** — $ISA[i]$ = rank of suffix starting at position $i$. Enables $O(1)$ lookup of a suffix's position in the sorted order.
- **Child table** — Enables top-down traversal of the implicit suffix tree in $O(1)$ per edge.

### Applications in Search

1. **Substring search** — Find documents containing an exact substring (useful for code search, log search)
2. **Longest repeated substring** — LCP array scan, useful for deduplication and plagiarism detection
3. **Document-level suffix arrays** — Concatenate all documents (with separators), build one suffix array. Each match maps back to a document ID. Used in some code search engines.
4. **Burrows-Wheeler Transform** — The BWT is derived from the suffix array and enables powerful compressed text indices (FM-index). Used in bioinformatics (read alignment) and compressed full-text indices.

### Trade-offs vs. Inverted Index

| Aspect | Suffix Array | Inverted Index |
|---|---|---|
| Query type | Substring / pattern matching | Term-level (whole word) matching |
| Space | $O(n)$ integers (4-8 bytes per character) | $O(T)$ postings (T = total term occurrences) |
| Query speed | $O(|P| \log n)$ | $O(|P|)$ dictionary lookup + posting list scan |
| Construction | Linear time, but large constant | Sort-based, well-pipelined |
| Updates | Expensive (rebuild) | Efficient (new segment) |
| Practical use | Code search, bioinformatics | General text search |

For general-purpose search, the inverted index wins. Suffix arrays shine when you need true substring matching (not just whole-word matching) or when the documents are structured data like source code or DNA sequences.

---

## B-Trees

B-trees are balanced search trees optimized for disk I/O, where each node contains multiple keys and has a high branching factor.

### Structure

A B-tree of order $m$:
- Each node has at most $m$ children
- Each non-root node has at least $\lceil m/2 \rceil$ children
- Each node contains between $\lceil m/2 \rceil - 1$ and $m - 1$ keys
- All leaves are at the same depth

```
B-tree of order 4 (max 3 keys per node):

              [20 | 40]
             /    |    \
    [5|10|15]  [25|30]  [45|50|60]
```

### Complexity

| Operation | Average | Worst Case |
|---|---|---|
| Search | $O(\log_m n)$ | $O(\log_m n)$ |
| Insert | $O(\log_m n)$ | $O(\log_m n)$ |
| Delete | $O(\log_m n)$ | $O(\log_m n)$ |
| Range scan | $O(\log_m n + k)$ | $O(\log_m n + k)$ |

Where $n$ = number of keys, $m$ = order, $k$ = number of results in range.

The key insight: by choosing $m$ so that a node fills one disk page (e.g., $m = 1000$ for 4KB pages with 4-byte keys), a B-tree of height 3 can index a billion keys with only 3 disk seeks.

### B+ Trees

The **B+ tree** variant stores all values in leaf nodes (internal nodes only contain keys for routing). Leaf nodes are linked in a doubly-linked list for efficient range scans.

```
B+ tree:
                [20 | 40]              ← internal (routing only)
               /    |    \
    [5|10|15]──[20|25|30]──[40|45|50]  ← leaves (data + linked list)
```

B+ trees are the standard for database indices and file systems (ext4, NTFS, HFS+).

### Role in Search Systems

B-trees are **not** the primary index structure for search (that's the inverted index), but they appear in:

1. **Database-backed search**: PostgreSQL's `ts_vector` full-text search uses GIN (Generalized Inverted Index), which is built on B-tree internals for the term dictionary.

2. **Metadata filtering**: When search systems need to filter by structured fields (price range, date range, category), B-tree indices on those fields provide efficient range queries.

3. **Lucene's sorted doc values**: When Lucene needs to sort results by a field (e.g., sort by price), it uses columnar storage with B-tree-like structures for efficient field value lookup.

4. **Term dictionary**: While Lucene uses an FST, some simpler search systems use a B-tree for the term dictionary. A B-tree supports prefix queries, range queries, and is straightforward to implement.

5. **Document stores**: Elasticsearch's doc values and stored fields are backed by memory-mapped files with B-tree-like access patterns.

### B-Tree vs. Hash Table for Term Dictionary

| Aspect | B-Tree | Hash Table |
|---|---|---|
| Exact lookup | $O(\log n)$ | $O(1)$ amortized |
| Prefix query | Supported | Not supported |
| Range query | Supported | Not supported |
| Disk-friendly | Yes (node = page) | No (random access) |
| Sorted iteration | Natural | Requires separate sort |
| Space overhead | Moderate | Low to moderate |

For term dictionaries in search, B-trees or FSTs are preferred over hash tables because prefix queries ("auto*") and sorted iteration (for segment merging) are essential operations.

---

## Skip Lists

A skip list is a probabilistic data structure that provides $O(\log n)$ search, insert, and delete in a sorted sequence, using a hierarchy of linked lists.

### Structure

```
Level 3: HEAD ────────────────────────────────── 67 ──── NIL
Level 2: HEAD ──── 12 ─────────── 33 ────────── 67 ──── NIL
Level 1: HEAD ──── 12 ──── 25 ── 33 ──── 41 ── 67 ──── NIL
Level 0: HEAD ── 5 ── 12 ── 25 ── 28 ── 33 ── 41 ── 67 ── 89 ── NIL
```

Each element appears in level 0. For each additional level, the element is promoted with probability $p$ (typically $p = 1/2$). Expected number of levels: $O(\log_{1/p} n)$.

### Complexity

| Operation | Expected | Worst Case |
|---|---|---|
| Search | $O(\log n)$ | $O(n)$ (vanishingly unlikely) |
| Insert | $O(\log n)$ | $O(n)$ |
| Delete | $O(\log n)$ | $O(n)$ |
| Space | $O(n)$ | $O(n \log n)$ |

Expected space with $p = 1/2$: $\frac{n}{1-p} = 2n$ pointers. Low overhead.

### Search Algorithm

```
SEARCH(skip_list, target):
    node ← skip_list.head
    for level ← max_level down to 0:
        while node.next[level] != NIL and node.next[level].key < target:
            node ← node.next[level]    // move forward
        // can't go further at this level, drop down
    node ← node.next[0]
    if node.key == target:
        return node
    return NOT_FOUND
```

The search starts at the highest level (fewest elements, biggest jumps) and works down. At each level, it moves forward as far as possible without overshooting the target, then drops to the next level for finer-grained search.

### Insert Algorithm

```
INSERT(skip_list, key, value):
    // Phase 1: Find the insertion position at each level
    update[max_level] ← array of predecessors at each level
    node ← skip_list.head
    for level ← max_level down to 0:
        while node.next[level] != NIL and node.next[level].key < key:
            node ← node.next[level]
        update[level] ← node

    // Phase 2: Randomly determine height of new node
    new_level ← 0
    while random() < p and new_level < max_level:
        new_level++

    // Phase 3: Insert node and update forward pointers
    new_node ← Node(key, value, level=new_level)
    for level ← 0 to new_level:
        new_node.next[level] ← update[level].next[level]
        update[level].next[level] ← new_node
```

### Role in Search Systems

**Posting list skip pointers.** As discussed in the inverted index section, skip pointers on posting lists are essentially a skip list structure overlaid on the sorted posting list. Lucene's `SkipListReader`/`SkipListWriter` implement multi-level skip lists over compressed posting blocks.

**In-memory indices.** Skip lists are used as the in-memory write buffer (memtable) in LSM-tree-based storage engines:
- **LevelDB/RocksDB** use skip lists for their memtable
- **Redis** sorted sets are implemented as skip lists
- **Lucene's in-memory buffer** before flushing to a segment uses a structure with skip-list-like properties

**Why skip lists over balanced BSTs?**

| Aspect | Skip List | Red-Black / AVL Tree |
|---|---|---|
| Implementation | Simpler (no rotations) | Complex (rebalancing logic) |
| Concurrency | Lock-free variants are straightforward | Lock-free BSTs are extremely complex |
| Range iteration | Natural (follow level-0 forward pointers) | In-order traversal (more pointer chasing) |
| Cache behavior | Good (sequential forward pointers) | Poor (tree pointers are scattered) |
| Deterministic | No (probabilistic) | Yes |
| Expected performance | Same $O(\log n)$ | Guaranteed $O(\log n)$ |

The concurrency advantage is the main reason modern systems choose skip lists. ConcurrentSkipListMap in Java, used in Lucene's indexing buffer, supports concurrent reads and writes without global locks.

---

## Comparison and Trade-offs

### When to Use What

| Data Structure | Best For | Time (Search) | Space | Key Advantage |
|---|---|---|---|---|
| **Inverted index** | Term → document lookup | $O(1)$ dict + $O(k)$ posting scan | $O(T)$ | Core of text search |
| **Trie / FST** | Prefix matching, term dictionary | $O(L)$ | $O(N \cdot L)$ naive, much less for FST | Prefix queries, compact |
| **Suffix array** | Substring matching | $O(P \log n)$ | $O(n)$ integers | True substring search |
| **B-tree** | Range queries, disk-based index | $O(\log_m n)$ | $O(n)$ | Disk-optimal, range scans |
| **Skip list** | In-memory sorted index, concurrent access | $O(\log n)$ expected | $O(n)$ | Simple, concurrent, sequential |

### How They Compose in a Real Search Engine

A search engine like Elasticsearch/Lucene uses ALL of these together:

```
Query: "running shoes" with price_filter: [50, 100]

1. TERM DICTIONARY (FST/Trie)
   Look up "run" (after stemming) → posting list offset
   Look up "shoe" (after stemming) → posting list offset

2. POSTING LISTS (Inverted Index + Skip Lists)
   Intersect posting list for "run" with posting list for "shoe"
   Use skip pointers to accelerate intersection
   → Candidate set: [doc12, doc45, doc89, doc102, ...]

3. NUMERIC INDEX (B-tree / BKD-tree)
   Filter candidates by price ∈ [50, 100]
   Lucene uses a BKD-tree (variant of k-d tree) for numeric/geo fields
   → Filtered set: [doc45, doc89, ...]

4. SCORING (Sparse vector dot product)
   Compute BM25(query, doc) for each candidate
   This is a dot product in the term-weight vector space

5. SORT + TOP-K (Priority queue / heap)
   Return top-10 by BM25 score
```

### Complexity Summary

| Structure | Build | Point Query | Range Query | Insert | Delete | Space |
|---|---|---|---|---|---|---|
| Inverted index | $O(T \log T)$ | $O(1)$ + $O(k)$ | N/A (term-level) | Append (new segment) | Bitset mark | $O(T)$ |
| Trie | $O(N \cdot L)$ | $O(L)$ | $O(L + k)$ prefix | $O(L)$ | $O(L)$ | $O(N \cdot L \cdot \|\Sigma\|)$ |
| FST | $O(N \cdot L)$ | $O(L)$ | $O(L + k)$ prefix | Immutable (rebuild) | Immutable | $O(N \cdot L)$ compressed |
| Suffix array | $O(n)$ (SA-IS) | $O(P \log n)$ | $O(P + \log n + k)$ with LCP | Expensive (rebuild) | Expensive | $O(n)$ |
| B-tree | $O(n \log_m n)$ | $O(\log_m n)$ | $O(\log_m n + k)$ | $O(\log_m n)$ | $O(\log_m n)$ | $O(n)$ |
| Skip list | $O(n \log n)$ | $O(\log n)$ | $O(\log n + k)$ | $O(\log n)$ | $O(\log n)$ | $O(n)$ |

Where:
- $T$ = total term occurrences in corpus
- $N$ = number of strings/terms
- $L$ = average string length
- $n$ = number of elements/characters
- $P$ = pattern/query length
- $k$ = number of results
- $m$ = B-tree order
- $|\Sigma|$ = alphabet size

## References

- Manning, C. D., Raghavan, P., & Schütze, H. — *Introduction to Information Retrieval*, Chapters 1-5 (Index construction, compression)
- Zobel, J. & Moffat, A. — "Inverted Files for Text Search Engines" (ACM Computing Surveys, 2006)
- Pugh, W. — "Skip Lists: A Probabilistic Alternative to Balanced Trees" (Communications of the ACM, 1990)
- Nong, G., Zhang, S., & Chan, W. H. — "Two Efficient Algorithms for Linear Time Suffix Array Construction" (IEEE Transactions on Computers, 2011)
- McCandless, M., Hatcher, E., & Gospodnetić, O. — *Lucene in Action* (Manning Publications)
- Lemire, D. & Boytsov, L. — "Decoding billions of integers per second through vectorization" (Software: Practice and Experience, 2015) — PForDelta and SIMD techniques
- Bayer, R. & McCreight, E. — "Organization and Maintenance of Large Ordered Indexes" (Acta Informatica, 1972) — Original B-tree paper
