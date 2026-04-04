# Search Knowledge Base

A comprehensive guide to search engineering — from fundamentals and data structures to modern vector search, learning to rank, and RAG.

## Table of Contents

### [01 – Fundamentals](./01-fundamentals)
- What is search (problem definition, types of search)
- Mathematical foundations (set theory, probability, linear algebra for search)
- Text representation basics (tokenization, stemming, normalization)
- Data structures for search (inverted index, tries, suffix arrays, B-trees, skip lists)

### [02 – Core Concepts](./02-core-concepts)
- Relevance and ranking
- Precision, recall, F1, MAP, NDCG, MRR
- TF-IDF and BM25
- Boolean retrieval model
- Vector space model
- Query processing pipeline (parsing, expansion, rewriting)

### [03 – Classical Search Algorithms](./03-classical-search)
- Full-text search (inverted index construction, posting list operations)
- Fuzzy search (edit distance, n-grams, phonetic matching)
- Faceted search and filtering
- Autocomplete and typeahead (prefix trees, FSTs)
- Geospatial search (R-trees, geohashing, Haversine)
- Sorting and pagination strategies

### [04 – Semantic & Vector Search](./04-semantic-vector-search)
- Word embeddings (Word2Vec, GloVe, FastText)
- Sentence/document embeddings (BERT, Sentence-BERT)
- Approximate nearest neighbor (LSH, HNSW, IVF, PQ)
- Dense retrieval vs sparse retrieval
- Hybrid search (combining keyword + vector)
- Cross-encoder re-ranking

### [05 – Search System Architecture](./05-architecture)
- Crawling and ingestion pipelines
- Indexing strategies (forward index, inverted index, columnar)
- Sharding and replication
- Real-time vs batch indexing
- Query routing and federation
- Caching strategies

### [06 – Search Engines & Tools](./06-engines-and-tools)
- Lucene internals
- Elasticsearch / OpenSearch
- Solr
- Meilisearch, Typesense
- Vector databases (Pinecone, Weaviate, Qdrant, Milvus, pgvector)
- Tantivy, Bleve and other embedded engines

### [07 – Learning to Rank & ML in Search](./07-learning-to-rank)
- Pointwise, pairwise, listwise approaches
- Feature engineering for ranking
- LambdaMART, RankNet, LambdaRank
- Click models and implicit feedback
- Neural ranking models (ColBERT, monoT5)
- Query understanding with ML (intent classification, entity recognition)

### [08 – Advanced Topics](./08-advanced-topics)
- Retrieval-Augmented Generation (RAG)
- Multi-modal search (image, audio, video)
- Conversational search
- Personalization and context-aware search
- Knowledge graphs in search
- Federated / distributed search
- Search result diversification

### [09 – Practical Guides](./09-practical-guides)
- Designing a search quality evaluation pipeline
- A/B testing for search
- Relevance tuning playbook
- Common failure modes and debugging
- Scaling search (10K → 1B documents)
- Building search for e-commerce / enterprise / logs

### [10 – Resources](./10-resources)
- Books (IR textbooks, Manning et al., Büttcher et al.)
- Courses and lectures
- Papers (PageRank, BM25, BERT for IR, ColBERT)
- Blogs and communities
- Datasets and benchmarks (TREC, MS MARCO, BEIR)

