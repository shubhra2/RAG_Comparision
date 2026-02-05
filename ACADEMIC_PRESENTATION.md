# Evaluating Graph-Based Retrieval-Augmented Generation
## Comparative Analysis and Quality Improvements over Standard RAG Systems

**Student:** Shubhra Gadhwala
**BITS ID:** 2023AA05750
**Course:** AIMLCZG628T - Dissertation/Project Work
**Supervisor:** Darshan Derasari
**Organization:** WeblineIndia, Ahmedabad

---

## Table of Contents

1. Introduction and Problem Statement
2. Research Objectives
3. Literature Review
4. System Architecture
5. Standard RAG Implementation
6. Graph-Based RAG Implementation
7. Evaluation Framework
8. Experimental Results
9. Discussion and Analysis
10. Conclusions and Future Work

---

## 1. Introduction and Problem Statement

### Background

- RAG enhances LLMs by grounding responses in retrieved context
- Standard RAG relies on vector similarity search
- Documents embedded into dense vector spaces

### Problem Statement

**Limitations of Standard RAG:**

- Cannot explicitly model entity relationships
- Struggles with multi-hop queries requiring multiple reasoning steps
- Entity linking failures across documents
- Context precision issues with broad queries

### Research Question

**Can graph-structured knowledge representations improve contextual grounding and reasoning quality in RAG systems, particularly for multi-hop question answering tasks?**

---

## 2. Research Objectives

### Primary Objectives

1. **Build Standard RAG System**

   - Vector-based retrieval using ChromaDB
   - Text chunking and embedding pipeline
   - Integration with local LLMs via Ollama

2. **Build Graph-Based RAG System**

   - Knowledge graph using Neo4j
   - Entity and relationship extraction
   - Two implementations: Chain-based and Agentic

3. **Comparative Evaluation**

   - Benchmark on multi-hop queries
   - Quantitative analysis using RAGAS metrics
   - Qualitative analysis of system behaviors

---

## 3. Literature Review

### Key Works

**RAG Foundations:**

- Lewis et al. (2020): Dense passage retrieval with sequence-to-sequence models
- Guu et al. (2020): Retrieval-augmented pre-training

**Graph-Based Retrieval:**

- Pan et al. (2023): Survey on LLMs with knowledge graphs
- Baek et al. (2023): Knowledge-augmented prompting for KG QA

**Evaluation:**

- Es et al. (2023): RAGAS framework for automated RAG evaluation
- Yang et al. (2018): HotpotQA dataset for multi-hop QA

### Research Gap

Limited systematic comparison between standard vector-based RAG and graph-based RAG on common benchmarks.

---

## 4. System Architecture

### Technology Stack

**Core Framework:**

- LangChain: RAG pipelines, LLM integration
- LangChain Neo4j: Graph database integration

**Storage:**

- ChromaDB: Vector database
- Neo4j: Graph database

**Embeddings & LLMs:**

- Sentence Transformers: `all-MiniLM-L6-v2`
- Ollama: Local LLMs (Qwen2, TinyLlama, Gemma)

**Evaluation:**

- RAGAS: Faithfulness, Answer Relevancy, Context Precision/Recall

**Frontend:**

- Streamlit: Interactive dashboard and chat playground

---

## 5. Standard RAG Implementation

### Pipeline

1. **Data Preprocessing:** CSV → Parquet caching
2. **Text Chunking:** 1000 chars, 200 overlap
3. **Embedding:** Sentence transformers (384-dim)
4. **Vector Storage:** ChromaDB with metadata
5. **Retrieval:** Top-K similarity search (K=4)
6. **Answer Generation:** LLM with retrieved context

### Key Features

- Fast semantic similarity search
- Simple architecture
- Efficient for straightforward queries

---

## 6. Graph-Based RAG Implementation

### Knowledge Graph Construction

**Entity Types:**

- Article nodes: title, content, topic
- Researcher nodes: name
- Topic nodes: name

**Relationships:**

- `PUBLISHED`: (Researcher) → (Article)
- `IN_TOPIC`: (Article) → (Topic)
- `RELATED_TO`: (Article) → (Article)

### Two Approaches

**Chain-Based (Initial):**

- Single Cypher query generation
- Limited retry logic
- Struggles with multi-hop queries

**Agentic (Enhanced):**

- Iterative query execution
- Tool calling: Cypher queries, graph statistics
- Multi-step reasoning
- Superior for complex queries

---

## 7. Evaluation Framework

### RAGAS Metrics

**Faithfulness (0-1):** Answer grounded in retrieved context
**Answer Relevancy (0-1):** Relevance to user question
**Context Precision (0-1):** Quality of retrieved context
**Context Recall (0-1):** Coverage of relevant information

### Datasets

**Primary:** HotpotQA

- ~113k training, ~7.6k dev questions
- Multi-hop QA benchmark

**Secondary:** Synthetic Articles

- Controlled evaluation environment
- Known entity relationships

---

## 8. Experimental Results

### Quantitative Results

| Metric | Standard RAG | Graph RAG (Agentic) | Winner |
|--------|--------------|---------------------|--------|
| Faithfulness | 0.60 - 1.00 | 0.57 - 1.00 | Tie |
| Answer Relevancy | 0.47 - 0.85 | 0.60 - 0.79 | Mixed |
| Context Precision | 0.00 - 0.80 | **0.99 - 1.00** | Graph RAG |
| Context Recall | **0.65 - 1.00** | 0.33 - 0.73 | Standard RAG |

### Key Observations

1. **Graph RAG:** Near-perfect context precision (0.99-1.00)
2. **Standard RAG:** Better context recall (0.65-1.00)
3. **Multi-hop Queries:** Agentic Graph RAG superior
4. **Single-hop Queries:** Standard RAG comparable/better

---

## 9. Discussion and Analysis

### When Graph RAG Performs Better

- **Multi-hop Reasoning:** Explicit relationship traversal
- **Relational Queries:** Direct access to entity relationships
- **Context Precision:** Targeted entity retrieval (0.99-1.00)
- **Entity-centric Questions:** Graph structure benefits

### When Standard RAG Performs Better

- **Simple Semantic Queries:** Fast vector similarity
- **Broad Topic Queries:** Better coverage
- **Speed:** Lower latency for straightforward queries
- **Resource Efficiency:** Simpler architecture

### Trade-offs

- **Setup Complexity:** Graph RAG requires entity extraction
- **Query Latency:** Graph queries may be slower
- **Storage:** Graph databases require more space
- **Maintenance:** Graph schemas need updates

---

## 10. Conclusions and Future Work

### Key Findings

**Graph RAG Advantages:**

- Superior multi-hop reasoning
- Improved context precision
- Better relational query handling

**Standard RAG Advantages:**

- Simplicity and speed
- Resource efficiency
- Broad coverage for simple queries

**Key Insight:** Choice depends on query characteristics and use case requirements.

### Future Work

1. **Hybrid RAG System:** Combine vector search + graph traversal
2. **Advanced Entity Extraction:** Transformer-based NER
3. **Query Optimization:** Better NL to Cypher translation
4. **Scalability Studies:** Larger datasets and graphs
5. **Larger LLM Models:** Evaluate with 7B+ parameters

---

## Acknowledgments

- **Supervisor:** Darshan Derasari
- **Organization:** WeblineIndia, Ahmedabad
- **BITS Pilani:** Academic framework
- **Open Source Community:** Tools and frameworks

---

## Questions and Discussion

**Contact:**

- Email: shubhra.gadhwala@gmail.com
- BITS Email: 2023aa05750@wilp.bits-pilani.ac.in

**Project Repository:** Available for review

---

## References

1. Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS.
2. Guu, K., et al. (2020). Retrieval Augmented Language Model Pre-Training. ICML.
3. Pan, S., et al. (2023). Unifying Large Language Models and Knowledge Graphs: A Survey.
4. Baek, J., et al. (2023). Knowledge-Augmented Language Model Prompting for Zero-Shot KG QA. ACL.
5. Es, S., et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation.
6. Yang, Z., et al. (2018). HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering. EMNLP.

---

## Thank You
