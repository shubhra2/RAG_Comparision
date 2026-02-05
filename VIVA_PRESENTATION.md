# Evaluating Graph-Based Retrieval-Augmented Generation
## Comparative Analysis and Quality Improvements over Standard RAG Systems

**Student:** Shubhra Gadhwala
**BITS ID:** 2023AA05750
**Course:** AIMLCZG628T - Dissertation/Project Work
**Supervisor:** Darshan Derasari
**Organization:** WeblineIndia, Ahmedabad

---

## Problem Statement

- Standard RAG systems rely on vector similarity search
- Struggles with:
  - Relational reasoning
  - Entity linking
  - Multi-hop queries
- Minor retrieval errors lead to incorrect answers
- **Research Question:** Can graph-structured knowledge improve contextual grounding and reasoning quality?

---

## Objectives

1. **Build Standard RAG System**
   - Vector-based retrieval using ChromaDB
   - Text chunking and embedding generation
   - Similarity-based document retrieval

2. **Build Graph-Based RAG System**
   - Knowledge graph construction using Neo4j
   - Entity and relationship extraction
   - Graph-based retrieval and traversal

3. **Comparative Evaluation**
   - Benchmark both systems on common dataset
   - Focus on multi-hop and complex queries
   - Quantitative and qualitative analysis

---

## System Architecture

**[DIAGRAM PLACEHOLDER: docs/diagrams/system_architecture.png]**

### Key Components:
- **User Interface:** Streamlit Dashboard with Chat Playground
- **RAG Systems:** Standard RAG & Graph-Based RAG
- **Retrieval:** Vector Store (ChromaDB) & Knowledge Graph (Neo4j)
- **LLM:** Local models (0.5B-1B parameters) via Ollama
- **Evaluation:** RAGAS metrics framework

---

## Standard RAG Pipeline

**[DIAGRAM PLACEHOLDER: docs/diagrams/standard_rag_pipeline.png]**

### Pipeline Steps:
1. **Document Ingestion** → Text chunking
2. **Embedding Generation** → Sentence Transformers
3. **Vector Storage** → ChromaDB indexing
4. **Similarity Retrieval** → Top-K document retrieval
5. **Answer Generation** → LLM with retrieved context

---

## Graph-Based RAG Pipeline

**[DIAGRAM PLACEHOLDER: docs/diagrams/graph_rag_pipeline.png]**

### Pipeline Steps:
1. **Entity Extraction** → Named entities from documents
2. **Relationship Extraction** → Entity relationships
3. **Knowledge Graph Construction** → Neo4j graph storage
4. **Graph Traversal** → Cypher query generation
5. **Context Assembly** → Related entities and relationships
6. **Answer Generation** → LLM with graph context

---

## Dataset & Tools

### Dataset
- **Synthetic Articles Dataset**
  - Research articles with structured information
  - Suitable for multi-hop reasoning
  - Lightweight for rapid experimentation

### Technology Stack
- **Framework:** LangChain
- **Vector Store:** ChromaDB
- **Graph Database:** Neo4j
- **LLMs:** Local models (Qwen2, Gemma, Phi) via Ollama
- **Embeddings:** Sentence Transformers (all-MiniLM-L6-v2)
- **Frontend:** Streamlit
- **Evaluation:** RAGAS metrics

---

## Current Progress

### ✅ Completed
- Literature review and problem statement finalization
- Codebase setup and environment configuration
- Dataset selection and validation
- Standard RAG system implementation
- Graph-Based RAG system implementation (in progress)
- Streamlit dashboard with chat playground

### 🔄 Ongoing
- Graph RAG pipeline refinement
- Query evaluation set preparation

### 📋 Pending
- Comprehensive testing and evaluation
- Comparative analysis
- Results documentation

---

## Expected Outcomes

1. **Working Comparison**
   - Functional Standard RAG system
   - Functional Graph-Based RAG system
   - Side-by-side performance comparison

2. **Quantitative Insights**
   - Retrieval accuracy metrics
   - Answer quality metrics (RAGAS)
   - Performance on multi-hop queries

3. **Qualitative Insights**
   - Scenarios where Graph-RAG excels
   - Limitations of each approach
   - Best practices and recommendations

---

## Data Flow

**[DIAGRAM PLACEHOLDER: docs/diagrams/data_flow.png]**

### Flow Overview:
- Documents → Preprocessing → Two parallel paths
- **Path 1:** Chunking → Embeddings → Vector Store
- **Path 2:** Entity Extraction → Graph Construction → Neo4j
- User Query → Retrieval (Vector/Graph) → Context → LLM → Answer

---

## Evaluation Framework

### Metrics (RAGAS)
- **Faithfulness:** Answer grounded in retrieved context
- **Answer Relevancy:** Answer relevance to question
- **Context Precision:** Quality of retrieved context

### Test Scenarios
- Single-hop queries
- Multi-hop queries
- Complex relational queries
- Edge cases and error handling

---

## Timeline

| Phase | Status |
|-------|--------|
| Literature Review | ✅ Completed |
| Initial Setup | ✅ Completed |
| Standard RAG Development | ✅ Completed |
| Graph RAG Development | 🔄 In Progress |
| Testing & Evaluation | 📋 Pending |
| Report Writing | 📋 Pending |

---

## Live Demo

**[SWITCH TO APPLICATION]**

### Demo Features:
1. **Streamlit Dashboard**
   - Dataset preview
   - Chat playground for both RAG systems
   - Side-by-side comparison

2. **Query Examples**
   - Single-hop questions
   - Multi-hop questions
   - Complex relational queries

3. **System Comparison**
   - Standard RAG responses
   - Graph-Based RAG responses
   - Retrieval context visualization

---

## Thank You

### Questions & Discussion

**Contact:**
shubhra.gadhwala@gmail.com
2023aa05750@wilp.bits-pilani.ac.in
