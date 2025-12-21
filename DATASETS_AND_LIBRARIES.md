# Datasets and Libraries Recommendations for RAG Comparison Project

Based on your project abstract, here are recommended datasets and libraries for building and evaluating both standard RAG and graph-based RAG systems.

## 📊 Recommended Datasets

### Primary QA Datasets (Multi-hop Focus)

1. **HotpotQA** ⭐ **RECOMMENDED**

   - **Description**: Multi-hop question answering dataset requiring reasoning over multiple documents
   - **Size**: ~113k questions (train), ~7.6k (dev)
   - **Why it fits**: Perfect for testing multi-hop queries where graph-based RAG should excel
   - **Source**: Hugging Face (`hotpot_qa`) or [hotpotqa.github.io](https://hotpotqa.github.io/)
   - **Format**: Questions with supporting paragraphs and answers
2. **2WikiMultihopQA**

   - **Description**: Multi-hop question answering over Wikipedia articles
   - **Size**: ~192k questions
   - **Why it fits**: Tests ability to connect facts across multiple documents
   - **Source**: Hugging Face (`2wikimultihopqa`)
3. **MuSiQue** (Multi-hop Questions)

   - **Description**: Multi-hop questions with explicit reasoning chains
   - **Size**: ~25k questions
   - **Why it fits**: Designed specifically for multi-hop reasoning evaluation
   - **Source**: Hugging Face (`muque`)
4. **QASPER** (Alternative)

   - **Description**: Question answering over scientific papers
   - **Size**: ~5k questions over 1.6k papers
   - **Why it fits**: Good for domain-specific evaluation if you want academic focus
   - **Source**: Hugging Face (`qasper`)

### Smaller Datasets (For Quick Testing)

5. **SQuAD 2.0** (Single-hop baseline)

   - **Description**: Standard QA dataset for baseline comparison
   - **Size**: ~150k questions
   - **Why it fits**: Good baseline to show where standard RAG works well
   - **Source**: Hugging Face (`squad_v2`)
6. **Natural Questions**

   - **Description**: Real user questions from Google search
   - **Size**: ~300k questions
   - **Why it fits**: Real-world question patterns
   - **Source**: Hugging Face (`natural_questions`)

**Recommendation**: Start with **HotpotQA** (dev set) for focused evaluation, as it's specifically designed for multi-hop reasoning where graph-based RAG should show advantages.

---

## 🛠️ Libraries and Frameworks

### Core RAG Frameworks

1. **LangChain** ⭐ **RECOMMENDED**

   - **Purpose**: Standard RAG pipeline, graph-based RAG, and orchestration
   - **Key modules**:
     - `langchain`: Core framework
     - `langchain-community`: Community integrations
     - `langchain-neo4j`: Neo4j graph database integration
     - `langchain-vectorstores`: Vector store integrations
   - **Install**: `pip install langchain langchain-community langchain-neo4j`
2. **LlamaIndex** (Alternative)

   - **Purpose**: Alternative RAG framework with graph support
   - **Key modules**:
     - `llama-index`: Core framework
     - `llama-index-vector-stores-*`: Vector store integrations
     - `llama-index-graph-stores-*`: Knowledge graph integrations
   - **Install**: `pip install llama-index`

**Recommendation**: Use **LangChain** as mentioned in your abstract. It has mature graph-based RAG support.

### Vector Stores (Standard RAG)

1. **ChromaDB** ⭐ **RECOMMENDED for prototyping**

   - **Why**: Easy to use, in-memory option, good for small datasets
   - **Install**: `pip install chromadb`
   - **Best for**: Quick prototyping and development
2. **FAISS** (Facebook AI Similarity Search)

   - **Why**: Fast, efficient, good for larger datasets
   - **Install**: `pip install faiss-cpu` (or `faiss-gpu`)
   - **Best for**: Production-like performance testing
3. **Qdrant**

   - **Why**: Production-ready, good performance
   - **Install**: `pip install qdrant-client`
   - **Best for**: Production deployments
4. **Weaviate**

   - **Why**: Hybrid search (vector + keyword), good for complex queries
   - **Install**: `pip install weaviate-client`
   - **Best for**: Advanced retrieval strategies

**Recommendation**: Start with **ChromaDB** for development, consider **FAISS** for performance testing.

### Knowledge Graph Databases (Graph-based RAG)

1. **Neo4j** ⭐ **RECOMMENDED**

   - **Why**: Most popular, excellent LangChain integration, Cypher query language
   - **Install**:
     - Database: Download from [neo4j.com](https://neo4j.com/download/)
     - Python client: `pip install neo4j langchain-neo4j`
   - **Best for**: Production graph-based RAG
2. **Neptune** (AWS)

   - **Why**: Managed service, scalable
   - **Best for**: Cloud deployments
3. **ArangoDB**

   - **Why**: Multi-model (graph + document)
   - **Install**: `pip install python-arango`
   - **Best for**: Hybrid document-graph storage
4. **Memgraph** (Lightweight alternative)

   - **Why**: In-memory, fast, good for smaller datasets
   - **Install**: `pip install gqlalchemy`
   - **Best for**: Quick prototyping

**Recommendation**: Use **Neo4j** (Community Edition is free) for graph-based RAG. It has the best LangChain integration.

### Embedding Models

1. **Sentence Transformers** ⭐ **RECOMMENDED**

   - **Why**: Easy to use, good quality embeddings
   - **Models**:
     - `all-MiniLM-L6-v2`: Fast, 384 dimensions
     - `all-mpnet-base-v2`: Better quality, 768 dimensions
   - **Install**: `pip install sentence-transformers`
2. **Hugging Face Transformers**

   - **Why**: Access to many embedding models
   - **Install**: `pip install transformers`

**Recommendation**: Use **Sentence Transformers** with `all-MiniLM-L6-v2` for speed or `all-mpnet-base-v2` for quality.

### Local LLM Models (0.5B - 1B parameters)

1. **TinyLlama** (1.1B) ⭐ **RECOMMENDED**

   - **Size**: 1.1B parameters
   - **Why**: Good balance of size and performance
   - **Source**: Hugging Face (`TinyLlama/TinyLlama-1.1B-Chat-v1.0`)
   - **Install**: Via `transformers` library
2. **Phi-2** (2.7B - slightly above range but excellent)

   - **Size**: 2.7B parameters
   - **Why**: Microsoft's small model, excellent quality
   - **Source**: Hugging Face (`microsoft/phi-2`)
3. **Qwen2-0.5B** (0.5B)

   - **Size**: 0.5B parameters
   - **Why**: Alibaba's efficient small model
   - **Source**: Hugging Face (`Qwen/Qwen2-0.5B-Instruct`)
4. **Gemma-2B** (2B - slightly above range)

   - **Size**: 2B parameters
   - **Why**: Google's open model
   - **Source**: Hugging Face (`google/gemma-2b-it`)
5. **StableLM-3B** (3B - above range but good baseline)

   - **Size**: 3B parameters
   - **Why**: Stability AI's model

**Recommendation**: Start with **TinyLlama-1.1B** or **Qwen2-0.5B** for strict parameter limit, or **Phi-2** if you can go slightly above 1B.

### LLM Inference Libraries

1. **Ollama** ⭐ **RECOMMENDED for local models**

   - **Why**: Easy local LLM deployment, supports many models
   - **Install**: Download from [ollama.ai](https://ollama.ai)
   - **Usage**: `ollama pull tinyllama` then use via LangChain
2. **llama.cpp** (via `llama-cpp-python`)

   - **Why**: Fast inference, quantized models
   - **Install**: `pip install llama-cpp-python`
3. **Transformers** (Hugging Face)

   - **Why**: Direct model loading
   - **Install**: `pip install transformers torch`

**Recommendation**: Use **Ollama** for easiest setup, or **transformers** for more control.

### Knowledge Graph Construction

1. **spaCy** + **spaCy-entity-linker**

   - **Purpose**: Named Entity Recognition (NER) and entity extraction
   - **Install**: `pip install spacy` + download models
2. **NetworkX**

   - **Purpose**: Graph manipulation and analysis
   - **Install**: `pip install networkx`
3. **pyvis**

   - **Purpose**: Graph visualization
   - **Install**: `pip install pyvis`
4. **LangChain Text Splitters**

   - **Purpose**: Document chunking for graph construction
   - **Part of**: `langchain` package

### Evaluation Libraries

1. **RAGAS** ⭐ **RECOMMENDED**

   - **Purpose**: RAG evaluation metrics (faithfulness, answer relevancy, context precision)
   - **Install**: `pip install ragas`
   - **Why**: Comprehensive RAG evaluation framework
2. **LangSmith** (LangChain's evaluation platform)

   - **Purpose**: Evaluation and monitoring
   - **Install**: `pip install langsmith`
3. **BLEU/ROGUE** (via `nltk` or `rouge-score`)

   - **Purpose**: Text similarity metrics
   - **Install**: `pip install nltk rouge-score`
4. **Exact Match / F1 Score**

   - **Purpose**: Standard QA metrics
   - **Implementation**: Custom or via `datasets` library

**Recommendation**: Use **RAGAS** for comprehensive evaluation, supplement with custom metrics.

### Utility Libraries

1. **datasets** (Hugging Face)

   - **Purpose**: Load and process datasets
   - **Install**: `pip install datasets`
2. **pandas**

   - **Purpose**: Data manipulation and analysis
   - **Install**: `pip install pandas`
3. **numpy**

   - **Purpose**: Numerical operations
   - **Install**: `pip install numpy`
4. **tqdm**

   - **Purpose**: Progress bars
   - **Install**: `pip install tqdm`
5. **python-dotenv**

   - **Purpose**: Environment variable management
   - **Install**: `pip install python-dotenv`

---

## 📦 Complete Installation Command

Here's a suggested installation command for the core stack:

```bash
# Core RAG framework
pip install langchain langchain-community langchain-neo4j

# Vector stores
pip install chromadb faiss-cpu

# Embeddings
pip install sentence-transformers

# LLM inference
pip install transformers torch ollama

# Knowledge graph
pip install neo4j networkx spacy

# Evaluation
pip install ragas datasets

# Utilities
pip install pandas numpy tqdm python-dotenv
```

---

## 🎯 Recommended Stack Summary

**For Standard RAG:**

- Framework: LangChain
- Vector Store: ChromaDB (dev) / FAISS (performance)
- Embeddings: Sentence Transformers (`all-MiniLM-L6-v2`)
- LLM: TinyLlama-1.1B via Ollama or Transformers

**For Graph-based RAG:**

- Framework: LangChain
- Graph DB: Neo4j Community Edition
- Entity Extraction: spaCy
- Graph Construction: LangChain + NetworkX

**For Evaluation:**

- Dataset: HotpotQA (dev set)
- Metrics: RAGAS + custom F1/EM scores

**For Development:**

- Python 3.9+
- Jupyter Notebooks or Python scripts
- Git for version control

---

## 📚 Additional Resources

1. **LangChain Graph RAG Documentation**: [langchain.com/docs/use_cases/graph/](https://python.langchain.com/docs/use_cases/graph/)
2. **Neo4j Graph RAG Guide**: [neo4j.com/developer-blog/graph-rag/](https://neo4j.com/developer-blog/graph-rag/)
3. **HotpotQA Paper**: [arxiv.org/abs/1809.09600](https://arxiv.org/abs/1809.09600)
4. **RAGAS Documentation**: [docs.ragas.io](https://docs.ragas.io/)

---

## 🚀 Quick Start Checklist

- [X] Install Python 3.9+
- [ ] Install core libraries (LangChain, ChromaDB, Neo4j)
- [ ] Download HotpotQA dataset
- [ ] Set up Neo4j database
- [ ] Download TinyLlama or Qwen2-0.5B model
- [ ] Set up evaluation framework (RAGAS)
- [ ] Create project structure
- [ ] Implement standard RAG pipeline
- [ ] Implement graph-based RAG pipeline
- [ ] Run comparative evaluation

---

*Last updated: Based on project abstract dated November 2025*
