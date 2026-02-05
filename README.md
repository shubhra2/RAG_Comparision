# RAG Comparison Project

Evaluating Graph-Based Retrieval-Augmented Generation: Comparative Analysis and Quality Improvements over Standard RAG Systems

## Project Overview

This dissertation research project compares standard RAG (Retrieval-Augmented Generation) systems with graph-based RAG approaches, focusing on knowledge graph construction and multi-hop reasoning using a synthetic research articles dataset.

## Features

- **Standard RAG**: Vector-based retrieval using ChromaDB/FAISS
- **Graph-Based RAG**: Knowledge graph enhanced retrieval using Neo4j
- **Comprehensive Evaluation**: RAGAS metrics and multi-hop QA datasets
- **Web Interface**: Streamlit dashboard with chat playground

## Installation

### Prerequisites

- Python 3.11 or higher
- pip or uv package manager

### Setup

1. Install dependencies:

```bash
pip install -e .
```

Or using uv:

```bash
uv pip install -e .
```

2. Run the Streamlit dashboard:

```bash
streamlit run src/rag_comparision/dashboard.py
```

The dashboard will be available at `http://localhost:8501`

## Project Structure

```
RAG_Comparision/
├── src/
│   └── rag_comparision/          # Main package
│       ├── __init__.py           # Package initialization
│       ├── dashboard.py          # Streamlit dashboard entry point
│       ├── config.py             # Application configuration
│       ├── pages/                # Streamlit pages
│       │   ├── 1_Dataset_Preview.py
│       │   ├── 2_Chat_Playground.py
│       │   └── 3_Ollama_Test.py
│       └── static/               # Static files (icons, images)
├── pyproject.toml                # Project configuration
└── README.md                     # This file
```

## Technology Stack

- **Framework**: Streamlit
- **UI**: Streamlit native components
- **RAG Framework**: LangChain
- **Vector Stores**: ChromaDB, FAISS
- **Graph Database**: Neo4j
- **Evaluation**: RAGAS

## Development

### Running in Development Mode

```bash
streamlit run src/rag_comparision/dashboard.py
```

### Running Tests

```bash
pytest
```

## License

MIT

## Author

shubhra2 (shubhra.gadhwala@gmail.com)
