# RAG Comparison Project

Evaluating Graph-Based Retrieval-Augmented Generation: Comparative Analysis and Quality Improvements over Standard RAG Systems

## Project Overview

This dissertation research project compares standard RAG (Retrieval-Augmented Generation) systems with graph-based RAG approaches, focusing on knowledge graph construction and multi-hop reasoning using a synthetic research articles dataset.

## Features

- **Standard RAG**: Vector-based retrieval using ChromaDB/FAISS
- **Graph-Based RAG**: Knowledge graph enhanced retrieval using Neo4j
- **Comprehensive Evaluation**: RAGAS metrics and multi-hop QA datasets
- **Web Interface**: FastAPI-based web application with Fomantic UI

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

2. Run the FastAPI application:

```bash
uvicorn src.rag_comparision.main:app --reload
```

The application will be available at `http://localhost:8000`

## Project Structure

```
RAG_Comparision/
├── src/
│   └── rag_comparision/          # Main package
│       ├── __init__.py           # Package initialization
│       ├── main.py               # FastAPI application entry point
│       ├── config.py             # Application configuration
│       ├── api/                  # API routes
│       │   └── __init__.py
│       ├── templates/            # HTML templates
│       │   └── index.html        # Main page template
│       └── static/               # Static files (CSS, JS, images)
├── pyproject.toml                # Project configuration
└── README.md                     # This file
```

## API Endpoints

- `GET /` - Main web interface
- `GET /health` - Health check endpoint
- `GET /api/v1/` - API root endpoint

## Technology Stack

- **Framework**: FastAPI
- **UI**: Fomantic UI (Semantic UI fork)
- **RAG Framework**: LangChain
- **Vector Stores**: ChromaDB, FAISS
- **Graph Database**: Neo4j
- **Evaluation**: RAGAS

## Development

### Running in Development Mode

```bash
uvicorn src.rag_comparision.main:app --reload --host 0.0.0.0 --port 8000
```

### Running Tests

```bash
pytest
```

## License

MIT

## Author

shubhra2 (shubhra.gadhwala@gmail.com)
