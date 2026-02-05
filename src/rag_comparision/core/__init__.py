"""Core business logic modules."""

from rag_comparision.core.data_loader import load_synthetic_articles_dataset
from rag_comparision.core.data_preprocessing import (
    preprocess_synthetic_articles_dataset,
    validate_preprocessed_data,
)
from rag_comparision.core.datasets import get_dataset_info
from rag_comparision.core.embeddings import (
    create_embedding_model,
    get_default_embedding_model,
)
from rag_comparision.core.llm_provider import (
    LLMProvider,
    LLMResponse,
    LLMStreamChunk,
)
from rag_comparision.core.ollama import (
    OllamaClient,
    OllamaResponse,
    OllamaStreamChunk,
)
from rag_comparision.core.rag import (
    GraphRAG,
    GraphRAGAgentic,
    Neo4jEmptyError,
    RAGResponse,
    RAGSystem,
    StandardRAG,
)
from rag_comparision.core.sidebar import render_sidebar
from rag_comparision.core.vector_store import (
    ChromaVectorStore,
    VectorStoreInterface,
)

__all__ = [
    # LLM Provider (new, recommended)
    'LLMProvider',
    'LLMResponse',
    'LLMStreamChunk',
    # Ollama Client (legacy, kept for backward compatibility)
    'OllamaClient',
    'OllamaResponse',
    'OllamaStreamChunk',
    # RAG Systems
    'RAGSystem',
    'RAGResponse',
    'StandardRAG',
    'GraphRAG',
    'GraphRAGAgentic',
    'Neo4jEmptyError',
    # Data and utilities
    'get_dataset_info',
    'load_synthetic_articles_dataset',
    'preprocess_synthetic_articles_dataset',
    'validate_preprocessed_data',
    'get_default_embedding_model',
    'create_embedding_model',
    'ChromaVectorStore',
    'VectorStoreInterface',
    'render_sidebar',
]
