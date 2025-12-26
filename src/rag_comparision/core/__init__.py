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
from rag_comparision.core.ollama import (
    OllamaClient,
    OllamaResponse,
    OllamaStreamChunk,
)
from rag_comparision.core.rag import RAGResponse, RAGSystem, StandardRAG
from rag_comparision.core.sidebar import render_sidebar
from rag_comparision.core.vector_store import (
    ChromaVectorStore,
    VectorStoreInterface,
)

__all__ = [
    'OllamaClient',
    'OllamaResponse',
    'OllamaStreamChunk',
    'RAGSystem',
    'RAGResponse',
    'StandardRAG',
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
