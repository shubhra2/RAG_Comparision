"""Application configuration."""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    # Application settings
    app_name: str = 'RAG Comparison Project'
    app_version: str = '0.1.0'
    debug: bool = False

    # Ollama/LLM Configuration
    ollama_base_url: str = 'http://localhost:11434'
    default_model: str = 'qwen3:0.6b'
    default_model_fallback: str = 'tinyllama'

    # RAG Configuration
    default_k_retrieval: int = 4
    default_chunk_size: int = 1000
    default_chunk_overlap: int = 200
    default_persist_directory: str = './chroma_db'
    chroma_collection_name: str = 'rag_documents'

    # Embedding Configuration
    default_embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2'
    embedding_device: str = 'cpu'
    embedding_normalize: bool = True

    # Data Configuration
    processed_data_path: str = 'data/processed/synthetic_articles.parquet'
    dataset_url: str = (
        'https://raw.githubusercontent.com/dcarpintero/ai-engineering/'
        'main/dataset/synthetic_articles.csv'
    )

    # RAG Types
    rag_type_standard: str = 'Standard RAG'
    rag_type_graph: str = 'Graph-Based RAG'

    # Confidence/Scoring
    default_confidence: float = 0.5
    confidence_min: float = 0.0
    confidence_max: float = 1.0

    # Timeout Configuration
    ollama_timeout: float = 10.0

    class Config:
        """Pydantic config."""

        env_file = '.env'
        env_file_encoding = 'utf-8'


settings = Settings()

# Module-level constants for easy import (derived from settings)
# These can be used directly without instantiating Settings

# Application
APP_NAME = settings.app_name
APP_VERSION = settings.app_version
DEBUG = settings.debug

# Ollama/LLM
OLLAMA_BASE_URL = settings.ollama_base_url
DEFAULT_MODEL = settings.default_model
DEFAULT_MODEL_FALLBACK = settings.default_model_fallback
# Available models list (not in settings, but useful constant)
AVAILABLE_MODELS = [
    'qwen3:0.6b',
    'qwen3:4b',
    'gemma3:1b',
    'phi4-mini',
    'deepscaler',
]

# RAG Configuration
DEFAULT_K_RETRIEVAL = settings.default_k_retrieval
DEFAULT_CHUNK_SIZE = settings.default_chunk_size
DEFAULT_CHUNK_OVERLAP = settings.default_chunk_overlap
DEFAULT_PERSIST_DIRECTORY = settings.default_persist_directory
CHROMA_COLLECTION_NAME = settings.chroma_collection_name

# Embedding Configuration
DEFAULT_EMBEDDING_MODEL = settings.default_embedding_model
EMBEDDING_DEVICE = settings.embedding_device
EMBEDDING_NORMALIZE = settings.embedding_normalize

# Data Configuration
PROCESSED_DATA_PATH = Path(settings.processed_data_path)
DATASET_URL = settings.dataset_url

# RAG Types
RAG_TYPE_STANDARD = settings.rag_type_standard
RAG_TYPE_GRAPH = settings.rag_type_graph
RAG_TYPES = [RAG_TYPE_STANDARD, RAG_TYPE_GRAPH]

# Confidence/Scoring
DEFAULT_CONFIDENCE = settings.default_confidence
CONFIDENCE_MIN = settings.confidence_min
CONFIDENCE_MAX = settings.confidence_max

# Timeout Configuration
OLLAMA_TIMEOUT = settings.ollama_timeout

# Prompt Templates
RAG_PROMPT_TEMPLATE = (
    'Use the following pieces of context to answer the question. '
    "If you don't know the answer, just say that you don't know, "
    "don't try to make up an answer.\n\n"
    'Context:\n{context}\n\n'
    'Question: {question}\n\n'
    'Answer:'
)

RAG_PROMPT_SIMPLE = (
    'Context:\n{context}\n\nQuestion: {question}\n\nAnswer based on the context above:'
)
