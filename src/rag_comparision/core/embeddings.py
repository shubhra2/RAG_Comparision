"""Embedding generation module."""

from typing import Any

try:
    # Try new langchain-huggingface package first (recommended)
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.embeddings import Embeddings
except ImportError:
    try:
        # Fallback to deprecated langchain_community version
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_core.embeddings import Embeddings
    except ImportError:
        HuggingFaceEmbeddings = None
        Embeddings = None


def get_default_embedding_model() -> 'Embeddings':
    """Get the default embedding model.

    Returns:
        Embeddings instance using sentence-transformers

    Raises:
        ImportError: If required packages are not installed
    """
    if HuggingFaceEmbeddings is None:
        raise ImportError(
            'langchain-huggingface or langchain-community is required for embeddings. '
            'Install it with: pip install langchain-huggingface'
        )

    # Use a lightweight, fast embedding model
    # all-MiniLM-L6-v2 is a good balance of quality and speed
    return HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True},
    )


def create_embedding_model(
    model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
    device: str = 'cpu',
    normalize: bool = True,
) -> 'Embeddings':
    """Create a custom embedding model.

    Args:
        model_name: Name of the embedding model
        device: Device to run the model on ('cpu' or 'cuda')
        normalize: Whether to normalize embeddings

    Returns:
        Embeddings instance

    Raises:
        ImportError: If required packages are not installed
    """
    if HuggingFaceEmbeddings is None:
        raise ImportError(
            'langchain-huggingface or langchain-community is required for embeddings. '
            'Install it with: pip install langchain-huggingface'
        )

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': normalize},
    )

