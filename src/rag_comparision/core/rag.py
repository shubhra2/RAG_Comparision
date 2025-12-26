"""RAG system core logic."""

from abc import ABC, abstractmethod
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings

try:
    from langchain.chains import RetrievalQA
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    from langchain_core.llms import LLM
    from langchain_core.prompts import PromptTemplate
except ImportError:
    RetrievalQA = None
    Document = None  # type: ignore
    Embeddings = None  # type: ignore
    LLM = None
    PromptTemplate = None

from rag_comparision.config import (
    CONFIDENCE_MAX,
    CONFIDENCE_MIN,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CONFIDENCE,
    DEFAULT_K_RETRIEVAL,
    DEFAULT_MODEL_FALLBACK,
    OLLAMA_BASE_URL,
    RAG_PROMPT_SIMPLE,
    RAG_PROMPT_TEMPLATE,
    RAG_TYPE_STANDARD,
)
from rag_comparision.core.data_loader import load_synthetic_articles_dataset
from rag_comparision.core.embeddings import get_default_embedding_model
from rag_comparision.core.ollama import OllamaClient, OllamaStreamChunk
from rag_comparision.core.vector_store import ChromaVectorStore, VectorStoreInterface


class RAGResponse:
    """Response from RAG system."""

    def __init__(
        self,
        content: str,
        rag_type: str,
        model: str,
        retrieved_chunks: int = 0,
        confidence: float = 0.0,
        metadata: dict | None = None,
    ):
        """Initialize RAG response.

        Args:
            content: Response content text
            rag_type: Type of RAG system used
            model: Model name used
            retrieved_chunks: Number of retrieved chunks
            confidence: Confidence score
            metadata: Additional metadata
        """
        self.content = content
        self.rag_type = rag_type
        self.model = model
        self.retrieved_chunks = retrieved_chunks
        self.confidence = confidence
        self.metadata = metadata or {}

    def to_dict(self) -> dict:
        """Convert response to dictionary."""
        return {
            'content': self.content,
            'rag_type': self.rag_type,
            'model': self.model,
            'retrieved_chunks': self.retrieved_chunks,
            'confidence': self.confidence,
            'metadata': self.metadata,
        }


class RAGSystem(ABC):
    """Abstract base class for RAG systems."""

    def __init__(self, rag_type: str, model: str = DEFAULT_MODEL_FALLBACK):
        """Initialize RAG system.

        Args:
            rag_type: Type of RAG system ("Standard RAG" or "Graph-Based RAG")
            model: Model name to use
        """
        self.rag_type = rag_type
        self.model = model

    @abstractmethod
    def query(self, prompt: str) -> RAGResponse:
        """Query the RAG system.

        Args:
            prompt: User query

        Returns:
            RAGResponse with answer and metadata
        """
        pass

    @abstractmethod
    def load_data(self, source: str | Path | None = None) -> None:
        """Load data into the RAG system.

        Args:
            source: Source of the data (file path, URL, etc.)
        """
        pass


class StandardRAG(RAGSystem):
    """Standard RAG implementation using ChromaDB and LangChain."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL_FALLBACK,
        embeddings: 'Embeddings | None' = None,
        vector_store: VectorStoreInterface | None = None,
        persist_directory: str | Path | None = None,
        ollama_base_url: str = OLLAMA_BASE_URL,
        k_retrieval: int = DEFAULT_K_RETRIEVAL,
    ):
        """Initialize Standard RAG system.

        Args:
            model: Ollama model name to use
            embeddings: Embedding model. If None, uses default.
            vector_store: Vector store instance. If None, creates ChromaDB.
            persist_directory: Directory to persist vector store
            ollama_base_url: Base URL for Ollama API
            k_retrieval: Number of documents to retrieve
        """
        super().__init__(rag_type=RAG_TYPE_STANDARD, model=model)
        self.embeddings = embeddings or get_default_embedding_model()
        self.persist_directory = persist_directory
        self.k_retrieval = k_retrieval

        # Initialize vector store
        if vector_store is None:
            self.vector_store = ChromaVectorStore(
                embeddings=self.embeddings,
                persist_directory=persist_directory,
            )
        else:
            self.vector_store = vector_store

        # Initialize Ollama client
        self.ollama_client = OllamaClient(
            model=model,
            base_url=ollama_base_url,
        )

        # Initialize retrieval chain (will be set up after data is loaded)
        self._qa_chain: Any = None
        self._retriever: Any = None
        self._prompt_template: PromptTemplate | None = None
        self._data_loaded = False

    def load_data(
        self,
        source: str | Path | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        force_reload: bool = False,
    ) -> None:
        """Load and index data into the vector store.

        Args:
            source: Source of the dataset. If None, loads synthetic articles.
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            force_reload: If True, reload data even if it already exists.
                If False, skip loading if data already exists in vector store.
        """
        # Check if data already exists in the vector store
        if not force_reload and self._data_loaded:
            # Data was already loaded in this session
            return

        # Check if vector store already has documents
        try:
            doc_count = self.vector_store.count()
            if doc_count > 0 and not force_reload:
                # Data already exists, just set up the retrieval chain
                self._setup_retrieval_chain()
                self._data_loaded = True
                return
        except Exception:
            # If count() fails, assume empty and proceed with loading
            pass

        # Load documents
        documents = load_synthetic_articles_dataset(
            source=source,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # Add to vector store
        self.vector_store.add_documents(documents)

        # Persist if directory is specified
        if self.persist_directory:
            self.vector_store.persist()

        # Set up retrieval chain
        self._setup_retrieval_chain()

        self._data_loaded = True

    def _setup_retrieval_chain(self) -> None:
        """Set up the LangChain retrieval QA chain."""
        if RetrievalQA is None:
            # Fallback: use simple retrieval without LangChain chain
            self._qa_chain = None
            return

        # Create retriever
        retriever = self.vector_store.as_retriever(
            search_kwargs={'k': self.k_retrieval},
        )

        # Create prompt template
        if PromptTemplate is not None:
            prompt_template = PromptTemplate(
                input_variables=['context', 'question'],
                template=RAG_PROMPT_TEMPLATE,
            )
        else:
            prompt_template = None

        # Note: RetrievalQA from langchain.chains is deprecated in newer versions
        # We'll use a simpler approach with direct retrieval + LLM
        self._retriever = retriever
        self._prompt_template = prompt_template

    def query(self, prompt: str) -> RAGResponse:
        """Query the RAG system.

        Args:
            prompt: User query

        Returns:
            RAGResponse with answer and metadata
        """
        if not self._data_loaded:
            # Auto-load data if not loaded
            self.load_data()

        # Retrieve relevant documents with scores
        try:
            docs_with_scores = self.vector_store.similarity_search_with_score(
                prompt,
                k=self.k_retrieval,
            )
        except Exception:
            # Fallback to search without scores
            docs_with_scores = [
                (doc, 0.0)
                for doc in self.vector_store.similarity_search(
                    prompt,
                    k=self.k_retrieval,
                )
            ]

        retrieved_docs = [doc for doc, _ in docs_with_scores]

        # Build context from retrieved documents
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            context_parts.append(f'[Document {i}]\n{doc.page_content}')

        context = '\n\n'.join(context_parts)

        # Create prompt with context
        if self._prompt_template:
            formatted_prompt = self._prompt_template.format(
                context=context,
                question=prompt,
            )
        else:
            # Simple prompt format
            formatted_prompt = RAG_PROMPT_SIMPLE.format(
                context=context,
                question=prompt,
            )

        # Generate response using Ollama
        try:
            ollama_response = self.ollama_client.chat(formatted_prompt)
            content = ollama_response.content
        except Exception as e:
            content = (
                f'Error generating response: {str(e)}\n\n'
                f'Retrieved {len(retrieved_docs)} documents:\n\n{context}'
            )

        # Calculate average confidence from scores
        if docs_with_scores:
            scores = [score for _, score in docs_with_scores]
            avg_distance = sum(scores) / len(scores)
            # Normalize cosine distance [0, 2] to [0, 1] then invert for confidence
            normalized_distance = min(1.0, avg_distance / 2.0)
            confidence = max(
                CONFIDENCE_MIN, min(CONFIDENCE_MAX, 1.0 - normalized_distance)
            )
        else:
            confidence = DEFAULT_CONFIDENCE

        # Extract metadata from retrieved documents with individual scores
        metadata = {
            'retrieved_documents': len(retrieved_docs),
            'retrieval_k': self.k_retrieval,
            'context': context,  # Full context sent to model
            'formatted_prompt': formatted_prompt,  # Exact prompt sent to LLM
            'documents': [
                {
                    'content': doc.page_content,  # Full content, not truncated
                    'metadata': doc.metadata,
                    'score': float(
                        score
                    ),  # ChromaDB distance score (lower = more similar)
                }
                for doc, score in docs_with_scores
            ],
        }

        return RAGResponse(
            content=content,
            rag_type=self.rag_type,
            model=self.model,
            retrieved_chunks=len(retrieved_docs),
            confidence=confidence,
            metadata=metadata,
        )

    def query_stream(
        self, prompt: str, reasoning: bool | None = None
    ) -> Generator[tuple[OllamaStreamChunk, dict], None, None]:
        """Query the RAG system with streaming response.

        Args:
            prompt: User query
            reasoning: Enable reasoning mode for supported models

        Yields:
            Tuples of (OllamaStreamChunk, metadata_dict) where metadata contains
            retrieval information that remains constant across chunks
        """
        if not self._data_loaded:
            # Auto-load data if not loaded
            self.load_data()

        # Retrieve relevant documents with scores
        try:
            docs_with_scores = self.vector_store.similarity_search_with_score(
                prompt,
                k=self.k_retrieval,
            )
        except Exception:
            # Fallback to search without scores
            docs_with_scores = [
                (doc, 0.0)
                for doc in self.vector_store.similarity_search(
                    prompt,
                    k=self.k_retrieval,
                )
            ]

        retrieved_docs = [doc for doc, _ in docs_with_scores]

        # Build context from retrieved documents
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            context_parts.append(f'[Document {i}]\n{doc.page_content}')

        context = '\n\n'.join(context_parts)

        # Create prompt with context
        if self._prompt_template:
            formatted_prompt = self._prompt_template.format(
                context=context,
                question=prompt,
            )
        else:
            # Simple prompt format
            formatted_prompt = RAG_PROMPT_SIMPLE.format(
                context=context,
                question=prompt,
            )

        # Calculate average confidence from scores
        if docs_with_scores:
            scores = [score for _, score in docs_with_scores]
            avg_distance = sum(scores) / len(scores)
            # Normalize cosine distance [0, 2] to [0, 1] then invert for confidence
            normalized_distance = min(1.0, avg_distance / 2.0)
            confidence = max(
                CONFIDENCE_MIN, min(CONFIDENCE_MAX, 1.0 - normalized_distance)
            )
        else:
            confidence = DEFAULT_CONFIDENCE

        # Extract metadata from retrieved documents with individual scores (constant across all chunks)
        metadata = {
            'retrieved_documents': len(retrieved_docs),
            'retrieval_k': self.k_retrieval,
            'confidence': confidence,
            'context': context,  # Full context sent to model
            'formatted_prompt': formatted_prompt,  # Exact prompt sent to LLM
            'documents': [
                {
                    'content': doc.page_content,  # Full content, not truncated
                    'metadata': doc.metadata,
                    'score': float(
                        score
                    ),  # ChromaDB distance score (lower = more similar)
                }
                for doc, score in docs_with_scores
            ],
        }

        # Stream response using Ollama
        try:
            for chunk in self.ollama_client.stream_chat(
                formatted_prompt, reasoning=reasoning
            ):
                yield chunk, metadata
        except Exception as e:
            # On error, yield a single chunk with error message
            error_content = (
                f'Error generating response: {str(e)}\n\n'
                f'Retrieved {len(retrieved_docs)} documents:\n\n{context}'
            )
            yield (
                OllamaStreamChunk(
                    content=error_content,
                    done=True,
                ),
                metadata,
            )

    def get_vector_store(self) -> VectorStoreInterface:
        """Get the vector store instance.

        Returns:
            Vector store instance
        """
        return self.vector_store
