"""RAG system core logic."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from typing import TYPE_CHECKING

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

from rag_comparision.core.data_loader import load_synthetic_articles_dataset
from rag_comparision.core.embeddings import get_default_embedding_model
from rag_comparision.core.ollama import OllamaClient
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

    def __init__(self, rag_type: str, model: str = 'tinyllama'):
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
        model: str = 'tinyllama',
        embeddings: 'Embeddings | None' = None,
        vector_store: VectorStoreInterface | None = None,
        persist_directory: str | Path | None = None,
        ollama_base_url: str = 'http://localhost:11434',
        k_retrieval: int = 4,
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
        super().__init__(rag_type='Standard RAG', model=model)
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
        self._data_loaded = False

    def load_data(
        self,
        source: str | Path | None = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> None:
        """Load and index data into the vector store.

        Args:
            source: Source of the dataset. If None, loads synthetic articles.
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
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
                template=(
                    'Use the following pieces of context to answer the question. '
                    'If you don\'t know the answer, just say that you don\'t know, '
                    'don\'t try to make up an answer.\n\n'
                    'Context:\n{context}\n\n'
                    'Question: {question}\n\n'
                    'Answer:'
                ),
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

        # Retrieve relevant documents
        retrieved_docs = self.vector_store.similarity_search(
            prompt,
            k=self.k_retrieval,
        )

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
            formatted_prompt = (
                f'Context:\n{context}\n\n'
                f'Question: {prompt}\n\n'
                f'Answer based on the context above:'
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

        # Calculate average similarity score (if available)
        try:
            docs_with_scores = self.vector_store.similarity_search_with_score(
                prompt,
                k=self.k_retrieval,
            )
            avg_score = sum(score for _, score in docs_with_scores) / len(
                docs_with_scores
            ) if docs_with_scores else 0.0
            # Convert distance to confidence (inverse relationship)
            confidence = max(0.0, min(1.0, 1.0 - avg_score))
        except Exception:
            confidence = 0.5  # Default confidence

        # Extract metadata from retrieved documents
        metadata = {
            'retrieved_documents': len(retrieved_docs),
            'retrieval_k': self.k_retrieval,
            'documents': [
                {
                    'content': doc.page_content[:200] + '...'
                    if len(doc.page_content) > 200
                    else doc.page_content,
                    'metadata': doc.metadata,
                }
                for doc in retrieved_docs
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

    def get_vector_store(self) -> VectorStoreInterface:
        """Get the vector store instance.

        Returns:
            Vector store instance
        """
        return self.vector_store
