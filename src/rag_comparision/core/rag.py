"""RAG system core logic."""

import logging
from abc import ABC, abstractmethod
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings

from rag_comparision.config import (
    CONFIDENCE_MAX,
    CONFIDENCE_MIN,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CONFIDENCE,
    DEFAULT_K_RETRIEVAL,
    DEFAULT_MODEL_FALLBACK,
    GEMINI_MODEL,
    GRAPH_RAG_CYPHER_PROMPT_TEMPLATE,
    GRAPH_RAG_PROMPT_TEMPLATE,
    LLM_PROVIDER,
    OLLAMA_BASE_URL,
    RAG_PROMPT_SIMPLE,
    RAG_PROMPT_TEMPLATE,
    RAG_TYPE_GRAPH,
    RAG_TYPE_STANDARD,
)
from rag_comparision.core.data_loader import load_synthetic_articles_dataset
from rag_comparision.core.embeddings import get_default_embedding_model
from rag_comparision.core.graph_loader import GraphDataLoader
from rag_comparision.core.llm_provider import LLMProvider, LLMStreamChunk
from rag_comparision.core.neo4j_graph import Neo4jConnectionError, Neo4jGraphManager
from rag_comparision.core.vector_store import ChromaVectorStore, VectorStoreInterface

# Import LangChain core components
try:
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    from langchain_core.prompts import PromptTemplate
except ImportError:
    Document = None  # type: ignore
    Embeddings = None  # type: ignore
    PromptTemplate = None

# Try to import Neo4j types for serialization
try:
    import neo4j.time

    HAS_NEO4J_TIME = True
except ImportError:
    HAS_NEO4J_TIME = False


def _serialize_neo4j_object(obj: Any) -> Any:
    """Serialize Neo4j objects (dates, times, etc.) to human-readable strings.

    Args:
        obj: Object to serialize (can be Neo4j date/time object, dict, list, etc.)

    Returns:
        Serialized object with Neo4j types converted to strings
    """
    if HAS_NEO4J_TIME:
        # Handle Neo4j date objects
        if isinstance(obj, neo4j.time.Date):
            return obj.iso_format()  # Returns YYYY-MM-DD format
        if isinstance(obj, neo4j.time.DateTime):
            return obj.iso_format()  # Returns ISO 8601 format
        if isinstance(obj, neo4j.time.Time):
            return obj.iso_format()  # Returns HH:MM:SS format
        if isinstance(obj, neo4j.time.Duration):
            return str(obj)  # Duration as string

    # Handle dicts recursively
    if isinstance(obj, dict):
        return {key: _serialize_neo4j_object(value) for key, value in obj.items()}

    # Handle lists/tuples recursively
    if isinstance(obj, (list, tuple)):
        return type(obj)(_serialize_neo4j_object(item) for item in obj)

    # Check for string representations that look like Neo4j objects
    # This handles cases where Neo4j objects were already converted to strings
    if isinstance(obj, str):
        import re

        # Pattern: neo4j.time.Date(YEAR, MONTH, DAY)
        date_pattern = r'neo4j\.time\.Date\((\d+),\s*(\d+),\s*(\d+)\)'
        match = re.search(date_pattern, obj)
        if match:
            year, month, day = match.groups()
            return f'{year}-{month.zfill(2)}-{day.zfill(2)}'

        # Pattern: neo4j.time.DateTime(...) - more complex, just return ISO if possible
        datetime_pattern = r'neo4j\.time\.DateTime\([^)]+\)'
        if re.search(datetime_pattern, obj):
            # For DateTime, try to extract and format if possible
            # This is a simplified version - full parsing would be more complex
            pass

    return obj


# Try to import GraphCypherQAChain from langchain_neo4j (preferred for Neo4j integration)
# GraphCypherQAChain is available directly from langchain_neo4j or via submodule path
GraphCypherQAChain = None  # type: ignore
try:
    # Try submodule path first (most reliable based on diagnostic)
    from langchain_neo4j.chains.graph_qa.cypher import GraphCypherQAChain
except ImportError:
    try:
        # Try direct import from langchain_neo4j
        from langchain_neo4j import GraphCypherQAChain
    except ImportError:
        try:
            # Fallback to langchain_community
            from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
        except ImportError:
            GraphCypherQAChain = None  # type: ignore


class ChromaDBEmptyError(Exception):
    """Raised when ChromaDB is empty and data needs to be loaded."""

    def __init__(
        self,
        message: str = 'ChromaDB is empty. Please load data first using load_data().',
        persist_directory: str | Path | None = None,
    ):
        """Initialize the error.

        Args:
            message: Error message
            persist_directory: Path to ChromaDB directory if available
        """
        self.message = message
        self.persist_directory = persist_directory
        super().__init__(self.message)


class Neo4jEmptyError(Exception):
    """Raised when Neo4j graph is empty and data needs to be loaded."""

    def __init__(
        self,
        message: str = 'Neo4j graph is empty. Please load data first using load_data().',
        uri: str | None = None,
    ):
        """Initialize the error.

        Args:
            message: Error message
            uri: Neo4j URI if available
        """
        self.message = message
        self.uri = uri
        super().__init__(self.message)


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
        model: str | None = None,
        embeddings: 'Embeddings | None' = None,
        vector_store: VectorStoreInterface | None = None,
        persist_directory: str | Path | None = None,
        llm_provider: str | None = None,
        ollama_base_url: str = OLLAMA_BASE_URL,
        gemini_api_key: str | None = None,
        k_retrieval: int = DEFAULT_K_RETRIEVAL,
    ):
        """Initialize Standard RAG system.

        Args:
            model: Model name to use. If None, uses provider default (Gemini: gemini-pro, Ollama: DEFAULT_MODEL_FALLBACK)
            embeddings: Embedding model. If None, uses default.
            vector_store: Vector store instance. If None, creates ChromaDB.
            persist_directory: Directory to persist vector store
            llm_provider: LLM provider name ("gemini" or "ollama"). If None, uses config default.
            ollama_base_url: Base URL for Ollama API (only used if provider is "ollama")
            gemini_api_key: Google API key for Gemini (only used if provider is "gemini")
            k_retrieval: Number of documents to retrieve
        """
        # Determine model based on provider
        provider = llm_provider or LLM_PROVIDER
        if model is None:
            if provider == 'gemini':
                model = GEMINI_MODEL
            else:
                model = DEFAULT_MODEL_FALLBACK

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

        # Initialize LLM provider (Gemini or Ollama)
        self.llm_provider = LLMProvider(
            provider=provider,
            model=model,
            gemini_api_key=gemini_api_key,
            ollama_base_url=ollama_base_url,
        )

        # Initialize retrieval chain (will be set up after data is loaded)
        self._qa_chain: Any = None
        self._retriever: Any = None
        self._prompt_template: PromptTemplate | None = None
        self._data_loaded = False

        # Check if ChromaDB is empty (but don't auto-load)
        self._check_chromadb_status()

    def _check_chromadb_status(self) -> None:
        """Check if ChromaDB has data and set up retrieval chain if available.

        Raises:
            ChromaDBEmptyError: If ChromaDB is empty
        """
        try:
            doc_count = self.vector_store.count()
            if doc_count > 0:
                # Data exists, set up the retrieval chain
                self._setup_retrieval_chain()
                self._data_loaded = True
            else:
                # ChromaDB is empty - don't raise error here, just mark as not loaded
                # Error will be raised when query is attempted
                self._data_loaded = False
        except Exception:
            # If count() fails, assume empty
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
        """Set up the retrieval chain components.

        Note: We use direct retrieval + LLM approach instead of the deprecated
        RetrievalQA chain from LangChain.
        """
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

        self._retriever = retriever
        self._prompt_template = prompt_template

    def query(self, prompt: str) -> RAGResponse:
        """Query the RAG system.

        Args:
            prompt: User query

        Returns:
            RAGResponse with answer and metadata

        Raises:
            ChromaDBEmptyError: If ChromaDB is empty and data hasn't been loaded
        """
        # Check if data is loaded
        if not self._data_loaded:
            # Re-check ChromaDB status in case it was populated externally
            try:
                doc_count = self.vector_store.count()
                if doc_count == 0:
                    raise ChromaDBEmptyError(
                        'ChromaDB is empty. Please load data first using load_data(). '
                        'You can load data from the Dataset Preview page.',
                        persist_directory=self.persist_directory,
                    )
                # Data exists, set up retrieval chain
                self._setup_retrieval_chain()
                self._data_loaded = True
            except ChromaDBEmptyError:
                raise
            except Exception as e:
                # If count() fails, assume empty
                raise ChromaDBEmptyError(
                    f'ChromaDB is empty or inaccessible. Error: {str(e)}. '
                    'Please load data first using load_data().',
                    persist_directory=self.persist_directory,
                ) from e

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

        # Generate response using LLM provider
        try:
            llm_response = self.llm_provider.chat(formatted_prompt)
            content = llm_response.content
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
    ) -> Generator[tuple[LLMStreamChunk, dict], None, None]:
        """Query the RAG system with streaming response.

        Args:
            prompt: User query
            reasoning: Enable reasoning mode for supported models

        Yields:
            Tuples of (LLMStreamChunk, metadata_dict) where metadata contains
            retrieval information that remains constant across chunks

        Raises:
            ChromaDBEmptyError: If ChromaDB is empty and data hasn't been loaded
        """
        # Check if data is loaded
        if not self._data_loaded:
            # Re-check ChromaDB status in case it was populated externally
            try:
                doc_count = self.vector_store.count()
                if doc_count == 0:
                    raise ChromaDBEmptyError(
                        'ChromaDB is empty. Please load data first using load_data(). '
                        'You can load data from the Dataset Preview page.',
                        persist_directory=self.persist_directory,
                    )
                # Data exists, set up retrieval chain
                self._setup_retrieval_chain()
                self._data_loaded = True
            except ChromaDBEmptyError:
                raise
            except Exception as e:
                # If count() fails, assume empty
                raise ChromaDBEmptyError(
                    f'ChromaDB is empty or inaccessible. Error: {str(e)}. '
                    'Please load data first using load_data().',
                    persist_directory=self.persist_directory,
                ) from e

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

        # Stream response using LLM provider
        try:
            # Note: reasoning parameter is Ollama-specific, pass it only if provider is Ollama
            stream_kwargs = {}
            if self.llm_provider.provider == 'ollama' and reasoning is not None:
                stream_kwargs['reasoning'] = reasoning

            for chunk in self.llm_provider.stream_chat(
                formatted_prompt, **stream_kwargs
            ):
                yield chunk, metadata
        except Exception as e:
            # On error, yield a single chunk with error message
            error_content = (
                f'Error generating response: {str(e)}\n\n'
                f'Retrieved {len(retrieved_docs)} documents:\n\n{context}'
            )
            yield (
                LLMStreamChunk(
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


class GraphRAG(RAGSystem):
    """Graph-Based RAG implementation using Neo4j and GraphCypherQAChain."""

    def __init__(
        self,
        model: str | None = None,
        graph_manager: Neo4jGraphManager | None = None,
        llm_provider: str | None = None,
        ollama_base_url: str = OLLAMA_BASE_URL,
        gemini_api_key: str | None = None,
    ):
        """Initialize Graph-Based RAG system.

        Args:
            model: Model name to use. If None, uses provider default (Gemini: gemini-pro, Ollama: DEFAULT_MODEL_FALLBACK)
            graph_manager: Neo4jGraphManager instance. If None, creates a new one.
            llm_provider: LLM provider name ("gemini" or "ollama"). If None, uses config default.
            ollama_base_url: Base URL for Ollama API (only used if provider is "ollama")
            gemini_api_key: Google API key for Gemini (only used if provider is "gemini")

        Raises:
            ImportError: If required dependencies are not installed
            Neo4jConnectionError: If Neo4j connection fails
        """
        # Determine model based on provider
        provider = llm_provider or LLM_PROVIDER
        if model is None:
            if provider == 'gemini':
                model = GEMINI_MODEL
            else:
                model = DEFAULT_MODEL_FALLBACK

        super().__init__(rag_type=RAG_TYPE_GRAPH, model=model)

        # Initialize Neo4j connection
        if graph_manager is None:
            self.graph_manager = Neo4jGraphManager()
        else:
            self.graph_manager = graph_manager

        # Initialize LLM provider (Gemini or Ollama)
        self.llm_provider = LLMProvider(
            provider=provider,
            model=model,
            gemini_api_key=gemini_api_key,
            ollama_base_url=ollama_base_url,
        )

        # Initialize graph data loader
        self.graph_loader = GraphDataLoader(self.graph_manager)

        # Initialize Cypher QA chain (will be set up after data is loaded)
        self._cypher_chain: Any = None
        self._data_loaded = False

        # Check if graph has data (but don't auto-load)
        self._check_graph_status()

    def _check_graph_status(self) -> None:
        """Check if Neo4j graph has data and set up chain if available.

        Raises:
            Neo4jConnectionError: If connection fails
        """
        try:
            if self.graph_manager.is_connected():
                node_count = self.graph_loader.get_node_count()
                if node_count > 0:
                    # Data exists, set up the Cypher chain
                    self._setup_cypher_chain()
                    self._data_loaded = True
                else:
                    # Graph is empty
                    self._data_loaded = False
            else:
                # Not connected yet, try to connect
                self.graph_manager.connect()
                node_count = self.graph_loader.get_node_count()
                if node_count > 0:
                    self._setup_cypher_chain()
                    self._data_loaded = True
                else:
                    self._data_loaded = False
        except Neo4jConnectionError:
            # Connection failed, will raise error when query is attempted
            self._data_loaded = False
        except Exception:
            # If check fails, assume empty
            self._data_loaded = False

    def load_data(
        self,
        source: str | Path | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        force_reload: bool = False,
    ) -> None:
        """Load and index data into the Neo4j graph.

        Args:
            source: Source of the dataset. If None, loads synthetic articles.
            chunk_size: Size of text chunks (not used for graph, kept for compatibility)
            chunk_overlap: Overlap between chunks (not used for graph, kept for compatibility)
            force_reload: If True, reload data even if it already exists.
                If False, skip loading if data already exists in graph.
        """
        # Check if data already exists in the graph
        if not force_reload and self._data_loaded:
            # Data was already loaded in this session
            return

        # Check if graph already has nodes
        try:
            node_count = self.graph_loader.get_node_count()
            if node_count > 0 and not force_reload:
                # Data already exists, just set up the Cypher chain
                self._setup_cypher_chain()
                self._data_loaded = True
                return
        except Exception:
            # If check fails, assume empty and proceed with loading
            pass

        # Load preprocessed data
        from rag_comparision.core.data_preprocessing import (
            preprocess_synthetic_articles_dataset,
        )

        df = preprocess_synthetic_articles_dataset(
            source=source,
            force_reprocess=False,
        )

        # Load into Neo4j graph
        _ = self.graph_loader.load_from_dataframe(df, force_reload=force_reload)

        # Set up Cypher chain
        self._setup_cypher_chain()

        self._data_loaded = True

    def _validate_cypher_query(self, cypher_query: str) -> tuple[bool, str]:
        """Validate Cypher query syntax and detect common errors.

        Args:
            cypher_query: The Cypher query to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not cypher_query or not isinstance(cypher_query, str):
            return False, 'Empty or invalid query'

        query_original = cypher_query.strip()
        query_lower = query_original.lower()

        # Check for common syntax errors
        # 1. Check for "sql" prefix (common LLM mistake) - can be on its own line or at start
        lines = query_original.split('\n')
        first_line = lines[0].strip().lower() if lines else ''
        if first_line.startswith('sql') or query_lower.startswith('sql'):
            return (
                False,
                "Query starts with 'sql' - should be pure Cypher, not SQL. Remove 'sql' prefix.",
            )

        if first_line.startswith('neo4j') or query_lower.startswith('neo4j'):
            return (
                False,
                "Query starts with 'neo4j' - should be pure Cypher, not Neo4j. Remove 'neo4j' prefix.",
            )

        # 2. Check for SQL keywords that shouldn't be in Cypher (at the start)
        sql_keywords = [
            'select',
            'from',
            'where',
            'join',
            'inner join',
            'left join',
            'right join',
        ]
        first_word = query_lower.split()[0] if query_lower.split() else ''
        # Only flag if it starts with SQL keyword and doesn't have MATCH nearby
        if first_word in sql_keywords:
            # Check if MATCH appears in first 100 chars (should be near the start)
            if 'match' not in query_lower[:100]:
                return (
                    False,
                    f"Query starts with SQL keyword '{first_word}' - should use Cypher syntax (MATCH, RETURN, etc.)",
                )

        # 3. Check for basic Cypher structure (only for substantial queries)
        if len(query_lower) > 20:
            if 'match' not in query_lower and 'return' not in query_lower:
                # Might be valid for some queries, but flag as suspicious
                return (
                    False,
                    'Query missing MATCH or RETURN clauses - required for Cypher queries',
                )

        # 4. Check for balanced parentheses and brackets
        paren_count = query_original.count('(') - query_original.count(')')
        bracket_count = query_original.count('[') - query_original.count(']')
        if paren_count != 0:
            return False, f'Unbalanced parentheses (difference: {paren_count})'
        if bracket_count != 0:
            return False, f'Unbalanced brackets (difference: {bracket_count})'

        # 5. Check for common Cypher syntax issues
        # Check for semicolon at end (not required but can cause issues)
        if query_original.endswith(';') and len(query_original) > 1:
            # Semicolon is usually fine, but some versions don't like it
            pass  # Allow semicolons

        return True, ''

    def _is_multi_hop_question(self, prompt: str) -> bool:
        """Detect if question requires multi-hop reasoning.

        Args:
            prompt: The user's question

        Returns:
            True if the question likely requires multi-hop reasoning
        """
        multi_hop_keywords = [
            'connected',
            'connection',
            'relationship',
            'relationships',
            'collaborate',
            'collaborator',
            'collaborators',
            'collaboration',
            'related',
            'relate',
            'relates',
            'relating',
            'through',
            'via',
            'between',
            'across',
            'chain',
            'theme',
            'themes',
            'pattern',
            'patterns',
            'how are',
            'how do',
            'what connects',
            'what links',
            'trace',
            'path',
            'paths',
            'traverse',
        ]
        prompt_lower = prompt.lower()

        # Check for multi-hop indicators
        has_keywords = any(keyword in prompt_lower for keyword in multi_hop_keywords)

        # Check for questions asking about connections between entities
        connection_patterns = [
            'how are X and Y connected',
            'what connects X to Y',
            'relationship between X and Y',
            'X connected to Y',
        ]
        has_patterns = any(
            pattern.replace('X', '').replace('Y', '') in prompt_lower
            for pattern in connection_patterns
        )

        return has_keywords or has_patterns

    def _setup_cypher_chain(self) -> None:
        """Set up the LangChain GraphCypherQAChain."""
        if GraphCypherQAChain is None:
            # Fallback: use simple query without chain
            self._cypher_chain = None
            return

        try:
            graph = self.graph_manager.get_graph()
            # Refresh schema to ensure it's up to date
            graph.refresh_schema()

            # Get LangChain LLM from provider for Cypher generation and QA
            # Both cypher_llm and qa_llm use the same LLM instance
            langchain_llm = self.llm_provider.get_langchain_llm()

            if langchain_llm is None:
                self._cypher_chain = None
                return

            # Use the same LLM for both Cypher generation and QA
            cypher_llm = langchain_llm
            qa_llm = langchain_llm

            # Create custom QA prompt template that enforces strict adherence to context
            qa_prompt = None
            if PromptTemplate is not None:
                qa_prompt = PromptTemplate(
                    input_variables=['context', 'question'],
                    template=GRAPH_RAG_PROMPT_TEMPLATE,
                )

            cypher_prompt = PromptTemplate(
                input_variables=['schema', 'question'],
                template=GRAPH_RAG_CYPHER_PROMPT_TEMPLATE,
            )

            # Create GraphCypherQAChain
            # verbose=True helps debug and shows Cypher queries being generated
            # allow_dangerous_requests=True is required for security acknowledgment
            # (the chain can execute arbitrary Cypher queries)
            # return_intermediate_steps=True enables access to Cypher query and graph results
            # Try to pass custom qa_prompt to enforce strict context adherence
            base_kwargs = {
                'cypher_llm': cypher_llm,
                'qa_llm': qa_llm,
                'graph': graph,
                'verbose': True,  # Set to True to see Cypher queries and context in logs
                'allow_dangerous_requests': True,  # Required: acknowledge security risks
                'validate_cypher': False,
                'return_intermediate_steps': True,  # Enable access to Cypher query and graph results
                'qa_prompt': qa_prompt,
                'cypher_prompt': cypher_prompt,
            }

            # Try to create chain with custom qa_prompt (different LangChain versions may use different parameter names)
            if qa_prompt is not None:
                self._cypher_chain = GraphCypherQAChain.from_llm(**base_kwargs)
            else:
                # No custom prompt available, use default
                self._cypher_chain = GraphCypherQAChain.from_llm(**base_kwargs)
        except Exception as e:
            # If chain creation fails, set to None and use fallback
            # Log the error for debugging (can be removed in production)
            import warnings

            warnings.warn(
                f'Failed to create GraphCypherQAChain: {type(e).__name__}: {e}. '
                'Falling back to simple query mode.',
                UserWarning,
                stacklevel=2,
            )
            self._cypher_chain = None

    def query(self, prompt: str) -> RAGResponse:
        """Query the Graph-Based RAG system.

        Args:
            prompt: User query

        Returns:
            RAGResponse with answer and metadata

        Raises:
            Neo4jEmptyError: If Neo4j graph is empty and data hasn't been loaded
            Neo4jConnectionError: If Neo4j connection fails
        """
        # Check if data is loaded
        if not self._data_loaded:
            # Re-check graph status in case it was populated externally
            try:
                node_count = self.graph_loader.get_node_count()
                if node_count == 0:
                    raise Neo4jEmptyError(
                        'Neo4j graph is empty. Please load data first using load_data(). '
                        'You can load data from the Dataset Preview page.',
                        uri=self.graph_manager.uri,
                    )
                # Data exists, set up Cypher chain
                self._setup_cypher_chain()
                self._data_loaded = True
            except Neo4jEmptyError:
                raise
            except Neo4jConnectionError:
                raise
            except Exception as e:
                # If check fails, assume empty
                raise Neo4jEmptyError(
                    f'Neo4j graph is empty or inaccessible. Error: {str(e)}. '
                    'Please load data first using load_data().',
                    uri=self.graph_manager.uri,
                ) from e

        # Use GraphCypherQAChain if available
        if self._cypher_chain is not None:
            try:
                # Retry logic for empty context and query errors
                max_retries = 3
                retry_count = 0
                result = None
                final_cypher_query = ''
                final_graph_results = []
                query_error = None
                last_cypher_query = ''
                last_query_error = None  # Keep track of last error for retry prompts

                # Detect if this is a multi-hop question
                is_multi_hop = self._is_multi_hop_question(prompt)

                while retry_count < max_retries:
                    # Modify prompt for retries based on error type and question type
                    if retry_count == 0:
                        if is_multi_hop:
                            # First attempt for multi-hop question - provide specialized guidance
                            query_prompt = (
                                f'{prompt}\n\n'
                                'IMPORTANT: This is a multi-hop question requiring traversal of multiple relationships. '
                                'Generate a Cypher query that:\n'
                                '- Traverses multiple relationships (e.g., Researcher -> Article -> Topic -> Article -> Researcher)\n'
                                '- Uses multiple MATCH clauses or path patterns to connect entities\n'
                                '- For questions about research themes: Find Topics connected to Articles, and group Articles by shared Topics\n'
                                '- For questions about collaborators: Find Researchers connected through shared Articles or Topics\n'
                                '- For questions about connections: Return all relevant nodes and relationships needed to trace the path\n'
                                '- Use OPTIONAL MATCH when relationships might not always exist\n'
                                '- Return enough information to analyze relationships and patterns'
                            )
                        else:
                            query_prompt = prompt
                    else:
                        if last_query_error:
                            # Query execution error - determine if syntax or runtime
                            error_str = last_query_error.lower()
                            is_syntax_err = any(
                                kw in error_str
                                for kw in [
                                    'syntax',
                                    'invalid',
                                    'unexpected',
                                    'expected',
                                    'parse',
                                ]
                            )
                            is_runtime_err = any(
                                kw in error_str
                                for kw in [
                                    'property',
                                    'relationship',
                                    'node',
                                    'label',
                                    'not found',
                                    'does not exist',
                                ]
                            )

                            if is_runtime_err:
                                # Runtime execution error - query syntax is OK but execution failed
                                if is_multi_hop:
                                    query_prompt = (
                                        f'{prompt}\n\n'
                                        f'ERROR: The previous Cypher query failed to execute: {last_query_error}\n'
                                        f'Previous query: {last_cypher_query[:200] if last_cypher_query else "N/A"}\n\n'
                                        'This is a multi-hop question. The query syntax was correct but execution failed. '
                                        'Please generate a new Cypher query that:\n'
                                        '- Uses only properties and relationships that exist in the graph schema\n'
                                        '- Checks node labels and relationship types carefully\n'
                                        '- Uses OPTIONAL MATCH if properties might not exist on all nodes\n'
                                        '- Traverses relationships that actually exist in the graph\n'
                                        '- Returns only properties that are guaranteed to exist'
                                    )
                                else:
                                    query_prompt = (
                                        f'{prompt}\n\n'
                                        f'ERROR: The previous Cypher query failed to execute: {last_query_error}\n'
                                        f'Previous query: {last_cypher_query[:200] if last_cypher_query else "N/A"}\n\n'
                                        'The query syntax was correct but execution failed. Please generate a new query that:\n'
                                        '- Uses only properties and relationships that exist in the graph schema\n'
                                        '- Checks node labels and relationship types carefully\n'
                                        '- Uses OPTIONAL MATCH if properties might not exist\n'
                                        '- Avoids accessing properties that may not exist on all nodes'
                                    )
                            elif is_syntax_err:
                                # Syntax error - ask to fix syntax
                                if is_multi_hop:
                                    query_prompt = (
                                        f'{prompt}\n\n'
                                        f'ERROR: The previous Cypher query had a syntax error: {last_query_error}\n'
                                        f'Previous query: {last_cypher_query[:200] if last_cypher_query else "N/A"}\n\n'
                                        'This is a multi-hop question. Please generate a CORRECT Cypher query that:\n'
                                        '- Uses proper Cypher syntax (MATCH, RETURN, WHERE, etc.)\n'
                                        '- Traverses multiple relationships to answer the multi-hop question\n'
                                        "- Does NOT include 'sql' or 'neo4j' prefix\n"
                                        '- Has balanced parentheses and brackets\n'
                                        '- Follows Neo4j Cypher query language rules'
                                    )
                                else:
                                    query_prompt = (
                                        f'{prompt}\n\n'
                                        f'ERROR: The previous Cypher query had a syntax error: {last_query_error}\n'
                                        f'Previous query: {last_cypher_query[:200] if last_cypher_query else "N/A"}\n\n'
                                        'Please generate a CORRECT Cypher query that:\n'
                                        '- Uses proper Cypher syntax (MATCH, RETURN, WHERE, etc.)\n'
                                        "- Does NOT include 'sql' or 'neo4j' prefix\n"
                                        '- Has balanced parentheses and brackets\n'
                                        '- Follows Neo4j Cypher query language rules'
                                    )
                            else:
                                # Generic error - try to fix both syntax and runtime issues
                                if is_multi_hop:
                                    query_prompt = (
                                        f'{prompt}\n\n'
                                        f'ERROR: The previous Cypher query failed: {last_query_error}\n'
                                        f'Previous query: {last_cypher_query[:200] if last_cypher_query else "N/A"}\n\n'
                                        'This is a multi-hop question. Please generate a new Cypher query that:\n'
                                        '- Uses proper Cypher syntax\n'
                                        '- Uses only existing properties and relationships\n'
                                        '- Traverses multiple relationships correctly\n'
                                        '- Follows Neo4j Cypher query language rules'
                                    )
                                else:
                                    query_prompt = (
                                        f'{prompt}\n\n'
                                        f'ERROR: The previous Cypher query failed: {last_query_error}\n'
                                        f'Previous query: {last_cypher_query[:200] if last_cypher_query else "N/A"}\n\n'
                                        'Please generate a new Cypher query that:\n'
                                        '- Uses proper Cypher syntax\n'
                                        '- Uses only existing properties and relationships\n'
                                        '- Follows Neo4j Cypher query language rules'
                                    )
                        else:
                            # Empty results - try different approach
                            if retry_count == 1:
                                if is_multi_hop:
                                    query_prompt = (
                                        f'{prompt}\n\n'
                                        'Note: The previous Cypher query returned no results for this multi-hop question. '
                                        'Please try a different approach:\n'
                                        '- Use broader relationship traversal (e.g., use variable-length paths with *)\n'
                                        '- Use OPTIONAL MATCH to include entities even if some relationships are missing\n'
                                        '- Try traversing through different relationship types\n'
                                        '- Use case-insensitive matching (toLower() function)\n'
                                        '- Return intermediate nodes in the path, not just endpoints\n'
                                        '- Consider using WITH clauses to build up the query step by step'
                                    )
                                else:
                                    query_prompt = (
                                        f'{prompt}\n\n'
                                        'Note: The previous Cypher query returned no results. '
                                        'Please try a different approach:\n'
                                        '- Use case-insensitive matching (toLower() function)\n'
                                        '- Use CONTAINS with partial keywords instead of full phrases\n'
                                        '- Try searching across different node properties\n'
                                        '- Consider using OPTIONAL MATCH for broader results'
                                    )
                            else:
                                # Second retry - even broader approach
                                if is_multi_hop:
                                    query_prompt = (
                                        f'{prompt}\n\n'
                                        'Note: Previous queries returned no results for this multi-hop question. '
                                        'Please use a very broad search approach:\n'
                                        '- Use variable-length paths (e.g., [:RELATIONSHIP*1..3])\n'
                                        '- Remove strict WHERE conditions\n'
                                        '- Return all nodes and relationships in the path\n'
                                        '- Search across all relationship types\n'
                                        '- Use UNION to combine multiple traversal strategies'
                                    )
                                else:
                                    query_prompt = (
                                        f'{prompt}\n\n'
                                        'Note: Previous queries returned no results. '
                                        'Please use a very broad search approach:\n'
                                        '- Remove strict WHERE conditions\n'
                                        '- Use wildcard patterns with CONTAINS\n'
                                        '- Search across all node types and properties\n'
                                        '- Return any related nodes even if not exact matches'
                                    )

                    # Reset error for this attempt (will be set if error occurs)
                    query_error = None

                    try:
                        # Invoke the chain
                        result = self._cypher_chain.invoke({'query': query_prompt})
                    except Exception as e:
                        # Catch query execution errors
                        error_str = str(e).lower()
                        query_error = str(e)
                        last_query_error = query_error

                        # Categorize the error type
                        is_syntax_error = any(
                            keyword in error_str
                            for keyword in [
                                'syntax',
                                'invalid',
                                'unexpected',
                                'expected',
                                'parse',
                            ]
                        )

                        is_runtime_error = any(
                            keyword in error_str
                            for keyword in [
                                'property',
                                'relationship',
                                'node',
                                'label',
                                'index',
                                'not found',
                                'does not exist',
                                'unknown',
                                'cannot',
                                'failed',
                                'execution',
                                'timeout',
                                'memory',
                            ]
                        )

                        is_connection_error = any(
                            keyword in error_str
                            for keyword in [
                                'connection',
                                'connect',
                                'network',
                                'timeout',
                                'refused',
                                'unreachable',
                                'authentication',
                                'unauthorized',
                            ]
                        )

                        # Handle different error types
                        if is_connection_error:
                            # Connection errors should not be retried - re-raise immediately
                            raise
                        elif is_syntax_error or is_runtime_error:
                            # Syntax or runtime errors - retry with fix prompt
                            # Mark as query error so we use the error-specific retry prompt
                            retry_count += 1
                            if retry_count >= max_retries:
                                # Last retry failed - will be handled below
                                break
                            continue
                        else:
                            # Unknown error type - try retrying once more
                            if retry_count < max_retries - 1:
                                retry_count += 1
                                last_query_error = (
                                    f'Query execution failed: {query_error}'
                                )
                                continue
                            else:
                                # Final retry failed - re-raise
                                raise

                    # Extract intermediate steps if available
                    # With return_intermediate_steps=True, intermediate_steps is a list of dicts:
                    # [{'query': "MATCH ..."}, {'context': [...]}]
                    if isinstance(result, dict):
                        intermediate_steps = result.get('intermediate_steps', [])
                        cypher_query = ''
                        graph_results = []

                        # Check if intermediate_steps exist and extract Cypher query and context
                        if intermediate_steps:
                            for step in intermediate_steps:
                                if isinstance(step, dict):
                                    # Step is a dict - check for 'query' or 'cypher' key
                                    if 'query' in step:
                                        cypher_query = step['query']
                                    elif 'cypher' in step:
                                        cypher_query = step['cypher']

                                    # Check for 'context' key containing graph results
                                    if 'context' in step:
                                        context = step['context']
                                        if isinstance(context, (list, tuple)):
                                            graph_results = [
                                                _serialize_neo4j_object(item)
                                                for item in context
                                            ]
                                        elif context:
                                            graph_results = [
                                                _serialize_neo4j_object(context)
                                            ]
                                elif isinstance(step, tuple) and len(step) >= 2:
                                    # Step is (action, observation) tuple (alternative format)
                                    action = step[0]
                                    observation = step[1]

                                    # Extract Cypher query from action
                                    if isinstance(action, dict):
                                        cypher_query = action.get(
                                            'query', action.get('cypher', cypher_query)
                                        )

                                    # Extract graph results from observation
                                    if observation:
                                        if isinstance(observation, (list, tuple)):
                                            graph_results = [
                                                _serialize_neo4j_object(item)
                                                for item in observation
                                            ]
                                        elif isinstance(observation, str):
                                            graph_results = [observation]
                                        else:
                                            graph_results = [
                                                _serialize_neo4j_object(observation)
                                            ]

                        # If no intermediate_steps found, try to get from result directly
                        if not cypher_query and not graph_results:
                            # Check if there's a 'query' or 'cypher' key in result
                            cypher_query = result.get('query', result.get('cypher', ''))
                            raw_graph_results = result.get(
                                'context', result.get('graph_results', [])
                            )
                            if not isinstance(raw_graph_results, list):
                                raw_graph_results = (
                                    [raw_graph_results] if raw_graph_results else []
                                )
                            graph_results = [
                                _serialize_neo4j_object(item)
                                for item in raw_graph_results
                            ]

                        # Validate Cypher query syntax
                        if cypher_query:
                            is_valid, validation_error = self._validate_cypher_query(
                                cypher_query
                            )
                            if not is_valid:
                                # Query has syntax error - set error and retry
                                query_error = validation_error
                                last_query_error = validation_error
                                last_cypher_query = cypher_query
                                retry_count += 1
                                continue

                        # Store the extracted values for this iteration
                        final_cypher_query = cypher_query
                        final_graph_results = graph_results
                        last_cypher_query = cypher_query

                        # Check if we got results - if yes, break out of retry loop
                        if graph_results and len(graph_results) > 0:
                            # Check if results are not just empty dicts/lists
                            non_empty_results = [
                                r
                                for r in graph_results
                                if r and (not isinstance(r, (dict, list)) or len(r) > 0)
                            ]
                            if non_empty_results:
                                break

                    retry_count += 1

                # Handle case where all retries failed and result is None
                if result is None:
                    # All retries exhausted - create error response
                    content = (
                        f'Unable to execute query after {retry_count} attempts.\n\n'
                        f'Last error: {last_query_error if last_query_error else "Unknown error"}\n\n'
                        f'Last attempted query: {last_cypher_query[:500] if last_cypher_query else "N/A"}\n\n'
                        'Please try rephrasing your question or check if the graph contains the requested data.'
                    )
                    return RAGResponse(
                        content=content,
                        rag_type=self.rag_type,
                        model=self.model,
                        retrieved_chunks=0,
                        confidence=DEFAULT_CONFIDENCE,
                        metadata={
                            'retrieved_documents': 0,
                            'confidence': DEFAULT_CONFIDENCE,
                            'cypher_query': last_cypher_query,
                            'graph_results': [],
                            'retry_count': retry_count,
                            'query_error': last_query_error
                            if last_query_error
                            else None,
                        },
                    )

                # Extract content and metadata
                # GraphCypherQAChain returns a dict with 'result' key containing the answer
                if isinstance(result, dict):
                    content = result.get('result', str(result))
                else:
                    # Result is a string
                    content = str(result)

                # Post-process content to fix any Neo4j object string representations
                # The chain might have formatted Neo4j objects as strings like "neo4j.time.Date(2024, 2, 14)"
                # Convert these to human-readable format
                import re

                # Pattern to match neo4j.time.Date(YEAR, MONTH, DAY)
                date_pattern = r'neo4j\.time\.Date\((\d+),\s*(\d+),\s*(\d+)\)'

                def replace_date(match):
                    year, month, day = match.groups()
                    return f'{year}-{month.zfill(2)}-{day.zfill(2)}'

                content = re.sub(date_pattern, replace_date, content)

                # Use the final extracted values (from last successful iteration or last attempt)
                cypher_query = final_cypher_query
                graph_results = final_graph_results

                # Count retrieved chunks from graph results
                if isinstance(graph_results, list):
                    retrieved_count = len(graph_results)
                elif graph_results:
                    retrieved_count = 1
                else:
                    retrieved_count = 0

                # Extract metadata
                metadata = {
                    'retrieved_documents': retrieved_count,
                    'confidence': DEFAULT_CONFIDENCE,
                    'cypher_query': cypher_query,
                    'graph_results': graph_results,
                    'retry_count': retry_count,
                    'query_error': last_query_error
                    if last_query_error
                    else (query_error if query_error else None),
                    'full_result': result
                    if isinstance(result, dict)
                    else {'result': result},
                }

                # If we still have a query error after all retries, add it to the content
                final_error = last_query_error if last_query_error else query_error
                if final_error and not graph_results:
                    content = (
                        f'Unable to generate a valid Cypher query after {retry_count} attempts.\n\n'
                        f'Last error: {final_error}\n\n'
                        f'Last attempted query: {cypher_query[:500] if cypher_query else "N/A"}\n\n'
                        'Please try rephrasing your question or check if the graph contains the requested data.'
                    )

                return RAGResponse(
                    content=content,
                    rag_type=self.rag_type,
                    model=self.model,
                    retrieved_chunks=retrieved_count,
                    confidence=DEFAULT_CONFIDENCE,  # Graph queries don't have similarity scores
                    metadata=metadata,
                )
            except Exception as e:
                # Fallback to simple query
                content = (
                    f'Error querying graph: {str(e)}\n\n'
                    'Please check your Neo4j connection and ensure data is loaded.'
                )
                return RAGResponse(
                    content=content,
                    rag_type=self.rag_type,
                    model=self.model,
                    retrieved_chunks=0,
                    confidence=DEFAULT_CONFIDENCE,
                    metadata={'error': str(e)},
                )
        else:
            # Fallback: direct Cypher query (simple implementation)
            try:
                graph = self.graph_manager.get_graph()
                # Simple query - just return schema info
                schema = graph.get_schema
                content = (
                    f'Graph schema available, but GraphCypherQAChain is not available. '
                    f'Schema: {schema}\n\n'
                    'Please install langchain-community with Neo4j support.'
                )
                return RAGResponse(
                    content=content,
                    rag_type=self.rag_type,
                    model=self.model,
                    retrieved_chunks=0,
                    confidence=DEFAULT_CONFIDENCE,
                    metadata={'schema': schema},
                )
            except Exception as e:
                content = f'Error accessing graph: {str(e)}'
                return RAGResponse(
                    content=content,
                    rag_type=self.rag_type,
                    model=self.model,
                    retrieved_chunks=0,
                    confidence=DEFAULT_CONFIDENCE,
                    metadata={'error': str(e)},
                )

    def query_stream(
        self, prompt: str, reasoning: bool | None = None
    ) -> Generator[tuple[LLMStreamChunk, dict], None, None]:
        """Query the Graph-Based RAG system with streaming response.

        Note: GraphCypherQAChain doesn't natively support streaming,
        so we'll execute the query and stream the final answer.

        Args:
            prompt: User query
            reasoning: Enable reasoning mode (Ollama-specific, not supported for graph queries)

        Yields:
            Tuples of (LLMStreamChunk, metadata_dict)

        Raises:
            Neo4jEmptyError: If Neo4j graph is empty and data hasn't been loaded
        """
        # Check if data is loaded
        if not self._data_loaded:
            try:
                node_count = self.graph_loader.get_node_count()
                if node_count == 0:
                    raise Neo4jEmptyError(
                        'Neo4j graph is empty. Please load data first using load_data(). '
                        'You can load data from the Dataset Preview page.',
                        uri=self.graph_manager.uri,
                    )
                self._setup_cypher_chain()
                self._data_loaded = True
            except Neo4jEmptyError:
                raise
            except Exception as e:
                raise Neo4jEmptyError(
                    f'Neo4j graph is empty or inaccessible. Error: {str(e)}. '
                    'Please load data first using load_data().',
                    uri=self.graph_manager.uri,
                ) from e

        # Execute query (non-streaming)
        rag_response = self.query(prompt)

        # Extract metadata
        metadata = {
            'retrieved_documents': rag_response.retrieved_chunks,
            'confidence': rag_response.confidence,
            'cypher_query': rag_response.metadata.get('cypher_query', ''),
            'graph_results': rag_response.metadata.get('graph_results', []),
        }

        # Stream the response content using LLM provider
        try:
            # Note: reasoning parameter is Ollama-specific, pass it only if provider is Ollama
            stream_kwargs = {}
            if self.llm_provider.provider == 'ollama' and reasoning is not None:
                stream_kwargs['reasoning'] = reasoning

            for chunk in self.llm_provider.stream_chat(
                rag_response.content, **stream_kwargs
            ):
                yield chunk, metadata
        except Exception as e:
            # On error, yield a single chunk with error message
            error_content = f'Error generating response: {str(e)}'
            yield (
                LLMStreamChunk(
                    content=error_content,
                    done=True,
                ),
                metadata,
            )

    def get_graph_manager(self) -> Neo4jGraphManager:
        """Get the Neo4j graph manager instance.

        Returns:
            Neo4j graph manager instance
        """
        return self.graph_manager


# Try to import LangChain agent components
try:
    from langchain.tools import tool

    HAS_LANGCHAIN_TOOL = True
except ImportError:
    try:
        from langchain_core.tools import tool

        HAS_LANGCHAIN_TOOL = True
    except ImportError:
        tool = None  # type: ignore
        HAS_LANGCHAIN_TOOL = False

try:
    from langchain.agents import create_agent

    HAS_CREATE_AGENT = True
except ImportError:
    try:
        from langchain_core.agents import create_agent

        HAS_CREATE_AGENT = True
    except ImportError:
        create_agent = None  # type: ignore
        HAS_CREATE_AGENT = False

# Try to import Neo4j driver
try:
    from neo4j import GraphDatabase

    HAS_NEO4J_DRIVER = True
except ImportError:
    GraphDatabase = None  # type: ignore
    HAS_NEO4J_DRIVER = False


class GraphRAGAgentic(RAGSystem):
    """Graph-Based RAG implementation using LangChain agents and Neo4j.

    This class uses an agentic workflow where the LLM maintains conversation context
    and can iteratively query the Neo4j graph using Cypher queries through a tool.
    This approach helps address context retention issues compared to single-call chains.
    """

    def __init__(
        self,
        model: str | None = None,
        graph_manager: Neo4jGraphManager | None = None,
        llm_provider: str | None = None,
        ollama_base_url: str = OLLAMA_BASE_URL,
        gemini_api_key: str | None = None,
        verbose: bool = True,
        reasoning: bool | str | None = None,
    ):
        """Initialize Graph-Based RAG system with agentic workflow.

        Args:
            model: Model name to use. If None, uses provider default (Gemini: gemini-pro, Ollama: DEFAULT_MODEL_FALLBACK)
            graph_manager: Neo4jGraphManager instance. If None, creates a new one.
            llm_provider: LLM provider name ("gemini" or "ollama"). If None, uses config default.
            ollama_base_url: Base URL for Ollama API (only used if provider is "ollama")
            gemini_api_key: Google API key for Gemini (only used if provider is "gemini")
            verbose: If True, enable detailed logging of agent steps and tool calls
            reasoning: Enable reasoning/thinking mode for supported models (Ollama-specific).
                True: Captures reasoning in additional_kwargs.reasoning_content
                False: Disables reasoning
                None: Uses model default behavior

        Raises:
            ImportError: If required dependencies are not installed
            Neo4jConnectionError: If Neo4j connection fails
        """
        # Determine model based on provider
        provider = llm_provider or LLM_PROVIDER
        if model is None:
            if provider == 'gemini':
                model = GEMINI_MODEL
            else:
                model = DEFAULT_MODEL_FALLBACK

        super().__init__(rag_type=RAG_TYPE_GRAPH, model=model)

        # Store verbose flag for logging
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)

        # Check for required dependencies
        if not HAS_LANGCHAIN_TOOL or tool is None:
            raise ImportError(
                'langchain.tools or langchain_core.tools is required for agentic workflow. '
                'Install with: pip install langchain langchain-core'
            )
        if not HAS_CREATE_AGENT or create_agent is None:
            raise ImportError(
                'langchain.agents or langchain_core.agents is required for agentic workflow. '
                'Install with: pip install langchain langchain-core'
            )
        if not HAS_NEO4J_DRIVER or GraphDatabase is None:
            raise ImportError(
                'neo4j driver is required. Install with: pip install neo4j'
            )

        # Initialize Neo4j connection
        if graph_manager is None:
            self.graph_manager = Neo4jGraphManager()
        else:
            self.graph_manager = graph_manager

        # Ensure connection is established
        self.graph_manager.connect()

        # Initialize Neo4j driver for direct session access (needed for tool)
        self.neo4j_driver = GraphDatabase.driver(
            self.graph_manager.uri,
            auth=(self.graph_manager.username, self.graph_manager.password),
        )

        # Store reasoning setting
        self.reasoning = reasoning

        # Initialize LLM provider (Gemini or Ollama)
        self.llm_provider = LLMProvider(
            provider=provider,
            model=model,
            gemini_api_key=gemini_api_key,
            ollama_base_url=ollama_base_url,
            reasoning=reasoning,
        )

        # Initialize graph data loader
        self.graph_loader = GraphDataLoader(self.graph_manager)

        # Initialize agent (will be set up after data is loaded)
        self._agent: Any = None
        self._data_loaded = False

        # Check if graph has data (but don't auto-load)
        self._check_graph_status()

    def _check_graph_status(self) -> None:
        """Check if Neo4j graph has data and set up agent if available.

        Raises:
            Neo4jConnectionError: If connection fails
        """
        try:
            if self.graph_manager.is_connected():
                node_count = self.graph_loader.get_node_count()
                if node_count > 0:
                    # Data exists, set up the agent
                    self._setup_agent()
                    self._data_loaded = True
                else:
                    # Graph is empty
                    self._data_loaded = False
            else:
                # Not connected yet, try to connect
                self.graph_manager.connect()
                node_count = self.graph_loader.get_node_count()
                if node_count > 0:
                    self._setup_agent()
                    self._data_loaded = True
                else:
                    self._data_loaded = False
        except Neo4jConnectionError:
            # Connection failed, will raise error when query is attempted
            self._data_loaded = False
        except Exception:
            # If check fails, assume empty
            self._data_loaded = False

    def _create_cypher_tool(self, verbose: bool = True):
        """Create a Cypher query tool for the agent.

        Args:
            verbose: If True, log tool calls and query execution

        Returns:
            Tool function decorated with @tool
        """
        # Capture driver reference and verbose flag in closure
        driver = self.neo4j_driver
        logger = logging.getLogger(__name__)

        @tool
        def run_cypher(query: str) -> str:
            """Run a Cypher query against Neo4j and return results.

            Use this tool to query the graph database. The query should be valid Cypher syntax.
            Returns a summary of results or "No results." if the query returns nothing.

            Args:
                query: A valid Cypher query string

            Returns:
                String representation of query results (limited to first 10 rows)
            """
            if verbose:
                logger.info('[TOOL CALL] run_cypher invoked')
                logger.info(f'[CYPHER QUERY] {query}')

            try:
                with driver.session() as session:
                    result = session.run(query)
                    rows = [dict(record) for record in result]

                    # Consume result to ensure notifications are available
                    # Notifications are only populated after consuming the result
                    summary = result.consume()

                    # Capture Neo4j notifications/warnings
                    notifications = []
                    # Try both result.notifications and summary.notifications
                    notification_source = None
                    if hasattr(summary, 'notifications'):
                        notification_source = summary.notifications
                    elif hasattr(result, 'notifications'):
                        notification_source = result.notifications

                    if notification_source:
                        for notification in notification_source:
                            # Extract relevant notification information
                            # Handle both dict-like and object-like notifications
                            if isinstance(notification, dict):
                                notif_info = {
                                    'severity': notification.get('severity', 'UNKNOWN'),
                                    'code': notification.get('code', ''),
                                    'title': notification.get('title', ''),
                                    'description': notification.get('description', ''),
                                }
                            else:
                                notif_info = {
                                    'severity': getattr(
                                        notification,
                                        'severity',
                                        getattr(
                                            notification, 'severityLevel', 'UNKNOWN'
                                        ),
                                    ),
                                    'code': getattr(notification, 'code', ''),
                                    'title': getattr(notification, 'title', ''),
                                    'description': getattr(
                                        notification, 'description', ''
                                    ),
                                }
                            notifications.append(notif_info)

                    # Build response with results and notifications
                    response_parts = []

                    # Add notifications/warnings first (so model sees them)
                    if notifications:
                        warning_messages = []
                        for notif in notifications:
                            warning_msg = f'WARNING: {notif.get("title", "")}'
                            if notif.get('description'):
                                warning_msg += f' - {notif.get("description", "")}'
                            if notif.get('code'):
                                warning_msg += f' (Code: {notif.get("code", "")})'
                            warning_messages.append(warning_msg)

                        response_parts.append('⚠️ NEO4J WARNINGS:')
                        response_parts.extend(warning_messages)
                        response_parts.append('')  # Empty line separator

                        if verbose:
                            logger.warning(
                                f'[NEO4J WARNINGS] {len(notifications)} warning(s) detected'
                            )
                            for notif in notifications:
                                logger.warning(
                                    f'[NEO4J WARNING] {notif.get("title", "")}: {notif.get("description", "")}'
                                )

                    # Serialize Neo4j objects and limit results
                    if not rows:
                        if verbose:
                            logger.info('[QUERY RESULT] No results returned')
                        if notifications:
                            # Return warnings even if no results
                            return '\n'.join(response_parts) + 'No results returned.'
                        return 'No results.'

                    # Serialize each row and limit to 10 rows
                    serialized_rows = []
                    for row in rows[:10]:
                        serialized_row = _serialize_neo4j_object(row)
                        serialized_rows.append(str(serialized_row))

                    result_str = '\n'.join(serialized_rows)

                    # Combine warnings and results
                    if response_parts:
                        response_parts.append('Query Results:')
                        response_parts.append(result_str)
                        final_response = '\n'.join(response_parts)
                    else:
                        final_response = result_str

                    if verbose:
                        logger.info(
                            f'[QUERY RESULT] Returned {len(rows)} row(s) (showing first 10)'
                        )
                        if notifications:
                            logger.debug(
                                f'[QUERY RESULT WITH WARNINGS] {final_response[:500]}...'
                            )
                        else:
                            logger.debug(
                                f'[QUERY RESULT DETAILS] {result_str[:500]}...'
                            )  # First 500 chars

                    return final_response
            except Exception as e:
                # Extract detailed error information from Neo4j exceptions
                error_parts = []
                error_parts.append('❌ QUERY EXECUTION ERROR:')

                # Try to extract Neo4j-specific error attributes
                error_code = None
                error_message = None

                # Check for Neo4j error attributes (code and message)
                if hasattr(e, 'code'):
                    error_code = getattr(e, 'code', None)
                if hasattr(e, 'message'):
                    error_message = getattr(e, 'message', None)

                # Also check if error is a dict-like object
                if isinstance(e, dict):
                    error_code = e.get('code', None)
                    error_message = e.get('message', None)

                # Format error details for the model
                if error_code:
                    error_parts.append(f'Error Code: {error_code}')
                if error_message:
                    error_parts.append(f'Error Message: {error_message}')

                # Always include the full string representation as fallback
                full_error_str = str(e)
                if error_code or error_message:
                    # If we extracted structured info, add full error as additional context
                    error_parts.append(f'Full Error Details: {full_error_str}')
                else:
                    # If no structured info, use full error string as primary message
                    error_parts.append(full_error_str)

                error_msg = '\n'.join(error_parts)

                if verbose:
                    logger.error(f'[QUERY ERROR] {error_msg}')

                # Return formatted error message - this will be passed to the model via ToolMessage
                return error_msg

        return run_cypher

    def _create_graph_statistics_tool(self, verbose: bool = True):
        """Create a graph statistics tool for the agent.

        Args:
            verbose: If True, log tool calls

        Returns:
            Tool function decorated with @tool
        """
        graph_manager = self.graph_manager
        logger = logging.getLogger(__name__)

        @tool
        def get_graph_statistics() -> str:
            """Get comprehensive statistics about the Neo4j graph database.

            Returns a summary including:
            - Total number of nodes
            - Total number of relationships
            - Node counts by label (e.g., Article, Researcher, Topic)
            - Relationship counts by type (e.g., PUBLISHED, IN_TOPIC)

            Returns:
                String representation of graph statistics
            """
            if verbose:
                logger.info('[TOOL CALL] get_graph_statistics invoked')

            try:
                stats = graph_manager.get_statistics()

                # Format statistics as a readable string
                result_parts = []
                result_parts.append('Graph Statistics:')
                result_parts.append(f'  Total Nodes: {stats["total_nodes"]}')
                result_parts.append(
                    f'  Total Relationships: {stats["total_relationships"]}'
                )

                if stats['nodes_by_label']:
                    result_parts.append('\nNodes by Label:')
                    for label, count in stats['nodes_by_label'].items():
                        result_parts.append(f'  {label}: {count}')

                if stats['relationships_by_type']:
                    result_parts.append('\nRelationships by Type:')
                    for rel_type, count in stats['relationships_by_type'].items():
                        result_parts.append(f'  {rel_type}: {count}')

                result_str = '\n'.join(result_parts)

                if verbose:
                    logger.info(
                        '[TOOL RESULT] get_graph_statistics returned statistics'
                    )
                    logger.debug(f'[TOOL RESULT DETAILS] {result_str[:500]}...')

                return result_str
            except Exception as e:
                # Extract detailed error information
                error_parts = []
                error_parts.append('❌ TOOL EXECUTION ERROR (get_graph_statistics):')

                # Try to extract error attributes
                error_code = None
                error_message = None

                if hasattr(e, 'code'):
                    error_code = getattr(e, 'code', None)
                if hasattr(e, 'message'):
                    error_message = getattr(e, 'message', None)

                if isinstance(e, dict):
                    error_code = e.get('code', None)
                    error_message = e.get('message', None)

                # Format error details for the model
                if error_code:
                    error_parts.append(f'Error Code: {error_code}')
                if error_message:
                    error_parts.append(f'Error Message: {error_message}')

                full_error_str = str(e)
                if error_code or error_message:
                    error_parts.append(f'Full Error Details: {full_error_str}')
                else:
                    error_parts.append(full_error_str)

                error_msg = '\n'.join(error_parts)

                if verbose:
                    logger.error(f'[TOOL ERROR] {error_msg}')
                return error_msg

        return get_graph_statistics

    def _create_node_count_tool(self, verbose: bool = True):
        """Create a node count tool for the agent.

        Args:
            verbose: If True, log tool calls

        Returns:
            Tool function decorated with @tool
        """
        graph_loader = self.graph_loader
        logger = logging.getLogger(__name__)

        @tool
        def get_node_count(label: str | None = None) -> str:
            """Get the count of nodes in the graph, optionally filtered by label.

            Use this tool to quickly check how many nodes exist in the graph,
            or how many nodes of a specific type (label) exist.

            Args:
                label: Optional node label to filter by (e.g., "Article", "Researcher", "Topic").
                    If None, returns total count of all nodes.

            Returns:
                String representation of node count
            """
            if verbose:
                logger.info(f'[TOOL CALL] get_node_count invoked with label={label}')

            try:
                count = graph_loader.get_node_count(label=label)

                if label:
                    result = f'Number of {label} nodes: {count}'
                else:
                    result = f'Total number of nodes: {count}'

                if verbose:
                    logger.info(f'[TOOL RESULT] get_node_count returned {count}')

                return result
            except Exception as e:
                # Extract detailed error information
                error_parts = []
                error_parts.append('❌ TOOL EXECUTION ERROR (get_node_count):')

                # Try to extract error attributes
                error_code = None
                error_message = None

                if hasattr(e, 'code'):
                    error_code = getattr(e, 'code', None)
                if hasattr(e, 'message'):
                    error_message = getattr(e, 'message', None)

                if isinstance(e, dict):
                    error_code = e.get('code', None)
                    error_message = e.get('message', None)

                # Format error details for the model
                if error_code:
                    error_parts.append(f'Error Code: {error_code}')
                if error_message:
                    error_parts.append(f'Error Message: {error_message}')

                full_error_str = str(e)
                if error_code or error_message:
                    error_parts.append(f'Full Error Details: {full_error_str}')
                else:
                    error_parts.append(full_error_str)

                error_msg = '\n'.join(error_parts)

                if verbose:
                    logger.error(f'[TOOL ERROR] {error_msg}')
                return error_msg

        return get_node_count

    def _create_relationship_count_tool(self, verbose: bool = True):
        """Create a relationship count tool for the agent.

        Args:
            verbose: If True, log tool calls

        Returns:
            Tool function decorated with @tool
        """
        graph_loader = self.graph_loader
        logger = logging.getLogger(__name__)

        @tool
        def get_relationship_count(rel_type: str | None = None) -> str:
            """Get the count of relationships in the graph, optionally filtered by type.

            Use this tool to quickly check how many relationships exist in the graph,
            or how many relationships of a specific type exist.

            Args:
                rel_type: Optional relationship type to filter by (e.g., "PUBLISHED", "IN_TOPIC").
                    If None, returns total count of all relationships.

            Returns:
                String representation of relationship count
            """
            if verbose:
                logger.info(
                    f'[TOOL CALL] get_relationship_count invoked with rel_type={rel_type}'
                )

            try:
                count = graph_loader.get_relationship_count(rel_type=rel_type)

                if rel_type:
                    result = f'Number of {rel_type} relationships: {count}'
                else:
                    result = f'Total number of relationships: {count}'

                if verbose:
                    logger.info(
                        f'[TOOL RESULT] get_relationship_count returned {count}'
                    )

                return result
            except Exception as e:
                # Extract detailed error information
                error_parts = []
                error_parts.append('❌ TOOL EXECUTION ERROR (get_relationship_count):')

                # Try to extract error attributes
                error_code = None
                error_message = None

                if hasattr(e, 'code'):
                    error_code = getattr(e, 'code', None)
                if hasattr(e, 'message'):
                    error_message = getattr(e, 'message', None)

                if isinstance(e, dict):
                    error_code = e.get('code', None)
                    error_message = e.get('message', None)

                # Format error details for the model
                if error_code:
                    error_parts.append(f'Error Code: {error_code}')
                if error_message:
                    error_parts.append(f'Error Message: {error_message}')

                full_error_str = str(e)
                if error_code or error_message:
                    error_parts.append(f'Full Error Details: {full_error_str}')
                else:
                    error_parts.append(full_error_str)

                error_msg = '\n'.join(error_parts)

                if verbose:
                    logger.error(f'[TOOL ERROR] {error_msg}')
                return error_msg

        return get_relationship_count

    def _setup_agent(self) -> None:
        """Set up the LangChain agent with graph query tools.

        Creates an agent with the following tools:
        - run_cypher: Execute Cypher queries
        - get_graph_statistics: Get comprehensive graph statistics
        - get_node_count: Get node counts (optionally filtered by label)
        - get_relationship_count: Get relationship counts (optionally filtered by type)
        """
        try:
            # Get LangChain LLM from provider
            langchain_llm = self.llm_provider.get_langchain_llm()

            if langchain_llm is None:
                self._agent = None
                return

            # Create tools
            cypher_tool = self._create_cypher_tool(verbose=self.verbose)
            statistics_tool = self._create_graph_statistics_tool(verbose=self.verbose)
            node_count_tool = self._create_node_count_tool(verbose=self.verbose)
            relationship_count_tool = self._create_relationship_count_tool(
                verbose=self.verbose
            )

            tools = [
                cypher_tool,
                statistics_tool,
                node_count_tool,
                relationship_count_tool,
            ]

            if self.verbose:
                self.logger.info(
                    f'[AGENT SETUP] Created agent with {len(tools)} tool(s)'
                )

            # Create system prompt for the agent
            system_prompt = (
                'You are a helpful assistant that can query a Neo4j graph database. '
                'You have access to several tools:\n'
                '- `run_cypher`: Execute Cypher queries against the graph\n'
                '- `get_graph_statistics`: Get comprehensive statistics about the graph (node/relationship counts by type)\n'
                '- `get_node_count`: Get count of nodes, optionally filtered by label\n'
                '- `get_relationship_count`: Get count of relationships, optionally filtered by type\n\n'
                'When answering questions:\n'
                '1. Use `get_graph_statistics` to get an overview of the graph structure\n'
                '2. Use `get_node_count` or `get_relationship_count` for quick counts\n'
                '3. Use `run_cypher` to query the graph when you need detailed information\n'
                '4. Analyze the results and provide a clear, concise answer\n'
                '5. If a query returns no results, try a different approach or rephrase the query\n'
                '6. For multi-hop questions, you may need to execute multiple queries to gather all necessary information\n'
                '7. Always base your answer on the actual query results, not on assumptions\n\n'
                'The graph contains research articles, researchers, topics, and their relationships. '
                'Use the available tools to explore these entities and their connections.'
            )

            # Create agent - try passing LLM object first, then try model string format
            try:
                # Try passing LangChain LLM object directly
                self._agent = create_agent(
                    model=langchain_llm,
                    tools=tools,
                    system_prompt=system_prompt,
                )
            except (TypeError, ValueError):
                # If that fails, try model string format (e.g., "ollama:qwen3:0.6b" or "gemini:gemini-pro")
                model_string = f'{self.llm_provider.provider}:{self.model}'
                self._agent = create_agent(
                    model=model_string,
                    tools=tools,
                    system_prompt=system_prompt,
                )
        except Exception as e:
            # If agent creation fails, set to None and use fallback
            import warnings

            warnings.warn(
                f'Failed to create agent: {type(e).__name__}: {e}. '
                'Falling back to simple query mode.',
                UserWarning,
                stacklevel=2,
            )
            self._agent = None

    def load_data(
        self,
        source: str | Path | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        force_reload: bool = False,
    ) -> None:
        """Load and index data into the Neo4j graph.

        Args:
            source: Source of the dataset. If None, loads synthetic articles.
            chunk_size: Size of text chunks (not used for graph, kept for compatibility)
            chunk_overlap: Overlap between chunks (not used for graph, kept for compatibility)
            force_reload: If True, reload data even if it already exists.
                If False, skip loading if data already exists in graph.
        """
        # Check if data already exists in the graph
        if not force_reload and self._data_loaded:
            # Data was already loaded in this session
            return

        # Check if graph already has nodes
        try:
            node_count = self.graph_loader.get_node_count()
            if node_count > 0 and not force_reload:
                # Data already exists, just set up the agent
                self._setup_agent()
                self._data_loaded = True
                return
        except Exception:
            # If check fails, assume empty and proceed with loading
            pass

        # Load preprocessed data
        from rag_comparision.core.data_preprocessing import (
            preprocess_synthetic_articles_dataset,
        )

        df = preprocess_synthetic_articles_dataset(
            source=source,
            force_reprocess=False,
        )

        # Load into Neo4j graph
        _ = self.graph_loader.load_from_dataframe(df, force_reload=force_reload)

        # Set up agent
        self._setup_agent()

        self._data_loaded = True

    def query(self, prompt: str) -> RAGResponse:
        """Query the Graph-Based RAG system using agentic workflow.

        Args:
            prompt: User query

        Returns:
            RAGResponse with answer and metadata

        Raises:
            Neo4jEmptyError: If Neo4j graph is empty and data hasn't been loaded
            Neo4jConnectionError: If Neo4j connection fails
        """
        # Check if data is loaded
        if not self._data_loaded:
            # Re-check graph status in case it was populated externally
            try:
                node_count = self.graph_loader.get_node_count()
                if node_count == 0:
                    raise Neo4jEmptyError(
                        'Neo4j graph is empty. Please load data first using load_data(). '
                        'You can load data from the Dataset Preview page.',
                        uri=self.graph_manager.uri,
                    )
                # Data exists, set up agent
                self._setup_agent()
                self._data_loaded = True
            except Neo4jEmptyError:
                raise
            except Neo4jConnectionError:
                raise
            except Exception as e:
                # If check fails, assume empty
                raise Neo4jEmptyError(
                    f'Neo4j graph is empty or inaccessible. Error: {str(e)}. '
                    'Please load data first using load_data().',
                    uri=self.graph_manager.uri,
                ) from e

        # Use agent if available
        if self._agent is not None:
            try:
                if self.verbose:
                    self.logger.info(f'[AGENT QUERY] Starting query: {prompt}')

                # Invoke agent with user message
                # The agent expects messages in a specific format
                result = self._agent.invoke(
                    {'messages': [{'role': 'user', 'content': prompt}]}
                )

                if self.verbose:
                    self.logger.info(
                        f'[AGENT RESULT] Agent returned result (type: {type(result).__name__})'
                    )

                # Extract content from agent response
                # The response format may vary by LangChain version
                if isinstance(result, dict):
                    # Check for messages in result
                    if 'messages' in result:
                        messages = result['messages']

                        if self.verbose:
                            self.logger.info(
                                f'[AGENT MESSAGES] Found {len(messages)} message(s) in conversation'
                            )
                            # Log each message - handle both dict and LangChain message objects
                            for i, msg in enumerate(messages):
                                # Determine message type and role
                                if isinstance(msg, dict):
                                    msg_type = msg.get('type', 'unknown')
                                    msg_role = msg.get('role', msg_type)
                                    msg_content = (
                                        str(msg.get('content', ''))[:200]
                                        if msg.get('content')
                                        else ''
                                    )

                                    # Check for tool calls
                                    tool_calls = msg.get(
                                        'tool_calls', msg.get('tool_calls', [])
                                    )
                                    if tool_calls:
                                        self.logger.info(
                                            f'[AGENT MSG {i}] Type: {msg_type}, Role: {msg_role}, Has {len(tool_calls)} tool call(s)'
                                        )
                                        for tc in tool_calls:
                                            tc_name = (
                                                tc.get(
                                                    'name',
                                                    tc.get('function', {}).get(
                                                        'name', 'unknown'
                                                    ),
                                                )
                                                if isinstance(tc, dict)
                                                else getattr(tc, 'name', 'unknown')
                                            )
                                            tc_args = (
                                                tc.get(
                                                    'args',
                                                    tc.get('function', {}).get(
                                                        'arguments', ''
                                                    ),
                                                )
                                                if isinstance(tc, dict)
                                                else getattr(tc, 'args', '')
                                            )
                                            if (
                                                isinstance(tc_args, dict)
                                                and 'query' in tc_args
                                            ):
                                                self.logger.info(
                                                    f'[TOOL CALL] {tc_name} with query: {tc_args.get("query", "")[:100]}...'
                                                )
                                            else:
                                                self.logger.info(
                                                    f'[TOOL CALL] {tc_name}'
                                                )
                                    else:
                                        if msg_content:
                                            self.logger.info(
                                                f'[AGENT MSG {i}] Type: {msg_type}, Role: {msg_role}, Content: {msg_content}...'
                                            )
                                        else:
                                            self.logger.info(
                                                f'[AGENT MSG {i}] Type: {msg_type}, Role: {msg_role}'
                                            )
                                else:
                                    # Handle LangChain message objects (HumanMessage, AIMessage, ToolMessage, etc.)
                                    msg_type = type(msg).__name__

                                    # Try to get role from message type
                                    if 'Human' in msg_type:
                                        msg_role = 'user'
                                    elif 'AI' in msg_type or 'Assistant' in msg_type:
                                        msg_role = 'assistant'
                                    elif 'Tool' in msg_type:
                                        msg_role = 'tool'
                                    else:
                                        msg_role = msg_type.lower().replace(
                                            'message', ''
                                        )

                                    # Get content
                                    msg_content = ''
                                    if hasattr(msg, 'content'):
                                        content_val = msg.content
                                        if isinstance(content_val, str):
                                            msg_content = content_val[:200]
                                        elif isinstance(content_val, list):
                                            # Handle list content (e.g., Gemini format)
                                            msg_content = str(content_val)[:200]
                                        else:
                                            msg_content = str(content_val)[:200]

                                    # Check for tool calls
                                    tool_calls = []
                                    if hasattr(msg, 'tool_calls'):
                                        tool_calls = msg.tool_calls or []
                                    elif hasattr(msg, 'tool_call_id'):
                                        # ToolMessage
                                        tool_calls = [{'name': 'tool_response'}]

                                    if tool_calls:
                                        self.logger.info(
                                            f'[AGENT MSG {i}] Type: {msg_type}, Role: {msg_role}, Has {len(tool_calls)} tool call(s)'
                                        )
                                        for tc in tool_calls:
                                            if isinstance(tc, dict):
                                                tc_name = tc.get(
                                                    'name',
                                                    tc.get('function', {}).get(
                                                        'name', 'unknown'
                                                    ),
                                                )
                                                tc_args = tc.get(
                                                    'args',
                                                    tc.get('function', {}).get(
                                                        'arguments', ''
                                                    ),
                                                )
                                                if (
                                                    isinstance(tc_args, dict)
                                                    and 'query' in tc_args
                                                ):
                                                    self.logger.info(
                                                        f'[TOOL CALL] {tc_name} with query: {tc_args.get("query", "")[:100]}...'
                                                    )
                                                else:
                                                    self.logger.info(
                                                        f'[TOOL CALL] {tc_name}'
                                                    )
                                            else:
                                                tc_name = getattr(
                                                    tc,
                                                    'name',
                                                    getattr(tc, 'function', {}).get(
                                                        'name', 'unknown'
                                                    )
                                                    if hasattr(tc, 'function')
                                                    else 'unknown',
                                                )
                                                self.logger.info(
                                                    f'[TOOL CALL] {tc_name}'
                                                )
                                    else:
                                        if msg_content:
                                            self.logger.info(
                                                f'[AGENT MSG {i}] Type: {msg_type}, Role: {msg_role}, Content: {msg_content}...'
                                            )
                                        else:
                                            self.logger.info(
                                                f'[AGENT MSG {i}] Type: {msg_type}, Role: {msg_role}'
                                            )

                        # Get the last message (should be the agent's response)
                        if messages and len(messages) > 0:
                            last_message = messages[-1]
                            if isinstance(last_message, dict):
                                content = last_message.get('content', str(result))
                                # Handle list content (e.g., Gemini format)
                                if isinstance(content, list):
                                    # Extract text from list items
                                    text_parts = []
                                    for item in content:
                                        if isinstance(item, dict):
                                            if 'text' in item:
                                                text_parts.append(str(item['text']))
                                            elif 'content' in item:
                                                text_parts.append(str(item['content']))
                                        elif isinstance(item, str):
                                            text_parts.append(item)
                                    content = (
                                        ''.join(text_parts)
                                        if text_parts
                                        else str(result)
                                    )
                            else:
                                # LangChain message object
                                content_val = getattr(
                                    last_message, 'content', str(result)
                                )
                                if isinstance(content_val, list):
                                    # Handle list content (e.g., Gemini format)
                                    text_parts = []
                                    for item in content_val:
                                        if isinstance(item, dict):
                                            if 'text' in item:
                                                text_parts.append(str(item['text']))
                                            elif 'content' in item:
                                                text_parts.append(str(item['content']))
                                        elif isinstance(item, str):
                                            text_parts.append(item)
                                    content = (
                                        ''.join(text_parts)
                                        if text_parts
                                        else str(content_val)
                                    )
                                elif isinstance(content_val, str):
                                    content = content_val
                                else:
                                    content = str(content_val)
                        else:
                            content = str(result)
                    else:
                        content = result.get(
                            'output', result.get('result', str(result))
                        )
                else:
                    content = str(result)

                if self.verbose:
                    self.logger.info(
                        f'[AGENT RESPONSE] Final content length: {len(content)} characters'
                    )

                # Extract metadata
                metadata = {
                    'retrieved_documents': 0,  # Agent queries are dynamic
                    'confidence': DEFAULT_CONFIDENCE,
                    'agent_response': result
                    if isinstance(result, dict)
                    else {'result': result},
                }

                return RAGResponse(
                    content=content,
                    rag_type=self.rag_type,
                    model=self.model,
                    retrieved_chunks=0,  # Agent queries are dynamic
                    confidence=DEFAULT_CONFIDENCE,
                    metadata=metadata,
                )
            except Exception as e:
                # Fallback to error response
                content = (
                    f'Error querying graph with agent: {str(e)}\n\n'
                    'Please check your Neo4j connection and ensure data is loaded.'
                )
                return RAGResponse(
                    content=content,
                    rag_type=self.rag_type,
                    model=self.model,
                    retrieved_chunks=0,
                    confidence=DEFAULT_CONFIDENCE,
                    metadata={'error': str(e)},
                )
        else:
            # Fallback: direct query (simple implementation)
            try:
                content = (
                    'Agent is not available. Please ensure LangChain agent dependencies are installed. '
                    'Install with: pip install langchain langchain-core'
                )
                return RAGResponse(
                    content=content,
                    rag_type=self.rag_type,
                    model=self.model,
                    retrieved_chunks=0,
                    confidence=DEFAULT_CONFIDENCE,
                    metadata={'error': 'Agent not initialized'},
                )
            except Exception as e:
                content = f'Error accessing graph: {str(e)}'
                return RAGResponse(
                    content=content,
                    rag_type=self.rag_type,
                    model=self.model,
                    retrieved_chunks=0,
                    confidence=DEFAULT_CONFIDENCE,
                    metadata={'error': str(e)},
                )

    def query_stream(
        self, prompt: str, reasoning: bool | None = None
    ) -> Generator[tuple[LLMStreamChunk, dict], None, None]:
        """Query the Graph-Based RAG system with streaming response.

        Note: Agent streaming support may vary by LangChain version.
        For now, we execute the query and stream the final answer.

        Args:
            prompt: User query
            reasoning: Enable reasoning mode (Ollama-specific, not supported for agent queries)

        Yields:
            Tuples of (LLMStreamChunk, metadata_dict)

        Raises:
            Neo4jEmptyError: If Neo4j graph is empty and data hasn't been loaded
        """
        # Check if data is loaded
        if not self._data_loaded:
            try:
                node_count = self.graph_loader.get_node_count()
                if node_count == 0:
                    raise Neo4jEmptyError(
                        'Neo4j graph is empty. Please load data first using load_data(). '
                        'You can load data from the Dataset Preview page.',
                        uri=self.graph_manager.uri,
                    )
                self._setup_agent()
                self._data_loaded = True
            except Neo4jEmptyError:
                raise
            except Exception as e:
                raise Neo4jEmptyError(
                    f'Neo4j graph is empty or inaccessible. Error: {str(e)}. '
                    'Please load data first using load_data().',
                    uri=self.graph_manager.uri,
                ) from e

        # Execute query (non-streaming for now)
        rag_response = self.query(prompt)

        # Extract metadata
        metadata = {
            'retrieved_documents': rag_response.retrieved_chunks,
            'confidence': rag_response.confidence,
            'agent_response': rag_response.metadata.get('agent_response', {}),
        }

        # Stream the response content using LLM provider
        try:
            # Note: reasoning parameter is Ollama-specific, pass it only if provider is Ollama
            stream_kwargs = {}
            if self.llm_provider.provider == 'ollama' and reasoning is not None:
                stream_kwargs['reasoning'] = reasoning

            for chunk in self.llm_provider.stream_chat(
                rag_response.content, **stream_kwargs
            ):
                yield chunk, metadata
        except Exception as e:
            # On error, yield a single chunk with error message
            error_content = f'Error generating response: {str(e)}'
            yield (
                LLMStreamChunk(
                    content=error_content,
                    done=True,
                ),
                metadata,
            )

    def get_graph_manager(self) -> Neo4jGraphManager:
        """Get the Neo4j graph manager instance.

        Returns:
            Neo4j graph manager instance
        """
        return self.graph_manager

    def close(self) -> None:
        """Close Neo4j driver connection."""
        if hasattr(self, 'neo4j_driver') and self.neo4j_driver:
            self.neo4j_driver.close()
