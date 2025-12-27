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

# Try to import GraphCypherQAChain (may be in different locations in different LangChain versions)
try:
    from langchain.chains import GraphCypherQAChain
except ImportError:
    try:
        from langchain.chains.graph_qa.cypher import GraphCypherQAChain
    except ImportError:
        GraphCypherQAChain = None  # type: ignore

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
    RAG_TYPE_GRAPH,
    RAG_TYPE_STANDARD,
)
from rag_comparision.core.data_loader import load_synthetic_articles_dataset
from rag_comparision.core.embeddings import get_default_embedding_model
from rag_comparision.core.graph_loader import GraphDataLoader
from rag_comparision.core.neo4j_graph import (
    Neo4jConnectionError,
    Neo4jGraphManager,
)
from rag_comparision.core.ollama import OllamaClient, OllamaStreamChunk
from rag_comparision.core.vector_store import ChromaVectorStore, VectorStoreInterface


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


class GraphRAG(RAGSystem):
    """Graph-Based RAG implementation using Neo4j and GraphCypherQAChain."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL_FALLBACK,
        graph_manager: Neo4jGraphManager | None = None,
        ollama_base_url: str = OLLAMA_BASE_URL,
    ):
        """Initialize Graph-Based RAG system.

        Args:
            model: Ollama model name to use
            graph_manager: Neo4jGraphManager instance. If None, creates a new one.
            ollama_base_url: Base URL for Ollama API

        Raises:
            ImportError: If required dependencies are not installed
            Neo4jConnectionError: If Neo4j connection fails
        """
        super().__init__(rag_type=RAG_TYPE_GRAPH, model=model)

        # Initialize Neo4j connection
        if graph_manager is None:
            self.graph_manager = Neo4jGraphManager()
        else:
            self.graph_manager = graph_manager

        # Initialize Ollama client
        self.ollama_client = OllamaClient(
            model=model,
            base_url=ollama_base_url,
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

            # Create Ollama LLM for Cypher generation and QA
            try:
                from langchain_ollama import ChatOllama
            except ImportError:
                # Fallback: try OllamaLLM if ChatOllama not available
                try:
                    from langchain_ollama import OllamaLLM as ChatOllama
                except ImportError:
                    ChatOllama = None

            if ChatOllama is None:
                self._cypher_chain = None
                return

            cypher_llm = ChatOllama(
                model=self.model,
                base_url=self.ollama_client.base_url,
            )
            qa_llm = ChatOllama(
                model=self.model,
                base_url=self.ollama_client.base_url,
            )

            # Create GraphCypherQAChain
            # verbose=True helps debug and shows Cypher queries being generated
            self._cypher_chain = GraphCypherQAChain.from_llm(
                cypher_llm=cypher_llm,
                qa_llm=qa_llm,
                graph=graph,
                verbose=True,  # Set to True to see Cypher queries and context in logs
            )
        except Exception:
            # If chain creation fails, set to None and use fallback
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
                # Invoke the chain - it should automatically pass context to QA LLM
                result = self._cypher_chain.invoke({'query': prompt})

                # Extract content and metadata
                # GraphCypherQAChain returns a dict with 'result' key containing the answer
                if isinstance(result, dict):
                    content = result.get('result', str(result))

                    # Try to extract intermediate steps if available
                    # The chain may store intermediate steps in different ways
                    intermediate_steps = result.get('intermediate_steps', [])
                    cypher_query = ''
                    graph_results = []

                    # Check if intermediate_steps exist and extract Cypher query and context
                    if intermediate_steps:
                        for step in intermediate_steps:
                            if isinstance(step, tuple) and len(step) >= 2:
                                # Step is (action, observation) tuple
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
                                        graph_results.extend(list(observation))
                                    elif isinstance(observation, str):
                                        graph_results.append(observation)
                                    else:
                                        graph_results.append(str(observation))
                            elif isinstance(step, dict):
                                # Step is a dict
                                cypher_query = step.get(
                                    'query', step.get('cypher', cypher_query)
                                )
                                context = step.get('context', step.get('result', []))
                                if context:
                                    if isinstance(context, (list, tuple)):
                                        graph_results.extend(list(context))
                                    else:
                                        graph_results.append(str(context))

                    # If no intermediate_steps, try to get from result directly
                    if not cypher_query and not graph_results:
                        # Check if there's a 'query' or 'cypher' key in result
                        cypher_query = result.get('query', result.get('cypher', ''))
                        graph_results = result.get(
                            'context', result.get('graph_results', [])
                        )
                else:
                    # Result is a string
                    content = str(result)
                    cypher_query = ''
                    graph_results = []

                # Extract metadata
                metadata = {
                    'cypher_query': cypher_query,
                    'graph_results': graph_results,
                    'full_result': result
                    if isinstance(result, dict)
                    else {'result': result},
                }

                # Count retrieved chunks
                if isinstance(graph_results, list):
                    retrieved_count = len(graph_results)
                elif graph_results:
                    retrieved_count = 1
                else:
                    retrieved_count = 0

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
    ) -> Generator[tuple[OllamaStreamChunk, dict], None, None]:
        """Query the Graph-Based RAG system with streaming response.

        Note: GraphCypherQAChain doesn't natively support streaming,
        so we'll execute the query and stream the final answer.

        Args:
            prompt: User query
            reasoning: Enable reasoning mode (not supported for graph queries)

        Yields:
            Tuples of (OllamaStreamChunk, metadata_dict)

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

        # Stream the response content using Ollama
        try:
            for chunk in self.ollama_client.stream_chat(
                rag_response.content, reasoning=reasoning
            ):
                yield chunk, metadata
        except Exception as e:
            # On error, yield a single chunk with error message
            error_content = f'Error generating response: {str(e)}'
            yield (
                OllamaStreamChunk(
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
