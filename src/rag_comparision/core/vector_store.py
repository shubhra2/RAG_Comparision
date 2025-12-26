"""Vector store interface and implementations."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings

try:
    # Try new langchain-chroma package first (recommended)
    from langchain_chroma import Chroma
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    from langchain_core.vectorstores import VectorStore
except ImportError:
    try:
        # Fallback to deprecated langchain_community version
        from langchain_community.vectorstores import Chroma
        from langchain_core.documents import Document
        from langchain_core.embeddings import Embeddings
        from langchain_core.vectorstores import VectorStore
    except ImportError:
        Chroma = None
        Document = None  # type: ignore
        Embeddings = None  # type: ignore
        VectorStore = None


class VectorStoreInterface(ABC):
    """Abstract interface for vector stores."""

    @abstractmethod
    def add_documents(self, documents: list['Document']) -> list[str]:
        """Add documents to the vector store.

        Args:
            documents: List of documents to add

        Returns:
            List of document IDs
        """
        pass

    @abstractmethod
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list['Document']:
        """Search for similar documents.

        Args:
            query: Query string
            k: Number of results to return
            **kwargs: Additional search parameters

        Returns:
            List of similar documents
        """
        pass

    @abstractmethod
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[tuple['Document', float]]:
        """Search for similar documents with similarity scores.

        Args:
            query: Query string
            k: Number of results to return
            **kwargs: Additional search parameters

        Returns:
            List of tuples (document, score)
        """
        pass

    @abstractmethod
    def as_retriever(self, **kwargs: Any) -> Any:
        """Get a retriever interface for this vector store.

        Args:
            **kwargs: Retriever configuration

        Returns:
            Retriever instance
        """
        pass

    @abstractmethod
    def delete(self, ids: list[str] | None = None) -> None:
        """Delete documents from the vector store.

        Args:
            ids: List of document IDs to delete. If None, deletes all.
        """
        pass

    @abstractmethod
    def persist(self) -> None:
        """Persist the vector store to disk."""
        pass


class ChromaVectorStore(VectorStoreInterface):
    """ChromaDB implementation of vector store."""

    def __init__(
        self,
        embeddings: 'Embeddings',
        persist_directory: str | Path | None = None,
        collection_name: str = 'rag_documents',
    ):
        """Initialize ChromaDB vector store.

        Args:
            embeddings: Embedding model to use
            persist_directory: Directory to persist the database.
                If None, uses in-memory storage.
            collection_name: Name of the ChromaDB collection

        Raises:
            ImportError: If chromadb or langchain-community is not installed
        """
        if Chroma is None:
            raise ImportError(
                'langchain-chroma or langchain-community and chromadb are required. '
                'Install with: pip install langchain-chroma chromadb'
            )

        self.embeddings = embeddings
        self.persist_directory = (
            str(persist_directory) if persist_directory else None
        )
        self.collection_name = collection_name
        self._vector_store: Chroma | None = None

    def _get_vector_store(self) -> Chroma:
        """Get or create the ChromaDB vector store instance.

        Returns:
            Chroma vector store instance
        """
        if self._vector_store is None:
            if self.persist_directory and Path(self.persist_directory).exists():
                # Load existing vector store
                self._vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings,
                    collection_name=self.collection_name,
                )
            else:
                # Create new vector store
                self._vector_store = Chroma(
                    embedding_function=self.embeddings,
                    collection_name=self.collection_name,
                    persist_directory=self.persist_directory,
                )
        return self._vector_store

    def add_documents(self, documents: list['Document']) -> list[str]:
        """Add documents to ChromaDB.

        Args:
            documents: List of documents to add

        Returns:
            List of document IDs

        Raises:
            ValueError: If documents list is empty or contains invalid documents
        """
        if not documents:
            raise ValueError('Cannot add empty list of documents')
        
        # Filter out documents with empty content
        valid_documents = [
            doc for doc in documents
            if doc.page_content and doc.page_content.strip()
        ]
        
        if not valid_documents:
            raise ValueError(
                'No valid documents to add. All documents have empty content.'
            )
        
        vector_store = self._get_vector_store()
        return vector_store.add_documents(valid_documents)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list['Document']:
        """Search for similar documents in ChromaDB.

        Args:
            query: Query string
            k: Number of results to return
            **kwargs: Additional search parameters

        Returns:
            List of similar documents
        """
        vector_store = self._get_vector_store()
        return vector_store.similarity_search(query, k=k, **kwargs)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Search for similar documents with scores.

        Args:
            query: Query string
            k: Number of results to return
            **kwargs: Additional search parameters

        Returns:
            List of tuples (document, score)
        """
        vector_store = self._get_vector_store()
        return vector_store.similarity_search_with_score(query, k=k, **kwargs)

    def as_retriever(self, **kwargs: Any) -> Any:
        """Get a retriever for ChromaDB.

        Args:
            **kwargs: Retriever configuration (search_kwargs, etc.)

        Returns:
            Retriever instance
        """
        vector_store = self._get_vector_store()
        return vector_store.as_retriever(**kwargs)

    def delete(self, ids: list[str] | None = None) -> None:
        """Delete documents from ChromaDB.

        Args:
            ids: List of document IDs to delete. If None, deletes all.
        """
        vector_store = self._get_vector_store()
        if ids is None:
            # Delete all documents by deleting the collection
            # Note: This is a limitation - Chroma doesn't have a direct
            # "delete all" method, so we'd need to get all IDs first
            # For now, we'll raise an error to prevent accidental deletion
            raise ValueError(
                'Deleting all documents requires explicit IDs. '
                'Use delete_collection() if you want to remove everything.'
            )
        vector_store.delete(ids=ids)

    def persist(self) -> None:
        """Persist ChromaDB to disk."""
        vector_store = self._get_vector_store()
        vector_store.persist()

    @classmethod
    def from_documents(
        cls,
        documents: list['Document'],
        embeddings: 'Embeddings',
        persist_directory: str | Path | None = None,
        collection_name: str = 'rag_documents',
    ) -> 'ChromaVectorStore':
        """Create a ChromaVectorStore from documents.

        Args:
            documents: List of documents to add
            embeddings: Embedding model to use
            persist_directory: Directory to persist the database
            collection_name: Name of the ChromaDB collection

        Returns:
            ChromaVectorStore instance with documents loaded
        """
        store = cls(
            embeddings=embeddings,
            persist_directory=persist_directory,
            collection_name=collection_name,
        )
        store.add_documents(documents)
        if persist_directory:
            store.persist()
        return store

