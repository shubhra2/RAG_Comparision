"""Vector store interface and implementations."""

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rag_comparision.config import CHROMA_COLLECTION_NAME

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

    @abstractmethod
    def count(self) -> int:
        """Get the number of documents in the vector store.

        Returns:
            Number of documents in the vector store
        """
        pass


class ChromaVectorStore(VectorStoreInterface):
    """ChromaDB implementation of vector store."""

    def __init__(
        self,
        embeddings: 'Embeddings',
        persist_directory: str | Path | None = None,
        collection_name: str = CHROMA_COLLECTION_NAME,
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
        self.persist_directory = str(persist_directory) if persist_directory else None
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
        """Add documents to ChromaDB with deduplication.

        Generates deterministic IDs based on document content to prevent duplicates.
        Only adds documents that don't already exist in the collection.

        Args:
            documents: List of documents to add

        Returns:
            List of document IDs (existing or newly added)

        Raises:
            ValueError: If documents list is empty or contains invalid documents
        """
        if not documents:
            raise ValueError('Cannot add empty list of documents')

        # Filter out documents with empty content
        valid_documents = [
            doc for doc in documents if doc.page_content and doc.page_content.strip()
        ]

        if not valid_documents:
            raise ValueError(
                'No valid documents to add. All documents have empty content.'
            )

        # Generate deterministic IDs based on content hash
        # This ensures the same document always gets the same ID
        document_ids = []
        for doc in valid_documents:
            # Create a hash from content and metadata for deterministic ID
            content_str = doc.page_content
            # Include relevant metadata in hash to make ID unique per document
            metadata_str = str(sorted(doc.metadata.items())) if doc.metadata else ''
            hash_input = f'{content_str}|{metadata_str}'.encode()
            doc_id = hashlib.sha256(hash_input).hexdigest()[:32]  # Use first 32 chars
            document_ids.append(doc_id)

        vector_store = self._get_vector_store()

        # Check which documents already exist
        existing_ids = set()
        try:
            if hasattr(vector_store, '_collection'):
                collection = vector_store._collection
                if hasattr(collection, 'get'):
                    # Get existing documents with these IDs
                    result = collection.get(ids=document_ids)
                    if result and 'ids' in result:
                        existing_ids = set(result['ids'])
        except Exception:
            # If check fails, assume none exist and proceed
            pass

        # Filter out documents that already exist
        new_documents = []
        new_ids = []
        for doc, doc_id in zip(valid_documents, document_ids, strict=True):
            if doc_id not in existing_ids:
                new_documents.append(doc)
                new_ids.append(doc_id)

        # Only add new documents
        if new_documents:
            # Try to add documents with explicit IDs to prevent duplicates
            # LangChain's Chroma add_documents supports ids parameter
            try:
                # Method 1: Try using ids parameter (preferred)
                vector_store.add_documents(documents=new_documents, ids=new_ids)
            except TypeError:
                # Method 2: If ids parameter not supported, use underlying collection
                try:
                    if hasattr(vector_store, '_collection'):
                        collection = vector_store._collection
                        # Get embeddings for new documents
                        texts = [doc.page_content for doc in new_documents]
                        embeddings = self.embeddings.embed_documents(texts)
                        metadatas = [doc.metadata for doc in new_documents]
                        # Add directly to collection with IDs
                        collection.add(
                            ids=new_ids,
                            documents=texts,
                            embeddings=embeddings,
                            metadatas=metadatas,
                        )
                    else:
                        # Fallback: add without IDs (may create duplicates)
                        vector_store.add_documents(documents=new_documents)
                except Exception:
                    # Last resort: add without IDs
                    vector_store.add_documents(documents=new_documents)

        # Return all IDs (existing + newly added)
        return document_ids

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
        """Persist ChromaDB to disk.

        Note: ChromaDB automatically persists when persist_directory is provided
        during initialization. This method is a no-op for compatibility with
        the VectorStoreInterface.
        """
        # ChromaDB automatically persists when persist_directory is provided
        # No explicit persist() method is needed or available
        if self.persist_directory:
            # Data is already persisted automatically
            pass
        # If no persist_directory, data is in-memory and cannot be persisted

    def count(self) -> int:
        """Get the number of documents in ChromaDB.

        Returns:
            Number of documents in the collection
        """
        vector_store = self._get_vector_store()
        # Access the underlying ChromaDB collection to get count
        # The Chroma vector store has a _collection attribute
        try:
            if hasattr(vector_store, '_collection'):
                collection = vector_store._collection
                # Try count() method first (most efficient)
                if hasattr(collection, 'count'):
                    return collection.count()
                # Fallback: get all IDs and count them
                if hasattr(collection, 'get'):
                    result = collection.get()
                    if result and 'ids' in result:
                        return len(result['ids'])
        except Exception:
            # If accessing collection fails, try alternative approach
            pass

        # Last resort: try to get all documents via get() with limit
        # This is less efficient but should work
        try:
            if hasattr(vector_store, '_collection'):
                collection = vector_store._collection
                if hasattr(collection, 'get'):
                    # Get all documents (with a reasonable limit)
                    result = collection.get(limit=100000)  # Large limit to get all
                    if result and 'ids' in result:
                        return len(result['ids'])
        except Exception:
            pass

        # If all else fails, return 0 (assume empty or inaccessible)
        return 0

    @classmethod
    def from_documents(
        cls,
        documents: list['Document'],
        embeddings: 'Embeddings',
        persist_directory: str | Path | None = None,
        collection_name: str = CHROMA_COLLECTION_NAME,
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
