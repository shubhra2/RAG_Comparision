"""Neo4j graph database connection and management."""

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_neo4j import Neo4jGraph

try:
    from langchain_neo4j import Neo4jGraph
except ImportError:
    Neo4jGraph = None

from rag_comparision.config import (
    NEO4J_DATABASE,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USERNAME,
)


class Neo4jConnectionError(Exception):
    """Raised when Neo4j connection fails."""

    def __init__(self, message: str, uri: str | None = None):
        """Initialize the error.

        Args:
            message: Error message
            uri: Neo4j URI that failed to connect
        """
        self.message = message
        self.uri = uri
        super().__init__(self.message)


class Neo4jGraphManager:
    """Manager for Neo4j graph database connections."""

    def __init__(
        self,
        uri: str | None = None,
        username: str | None = None,
        password: str | None = None,
        database: str | None = None,
    ):
        """Initialize Neo4j connection manager.

        Args:
            uri: Neo4j URI. If None, tries environment variable NEO4J_URI,
                then defaults to config value.
            username: Neo4j username. If None, tries environment variable
                NEO4J_USERNAME, then defaults to config value.
            password: Neo4j password. If None, tries environment variable
                NEO4J_PASSWORD, then defaults to config value.
            database: Neo4j database name. If None, uses config value.

        Raises:
            ImportError: If langchain-community is not installed
            Neo4jConnectionError: If connection fails
        """
        if Neo4jGraph is None:
            raise ImportError(
                'langchain-community is required for Neo4j support. '
                'Install with: pip install langchain-community neo4j'
            )

        # Get connection parameters (env vars take precedence, then config)
        self.uri = uri or os.getenv('NEO4J_URI', NEO4J_URI)
        self.username = username or os.getenv('NEO4J_USERNAME', NEO4J_USERNAME)
        self.password = password or os.getenv('NEO4J_PASSWORD', NEO4J_PASSWORD)
        self.database = database or NEO4J_DATABASE

        self._graph: Neo4jGraph | None = None
        self._connected = False

    def connect(self) -> 'Neo4jGraph':
        """Connect to Neo4j database.

        Returns:
            Neo4jGraph instance

        Raises:
            Neo4jConnectionError: If connection fails
        """
        if self._graph is not None and self._connected:
            return self._graph

        try:
            self._graph = Neo4jGraph(
                url=self.uri,
                username=self.username,
                password=self.password,
                database=self.database,
            )
            # Test connection by refreshing schema
            self._graph.refresh_schema()
            self._connected = True
            return self._graph
        except Exception as e:
            self._connected = False
            error_msg = (
                f'Failed to connect to Neo4j at {self.uri}. '
                f'Error: {str(e)}\n\n'
                'Please ensure Neo4j is running and connection details are correct. '
                'You can set NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD environment variables.'
            )
            raise Neo4jConnectionError(error_msg, uri=self.uri) from e

    def get_graph(self) -> 'Neo4jGraph':
        """Get Neo4j graph instance, connecting if necessary.

        Returns:
            Neo4jGraph instance

        Raises:
            Neo4jConnectionError: If connection fails
        """
        if self._graph is None or not self._connected:
            return self.connect()
        return self._graph

    def is_connected(self) -> bool:
        """Check if connected to Neo4j.

        Returns:
            True if connected, False otherwise
        """
        if not self._connected or self._graph is None:
            return False

        try:
            # Try to refresh schema to verify connection
            self._graph.refresh_schema()
            return True
        except Exception:
            self._connected = False
            return False

    def refresh_schema(self) -> dict[str, Any]:
        """Refresh and get the graph schema.

        Returns:
            Dictionary containing schema information

        Raises:
            Neo4jConnectionError: If not connected or connection fails
        """
        graph = self.get_graph()
        graph.refresh_schema()
        return graph.get_schema

    def get_statistics(self) -> dict[str, Any]:
        """Get graph statistics (node counts, relationship counts).

        Returns:
            Dictionary with statistics

        Raises:
            Neo4jConnectionError: If not connected or connection fails
        """
        graph = self.get_graph()

        # Query for node counts by label
        node_query = """
        MATCH (n)
        RETURN labels(n) as labels, count(n) as count
        ORDER BY count DESC
        """
        node_results = graph.query(node_query)

        # Query for relationship counts by type
        rel_query = """
        MATCH ()-[r]->()
        RETURN type(r) as type, count(r) as count
        ORDER BY count DESC
        """
        rel_results = graph.query(rel_query)

        # Calculate total nodes and relationships
        total_nodes = sum(row['count'] for row in node_results)
        total_relationships = sum(row['count'] for row in rel_results)

        return {
            'total_nodes': total_nodes,
            'total_relationships': total_relationships,
            'nodes_by_label': {
                str(row['labels'][0]): row['count']
                for row in node_results
                if row['labels']
            },
            'relationships_by_type': {row['type']: row['count'] for row in rel_results},
        }

    def clear_graph(self) -> None:
        """Clear all nodes and relationships from the graph.

        WARNING: This will delete all data in the database!

        Raises:
            Neo4jConnectionError: If not connected or connection fails
        """
        graph = self.get_graph()
        graph.query('MATCH (n) DETACH DELETE n')

    def close(self) -> None:
        """Close the connection."""
        self._graph = None
        self._connected = False
