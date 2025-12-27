"""Graph data loading module for Neo4j."""

from typing import Any

try:
    import pandas as pd
except ImportError:
    pd = None

from rag_comparision.core.neo4j_graph import Neo4jGraphManager


class GraphDataLoader:
    """Load data into Neo4j graph database."""

    def __init__(self, graph_manager: Neo4jGraphManager):
        """Initialize graph data loader.

        Args:
            graph_manager: Neo4jGraphManager instance
        """
        self.graph_manager = graph_manager

    def load_from_dataframe(
        self, df: 'pd.DataFrame', force_reload: bool = False
    ) -> dict[str, Any]:
        """Load DataFrame into Neo4j graph.

        Creates nodes:
        - Researcher (from Authors field)
        - Article (from Title, Abstract, Publication_Date)
        - Topic (from Topic field)

        Creates relationships:
        - Researcher --[PUBLISHED]--> Article
        - Article --[IN_TOPIC]--> Topic

        Args:
            df: DataFrame with columns: Title, Abstract, Topic, Authors, Publication_Date
            force_reload: If True, clears graph before loading

        Returns:
            Dictionary with loading statistics

        Raises:
            ImportError: If pandas is not installed
            ValueError: If DataFrame is missing required columns
        """
        if pd is None:
            raise ImportError(
                'pandas is required for graph loading. Install with: pip install pandas'
            )

        # Validate required columns
        required_columns = ['Title', 'Abstract', 'Topic', 'Authors']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(
                f'DataFrame is missing required columns: {missing_columns}'
            )

        graph = self.graph_manager.get_graph()

        # Clear graph if force_reload
        if force_reload:
            self.graph_manager.clear_graph()

        # Statistics
        stats = {
            'articles_created': 0,
            'researchers_created': 0,
            'topics_created': 0,
            'published_relationships': 0,
            'in_topic_relationships': 0,
        }

        # Process each row
        for _, row in df.iterrows():
            # Skip rows with missing Title or Abstract
            title = row.get('Title', '')
            abstract = row.get('Abstract', '')
            if pd.isna(title) or pd.isna(abstract) or not str(title).strip():
                continue

            title = str(title).strip()
            abstract = str(abstract).strip() if not pd.isna(abstract) else ''

            # Create or merge Article node
            article_query = """
            MERGE (a:Article {title: $title})
            SET a.abstract = $abstract
            """
            params: dict[str, Any] = {'title': title, 'abstract': abstract}

            # Add publication date if available
            pub_date = row.get('Publication_Date', None)
            if pd.notna(pub_date):
                try:
                    # Try to format as date string
                    if hasattr(pub_date, 'strftime'):
                        params['publication_date'] = pub_date.strftime('%Y-%m-%d')
                        article_query += (
                            ', a.publication_date = date($publication_date)'
                        )
                    else:
                        params['publication_date'] = str(pub_date)
                        article_query += (
                            ', a.publication_date = date($publication_date)'
                        )
                except Exception:
                    # If date parsing fails, skip it
                    pass

            article_query += ' RETURN a'
            result = graph.query(article_query, params=params)
            if result:
                stats['articles_created'] += 1

            # Create Researcher nodes and PUBLISHED relationships
            authors_str = row.get('Authors', '')
            if pd.notna(authors_str) and str(authors_str).strip():
                authors = [
                    author.strip()
                    for author in str(authors_str).split(',')
                    if author.strip()
                ]

                for author in authors:
                    # Create or merge Researcher node
                    researcher_query = """
                    MERGE (r:Researcher {name: $name})
                    RETURN r
                    """
                    result = graph.query(researcher_query, params={'name': author})
                    if result:
                        stats['researchers_created'] += 1

                    # Create PUBLISHED relationship
                    published_query = """
                    MATCH (r:Researcher {name: $author_name})
                    MATCH (a:Article {title: $article_title})
                    MERGE (r)-[:PUBLISHED]->(a)
                    RETURN r, a
                    """
                    result = graph.query(
                        published_query,
                        params={'author_name': author, 'article_title': title},
                    )
                    if result:
                        stats['published_relationships'] += 1

            # Create Topic node and IN_TOPIC relationship
            topic = row.get('Topic', '')
            if pd.notna(topic) and str(topic).strip():
                topic = str(topic).strip()

                # Create or merge Topic node
                topic_query = """
                MERGE (t:Topic {name: $name})
                RETURN t
                """
                result = graph.query(topic_query, params={'name': topic})
                if result:
                    stats['topics_created'] += 1

                # Create IN_TOPIC relationship
                in_topic_query = """
                MATCH (a:Article {title: $article_title})
                MATCH (t:Topic {name: $topic_name})
                MERGE (a)-[:IN_TOPIC]->(t)
                RETURN a, t
                """
                result = graph.query(
                    in_topic_query,
                    params={'article_title': title, 'topic_name': topic},
                )
                if result:
                    stats['in_topic_relationships'] += 1

        # Refresh schema after loading
        self.graph_manager.refresh_schema()

        return stats

    def get_node_count(self, label: str | None = None) -> int:
        """Get count of nodes in the graph.

        Args:
            label: Node label to count. If None, counts all nodes.

        Returns:
            Number of nodes
        """
        graph = self.graph_manager.get_graph()

        if label:
            query = f'MATCH (n:{label}) RETURN count(n) as count'
        else:
            query = 'MATCH (n) RETURN count(n) as count'

        result = graph.query(query)
        if result and len(result) > 0:
            return result[0]['count']
        return 0

    def get_relationship_count(self, rel_type: str | None = None) -> int:
        """Get count of relationships in the graph.

        Args:
            rel_type: Relationship type to count. If None, counts all relationships.

        Returns:
            Number of relationships
        """
        graph = self.graph_manager.get_graph()

        if rel_type:
            query = f'MATCH ()-[r:{rel_type}]->() RETURN count(r) as count'
        else:
            query = 'MATCH ()-[r]->() RETURN count(r) as count'

        result = graph.query(query)
        if result and len(result) > 0:
            return result[0]['count']
        return 0
