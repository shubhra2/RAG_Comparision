"""Dataset Preview page for RAG Comparison Project."""

import streamlit as st

try:
    import pandas as pd
except ImportError:
    pd = None

from rag_comparision.config import (
    DEFAULT_PERSIST_DIRECTORY,
    PROCESSED_DATA_PATH,
)
from rag_comparision.core.data_preprocessing import (
    preprocess_synthetic_articles_dataset,
)
from rag_comparision.core.embeddings import get_default_embedding_model
from rag_comparision.core.rag import GraphRAG, StandardRAG
from rag_comparision.core.sidebar import render_sidebar
from rag_comparision.core.vector_store import ChromaVectorStore

# Render shared sidebar
render_sidebar()

st.title(' Dataset Preview')

if pd is None:
    st.error(
        ' **pandas is required for dataset preview.**\n\n'
        'Install it with: `pip install pandas pyarrow`'
    )
    st.stop()


# Load dataset
@st.cache_data
def load_dataset():
    """Load the processed dataset."""
    try:
        # Try to load preprocessed data first
        if PROCESSED_DATA_PATH.exists():
            df = pd.read_parquet(PROCESSED_DATA_PATH)
        else:
            # If not exists, preprocess it
            st.info(' Preprocessed data not found. Processing dataset...')
            df = preprocess_synthetic_articles_dataset()
        return df
    except Exception as e:
        st.error(f' **Error loading dataset:**\n\n`{str(e)}`')
        return None


with st.spinner('Loading dataset...'):
    df = load_dataset()

if df is not None and not df.empty:
    # Display basic stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric('Total Articles', len(df))
    with col2:
        st.metric('Columns', len(df.columns))
    with col3:
        missing_count = df.isnull().sum().sum()
        st.metric('Missing Values', missing_count)

    st.markdown('---')

    # Display the table
    st.subheader('Dataset Table')
    st.dataframe(df, use_container_width=True, hide_index=False)

    # Show column info
    with st.expander(' Column Information'):
        st.write('**Columns:**', ', '.join(df.columns.tolist()))
        st.write('**Data Types:**')
        st.write(df.dtypes)

    st.markdown('---')

    # ChromaDB Status Section
    st.subheader(' ChromaDB Status')

    # Get persist directory from session state or use default
    persist_directory = st.session_state.get(
        'persist_directory', DEFAULT_PERSIST_DIRECTORY
    )

    # Check ChromaDB status
    try:
        embeddings = get_default_embedding_model()
        vector_store = ChromaVectorStore(
            embeddings=embeddings,
            persist_directory=persist_directory if persist_directory else None,
        )

        doc_count = vector_store.count()

        col1, col2 = st.columns([2, 1])
        with col1:
            if doc_count > 0:
                st.success(f' ChromaDB contains **{doc_count}** documents.')
            else:
                st.warning('️ ChromaDB is **empty**. Load data to enable RAG queries.')
        with col2:
            if doc_count == 0:
                if st.button(' Load Data into ChromaDB', type='primary'):
                    with st.spinner(
                        'Loading data into ChromaDB... This may take a few minutes.'
                    ):
                        try:
                            # Initialize RAG system and load data
                            rag = StandardRAG(
                                persist_directory=persist_directory
                                if persist_directory
                                else None,
                            )
                            rag.load_data(force_reload=False)
                            st.success(' Successfully loaded data into ChromaDB!')
                            st.rerun()
                        except Exception as e:
                            st.error(f' **Error loading data:**\n\n`{str(e)}`')
            else:
                if st.button(' Reload Data', type='secondary'):
                    with st.spinner(
                        'Reloading data into ChromaDB... This may take a few minutes.'
                    ):
                        try:
                            # Initialize RAG system and reload data
                            rag = StandardRAG(
                                persist_directory=persist_directory
                                if persist_directory
                                else None,
                            )
                            rag.load_data(force_reload=True)
                            st.success(' Successfully reloaded data into ChromaDB!')
                            st.rerun()
                        except Exception as e:
                            st.error(f' **Error reloading data:**\n\n`{str(e)}`')

        # Show ChromaDB path
        if persist_directory:
            st.caption(f'ChromaDB location: `{persist_directory}`')
        else:
            st.caption('ChromaDB: In-memory (not persisted)')

    except Exception as e:
        st.error(f' **Error checking ChromaDB status:**\n\n`{str(e)}`')
        st.info(
            'You may need to install required packages: `pip install chromadb langchain-chroma`'
        )

    st.markdown('---')

    # Neo4j Status Section
    st.subheader(' Neo4j Graph Status')

    # Check Neo4j status
    try:
        from rag_comparision.core.graph_loader import GraphDataLoader
        from rag_comparision.core.neo4j_graph import (
            Neo4jConnectionError,
            Neo4jGraphManager,
        )

        graph_manager = Neo4jGraphManager()

        # Check connection status
        try:
            graph = graph_manager.connect()
            connection_status = '✅ Connected'
            connection_color = 'green'
        except Neo4jConnectionError:
            connection_status = '❌ Not Connected'
            connection_color = 'red'
            graph = None

        # Display connection status
        st.markdown(
            f'**Connection Status:** <span style="color:{connection_color}">{connection_status}</span>',
            unsafe_allow_html=True,
        )

        if graph is not None:
            graph_loader = GraphDataLoader(graph_manager)

            try:
                # Try to connect and get node count
                node_count = graph_loader.get_node_count()
                rel_count = graph_loader.get_relationship_count()

                col1, col2 = st.columns([2, 1])
                with col1:
                    if node_count > 0:
                        st.success(
                            f' Neo4j graph contains **{node_count}** nodes and **{rel_count}** relationships.'
                        )
                    else:
                        st.warning(
                            '️ Neo4j graph is **empty**. Load data to enable Graph-Based RAG queries.'
                        )
                with col2:
                    if node_count == 0:
                        if st.button(
                            ' Load Data from CSV', type='primary', key='load_neo4j_csv'
                        ):
                            with st.spinner(
                                'Loading data from CSV into Neo4j... This may take a few minutes.'
                            ):
                                try:
                                    # Load data directly from CSV using Cypher query
                                    csv_url = (
                                        'https://raw.githubusercontent.com/dcarpintero/'
                                        'generative-ai-101/main/dataset/synthetic_articles.csv'
                                    )

                                    q_load_articles = f"""
                                    LOAD CSV WITH HEADERS
                                    FROM '{csv_url}'
                                    AS row
                                    FIELDTERMINATOR ';'
                                    MERGE (a:Article {{title:row.Title}})
                                    SET a.abstract = row.Abstract,
                                        a.publication_date = date(row.Publication_Date)
                                    FOREACH (researcher in split(row.Authors, ',') |
                                        MERGE (p:Researcher {{name:trim(researcher)}})
                                        MERGE (p)-[:PUBLISHED]->(a))
                                    FOREACH (topic in [row.Topic] |
                                        MERGE (t:Topic {{name:trim(topic)}})
                                        MERGE (a)-[:IN_TOPIC]->(t))
                                    """

                                    graph.query(q_load_articles)
                                    st.success(
                                        ' Successfully loaded data from CSV into Neo4j!'
                                    )
                                    st.rerun()
                                except Neo4jConnectionError as e:
                                    st.error(
                                        f' **Neo4j connection error:**\n\n`{str(e)}`\n\n'
                                        'Please ensure Neo4j is running. You can set connection details via '
                                        'environment variables: NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD'
                                    )
                                except Exception as e:
                                    st.error(f' **Error loading data:**\n\n`{str(e)}`')

                        if st.button(
                            ' Load Data via GraphRAG',
                            type='secondary',
                            key='load_neo4j_rag',
                        ):
                            with st.spinner(
                                'Loading data into Neo4j via GraphRAG... This may take a few minutes.'
                            ):
                                try:
                                    # Initialize GraphRAG and load data
                                    graph_rag = GraphRAG()
                                    graph_rag.load_data(force_reload=False)
                                    st.success(' Successfully loaded data into Neo4j!')
                                    st.rerun()
                                except Neo4jConnectionError as e:
                                    st.error(
                                        f' **Neo4j connection error:**\n\n`{str(e)}`\n\n'
                                        'Please ensure Neo4j is running. You can set connection details via '
                                        'environment variables: NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD'
                                    )
                                except Exception as e:
                                    st.error(f' **Error loading data:**\n\n`{str(e)}`')
                    else:
                        if st.button(
                            ' Reload Data from CSV',
                            type='secondary',
                            key='reload_neo4j_csv',
                        ):
                            with st.spinner(
                                'Reloading data from CSV into Neo4j... This may take a few minutes.'
                            ):
                                try:
                                    # Clear graph first
                                    graph_manager.clear_graph()

                                    # Load data directly from CSV using Cypher query
                                    csv_url = (
                                        'https://raw.githubusercontent.com/dcarpintero/'
                                        'generative-ai-101/main/dataset/synthetic_articles.csv'
                                    )

                                    q_load_articles = f"""
                                    LOAD CSV WITH HEADERS
                                    FROM '{csv_url}'
                                    AS row
                                    FIELDTERMINATOR ';'
                                    MERGE (a:Article {{title:row.Title}})
                                    SET a.abstract = row.Abstract,
                                        a.publication_date = date(row.Publication_Date)
                                    FOREACH (researcher in split(row.Authors, ',') |
                                        MERGE (p:Researcher {{name:trim(researcher)}})
                                        MERGE (p)-[:PUBLISHED]->(a))
                                    FOREACH (topic in [row.Topic] |
                                        MERGE (t:Topic {{name:trim(topic)}})
                                        MERGE (a)-[:IN_TOPIC]->(t))
                                    """

                                    graph.query(q_load_articles)
                                    st.success(
                                        ' Successfully reloaded data from CSV into Neo4j!'
                                    )
                                    st.rerun()
                                except Neo4jConnectionError as e:
                                    st.error(
                                        f' **Neo4j connection error:**\n\n`{str(e)}`\n\n'
                                        'Please ensure Neo4j is running.'
                                    )
                                except Exception as e:
                                    st.error(
                                        f' **Error reloading data:**\n\n`{str(e)}`'
                                    )

                        if st.button(
                            ' Reload Data via GraphRAG',
                            type='secondary',
                            key='reload_neo4j_rag',
                        ):
                            with st.spinner(
                                'Reloading data into Neo4j via GraphRAG... This may take a few minutes.'
                            ):
                                try:
                                    # Initialize GraphRAG and reload data
                                    graph_rag = GraphRAG()
                                    graph_rag.load_data(force_reload=True)
                                    st.success(
                                        ' Successfully reloaded data into Neo4j!'
                                    )
                                    st.rerun()
                                except Neo4jConnectionError as e:
                                    st.error(
                                        f' **Neo4j connection error:**\n\n`{str(e)}`\n\n'
                                        'Please ensure Neo4j is running.'
                                    )
                                except Exception as e:
                                    st.error(
                                        f' **Error reloading data:**\n\n`{str(e)}`'
                                    )

                # Show Neo4j connection info
                st.caption(f'Neo4j URI: `{graph_manager.uri}`')

            except Neo4jConnectionError as e:
                st.error(
                    f' **Neo4j connection failed:**\n\n`{str(e)}`\n\n'
                    'Please ensure Neo4j is running and accessible. '
                    'You can set connection details via environment variables: '
                    'NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD'
                )
                st.info(
                    'To start Neo4j locally, you can use Docker: '
                    '`docker run -p 7474:7474 -p 7687:7687 neo4j:latest`'
                )
        else:
            st.info(
                'To start Neo4j locally, you can use Docker: '
                '`docker run -p 7474:7474 -p 7687:7687 neo4j:latest`'
            )

    except ImportError as e:
        st.warning(
            f'️ **Neo4j support not available:**\n\n`{str(e)}`\n\n'
            'Install required packages: `pip install langchain-community neo4j`'
        )
    except Exception as e:
        st.error(f' **Error checking Neo4j status:**\n\n`{str(e)}`')

else:
    st.warning('️ No data available to display.')
