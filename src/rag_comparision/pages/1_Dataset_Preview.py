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
from rag_comparision.core.rag import StandardRAG
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
else:
    st.warning('️ No data available to display.')
