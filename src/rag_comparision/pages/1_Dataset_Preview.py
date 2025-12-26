"""Dataset Preview page for RAG Comparison Project."""

import streamlit as st

try:
    import pandas as pd
except ImportError:
    pd = None

from rag_comparision.config import PROCESSED_DATA_PATH
from rag_comparision.core.data_preprocessing import (
    preprocess_synthetic_articles_dataset,
)
from rag_comparision.core.sidebar import render_sidebar

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
else:
    st.warning('️ No data available to display.')
