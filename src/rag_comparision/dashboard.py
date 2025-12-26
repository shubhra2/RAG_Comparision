"""Streamlit dashboard and chat playground for RAG Comparison Project."""

import streamlit as st

from rag_comparision.core import get_dataset_info
from rag_comparision.core.sidebar import render_sidebar

# Page configuration
st.set_page_config(
    page_title='RAG Comparison Dashboard',
    page_icon='🤖',
    layout='wide',
    initial_sidebar_state='expanded',
)

# Render shared sidebar
render_sidebar()

# Dashboard content (main page)
st.title('📊 RAG Comparison Dashboard')

# Dashboard metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric('Total Queries', '0', delta=None)

with col2:
    st.metric('Standard RAG Accuracy', '0%', delta=None)

with col3:
    st.metric('Graph RAG Accuracy', '0%', delta=None)

with col4:
    st.metric('Improvement', '0%', delta=None)

st.markdown('---')

# Charts section
st.subheader('Performance Comparison')

col1, col2 = st.columns(2)

with col1:
    st.write('**Accuracy Over Time**')
    st.line_chart(
        {
            'Standard RAG': [0.65, 0.68, 0.70],
            'Graph RAG': [0.72, 0.75, 0.78],
        }
    )

with col2:
    st.write('**Metric Comparison**')
    st.bar_chart(
        {
            'Faithfulness': [0.85, 0.92],
            'Answer Relevancy': [0.78, 0.88],
            'Context Precision': [0.82, 0.90],
        }
    )

st.markdown('---')

# Dataset information
st.subheader('Dataset Information')

dataset_info = get_dataset_info()

for dataset, info in dataset_info.items():
    with st.expander(f' {dataset}'):
        for key, value in info.items():
            st.write(f'**{key}:** {value}')
