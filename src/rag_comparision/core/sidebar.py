"""Shared sidebar component for Streamlit dashboard."""

from pathlib import Path

import streamlit as st

from rag_comparision.config import (
    AVAILABLE_MODELS,
    DEFAULT_K_RETRIEVAL,
    DEFAULT_MODEL,
    DEFAULT_PERSIST_DIRECTORY,
    OLLAMA_BASE_URL,
    RAG_TYPES,
)


def render_sidebar() -> None:
    """Render the shared sidebar with settings across all pages."""
    # Get static directory path
    BASE_DIR = Path(__file__).parent.parent.parent.parent
    STATIC_DIR = BASE_DIR / 'src' / 'rag_comparision' / 'static'

    with st.sidebar:
        st.title(' RAG Comparison')
        st.markdown('---')

        # Settings
        st.subheader('Settings')

        # Store settings in session state so pages can access them
        st.session_state.rag_type = st.selectbox(
            'RAG Type',
            RAG_TYPES,
            help='Select the RAG system to use',
            key='rag_type_selectbox',
        )

        st.session_state.model = st.selectbox(
            'Model',
            AVAILABLE_MODELS,
            help='Select the LLM model (used for both RAG and Ollama Test)',
            key='model_selectbox',
            index=AVAILABLE_MODELS.index(DEFAULT_MODEL)
            if DEFAULT_MODEL in AVAILABLE_MODELS
            else 0,
        )

        # RAG-specific settings
        st.session_state.k_retrieval = st.number_input(
            'Retrieval Count (k)',
            min_value=1,
            max_value=20,
            value=DEFAULT_K_RETRIEVAL,
            help='Number of documents to retrieve for RAG',
            key='k_retrieval_input',
        )

        st.session_state.persist_directory = st.text_input(
            'Persist Directory',
            value=DEFAULT_PERSIST_DIRECTORY,
            help='Directory to persist the vector store (leave empty for in-memory)',
            key='persist_directory_input',
        )

        # Ollama settings
        st.session_state.ollama_base_url = st.text_input(
            'Ollama Base URL',
            value=OLLAMA_BASE_URL,
            help='Ollama API endpoint URL',
            key='ollama_base_url_input',
        )

        st.session_state.enable_streaming = st.checkbox(
            'Enable Streaming',
            value=False,
            help='Stream responses token by token for better UX',
            key='enable_streaming_checkbox',
        )

        st.session_state.enable_reasoning = st.checkbox(
            'Enable Reasoning',
            value=False,
            help='Capture thinking/reasoning tokens (for supported models like qwen3:4b)',
            key='enable_reasoning_checkbox',
        )

        # Display logo/icon if available
        logo_path = STATIC_DIR / 'logo.png'
        if logo_path.exists():
            st.image(str(logo_path), use_container_width=True)

        st.markdown('---')
        st.markdown('**RAG Comparison Project**')
        st.caption('Evaluating Graph-Based RAG vs Standard RAG')
