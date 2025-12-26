"""Chat Playground page for RAG Comparison Project."""

import streamlit as st

from rag_comparision.config import (
    DEFAULT_K_RETRIEVAL,
    DEFAULT_MODEL,
    DEFAULT_PERSIST_DIRECTORY,
    OLLAMA_BASE_URL,
    RAG_TYPE_GRAPH,
    RAG_TYPE_STANDARD,
)
from rag_comparision.core import StandardRAG
from rag_comparision.core.sidebar import render_sidebar


def display_raw_data(formatted_prompt: str | None = None):
    """Display the exact raw data sent to the model, word-for-word.

    Args:
        context: The exact context string sent to the model
        formatted_prompt: The exact formatted prompt sent to the LLM
    """
    st.markdown('###  Raw Data Sent to Model')

    if formatted_prompt:
        st.markdown('#### Full Prompt (exactly as sent to LLM):')
        st.code(formatted_prompt, language='text')


def display_retrieved_documents(documents: list[dict] | None = None):
    """Display retrieved documents with their ChromaDB scores.

    Args:
        documents: List of document dictionaries with 'content', 'metadata', and 'score' keys
    """
    if not documents:
        return

    st.markdown('###  Retrieved Documents')
    st.caption(
        'Documents retrieved from the vector store with similarity scores (lower score = more similar)'
    )

    for i, doc in enumerate(documents, 1):
        score = doc.get('score', None)
        content = doc.get('content', '')
        doc_metadata = doc.get('metadata', {})

        # Create expander title with score if available
        if score is not None:
            expander_title = f' Document {i} - Score: {score:.4f}'
        else:
            expander_title = f' Document {i}'

        with st.expander(expander_title):
            if score is not None:
                # Display score with color coding
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown('**Distance Score:**')
                with col2:
                    # Color code: lower is better (green for low, red for high)
                    if score < 0.5:
                        score_color = ''
                    elif score < 1.0:
                        score_color = ''
                    else:
                        score_color = ''
                    st.markdown(f'{score_color} **{score:.4f}**')
                    st.caption('Lower = more similar')

            # Display document content
            st.markdown('**Content:**')
            st.text(content)

            # Display metadata if available
            if doc_metadata:
                st.markdown('**Metadata:**')
                st.json(doc_metadata)


# Render shared sidebar
render_sidebar()

st.title(' Chat Playground')

# Get settings from session state (set in sidebar)
rag_type = st.session_state.get('rag_type', RAG_TYPE_STANDARD)
model = st.session_state.get('model', DEFAULT_MODEL)
k_retrieval = st.session_state.get('k_retrieval', DEFAULT_K_RETRIEVAL)
persist_directory = st.session_state.get('persist_directory', DEFAULT_PERSIST_DIRECTORY)
ollama_base_url = st.session_state.get('ollama_base_url', OLLAMA_BASE_URL)
enable_streaming = st.session_state.get('enable_streaming', True)
enable_reasoning = st.session_state.get('enable_reasoning', False)

st.info(
    f'Using **{rag_type}** with **{model}** model. '
    f'Retrieving **{k_retrieval}** documents per query. '
    'Ask questions to test the RAG system performance.'
)

# Initialize chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Check if Graph-Based RAG is selected (not yet implemented)
if rag_type == RAG_TYPE_GRAPH:
    st.warning(
        '️ **Graph-Based RAG is not yet implemented.** '
        'Please select "Standard RAG" to use the RAG system.'
    )
    st.stop()

# Initialize RAG system in session state (reuse if settings haven't changed)
rag_key = f'rag_{rag_type}_{model}_{k_retrieval}_{persist_directory}_{ollama_base_url}'
if rag_key not in st.session_state:
    with st.spinner('Initializing RAG system...'):
        try:
            st.session_state[rag_key] = StandardRAG(
                model=model,
                persist_directory=persist_directory if persist_directory else None,
                ollama_base_url=ollama_base_url,
                k_retrieval=k_retrieval,
            )
            # Load data on first initialization
            st.session_state[rag_key].load_data()
            st.success(' RAG system initialized and data loaded!')
        except Exception as e:
            st.error(f' **Error initializing RAG system:**\n\n`{str(e)}`')
            st.session_state[rag_key] = None
else:
    # Check if the cached instance has query_stream method (in case code was updated)
    rag_system = st.session_state.get(rag_key)
    if rag_system and not hasattr(rag_system, 'query_stream'):
        # Reinitialize if method is missing (code was updated)
        with st.spinner('Reinitializing RAG system (code updated)...'):
            try:
                st.session_state[rag_key] = StandardRAG(
                    model=model,
                    persist_directory=persist_directory if persist_directory else None,
                    ollama_base_url=ollama_base_url,
                    k_retrieval=k_retrieval,
                )
                # Load data on reinitialization
                st.session_state[rag_key].load_data()
                st.success(' RAG system reinitialized!')
            except Exception as e:
                st.error(f' **Error reinitializing RAG system:**\n\n`{str(e)}`')
                st.session_state[rag_key] = None

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        # Show reasoning first if available
        if message.get('reasoning'):
            with st.expander(' Reasoning Process', expanded=False):
                st.text(message['reasoning'])
        # Show content after reasoning
        st.markdown(message['content'])

        # Display raw data sent to model (for assistant messages)
        if message['role'] == 'assistant' and 'metadata' in message:
            metadata = message['metadata']
            if isinstance(metadata, dict):
                # Get context and formatted prompt from metadata
                context = None
                formatted_prompt = None

                if 'metadata' in metadata and isinstance(metadata['metadata'], dict):
                    context = metadata['metadata'].get('context')
                    formatted_prompt = metadata['metadata'].get('formatted_prompt')
                elif 'context' in metadata:
                    context = metadata.get('context')
                    formatted_prompt = metadata.get('formatted_prompt')

                if context or formatted_prompt:
                    with st.expander(' Raw Data Sent to Model', expanded=False):
                        display_raw_data(formatted_prompt=formatted_prompt)

        # Display metadata if available
        if 'metadata' in message:
            with st.expander(' Response Metadata', expanded=False):
                st.json(message['metadata'])

# Accept user input
if prompt := st.chat_input('Ask a question about the knowledge base...'):
    # Add user message to chat history
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    # Display user message
    with st.chat_message('user'):
        st.markdown(prompt)

    # Generate assistant response using RAG system
    with st.chat_message('assistant'):
        try:
            rag_system = st.session_state.get(rag_key)
            if rag_system is None:
                raise ValueError('RAG system not initialized. Please check settings.')

            # Check if streaming is supported
            has_streaming = hasattr(rag_system, 'query_stream')
            if enable_streaming and not has_streaming:
                st.warning(
                    '️ Streaming not available. Please restart Streamlit to load the latest code.'
                )
                enable_streaming = False

            if enable_streaming and has_streaming:
                # Streaming mode
                response_text = ''
                reasoning_text = ''
                response_metadata = {}

                # Create reasoning expander and placeholder if reasoning is enabled (show first)
                reasoning_display_placeholder = None
                if enable_reasoning:
                    reasoning_expander = st.expander(
                        ' Reasoning Process', expanded=True
                    )
                    reasoning_display_placeholder = reasoning_expander.empty()

                # Create placeholder for streaming content (show after reasoning)
                content_placeholder = st.empty()

                # Stream chunks
                for chunk, _ in rag_system.query_stream(
                    prompt,
                    reasoning=enable_reasoning if enable_reasoning else None,
                ):
                    # Handle reasoning content first
                    if (
                        chunk.reasoning_content
                        and reasoning_display_placeholder is not None
                    ):
                        new_reasoning = chunk.reasoning_content

                        # Check if reasoning_content is cumulative (contains previous content)
                        if reasoning_text and new_reasoning.startswith(reasoning_text):
                            # Cumulative: replace with new full content
                            reasoning_text = new_reasoning
                        elif reasoning_text and reasoning_text in new_reasoning:
                            # Contains previous but not at start - likely cumulative
                            reasoning_text = new_reasoning
                        else:
                            # Incremental: append
                            reasoning_text += new_reasoning

                        reasoning_display_placeholder.text(reasoning_text)

                    # Update response content display
                    response_text += chunk.content
                    content_placeholder.markdown(response_text)

                    # Capture final metadata from Ollama response
                    if chunk.done and chunk.response_metadata:
                        response_metadata = chunk.response_metadata

                # Use retrieval metadata (constant across chunks)
                retrieval_metadata = metadata
                retrieved_docs = retrieval_metadata.get('retrieved_documents', 0)
                confidence = retrieval_metadata.get('confidence', 0.0)

                # Display key metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric('Retrieved Documents', retrieved_docs)
                with col2:
                    st.metric('Confidence', f'{confidence:.2%}')

                # Display retrieved documents with scores
                documents = retrieval_metadata.get('documents', [])
                if documents:
                    st.markdown('---')
                    display_retrieved_documents(documents)

                # Display raw data sent to model
                context = retrieval_metadata.get('context')
                formatted_prompt = retrieval_metadata.get('formatted_prompt')
                if context or formatted_prompt:
                    st.markdown('---')
                    display_raw_data(formatted_prompt=formatted_prompt)

                # Build full metadata dict
                full_metadata = {
                    'content': response_text,
                    'rag_type': rag_type,
                    'model': model,
                    'retrieved_chunks': retrieved_docs,
                    'confidence': confidence,
                    'metadata': retrieval_metadata,
                }
                if reasoning_text:
                    full_metadata['reasoning_content'] = reasoning_text
                if response_metadata:
                    full_metadata['response_metadata'] = response_metadata

                with st.expander(' Response Metadata'):
                    st.json(full_metadata)

                # Add assistant response to chat history
                st.session_state.messages.append(
                    {
                        'role': 'assistant',
                        'content': response_text,
                        'metadata': full_metadata,
                        'reasoning': reasoning_text if reasoning_text else None,
                    }
                )
            else:
                # Non-streaming mode
                with st.spinner('Thinking...'):
                    rag_response = rag_system.query(prompt)

                    # Show reasoning first if available (though non-streaming doesn't support it)
                    st.markdown(rag_response.content)

                    # Display key metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric('Retrieved Documents', rag_response.retrieved_chunks)
                    with col2:
                        st.metric('Confidence', f'{rag_response.confidence:.2%}')

                    # Display retrieved documents with scores
                    documents = rag_response.metadata.get('documents', [])
                    if documents:
                        st.markdown('---')
                        display_retrieved_documents(documents)

                    # Display raw data sent to model
                    context = rag_response.metadata.get('context')
                    formatted_prompt = rag_response.metadata.get('formatted_prompt')
                    if context or formatted_prompt:
                        st.markdown('---')
                        display_raw_data(formatted_prompt=formatted_prompt)

                    metadata = rag_response.to_dict()

                    with st.expander(' Response Metadata'):
                        st.json(metadata)

                    # Add assistant response to chat history
                    st.session_state.messages.append(
                        {
                            'role': 'assistant',
                            'content': rag_response.content,
                            'metadata': metadata,
                        }
                    )
        except Exception as e:
            error_msg = f' **Error:** {str(e)}'
            st.error(error_msg)
            st.session_state.messages.append(
                {
                    'role': 'assistant',
                    'content': error_msg,
                }
            )

# Clear chat button
if st.session_state.messages:
    if st.button('️ Clear Chat', type='secondary'):
        st.session_state.messages = []
        st.rerun()
