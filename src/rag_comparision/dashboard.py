"""Streamlit dashboard and chat playground for RAG Comparison Project."""

from pathlib import Path

import streamlit as st

from rag_comparision.core import OllamaClient, RAGSystem, get_dataset_info

# Page configuration
st.set_page_config(
    page_title='RAG Comparison Dashboard',
    page_icon='🤖',
    layout='wide',
    initial_sidebar_state='expanded',
)

# Get static directory path
BASE_DIR = Path(__file__).parent.parent.parent
STATIC_DIR = BASE_DIR / 'src' / 'rag_comparision' / 'static'


def load_image(image_path: Path) -> str | None:
    """Load image from static directory if it exists."""
    if image_path.exists():
        return str(image_path)
    return None


# Sidebar
with st.sidebar:
    st.title('🤖 RAG Comparison')
    st.markdown('---')

    # Navigation
    page = st.radio(
        'Navigation',
        ['Dashboard', 'Chat Playground', 'Ollama Test'],
        label_visibility='collapsed',
    )

    st.markdown('---')

    # Settings
    st.subheader('Settings')
    rag_type = st.selectbox(
        'RAG Type',
        ['Standard RAG', 'Graph-Based RAG'],
        help='Select the RAG system to use',
    )

    model = st.selectbox(
        'Model',
        ['qwen3:0.6b', 'qwen3:4b', 'gemma3:1b', 'phi4-mini', 'deepscaler'],
        help='Select the LLM model (used for both RAG and Ollama Test)',
    )

    # Ollama settings
    ollama_base_url = st.text_input(
        'Ollama Base URL',
        value='http://localhost:11434',
        help='Ollama API endpoint URL',
    )

    enable_streaming = st.checkbox(
        'Enable Streaming',
        value=True,
        help='Stream responses token by token for better UX',
    )

    enable_reasoning = st.checkbox(
        'Enable Reasoning',
        value=False,
        help='Capture thinking/reasoning tokens (for supported models like qwen3:4b)',
    )

    # Display logo/icon if available
    logo_path = STATIC_DIR / 'logo.png'
    if logo_path.exists():
        st.image(str(logo_path), use_container_width=True)

    st.markdown('---')
    st.markdown('**RAG Comparison Project**')
    st.caption('Evaluating Graph-Based RAG vs Standard RAG')


# Main content area
if page == 'Dashboard':
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
        with st.expander(f'📚 {dataset}'):
            for key, value in info.items():
                st.write(f'**{key}:** {value}')

elif page == 'Chat Playground':
    st.title('💬 Chat Playground')

    st.info(
        f'Using **{rag_type}** with **{model}** model. '
        'Ask questions to test the RAG system performance.'
    )

    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

            # Display metadata if available
            if 'metadata' in message:
                with st.expander('📋 Response Metadata'):
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
            with st.spinner('Thinking...'):
                try:
                    rag_system = RAGSystem(rag_type=rag_type, model=model)
                    rag_response = rag_system.query(prompt)

                    st.markdown(rag_response.content)

                    metadata = rag_response.to_dict()

                    with st.expander('📋 Response Metadata'):
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
                    error_msg = f'❌ **Error:** {str(e)}'
                    st.error(error_msg)
                    st.session_state.messages.append(
                        {
                            'role': 'assistant',
                            'content': error_msg,
                        }
                    )

    # Clear chat button
    if st.session_state.messages:
        if st.button('🗑️ Clear Chat', type='secondary'):
            st.session_state.messages = []
            st.rerun()

elif page == 'Ollama Test':
    st.title('🔧 Ollama Direct Test')

    st.info(
        "This page directly calls Ollama to test if it's running. "
        'Make sure Ollama is running locally before using this feature.'
    )

    # Check if Ollama is available
    if not OllamaClient.is_available():
        st.error(
            '❌ `langchain-ollama` is not installed. '
            'Please install it: `pip install langchain-ollama`'
        )
        st.stop()

    # Initialize Ollama client
    try:
        ollama_client = OllamaClient(
            model=model,
            base_url=ollama_base_url,
            reasoning=enable_reasoning if enable_reasoning else None,
        )
    except Exception as e:
        st.error(f'❌ **Error initializing Ollama client:**\n\n`{str(e)}`')
        st.stop()

    # Test connection button
    col1, col2 = st.columns([1, 4])
    with col1:
        test_connection = st.button('🔍 Test Connection', type='primary')

    if test_connection:
        try:
            with st.spinner(f'Testing connection to Ollama at {ollama_base_url}...'):
                test_result = ollama_client.test_connection()

                if test_result['status'] == 'success':
                    st.success('✅ **Connection successful!**')

                    # Display model information
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric('Available Models', test_result['model_count'])
                    with col2:
                        model_status = (
                            '✅ Available'
                            if test_result['model_available']
                            else '❌ Not Found'
                        )
                        st.metric('Configured Model', model_status)

                    # Show available models
                    if test_result['available_models']:
                        with st.expander('📋 Available Models', expanded=False):
                            for model_name in test_result['available_models']:
                                st.text(f'• {model_name}')
                    else:
                        st.warning(
                            '⚠️ No models found. Pull a model with: `ollama pull <model_name>`'
                        )

                    # Show warning if configured model is not available
                    if not test_result['model_available']:
                        st.warning(
                            f'⚠️ **Model `{model}` not found in available models.**\n\n'
                            f'Pull it with: `ollama pull {model}`'
                        )
                else:
                    st.error(
                        f'❌ **Connection failed:** {test_result.get("error", "Unknown error")}'
                    )

        except Exception as e:
            st.error(f'❌ **Connection failed:**\n\n`{str(e)}`')
            st.info(
                '**Troubleshooting:**\n'
                f'1. Make sure Ollama is running: `ollama serve`\n'
                '2. Check if the model is pulled: `ollama list`\n'
                f'3. Pull the model if needed: `ollama pull {model}`\n'
                '4. Verify the base URL is correct'
            )

    st.markdown('---')

    # Initialize chat history for Ollama
    if 'ollama_messages' not in st.session_state:
        st.session_state.ollama_messages = []

    # Display chat messages from history
    for message in st.session_state.ollama_messages:
        with st.chat_message(message['role']):
            # Show reasoning first if available
            if message.get('reasoning'):
                with st.expander('🧠 Reasoning Process', expanded=False):
                    st.text(message['reasoning'])
            # Show content after reasoning
            st.markdown(message['content'])

    # Accept user input
    if prompt := st.chat_input('Type a message to test Ollama...'):
        # Add user message to chat history
        st.session_state.ollama_messages.append({'role': 'user', 'content': prompt})

        # Display user message
        with st.chat_message('user'):
            st.markdown(prompt)

        # Generate assistant response using Ollama
        with st.chat_message('assistant'):
            try:
                if enable_streaming:
                    # Streaming mode
                    response_text = ''
                    reasoning_text = ''
                    final_metadata = {}

                    # Create reasoning expander and placeholder if reasoning is enabled (show first)
                    reasoning_display_placeholder = None
                    if enable_reasoning:
                        reasoning_expander = st.expander(
                            '🧠 Reasoning Process', expanded=True
                        )
                        reasoning_display_placeholder = reasoning_expander.empty()

                    # Create placeholder for streaming content (show after reasoning)
                    content_placeholder = st.empty()

                    # Stream chunks
                    for chunk in ollama_client.stream_chat(
                        prompt, reasoning=enable_reasoning if enable_reasoning else None
                    ):
                        # Handle reasoning content first
                        # reasoning_content may be cumulative (full text) or incremental
                        if (
                            chunk.reasoning_content
                            and reasoning_display_placeholder is not None
                        ):
                            new_reasoning = chunk.reasoning_content

                            # Check if reasoning_content is cumulative (contains previous content)
                            if reasoning_text and new_reasoning.startswith(
                                reasoning_text
                            ):
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

                        # Capture final metadata
                        if chunk.done and chunk.response_metadata:
                            final_metadata = chunk.response_metadata

                    # Show final model info
                    with st.expander('ℹ️ Model Info'):
                        info_dict = {
                            'content': response_text,
                            'model': model,
                            'base_url': ollama_base_url,
                        }
                        if reasoning_text:
                            info_dict['reasoning_content'] = reasoning_text
                        if final_metadata:
                            info_dict['response_metadata'] = final_metadata
                        st.json(info_dict)

                    # Add assistant response to chat history
                    st.session_state.ollama_messages.append(
                        {
                            'role': 'assistant',
                            'content': response_text,
                            'reasoning': reasoning_text if reasoning_text else None,
                        }
                    )
                else:
                    # Non-streaming mode
                    with st.spinner('Waiting for Ollama response...'):
                        response = ollama_client.chat(
                            prompt,
                            reasoning=enable_reasoning if enable_reasoning else None,
                        )
                        response_text = response.content

                        # Show reasoning first if available
                        if response.reasoning_content:
                            with st.expander('🧠 Reasoning Process', expanded=True):
                                st.text(response.reasoning_content)

                        # Show response content after reasoning
                        st.markdown(response_text)

                        # Show model info
                        with st.expander('ℹ️ Model Info'):
                            st.json(response.to_dict())

                    # Add assistant response to chat history
                    st.session_state.ollama_messages.append(
                        {
                            'role': 'assistant',
                            'content': response_text,
                            'reasoning': response.reasoning_content
                            if response.reasoning_content
                            else None,
                        }
                    )

            except Exception as e:
                error_msg = f'❌ **Error calling Ollama:**\n\n`{str(e)}`'
                st.error(error_msg)

                st.info(
                    '**Possible issues:**\n'
                    '- Ollama is not running\n'
                    f'- Model not found (try pulling it: `ollama pull {model}`)\n'
                    '- Incorrect base URL\n'
                    '- Network connectivity issues'
                )

                # Add error to chat history
                st.session_state.ollama_messages.append(
                    {
                        'role': 'assistant',
                        'content': error_msg,
                    }
                )

    # Clear chat button
    if st.session_state.ollama_messages:
        if st.button('🗑️ Clear Ollama Chat', type='secondary', key='clear_ollama'):
            st.session_state.ollama_messages = []
            st.rerun()
