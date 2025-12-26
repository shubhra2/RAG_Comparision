"""Ollama Test page for RAG Comparison Project."""

import streamlit as st

from rag_comparision.config import DEFAULT_MODEL, OLLAMA_BASE_URL
from rag_comparision.core import OllamaClient
from rag_comparision.core.sidebar import render_sidebar

# Render shared sidebar
render_sidebar()

st.title(' Ollama Direct Test')

st.info(
    "This page directly calls Ollama to test if it's running. "
    'Make sure Ollama is running locally before using this feature.'
)

# Get settings from session state (set in sidebar)
model = st.session_state.get('model', DEFAULT_MODEL)
ollama_base_url = st.session_state.get('ollama_base_url', OLLAMA_BASE_URL)
enable_streaming = st.session_state.get('enable_streaming', True)
enable_reasoning = st.session_state.get('enable_reasoning', False)

# Check if Ollama is available
if not OllamaClient.is_available():
    st.error(
        ' `langchain-ollama` is not installed. '
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
    st.error(f' **Error initializing Ollama client:**\n\n`{str(e)}`')
    st.stop()

# Test connection button
col1, col2 = st.columns([1, 4])
with col1:
    test_connection = st.button(' Test Connection', type='primary')

if test_connection:
    try:
        with st.spinner(f'Testing connection to Ollama at {ollama_base_url}...'):
            test_result = ollama_client.test_connection()

            if test_result['status'] == 'success':
                st.success(' **Connection successful!**')

                # Display model information
                col1, col2 = st.columns(2)
                with col1:
                    st.metric('Available Models', test_result['model_count'])
                with col2:
                    model_status = (
                        ' Available' if test_result['model_available'] else ' Not Found'
                    )
                    st.metric('Configured Model', model_status)

                # Show available models
                if test_result['available_models']:
                    with st.expander(' Available Models', expanded=False):
                        for model_name in test_result['available_models']:
                            st.text(f'• {model_name}')
                else:
                    st.warning(
                        '️ No models found. Pull a model with: `ollama pull <model_name>`'
                    )

                # Show warning if configured model is not available
                if not test_result['model_available']:
                    st.warning(
                        f'️ **Model `{model}` not found in available models.**\n\n'
                        f'Pull it with: `ollama pull {model}`'
                    )
            else:
                st.error(
                    f' **Connection failed:** {test_result.get("error", "Unknown error")}'
                )

    except Exception as e:
        st.error(f' **Connection failed:**\n\n`{str(e)}`')
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
            with st.expander(' Reasoning Process', expanded=False):
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
                        ' Reasoning Process', expanded=True
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

                    # Capture final metadata
                    if chunk.done and chunk.response_metadata:
                        final_metadata = chunk.response_metadata

                # Show final model info
                with st.expander('️ Model Info'):
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
                        with st.expander(' Reasoning Process', expanded=True):
                            st.text(response.reasoning_content)

                    # Show response content after reasoning
                    st.markdown(response_text)

                    # Show model info
                    with st.expander('️ Model Info'):
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
            error_msg = f' **Error calling Ollama:**\n\n`{str(e)}`'
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
    if st.button('️ Clear Ollama Chat', type='secondary', key='clear_ollama'):
        st.session_state.ollama_messages = []
        st.rerun()
