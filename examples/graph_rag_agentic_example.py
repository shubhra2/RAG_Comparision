"""Example usage of Graph-Based RAG with Agentic Workflow (LangChain Agents).

This script demonstrates the GraphRAGAgentic class which uses LangChain agents
to maintain conversation context and iteratively query the Neo4j graph database.

Usage:
    # From project root directory
    python examples/graph_rag_agentic_example.py

    # Or if package is installed
    python -m examples.graph_rag_agentic_example

Prerequisites:
    1. Neo4j must be running (default: bolt://localhost:7687)
    2. Ollama must be running (default: http://localhost:11434)
    3. For Gemini tests: Set GOOGLE_API_KEY or GOOGLE_AI_API_KEY env var

The agentic workflow helps maintain context across multiple queries, making it
especially useful for multi-hop questions that require traversing relationships
in the knowledge graph.

Thinking Tokens (Reasoning Mode):
    For models that support thinking tokens (e.g., qwen3:4b), you can enable
    reasoning mode by setting reasoning=True when initializing GraphRAGAgentic.
    This captures the model's intermediate reasoning steps in the response.
    Set reasoning=False to disable, or reasoning=None to use model default.
"""

import logging
import os
import sys

from rag_comparision.core.rag import GraphRAGAgentic

# Configure logging to show intermediate steps
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
    datefmt='%H:%M:%S',
)

# Set logger for rag_comparision to INFO level to see agent steps
logging.getLogger('rag_comparision.core.rag').setLevel(logging.INFO)

# Suppress noisy logs from dependencies
logging.getLogger('httpx').setLevel(logging.WARNING)  # Suppress HTTP request logs
logging.getLogger('google_genai').setLevel(
    logging.WARNING
)  # Suppress Gemini retry logs
logging.getLogger('neo4j').setLevel(
    logging.WARNING
)  # Suppress Neo4j notification warnings


def test_ollama_provider(queries):
    """Test GraphRAGAgentic with Ollama provider."""
    print('\n' + '=' * 70)
    print('Testing GraphRAGAgentic with Ollama Provider')
    print('=' * 70)

    # Initialize the Graph RAG Agentic system with Ollama
    print('\nInitializing GraphRAGAgentic with Ollama...')
    rag = GraphRAGAgentic(
        model='gpt-oss:20b',  # or 'qwen3:4b' or any Ollama model
        llm_provider='ollama',
        verbose=True,  # Enable detailed logging of intermediate steps
        reasoning=None,  # Enable thinking tokens for models that support it (e.g., qwen3:4b)
    )

    # Load the synthetic articles dataset into Neo4j
    print('\nLoading dataset into Neo4j graph...')
    try:
        rag.load_data()  # Uses default synthetic articles dataset
        print('✓ Dataset loaded and indexed into Neo4j!')
    except Exception as e:
        print(f'✗ Error loading data: {e}')
        print('Make sure Neo4j is running and accessible.')
        return

    print('\n' + '-' * 70)
    print('Running Queries')
    print('-' * 70)

    for i, query in enumerate(queries, 1):
        print(f'\n[{i}/{len(queries)}] Query: {query}')
        print('-' * 70)

        try:
            response = rag.query(query)

            print(f'\nAnswer:\n{response.content}')
            print(f'\nModel: {response.model}')
            print(f'RAG Type: {response.rag_type}')
            print(f'Retrieved chunks: {response.retrieved_chunks}')
            print(f'Confidence: {response.confidence:.2f}')

            # Display agent-specific metadata
            if response.metadata:
                # Check for reasoning content in metadata (if thinking tokens were captured)
                if 'reasoning_content' in response.metadata:
                    print(
                        f'\nReasoning/Thinking:\n{response.metadata["reasoning_content"]}'
                    )
                if 'agent_response' in response.metadata:
                    agent_resp = response.metadata['agent_response']
                    if isinstance(agent_resp, dict) and 'messages' in agent_resp:
                        print(
                            f'\nAgent Messages: {len(agent_resp["messages"])} messages in conversation'
                        )
                        # Show tool calls if available
                        for msg in agent_resp['messages']:
                            if isinstance(msg, dict) and msg.get('role') == 'assistant':
                                if 'tool_calls' in msg or 'tool_calls' in str(msg):
                                    print('  Tool calls detected in agent workflow')

        except Exception as e:
            print(f'✗ Error querying graph: {e}')
            import traceback

            traceback.print_exc()

    # # Test streaming query
    # print('\n' + '-' * 70)
    # print('Testing Streaming Query')
    # print('-' * 70)

    # test_query = 'What are the main research topics and how many articles cover each topic?'
    # print(f'\nQuery: {test_query}')
    # print('Streaming response:')
    # print('-' * 70)

    # try:
    #     full_content = ''
    #     for chunk, metadata in rag.query_stream(test_query):
    #         if chunk.content:
    #             print(chunk.content, end='', flush=True)
    #             full_content += chunk.content
    #         if chunk.done:
    #             print('\n\n✓ Streaming completed')
    #             break
    # except Exception as e:
    #     print(f'\n✗ Error in streaming: {e}')

    # Cleanup
    print('\nClosing connections...')
    rag.close()
    print('✓ Done')


def test_gemini_provider(queries):
    """Test GraphRAGAgentic with Gemini provider."""
    print('\n' + '=' * 70)
    print('Testing GraphRAGAgentic with Gemini Provider')
    print('=' * 70)

    # Check for API key
    gemini_api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GOOGLE_AI_API_KEY')
    if not gemini_api_key:
        print('\n⚠ Skipping Gemini test: GOOGLE_API_KEY or GOOGLE_AI_API_KEY not set')
        print('Set the environment variable to test Gemini provider.')
        return

    # Initialize the Graph RAG Agentic system with Gemini
    print('\nInitializing GraphRAGAgentic with Gemini...')
    try:
        rag = GraphRAGAgentic(
            model='gemini-2.5-flash',  # or 'gemini-flash-latest'
            llm_provider='gemini',
            gemini_api_key=gemini_api_key,
            verbose=True,  # Enable detailed logging of intermediate steps
            reasoning=None,  # Reasoning not supported for Gemini (Ollama-specific)
        )
    except Exception as e:
        print(f'✗ Error initializing Gemini: {e}')
        return

    # Load the synthetic articles dataset into Neo4j
    print('\nLoading dataset into Neo4j graph...')
    try:
        rag.load_data()  # Uses default synthetic articles dataset
        print('✓ Dataset loaded and indexed into Neo4j!')
    except Exception as e:
        print(f'✗ Error loading data: {e}')
        print('Make sure Neo4j is running and accessible.')
        return

    print('\n' + '-' * 70)
    print('Running Queries')
    print('-' * 70)

    for i, query in enumerate(queries, 1):
        print(f'\n[{i}/{len(queries)}] Query: {query}')
        print('-' * 70)

        try:
            response = rag.query(query)

            print(f'\nAnswer:\n{response.content}')
            print(f'\nModel: {response.model}')
            print(f'RAG Type: {response.rag_type}')

        except Exception as e:
            print(f'✗ Error querying graph: {e}')
            import traceback

            traceback.print_exc()

    # Cleanup
    print('\nClosing connections...')
    rag.close()
    print('✓ Done')


def main():
    """Main function to run all tests."""
    print('=' * 70)
    print('GraphRAGAgentic Example Script')
    print('Testing Agentic Workflow with LangChain Agents')
    print('=' * 70)

    print('\nThis script demonstrates the GraphRAGAgentic class which uses')
    print('LangChain agents to maintain conversation context and iteratively')
    print('query the Neo4j graph database.')

    print('\nPrerequisites:')
    print('1. Neo4j must be running (default: bolt://localhost:7687)')
    print('2. Ollama must be running (default: http://localhost:11434)')
    print('3. For Gemini tests: Set GOOGLE_API_KEY or GOOGLE_AI_API_KEY env var')

    # Test queries - including multi-hop questions that benefit from agentic workflow
    # These queries are based on the actual dataset (AI/ML research articles)
    queries = [
        # Basic queries
        # 'What are the main topics covered in the articles?',
        # 'Who are the authors mentioned in the articles?',
        # # Multi-hop questions that benefit from agentic workflow
        # 'Which researchers have published articles on both model architectures and AI ethics?',
        # 'Find researchers who have collaborated on similar topics. Show me their connections.',
        # 'What research themes connect multiple articles together?',
        'Identify papers that share at least one author with the sparse attention paper and summarize their focus areas.',
        # 'Which authors have worked on both model optimization and ethical considerations?',
        # 'What are the different approaches to improving transformer architectures mentioned across the articles?',
    ]

    # Test with Ollama (default)
    try:
        test_ollama_provider(queries)
    except KeyboardInterrupt:
        print('\n\n⚠ Test interrupted by user')
    except Exception as e:
        print(f'\n✗ Error in Ollama test: {e}')
        import traceback

        traceback.print_exc()

    # # Test with Gemini (optional)
    # try:
    #     test_gemini_provider(queries)
    # except KeyboardInterrupt:
    #     print('\n\n⚠ Test interrupted by user')
    # except Exception as e:
    #     print(f'\n✗ Error in Gemini test: {e}')
    #     import traceback
    #     traceback.print_exc()

    print('\n' + '=' * 70)
    print('All tests completed!')
    print('=' * 70)


if __name__ == '__main__':
    main()
