"""Example usage of Graph-Based RAG with Neo4j."""

from rag_comparision.core.rag import GraphRAG


def main():
    """Example of using Graph-Based RAG system."""
    # Initialize the Graph RAG system
    rag = GraphRAG(
        model='qwen3:4b',  # or 'qwen3:0.6b' or any Ollama model
    )

    # Load the synthetic articles dataset into Neo4j
    print('Loading dataset into Neo4j graph...')
    try:
        rag.load_data()  # Uses default synthetic articles dataset
        print('Dataset loaded and indexed into Neo4j!')
    except Exception as e:
        print(f'Error loading data: {e}')
        print('Make sure Neo4j is running and accessible.')
        return

    # Query the Graph RAG system
    queries = [
        'What are the main topics covered in the articles?',
        'Who are the authors mentioned?',
        'What articles are related to machine learning?',
        'Which researchers have published articles on artificial intelligence?',
        'What is the relationship between articles and topics?',
    ]

    for query in queries:
        print(f'\n{"=" * 60}')
        print(f'Query: {query}')
        print('-' * 60)

        try:
            response = rag.query(query)

            print(f'Answer: {response.content}')
            print(f'\nRetrieved {response.retrieved_chunks} graph results')
            print(f'Confidence: {response.confidence:.2f}')

            # Display metadata if available
            if response.metadata:
                if (
                    'cypher_query' in response.metadata
                    and response.metadata['cypher_query']
                ):
                    print('\nCypher Query Generated:')
                    print(response.metadata['cypher_query'])

                if (
                    'graph_results' in response.metadata
                    and response.metadata['graph_results']
                ):
                    print(
                        f'\nGraph Results ({len(response.metadata["graph_results"])} items):'
                    )
                    for i, result in enumerate(
                        response.metadata['graph_results'][:3], 1
                    ):  # Show first 3
                        print(
                            f'  {i}. {str(result)[:100]}...'
                            if len(str(result)) > 100
                            else f'  {i}. {result}'
                        )

        except Exception as e:
            print(f'Error querying graph: {e}')
            print('Make sure Neo4j is running and data is loaded.')


if __name__ == '__main__':
    main()
