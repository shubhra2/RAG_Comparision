"""Example usage of Standard RAG with ChromaDB."""

from pathlib import Path

from rag_comparision.core.rag import StandardRAG


def main():
    """Example of using Standard RAG system."""
    # Initialize the RAG system
    rag = StandardRAG(
        model="qwen3:0.6b",  # or 'qwen3:0.6b' or any Ollama model
        persist_directory="./chroma_db",  # Optional: persist to disk
        k_retrieval=4,  # Number of documents to retrieve
    )

    # Load the synthetic articles dataset
    print("Loading dataset...")
    rag.load_data()  # Uses default synthetic articles dataset
    print("Dataset loaded and indexed!")

    # Query the RAG system
    queries = [
        "What are the main topics covered in the articles?",
        "Who are the authors mentioned?",
        "What is machine learning?",
    ]

    for query in queries:
        print(f'\n{"="*60}')
        print(f"Query: {query}")
        print("-" * 60)

        response = rag.query(query)

        print(f"Answer: {response.content}")
        print(f"\nRetrieved {response.retrieved_chunks} documents")
        print(f"Confidence: {response.confidence:.2f}")


if __name__ == "__main__":
    main()
