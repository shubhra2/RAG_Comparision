"""Example script for evaluating Standard RAG and Graph RAG using RAGAS."""

import json

from rag_comparision.core.evaluation import RAGEvaluator
from rag_comparision.core.rag import GraphRAGAgentic, StandardRAG

# RAG systems use Ollama by default (for local inference)
rag_model = 'gpt-oss:20b'  # Ollama model
eval_model = 'granite4:7b-a1b-h'


def main():
    """Run evaluation on both RAG systems."""
    # Initialize RAG systems with Ollama (default)
    print('Initializing RAG systems...')
    standard_rag = StandardRAG(model=rag_model, llm_provider='ollama')
    graph_rag_agentic = GraphRAGAgentic(
        model=rag_model, llm_provider='ollama', reasoning='high'
    )
    # Load data into both systems
    print('\nLoading data into Standard RAG...')
    standard_rag.load_data()

    print('Loading data into Graph RAG Agentic...')
    graph_rag_agentic.load_data()

    # Prepare test queries with ground truth answers
    # These are based on the actual dataset (AI/ML research articles)
    test_queries = [
        {
            'query': 'What are the main topics covered in the articles?',
            'reference': 'The main topics are Foundations of Language Models and AI Ethics. Foundations of Language Models includes subtopics like Model Architectures, Model Optimization. AI Ethics includes subtopics like Social Impact and Safety.',
        },
        {
            'query': 'Which authors have published articles on both model architectures and AI ethics?',
            'reference': "Emily Chen has published on both: 'Transformer Architecture Innovations' (Model Architectures) and 'Ensuring Robustness in AI Systems' (AI Ethics - Safety). David Johnson has published on 'Transformer Architecture Innovations' (Model Architectures) and 'The Role of AI in Combating Climate Change' (AI Ethics - Social Impact). John Smith has published on 'Attention Mechanism Enhancements' (Model Architectures) and 'AI and Privacy' (AI Ethics - Social Impact).",
        },
        {
            'query': 'What techniques are mentioned for optimizing language models?',
            'reference': 'Several optimization techniques are discussed: compressing and optimizing LLMs for edge devices, efficient fine-tuning strategies for domain-specific models, sparse attention mechanisms for efficient NLP, quantum-inspired algorithms for training, federated learning for privacy-preserving training, and scaling laws in language model training.',
        },
        {
            'query': 'Which researchers have worked on both model optimization and ethical considerations?',
            'reference': "Emily Chen worked on 'Optimizing Large Language Models for Edge Devices' (Model Optimization) and 'Ensuring Robustness in AI Systems' (AI Ethics - Safety). Sarah Lee worked on 'Efficient Fine-tuning Strategies for Domain-Specific Language Models' (Model Optimization) and 'The Impact of AI on Employment' (AI Ethics - Social Impact). John Smith worked on 'Optimizing Large Language Models for Edge Devices' (Model Optimization) and 'AI and Privacy' (AI Ethics - Social Impact).",
        },
        {
            'query': 'What are the different approaches to improving transformer architectures?',
            'reference': 'The articles discuss several approaches: novel modifications to transformer architecture for processing long sequences, attention mechanism enhancements for improved language understanding and long-range dependencies, sparse attention for efficient NLP, dynamic neural network architectures for adaptive language processing, and multilingual pretraining for universal language understanding.',
        },
    ]

    # Initialize evaluator
    # RAGAS evaluator uses Gemini by default (for compatibility)
    # RAG systems use Ollama (already initialized above)
    print('\nInitializing RAGAS evaluator...')
    evaluator = RAGEvaluator(
        standard_rag=standard_rag,
        graph_rag=graph_rag_agentic,
        eval_llm_provider='ollama',
        eval_llm_model=eval_model,
        # RAG systems already initialized with Ollama above
        # RAGAS evaluator will use Gemini by default
    )

    # Run comparison (Standard RAG vs Graph RAG Agentic)
    print('\nRunning evaluation on both systems...')
    print('This may take a while as it evaluates each query...')
    comparison_results = evaluator.compare_rag_systems(test_queries)

    # Rename graph_rag to graph_rag_agentic in results for proper labeling
    if 'graph_rag' in comparison_results:
        comparison_results['graph_rag_agentic'] = comparison_results.pop('graph_rag')

    # Update summary to use graph_rag_agentic label
    if 'summary' in comparison_results:
        for _, comparison in comparison_results['summary'].items():
            # Rename graph_rag to graph_rag_agentic in summary
            if 'graph_rag' in comparison:
                comparison['graph_rag_agentic'] = comparison.pop('graph_rag')
            # Update winner label if needed
            if comparison.get('winner') == 'Graph RAG':
                comparison['winner'] = 'Graph RAG (Agentic)'

    # Save results to JSON
    print('\nSaving evaluation results to JSON...')
    results_path = evaluator.save_evaluation_results(comparison_results)
    print(f'Results saved to: {results_path}')

    # Display results
    print('\n' + '=' * 80)
    print('EVALUATION RESULTS')
    print('=' * 80)

    # Standard RAG results
    print('\n--- Standard RAG Metrics ---')
    standard_metrics = comparison_results['standard_rag']['metrics']
    for metric_name, score in standard_metrics.items():
        print(f'{metric_name}: {score:.4f}')

    # Graph RAG Agentic results
    print('\n--- Graph RAG (Agentic) Metrics ---')
    graph_agentic_metrics = comparison_results['graph_rag_agentic']['metrics']
    for metric_name, score in graph_agentic_metrics.items():
        print(f'{metric_name}: {score:.4f}')

    # Raw RAGAS results (unaggregated) from both systems
    print('\n--- Raw RAGAS Results (Standard RAG) ---')
    standard_raw = comparison_results['standard_rag'].get('raw_results')
    try:
        standard_df = standard_raw.to_pandas()  # type: ignore[assignment]
        print(json.dumps(standard_df.to_dict(), indent=4))
    except Exception:
        # Fallback to plain repr if DataFrame conversion is not available
        print(standard_raw)

    print('\n--- Raw RAGAS Results (Graph RAG Agentic) ---')
    graph_raw = comparison_results['graph_rag_agentic'].get('raw_results')
    try:
        graph_df = graph_raw.to_pandas()  # type: ignore[assignment]
        print(json.dumps(graph_df.to_dict(), indent=4))
    except Exception:
        print(graph_raw)

    # Comparison summary
    print('\n--- Comparison Summary ---')
    summary = comparison_results['summary']
    for metric_name, comparison in summary.items():
        print(f'\n{metric_name}:')
        print(f'  Standard RAG:        {comparison["standard_rag"]:.4f}')
        print(f'  Graph RAG (Agentic): {comparison["graph_rag_agentic"]:.4f}')
        print(f'  Difference:         {comparison["difference"]:+.4f}')
        print(f'  Winner:             {comparison["winner"]}')

    # Detailed results for each query
    print('\n--- Detailed Results (First 2 queries) ---')
    for i, (standard_data, graph_agentic_data) in enumerate(
        zip(
            comparison_results['standard_rag']['evaluation_data'][:2],
            comparison_results['graph_rag_agentic']['evaluation_data'][:2],
            strict=False,
        )
    ):
        print(f'\nQuery {i + 1}: {standard_data["user_input"]}')
        print(f'  Standard RAG Response:        {standard_data["response"][:100]}...')
        print(
            f'  Graph RAG (Agentic) Response: {graph_agentic_data["response"][:100]}...'
        )
        print(f'  Reference:                    {standard_data["reference"][:100]}...')


if __name__ == '__main__':
    main()
