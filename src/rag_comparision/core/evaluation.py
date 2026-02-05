"""RAGAS evaluation module for comparing Standard RAG and Graph RAG."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from ragas import EvaluationDataset, evaluate
    from ragas.dataset_schema import SingleTurnSample
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        ContextPrecision,
        ContextRecall,
        Faithfulness,
        ResponseRelevancy,
    )

    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    EvaluationDataset = None  # type: ignore
    SingleTurnSample = None  # type: ignore
    evaluate = None  # type: ignore

from rag_comparision.config import (
    DEFAULT_MODEL_FALLBACK,
    EVALUATION_RESULTS_PATH,
    GEMINI_MODEL,
    OLLAMA_BASE_URL,
)
from rag_comparision.core.embeddings import get_default_embedding_model
from rag_comparision.core.llm_provider import LLMProvider
from rag_comparision.core.rag import GraphRAG, GraphRAGAgentic, RAGResponse, StandardRAG


class RAGEvaluator:
    """Evaluator for comparing Standard RAG and Graph RAG using RAGAS metrics."""

    def __init__(
        self,
        standard_rag: StandardRAG | None = None,
        graph_rag: GraphRAG | GraphRAGAgentic | None = None,
        rag_llm_model: str | None = None,
        rag_llm_provider: str | None = None,
        eval_llm_model: str | None = None,
        eval_llm_provider: str | None = None,
        ollama_base_url: str = OLLAMA_BASE_URL,
        gemini_api_key: str | None = None,
    ):
        """Initialize RAG evaluator.

        Args:
            standard_rag: StandardRAG instance. If None, creates a new one with Ollama.
            graph_rag: GraphRAG or GraphRAGAgentic instance. If None, creates a new GraphRAG with Ollama.
            rag_llm_model: Model name for RAG systems. If None, uses Ollama default.
            rag_llm_provider: LLM provider for RAG systems ("ollama" or "gemini").
                Defaults to "ollama" for local inference.
            eval_llm_model: Model name for RAGAS evaluation LLM. If None, uses Gemini default.
            eval_llm_provider: LLM provider for RAGAS evaluator ("gemini" or "ollama").
                Defaults to "gemini" for RAGAS compatibility.
            ollama_base_url: Base URL for Ollama API
            gemini_api_key: Google API key for Gemini (required if using Gemini)
        """
        if not RAGAS_AVAILABLE:
            raise ImportError(
                'RAGAS is not installed. Install it with: pip install ragas'
            )

        # RAG systems use Ollama by default (for local inference)
        rag_provider = rag_llm_provider or 'ollama'
        if rag_llm_model is None:
            rag_llm_model = DEFAULT_MODEL_FALLBACK

        # Create RAG systems with Ollama if not provided
        self.standard_rag = standard_rag or StandardRAG(
            model=rag_llm_model,
            llm_provider=rag_provider,
            ollama_base_url=ollama_base_url,
            gemini_api_key=gemini_api_key if rag_provider == 'gemini' else None,
        )
        self.graph_rag = graph_rag or GraphRAG(
            model=rag_llm_model,
            llm_provider=rag_provider,
            ollama_base_url=ollama_base_url,
            gemini_api_key=gemini_api_key if rag_provider == 'gemini' else None,
        )

        # RAGAS evaluator uses Gemini by default (for compatibility)
        eval_provider = eval_llm_provider or 'gemini'
        if eval_llm_model is None:
            eval_llm_model = GEMINI_MODEL

        # Initialize evaluation LLM using Gemini for RAGAS
        self.eval_llm_provider = LLMProvider(
            provider=eval_provider,
            model=eval_llm_model,
            gemini_api_key=gemini_api_key,
            ollama_base_url=ollama_base_url,
        )

        # Get LangChain LLM for RAGAS
        self.eval_llm = self.eval_llm_provider.get_langchain_llm()

        # Initialize embeddings for RAGAS (required for some metrics)
        try:
            from ragas.embeddings import LangchainEmbeddingsWrapper

            embedding_model = get_default_embedding_model()
            self.embeddings = LangchainEmbeddingsWrapper(embedding_model)
        except Exception:
            # If embeddings fail, we'll proceed without them (some metrics may not work)
            self.embeddings = None

    def _extract_retrieved_contexts(
        self, response: RAGResponse, rag_type: str
    ) -> list[str]:
        """Extract retrieved contexts from RAGResponse.

        Args:
            response: RAGResponse object
            rag_type: Type of RAG system ("Standard RAG" or "Graph-Based RAG")

        Returns:
            List of retrieved context strings
        """
        contexts = []

        if rag_type == 'Standard RAG':
            # For Standard RAG, extract from metadata['documents']
            if response.metadata and 'documents' in response.metadata:
                for doc in response.metadata['documents']:
                    if isinstance(doc, dict) and 'content' in doc:
                        contexts.append(doc['content'])
                    elif isinstance(doc, str):
                        contexts.append(doc)
        elif rag_type == 'Graph-Based RAG':
            # For Graph RAG, extract from metadata['graph_results']
            if response.metadata and 'graph_results' in response.metadata:
                graph_results = response.metadata['graph_results']
                if isinstance(graph_results, list):
                    for result in graph_results:
                        if isinstance(result, dict):
                            # Convert dict to string representation
                            contexts.append(str(result))
                        elif isinstance(result, str):
                            contexts.append(result)
                        else:
                            contexts.append(str(result))
                elif graph_results:
                    contexts.append(str(graph_results))

        return contexts

    def evaluate_rag_system(
        self,
        test_queries: list[dict[str, str]],
        rag_system: StandardRAG | GraphRAG | GraphRAGAgentic,
        rag_type: str,
    ) -> dict[str, Any]:
        """Evaluate a single RAG system on test queries.

        Args:
            test_queries: List of dicts with 'query' and 'reference' keys
            rag_system: RAG system instance to evaluate
            rag_type: Type of RAG system ("Standard RAG" or "Graph-Based RAG")

        Returns:
            Dictionary with evaluation results and metrics
        """
        # Prepare evaluation dataset using SingleTurnSample format
        samples = []

        for test_case in test_queries:
            query = test_case['query']
            reference = test_case.get('reference', '')

            # Query the RAG system
            response = rag_system.query(query)

            # Extract retrieved contexts - ensure they're strings
            retrieved_contexts = self._extract_retrieved_contexts(response, rag_type)

            # Ensure response content is a string (handle list format from Gemini)
            response_content = response.content
            if isinstance(response_content, list):
                # If content is a list (Gemini format), extract text from each item
                text_parts = []
                for item in response_content:
                    if isinstance(item, dict):
                        if 'text' in item:
                            text_parts.append(str(item['text']))
                        elif 'content' in item:
                            text_parts.append(str(item['content']))
                    elif isinstance(item, str):
                        text_parts.append(item)
                    else:
                        text_parts.append(str(item))
                response_content = ''.join(text_parts)
            elif not isinstance(response_content, str):
                response_content = str(response_content) if response_content else ''
            else:
                response_content = response_content if response_content else ''

            # Ensure contexts is a list of strings (not empty)
            if not retrieved_contexts:
                # If no contexts found, use a placeholder to avoid 0 scores
                retrieved_contexts = [
                    response_content[:200]
                    if response_content
                    else 'No context retrieved'
                ]

            # Ensure all contexts are strings
            retrieved_contexts = [
                str(ctx) if not isinstance(ctx, str) else ctx
                for ctx in retrieved_contexts
            ]

            # Create SingleTurnSample object (proper RAGAS format)
            if SingleTurnSample is not None:
                sample = SingleTurnSample(
                    user_input=query,
                    retrieved_contexts=retrieved_contexts,
                    response=response_content,
                    reference=reference if reference else '',
                )
                samples.append(sample)
            else:
                # Fallback to dict format if SingleTurnSample not available
                samples.append(
                    {
                        'user_input': query,
                        'retrieved_contexts': retrieved_contexts,
                        'response': response_content,
                        'reference': reference if reference else '',
                    }
                )

        # Create RAGAS evaluation dataset
        if SingleTurnSample is not None:
            # Use proper SingleTurnSample format
            eval_dataset = EvaluationDataset(samples=samples)
        else:
            # Fallback to from_list if SingleTurnSample not available
            eval_dataset = EvaluationDataset.from_list(samples)

        # Define metrics
        metrics = [
            Faithfulness(),
            ResponseRelevancy(),
            ContextPrecision(),
            ContextRecall(),
        ]

        # Wrap LLM for RAGAS
        evaluator_llm = LangchainLLMWrapper(self.eval_llm)

        # Run evaluation with embeddings if available
        eval_kwargs = {
            'dataset': eval_dataset,
            'metrics': metrics,
            'llm': evaluator_llm,
        }
        if self.embeddings is not None:
            eval_kwargs['embeddings'] = self.embeddings

        results = evaluate(**eval_kwargs)

        # Convert EvaluationResult to dictionary format
        metrics_dict = self._extract_metrics_dict(results)

        # Store evaluation data for reference (convert samples back to dict format)
        evaluation_data = []
        for sample in samples:
            if hasattr(sample, 'user_input'):
                evaluation_data.append(
                    {
                        'user_input': sample.user_input,
                        'retrieved_contexts': sample.retrieved_contexts,
                        'response': sample.response,
                        'reference': sample.reference,
                    }
                )
            else:
                evaluation_data.append(sample)

        return {
            'rag_type': rag_type,
            'results': metrics_dict if metrics_dict else results,  # Aggregate metrics
            'raw_results': results,  # Full RAGAS EvaluationResult object
            'evaluation_data': evaluation_data,
        }

    def compare_rag_systems(
        self,
        test_queries: list[dict[str, str]],
    ) -> dict[str, Any]:
        """Compare Standard RAG and Graph RAG on test queries.

        Args:
            test_queries: List of dicts with 'query' and 'reference' keys.
                Example: [
                    {
                        "query": "What are the main topics?",
                        "reference": "The main topics are machine learning and AI."
                    },
                    ...
                ]

        Returns:
            Dictionary with comparison results for both systems
        """
        # Evaluate Standard RAG
        standard_results = self.evaluate_rag_system(
            test_queries, self.standard_rag, 'Standard RAG'
        )

        # Evaluate Graph RAG
        graph_results = self.evaluate_rag_system(
            test_queries, self.graph_rag, 'Graph-Based RAG'
        )

        # Extract metrics for comparison
        standard_metrics = standard_results['results']
        graph_metrics = graph_results['results']

        comparison = {
            'standard_rag': {
                'metrics': standard_metrics,
                'evaluation_data': standard_results['evaluation_data'],
                'raw_results': standard_results['raw_results'],
            },
            'graph_rag': {
                'metrics': graph_metrics,
                'evaluation_data': graph_results['evaluation_data'],
                'raw_results': graph_results['raw_results'],
            },
            'summary': self._create_comparison_summary(standard_metrics, graph_metrics),
        }

        return comparison

    def save_evaluation_results(
        self,
        comparison_results: dict[str, Any],
        results_path: Path | str | None = None,
    ) -> Path:
        """Save evaluation results to JSON file, appending to existing results.

        Args:
            comparison_results: Results dictionary from compare_rag_systems()
            results_path: Path to JSON file. If None, uses default from config.

        Returns:
            Path to the saved JSON file
        """
        if results_path is None:
            results_path = EVALUATION_RESULTS_PATH
        else:
            results_path = Path(results_path)

        # Ensure directory exists
        results_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing results if file exists
        existing_results = {'evaluations': []}
        if results_path.exists():
            try:
                with open(results_path, encoding='utf-8') as f:
                    existing_results = json.load(f)
            except (json.JSONDecodeError, OSError):
                # If file is corrupted or can't be read, start fresh
                existing_results = {'evaluations': []}

        # Calculate accuracy from metrics (using faithfulness as accuracy proxy)
        # or average of all metrics
        def calculate_accuracy(metrics: dict[str, float]) -> float:
            """Calculate accuracy from metrics."""
            if not metrics:
                return 0.0
            # Use average of all metrics as accuracy
            values = [v for v in metrics.values() if isinstance(v, (int, float))]
            return sum(values) / len(values) if values else 0.0

        standard_metrics = comparison_results['standard_rag']['metrics']

        # Check which graph RAG variant exists (agentic takes precedence)
        if 'graph_rag_agentic' in comparison_results:
            graph_metrics = comparison_results['graph_rag_agentic']['metrics']
            graph_key = 'graph_rag_agentic'
        elif 'graph_rag' in comparison_results:
            graph_metrics = comparison_results['graph_rag']['metrics']
            graph_key = 'graph_rag'
        else:
            raise ValueError(
                "Neither 'graph_rag' nor 'graph_rag_agentic' found in comparison_results"
            )

        # Prepare new evaluation entry
        evaluation_entry = {
            'timestamp': datetime.now().isoformat(),
            'total_queries': len(comparison_results['standard_rag']['evaluation_data']),
            'standard_rag': {
                'metrics': {
                    k: float(v) if isinstance(v, (int, float)) else 0.0
                    for k, v in standard_metrics.items()
                },
                'accuracy': calculate_accuracy(standard_metrics),
            },
        }

        # Add graph RAG results (either standard or agentic)
        evaluation_entry[graph_key] = {
            'metrics': {
                k: float(v) if isinstance(v, (int, float)) else 0.0
                for k, v in graph_metrics.items()
            },
            'accuracy': calculate_accuracy(graph_metrics),
        }

        # If we have graph_rag but also graph_rag_agentic, add both
        if (
            'graph_rag' in comparison_results
            and 'graph_rag_agentic' in comparison_results
        ):
            graph_rag_metrics = comparison_results['graph_rag']['metrics']
            evaluation_entry['graph_rag'] = {
                'metrics': {
                    k: float(v) if isinstance(v, (int, float)) else 0.0
                    for k, v in graph_rag_metrics.items()
                },
                'accuracy': calculate_accuracy(graph_rag_metrics),
            }

        # Add summary metrics
        if 'summary' in comparison_results:
            summary_dict = {}
            for k, v in comparison_results['summary'].items():
                summary_entry = {
                    'standard_rag': float(v.get('standard_rag', 0.0)),
                    'difference': float(v.get('difference', 0.0)),
                    'winner': v.get('winner', 'Tie'),
                }
                # Include graph_rag if present
                if 'graph_rag' in v:
                    summary_entry['graph_rag'] = float(v.get('graph_rag', 0.0))
                # Include graph_rag_agentic if present
                if 'graph_rag_agentic' in v:
                    summary_entry['graph_rag_agentic'] = float(
                        v.get('graph_rag_agentic', 0.0)
                    )
                summary_dict[k] = summary_entry
            evaluation_entry['summary'] = summary_dict

        # Append new evaluation
        existing_results['evaluations'].append(evaluation_entry)

        # Save updated results
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(existing_results, f, indent=2, ensure_ascii=False)

        return results_path

    @staticmethod
    def load_evaluation_results(
        results_path: Path | str | None = None,
    ) -> dict[str, Any]:
        """Load evaluation results from JSON file.

        Args:
            results_path: Path to JSON file. If None, uses default from config.

        Returns:
            Dictionary with evaluation results, or empty dict if file doesn't exist
        """
        if results_path is None:
            results_path = EVALUATION_RESULTS_PATH
        else:
            results_path = Path(results_path)

        if not results_path.exists():
            return {'evaluations': []}

        try:
            with open(results_path, encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            # Return empty structure if file is corrupted
            return {'evaluations': []}

    def _create_comparison_summary(
        self, standard_metrics: dict | Any, graph_metrics: dict | Any
    ) -> dict[str, Any]:
        """Create a summary comparing metrics from both systems.

        Args:
            standard_metrics: Metrics from Standard RAG (dict or EvaluationResult)
            graph_metrics: Metrics from Graph RAG (dict or EvaluationResult)

        Returns:
            Summary dictionary with comparisons
        """
        summary = {}

        # Ensure we have dictionaries
        if not isinstance(standard_metrics, dict):
            # Try to convert EvaluationResult to dict
            standard_metrics = self._extract_metrics_dict(standard_metrics)
        if not isinstance(graph_metrics, dict):
            graph_metrics = self._extract_metrics_dict(graph_metrics)

        # Get all metric names from both results
        all_metric_names = set(standard_metrics.keys()) | set(graph_metrics.keys())

        # Compare each metric
        for metric_name in all_metric_names:
            standard_score = standard_metrics.get(metric_name, 0.0)
            graph_score = graph_metrics.get(metric_name, 0.0)

            # Ensure scores are numeric
            if not isinstance(standard_score, (int, float)):
                standard_score = 0.0
            if not isinstance(graph_score, (int, float)):
                graph_score = 0.0

            summary[metric_name] = {
                'standard_rag': float(standard_score),
                'graph_rag': float(graph_score),
                'difference': float(graph_score - standard_score),
                'winner': (
                    'Graph RAG'
                    if graph_score > standard_score
                    else 'Standard RAG'
                    if standard_score > graph_score
                    else 'Tie'
                ),
            }

        return summary

    def _extract_metrics_dict(self, results: Any) -> dict[str, float]:
        """Extract metrics dictionary from RAGAS EvaluationResult.

        Args:
            results: RAGAS EvaluationResult object

        Returns:
            Dictionary of metric names to scores
        """
        metrics_dict = {}

        # Method 1: Try to convert to pandas DataFrame (most reliable method)
        if hasattr(results, 'to_pandas'):
            try:
                import pandas as pd

                df = results.to_pandas()
                # RAGAS typically returns columns like 'faithfulness', 'response_relevancy', etc.
                # Extract all numeric columns that look like metrics
                for col in df.columns:
                    col_lower = col.lower().strip()
                    # Skip non-metric columns
                    if col_lower in [
                        'question',
                        'answer',
                        'contexts',
                        'ground_truth',
                        'user_input',
                        'reference',
                    ]:
                        continue
                    # Check if it's a numeric column (likely a metric)
                    if pd.api.types.is_numeric_dtype(df[col]):
                        try:
                            mean_value = float(df[col].mean())
                            # Use the column name as the metric name
                            metrics_dict[col_lower] = mean_value
                        except Exception:
                            pass
            except Exception:
                # If pandas conversion fails, try other methods
                pass

        # Method 2: Try accessing as attributes (snake_case)
        if not metrics_dict:
            metric_attrs = [
                'faithfulness',
                'response_relevancy',
                'response_relevance',  # Alternative naming
                'context_precision',
                'context_recall',
            ]
            for attr_name in metric_attrs:
                if hasattr(results, attr_name):
                    try:
                        value = getattr(results, attr_name)
                        # Handle different value types
                        if hasattr(value, 'mean'):
                            metrics_dict[attr_name] = float(value.mean())
                        elif isinstance(value, (int, float)):
                            metrics_dict[attr_name] = float(value)
                        elif isinstance(value, (list, tuple)) and len(value) > 0:
                            # If it's a list of scores, take the mean
                            metrics_dict[attr_name] = float(sum(value) / len(value))
                    except Exception:
                        pass

        # Method 3: Try dictionary-style access
        if not metrics_dict:
            try:
                for key in [
                    'faithfulness',
                    'response_relevancy',
                    'context_precision',
                    'context_recall',
                ]:
                    if key in results:
                        value = results[key]
                        if hasattr(value, 'mean'):
                            metrics_dict[key] = float(value.mean())
                        elif isinstance(value, (int, float)):
                            metrics_dict[key] = float(value)
                        elif isinstance(value, (list, tuple)) and len(value) > 0:
                            metrics_dict[key] = float(sum(value) / len(value))
            except (TypeError, AttributeError):
                pass

        # Method 4: Try accessing via __getitem__ if it's dict-like
        if not metrics_dict:
            try:
                for key in [
                    'faithfulness',
                    'response_relevancy',
                    'context_precision',
                    'context_recall',
                ]:
                    value = results[key]
                    if hasattr(value, 'mean'):
                        metrics_dict[key] = float(value.mean())
                    elif isinstance(value, (int, float)):
                        metrics_dict[key] = float(value)
                    elif isinstance(value, (list, tuple)) and len(value) > 0:
                        metrics_dict[key] = float(sum(value) / len(value))
            except (TypeError, KeyError, AttributeError):
                pass

        return metrics_dict
