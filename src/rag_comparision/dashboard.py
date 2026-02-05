"""Streamlit dashboard and chat playground for RAG Comparison Project."""

import streamlit as st

from rag_comparision.core import get_dataset_info
from rag_comparision.core.evaluation import RAGEvaluator
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

# Load evaluation results
evaluation_results = RAGEvaluator.load_evaluation_results()
evaluations = evaluation_results.get('evaluations', [])

# Dashboard metrics
col1, col2, col3, col4, col5 = st.columns(5)

# Calculate metrics from latest evaluation
if evaluations:
    latest = evaluations[-1]
    total_queries = latest.get('total_queries', 0)
    standard_accuracy = latest.get('standard_rag', {}).get('accuracy', 0.0)
    graph_accuracy = latest.get('graph_rag', {}).get('accuracy', 0.0)
    graph_agentic_accuracy = latest.get('graph_rag_agentic', {}).get('accuracy', 0.0)

    # Calculate best improvement (max improvement over standard RAG)
    improvements = [
        graph_accuracy - standard_accuracy,
        graph_agentic_accuracy - standard_accuracy,
    ]
    best_improvement = max(improvements) if improvements else 0.0
else:
    total_queries = 0
    standard_accuracy = 0.0
    graph_accuracy = 0.0
    graph_agentic_accuracy = 0.0
    best_improvement = 0.0

with col1:
    st.metric('Total Queries', total_queries, delta=None)

with col2:
    st.metric(
        'Standard RAG Accuracy',
        f'{standard_accuracy * 100:.1f}%',
        delta=None,
    )

with col3:
    st.metric(
        'Graph RAG Accuracy',
        f'{graph_accuracy * 100:.1f}%',
        delta=None,
    )

with col4:
    st.metric(
        'Graph RAG (Agentic) Accuracy',
        f'{graph_agentic_accuracy * 100:.1f}%',
        delta=None,
    )

with col5:
    st.metric(
        'Best Improvement',
        f'{best_improvement * 100:.1f}%',
        delta=None,
    )

st.markdown('---')

# Charts section
st.subheader('Performance Comparison')

col1, col2 = st.columns(2)

with col1:
    st.write('**Accuracy Over Time**')
    if evaluations:
        # Prepare data for line chart
        accuracy_data = {
            'Standard RAG': [],
            'Graph RAG': [],
            'Graph RAG (Agentic)': [],
        }

        for eval_entry in evaluations:
            accuracy_data['Standard RAG'].append(
                eval_entry.get('standard_rag', {}).get('accuracy', 0.0)
            )
            accuracy_data['Graph RAG'].append(
                eval_entry.get('graph_rag', {}).get('accuracy', 0.0)
            )
            if 'graph_rag_agentic' in eval_entry:
                accuracy_data['Graph RAG (Agentic)'].append(
                    eval_entry.get('graph_rag_agentic', {}).get('accuracy', 0.0)
                )
            else:
                # If agentic not in this evaluation, use graph_rag value
                accuracy_data['Graph RAG (Agentic)'].append(
                    eval_entry.get('graph_rag', {}).get('accuracy', 0.0)
                )

        st.line_chart(accuracy_data)
    else:
        st.info('No evaluation data available. Run evaluations to see charts.')

with col2:
    st.write('**Metric Comparison**')
    if evaluations:
        latest = evaluations[-1]
        # Get metrics from latest evaluation
        standard_metrics = latest.get('standard_rag', {}).get('metrics', {})
        graph_metrics = latest.get('graph_rag', {}).get('metrics', {})
        graph_agentic_metrics = latest.get('graph_rag_agentic', {}).get('metrics', {})

        # Prepare data for bar chart (stacked)
        # Use common metrics: faithfulness, response_relevancy, context_precision
        metric_names = ['Faithfulness', 'Answer Relevancy', 'Context Precision']
        metric_keys = ['faithfulness', 'response_relevancy', 'context_precision']

        bar_data = {}
        for display_name, key in zip(metric_names, metric_keys, strict=False):
            values = []
            if key in standard_metrics:
                values.append(standard_metrics[key])
            else:
                values.append(0.0)

            if key in graph_metrics:
                values.append(graph_metrics[key])
            else:
                values.append(0.0)

            if graph_agentic_metrics and key in graph_agentic_metrics:
                values.append(graph_agentic_metrics[key])
            elif key in graph_metrics:
                values.append(graph_metrics[key])
            else:
                values.append(0.0)

            bar_data[display_name] = values

        # Create DataFrame for better chart display
        import pandas as pd

        df = pd.DataFrame(
            bar_data, index=['Standard RAG', 'Graph RAG', 'Graph RAG (Agentic)']
        )
        st.bar_chart(df)
    else:
        st.info('No evaluation data available. Run evaluations to see charts.')

st.markdown('---')

# Dataset information
st.subheader('Dataset Information')

dataset_info = get_dataset_info()

for dataset, info in dataset_info.items():
    with st.expander(f' {dataset}'):
        for key, value in info.items():
            st.write(f'**{key}:** {value}')
