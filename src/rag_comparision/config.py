"""Application configuration."""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    # Application settings
    app_name: str = 'RAG Comparison Project'
    app_version: str = '0.1.0'
    debug: bool = False

    # LLM Provider Configuration
    # Note: RAG systems default to 'ollama' for local inference
    # RAGAS evaluator defaults to 'gemini' for compatibility
    # This setting is the general default, but can be overridden per component
    llm_provider: str = 'ollama'  # 'gemini' or 'ollama' (default for RAG systems)

    # Gemini Configuration
    gemini_api_key: str = ''  # Set via GOOGLE_API_KEY or GOOGLE_AI_API_KEY env var
    gemini_model: str = 'gemini-flash-lite-latest'  # Default Gemini model

    # Ollama Configuration
    ollama_base_url: str = 'http://localhost:11434'
    default_model: str = 'qwen3:0.6b'
    default_model_fallback: str = 'tinyllama'

    # RAG Configuration
    default_k_retrieval: int = 4
    default_chunk_size: int = 1000
    default_chunk_overlap: int = 200
    default_persist_directory: str = './chroma_db'
    chroma_collection_name: str = 'rag_documents'

    # Embedding Configuration
    default_embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2'
    embedding_device: str = 'cpu'
    embedding_normalize: bool = True

    # Data Configuration
    processed_data_path: str = 'data/processed/synthetic_articles.parquet'
    dataset_url: str = (
        'https://raw.githubusercontent.com/dcarpintero/ai-engineering/'
        'main/dataset/synthetic_articles.csv'
    )
    evaluation_results_path: str = 'data/results/evaluation_results.json'

    # RAG Types
    rag_type_standard: str = 'Standard RAG'
    rag_type_graph: str = 'Graph-Based RAG'
    rag_type_graph_agentic: str = 'Graph-Based RAG (Agentic)'

    # Confidence/Scoring
    default_confidence: float = 0.5
    confidence_min: float = 0.0
    confidence_max: float = 1.0

    # Timeout Configuration
    ollama_timeout: float = 10.0

    # Neo4j Configuration
    neo4j_uri: str = 'bolt://localhost:7687'
    neo4j_username: str = 'neo4j'
    neo4j_password: str = 'admin12345'
    neo4j_database: str = 'neo4j'

    class Config:
        """Pydantic config."""

        env_file = '.env'
        env_file_encoding = 'utf-8'


settings = Settings()

# Module-level constants for easy import (derived from settings)
# These can be used directly without instantiating Settings

# Application
APP_NAME = settings.app_name
APP_VERSION = settings.app_version
DEBUG = settings.debug

# LLM Provider Configuration
LLM_PROVIDER = settings.llm_provider

# Gemini Configuration
GEMINI_API_KEY = settings.gemini_api_key
GEMINI_MODEL = settings.gemini_model

# Ollama Configuration
OLLAMA_BASE_URL = settings.ollama_base_url
DEFAULT_MODEL = settings.default_model
DEFAULT_MODEL_FALLBACK = settings.default_model_fallback
# Available models list (not in settings, but useful constant)
AVAILABLE_MODELS = [
    'qwen3:0.6b',
    'qwen3:4b',
    'gemma3:1b',
    'phi4-mini',
    'granite4:3b',
    'granite4:7b-a1b-h',
    'olmo-3:7b-instruct',
    'gpt-oss:20b',
    'gemini-flash-lite-latest',
    'gemini-flash-latest',
]

# RAG Configuration
DEFAULT_K_RETRIEVAL = settings.default_k_retrieval
DEFAULT_CHUNK_SIZE = settings.default_chunk_size
DEFAULT_CHUNK_OVERLAP = settings.default_chunk_overlap
DEFAULT_PERSIST_DIRECTORY = settings.default_persist_directory
CHROMA_COLLECTION_NAME = settings.chroma_collection_name

# Embedding Configuration
DEFAULT_EMBEDDING_MODEL = settings.default_embedding_model
EMBEDDING_DEVICE = settings.embedding_device
EMBEDDING_NORMALIZE = settings.embedding_normalize

# Data Configuration
PROCESSED_DATA_PATH = Path(settings.processed_data_path)
DATASET_URL = settings.dataset_url
EVALUATION_RESULTS_PATH = Path(settings.evaluation_results_path)

# RAG Types
RAG_TYPE_STANDARD = settings.rag_type_standard
RAG_TYPE_GRAPH = settings.rag_type_graph
RAG_TYPE_GRAPH_AGENTIC = settings.rag_type_graph_agentic
RAG_TYPES = [RAG_TYPE_STANDARD, RAG_TYPE_GRAPH, RAG_TYPE_GRAPH_AGENTIC]

# Confidence/Scoring
DEFAULT_CONFIDENCE = settings.default_confidence
CONFIDENCE_MIN = settings.confidence_min
CONFIDENCE_MAX = settings.confidence_max

# Timeout Configuration
OLLAMA_TIMEOUT = settings.ollama_timeout

# Neo4j Configuration
NEO4J_URI = settings.neo4j_uri
NEO4J_USERNAME = settings.neo4j_username
NEO4J_PASSWORD = settings.neo4j_password
NEO4J_DATABASE = settings.neo4j_database

# Prompt Templates
RAG_PROMPT_TEMPLATE = (
    'Use the following pieces of context to answer the question. '
    "If you don't know the answer, just say that you don't know, "
    "don't try to make up an answer.\n\n"
    'Context:\n{context}\n\n'
    'Question: {question}\n\n'
    'Answer:'
)

RAG_PROMPT_SIMPLE = (
    'Context:\n{context}\n\nQuestion: {question}\n\nAnswer based on the context above:'
)

# Graph RAG Prompt Template - Strict Cypher and Evidence-Grounded Reasoning
GRAPH_RAG_PROMPT_TEMPLATE = (
    'You are a Graph-RAG answering system.\n'
    '\n'
    'You MUST follow these rules strictly and in order of priority:\n'
    '\n'
    '────────────────────────────────────────\n'
    'You are NOT allowed to:\n'
    '- evaluate, review, praise, or critique Cypher queries\n'
    '- explain why a query is good or bad\n'
    '- suggest query improvements\n'
    '- discuss edge cases unless directly required to answer the question\n'
    '\n'
    'Your ONLY task is to answer the user’s question using the query results.\n'
    '\n'
    '────────────────────────────────────────\n'
    '1. Evidence Priority (Highest Priority)\n'
    '────────────────────────────────────────\n'
    'If a Cypher query is executed and RETURNS RESULTS, you MUST treat the graph as non-empty.\n'
    'You are REQUIRED to use the returned results as valid evidence, even if:\n'
    '- the schema is incomplete,\n'
    '- expected node types are missing,\n'
    '- relationships differ from your assumed ontology.\n'
    '\n'
    'You MUST NOT claim that information is missing if results are present.\n'
    '\n'
    '────────────────────────────────────────\n'
    '2. Schema Flexibility\n'
    '────────────────────────────────────────\n'
    'Do NOT assume a fixed schema (e.g., Researcher → Article → Topic).\n'
    'Authors, documents, and entities may appear as:\n'
    '- node properties,\n'
    '- inferred document attributes,\n'
    '- indirectly connected nodes.\n'
    '\n'
    'Lack of explicit entity nodes does NOT imply lack of information.\n'
    '\n'
    '────────────────────────────────────────\n'
    '3. Grounded Answering\n'
    '────────────────────────────────────────\n'
    'You may ONLY use:\n'
    '- Cypher query results,\n'
    '- retrieved document chunks,\n'
    '- explicitly provided metadata.\n'
    '\n'
    'You MUST NOT use external knowledge or inference beyond this evidence.\n'
    '\n'
    'If information is partial, answer using available evidence and explicitly state limitations.\n'
    'Do NOT refuse if partial evidence exists.\n'
    '\n'
    '────────────────────────────────────────\n'
    '4. Refusal Condition (Lowest Priority)\n'
    '────────────────────────────────────────\n'
    'You MAY state that a question is unanswerable ONLY IF:\n'
    '- Cypher query returns ZERO results AND\n'
    '- retrieved document chunks contain NO relevant information.\n'
    '\n'
    'If either source contains evidence, you MUST answer.\n'
    '\n'
    '────────────────────────────────────────\n'
    '5. Conflict Resolution\n'
    '────────────────────────────────────────\n'
    'If your internal assumptions conflict with observed data:\n'
    '- TRUST the observed data.\n'
    '- Discard your assumptions.\n'
    '\n'
    '────────────────────────────────────────\n'
    '6. Answer Style\n'
    '────────────────────────────────────────\n'
    '- Be concise and factual.\n'
    '- Do NOT praise, validate, or comment on user analysis.\n'
    '- Do NOT mention instructions, policies, or internal reasoning.\n'
    '- Clearly separate facts from limitations.\n'
    '\n'
    '────────────────────────────────────────\n'
    '7. Output Format\n'
    '────────────────────────────────────────\n'
    'Respond with:\n'
    '- A direct answer based on evidence\n'
    '- A brief limitation note if needed\n'
    '\n'
    'Context (may include Cypher query results, document chunks, and metadata):\n{context}\n\n'
    'Question: {question}\n\n'
    'Answer:'
)

GRAPH_RAG_CYPHER_PROMPT_TEMPLATE = """Task: Generate a Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.

Schema:
{schema}

# Examples
# Count how many streamers are from Norway
MATCH (s:Stream)-[:HAS_LANGUAGE]->(:Language {{name: 'no'}})
RETURN count(s) AS streamers

# Recommend similar streamers if I like kimdoe
MATCH (s:Stream)
WHERE s.name = "kimdoe"
WITH collect(s) AS sourceNodes
CALL gds.pageRank.stream(
  "shared-audience",
  {{
    sourceNodes: sourceNodes,
    relationshipTypes: ['SHARED_AUDIENCE'],
    nodeLabels: ['Stream']
  }}
)
YIELD nodeId, score
WITH gds.util.asNode(nodeId) AS node, score
WHERE NOT node IN sourceNodes
RETURN node.name AS streamer, score
ORDER BY score DESC LIMIT 3

# Note: Do not include any explanations or apologies in your responses.
# Do not respond to anything other than a Cypher statement for the question below.
# Only output the generated Cypher statement — no extra text.

The question is:
{question}"""
