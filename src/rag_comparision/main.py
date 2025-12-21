"""FastAPI application entry point for API endpoints."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

# Get the base directory
BASE_DIR = Path(__file__).parent.parent.parent
STATIC_DIR = BASE_DIR / 'src' / 'rag_comparision' / 'static'

app = FastAPI(
    title='RAG Comparison Project API',
    description='Evaluating Graph-Based Retrieval-Augmented Generation: Comparative Analysis and Quality Improvements over Standard RAG Systems',
    version='0.1.0',
)

# Mount static files (for icons/images used by Streamlit)
if STATIC_DIR.exists():
    app.mount('/static', StaticFiles(directory=str(STATIC_DIR)), name='static')

# Include routers
try:
    from rag_comparision.api import router as api_router

    app.include_router(api_router)
except ImportError:
    # API router not yet implemented
    pass
