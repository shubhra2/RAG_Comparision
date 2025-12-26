from fastapi import APIRouter
from fastapi.openapi.docs import get_redoc_html
from fastapi.responses import HTMLResponse

from rag_comparision.config import APP_VERSION, PORT

router = APIRouter()


@router.get('/')
def read_root():
    """Default route for API router."""
    return {
        'message': f'RAG Comparison API Router is active. Docs: http://localhost:{PORT}/docs'
    }


@router.get('/health')
async def health_check():
    """Health check endpoint."""
    return {'status': 'healthy', 'version': APP_VERSION}


@router.get('/redocs', response_class=HTMLResponse)
def custom_redoc():
    return get_redoc_html(
        openapi_url='/openapi.json',
        title='Custom API Docs',
        redoc_js_url='https://cdn.jsdelivr.net/npm/redoc@2/bundles/redoc.standalone.js',
        redoc_favicon_url='https://example.com/favicon.ico',
        with_google_fonts=True,
    )
