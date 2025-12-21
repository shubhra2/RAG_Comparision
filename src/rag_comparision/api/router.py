from fastapi import APIRouter
from fastapi.openapi.docs import get_redoc_html
from fastapi.responses import HTMLResponse

router = APIRouter()


@router.get('/')
def read_root():
    """Default route for API router."""
    return {
        'message': 'RAG Comparison API Router is active. Docs: http://localhost:8000/docs'
    }


@router.get('/health')
async def health_check():
    """Health check endpoint."""
    return {'status': 'healthy', 'version': '0.1.0'}


@router.get('/redocs', response_class=HTMLResponse)
def custom_redoc():
    return get_redoc_html(
        openapi_url='/openapi.json',
        title='Custom API Docs',
        redoc_js_url='https://cdn.jsdelivr.net/npm/redoc@2/bundles/redoc.standalone.js',
        redoc_favicon_url='https://example.com/favicon.ico',
        with_google_fonts=True,
    )
