"""Launcher script to run both FastAPI and Streamlit applications together."""

import subprocess
import sys
import time
from pathlib import Path

from rag_comparision.config import settings


def run_fastapi():
    """Run FastAPI application in a subprocess."""
    cmd = [
        sys.executable,
        '-m',
        'uvicorn',
        'rag_comparision.main:app',
        '--host',
        settings.host,
        '--port',
        str(settings.port),
    ]

    if settings.debug:
        cmd.append('--reload')
        cmd.extend(['--log-level', 'info'])
    else:
        cmd.extend(['--log-level', 'warning'])

    # Start FastAPI in a subprocess
    subprocess.Popen(cmd)


def run_streamlit():
    """Run Streamlit application using subprocess."""
    # Get the path to the dashboard script
    dashboard_path = Path(__file__).parent / 'dashboard.py'
    app_path = str(dashboard_path)

    # Run Streamlit using subprocess (this will block)
    cmd = [
        sys.executable,
        '-m',
        'streamlit',
        'run',
        app_path,
        '--server.headless',
        'true',
    ]

    subprocess.run(cmd)


def main():
    """Main entry point to start both applications."""
    print('=' * 60)
    print('Starting RAG Comparison Project Services')
    print('=' * 60)
    print(f'FastAPI server: http://{settings.host}:{settings.port}')
    print('Streamlit dashboard: Starting...')
    print('-' * 60)

    # Start FastAPI in a background subprocess
    run_fastapi()

    # Give FastAPI a moment to start
    time.sleep(1)

    # Run Streamlit in the main thread (blocking)
    # This will keep the process alive
    run_streamlit()


if __name__ == '__main__':
    main()
