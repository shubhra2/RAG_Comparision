"""Launcher script to run the Streamlit dashboard for RAG Comparison Project."""

import subprocess
import sys
from pathlib import Path


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
    """Main entry point to start the Streamlit dashboard."""
    print('=' * 60)
    print('Starting RAG Comparison Project Streamlit Dashboard')
    print('=' * 60)
    print('Streamlit dashboard: Starting...')
    print('-' * 60)

    run_streamlit()


if __name__ == '__main__':
    main()
