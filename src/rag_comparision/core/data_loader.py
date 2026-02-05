"""Data loading and preprocessing module."""

from pathlib import Path
from typing import Any

try:
    import pandas as pd
except ImportError:
    pd = None

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_comparision.config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    PROCESSED_DATA_PATH,
)
from rag_comparision.core.data_preprocessing import (
    preprocess_synthetic_articles_dataset,
)


def load_synthetic_articles_dataset(
    source: str | Path | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    processed_data_path: str | Path | None = None,
    force_reprocess: bool = False,
) -> list[Document]:
    """Load and chunk the synthetic articles dataset.

    Uses preprocessed parquet data if available, otherwise loads and processes CSV.

    Args:
        source: Path to local CSV file or URL. If None, uses the default source
            from dataset info. Only used if processed data doesn't exist.
        chunk_size: Size of text chunks for splitting
        chunk_overlap: Overlap between chunks
        processed_data_path: Path to preprocessed parquet file. If None, uses
            default location from config
        force_reprocess: If True, reprocesses data even if parquet exists

    Returns:
        List of Document objects ready for embedding

    Raises:
        ImportError: If pandas is not installed
        ValueError: If dataset cannot be loaded
    """
    if pd is None:
        raise ImportError(
            'pandas is required for loading datasets. '
            'Install it with: pip install pandas pyarrow'
        )

    try:
        # Load preprocessed data (or process if needed)
        if processed_data_path is None:
            processed_data_path = PROCESSED_DATA_PATH
        df = preprocess_synthetic_articles_dataset(
            source=source,
            output_path=processed_data_path,
            force_reprocess=force_reprocess,
        )

        # Debug: Print column names to help diagnose issues
        print(f'DataFrame columns: {list(df.columns)}')
        print(f'DataFrame shape: {df.shape}')

        # Convert DataFrame to list of documents
        # Combine relevant fields into text content
        documents = []
        skipped_count = 0

        for _, row in df.iterrows():
            # Create a comprehensive text representation
            text_parts = []

            # Check columns exist in DataFrame, not in row (row is a Series)
            if 'Title' in df.columns:
                val = row.get('Title', '')
                if pd.notna(val) and str(val).strip():
                    text_parts.append(f'Title: {str(val).strip()}')

            if 'Abstract' in df.columns:
                val = row.get('Abstract', '')
                if pd.notna(val) and str(val).strip():
                    text_parts.append(f'Abstract: {str(val).strip()}')

            if 'Topic' in df.columns:
                val = row.get('Topic', '')
                if pd.notna(val) and str(val).strip():
                    text_parts.append(f'Topic: {str(val).strip()}')

            if 'Subtopic' in df.columns:
                val = row.get('Subtopic', '')
                if pd.notna(val) and str(val).strip():
                    text_parts.append(f'Subtopic: {str(val).strip()}')

            if 'Authors' in df.columns:
                val = row.get('Authors', '')
                if pd.notna(val) and str(val).strip():
                    text_parts.append(f'Authors: {str(val).strip()}')

            if 'Publication_Date' in df.columns:
                val = row.get('Publication_Date', '')
                if pd.notna(val):
                    pub_date = str(val).strip()
                    if (
                        pub_date and pub_date != 'NaT' and pub_date != 'nan'
                    ):  # Handle datetime NaT
                        text_parts.append(f'Publication Date: {pub_date}')

            text = '\n'.join(text_parts)

            # Skip documents with empty content
            if not text or not text.strip():
                skipped_count += 1
                continue

            # Create metadata from all columns
            metadata: dict[str, Any] = {}
            for col in df.columns:
                val = row[col]
                if pd.notna(val):
                    # Convert datetime to string for metadata
                    if pd.api.types.is_datetime64_any_dtype(type(val)):
                        metadata[col] = str(val)
                    else:
                        val_str = str(val).strip()
                        if val_str and val_str != 'nan':
                            metadata[col] = val_str

            documents.append(Document(page_content=text, metadata=metadata))

        if not documents:
            # Provide more detailed error message
            sample_row = df.iloc[0] if len(df) > 0 else None
            error_msg = (
                f'No valid documents created from dataset. '
                f'DataFrame has {len(df)} rows, {skipped_count} rows were skipped. '
                f'Columns found: {list(df.columns)}. '
            )
            if sample_row is not None:
                error_msg += f'Sample row values: {dict(sample_row)}'
            raise ValueError(error_msg)

        print(
            f'Created {len(documents)} documents from {len(df)} rows ({skipped_count} skipped)'
        )

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

        chunks = text_splitter.split_documents(documents)

        # Filter out empty chunks
        filtered_chunks = [
            chunk
            for chunk in chunks
            if chunk.page_content and chunk.page_content.strip()
        ]

        if not filtered_chunks:
            raise ValueError(
                'No valid documents found after processing. '
                'Check that the dataset contains non-empty text fields.'
            )

        print(
            f'Created {len(filtered_chunks)} document chunks from {len(documents)} documents'
        )
        return filtered_chunks

    except Exception as e:
        raise ValueError(f'Failed to load dataset: {str(e)}') from e
