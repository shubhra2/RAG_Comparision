"""Data cleaning and preprocessing module."""

import io
from pathlib import Path
from typing import Any
from urllib.request import urlopen

try:
    import pandas as pd
except ImportError:
    pd = None

from rag_comparision.config import DATASET_URL, PROCESSED_DATA_PATH


def clean_text(text: str | None) -> str:
    """Clean and normalize text.

    Args:
        text: Text to clean

    Returns:
        Cleaned text
    """
    if text is None:
        return ''

    if pd is not None and pd.isna(text):
        return ''

    text = str(text).strip()
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text


def preprocess_synthetic_articles_dataset(
    source: str | Path | None = None,
    output_path: str | Path | None = None,
    force_reprocess: bool = False,
) -> pd.DataFrame:
    """Load, clean, and preprocess the synthetic articles dataset.

    Args:
        source: Path to local CSV file or URL. If None, uses the default source.
        output_path: Path to save processed parquet file. If None, saves to
            'data/processed/synthetic_articles.parquet'
        force_reprocess: If True, reprocesses even if parquet file exists

    Returns:
        Cleaned and preprocessed DataFrame

    Raises:
        ImportError: If pandas is not installed
        ValueError: If dataset cannot be loaded or processed
    """
    if pd is None:
        raise ImportError(
            'pandas is required for data preprocessing. '
            'Install it with: pip install pandas pyarrow'
        )

    # Default source URL from dataset info
    if source is None:
        source = DATASET_URL

    # Set default output path
    if output_path is None:
        output_path = PROCESSED_DATA_PATH
    else:
        output_path = Path(output_path)

    # Check if processed file exists and is newer than source
    if not force_reprocess and output_path.exists():
        try:
            print(f'Loading preprocessed data from {output_path}')
            df = pd.read_parquet(output_path)
            print(f'Loaded {len(df)} preprocessed records')
            return df
        except Exception as e:
            print(f'Warning: Could not load preprocessed file: {e}')
            print('Reprocessing from source...')

    # Load raw CSV data
    print(f'Loading raw data from {source}...')
    df = _load_csv_with_error_handling(source)
    print(f'Loaded {len(df)} raw records')

    # Clean and preprocess
    print('Cleaning and preprocessing data...')
    print(f'Columns before cleaning: {list(df.columns)}')
    df = _clean_dataframe(df)
    print(f'After cleaning: {len(df)} records')
    print(f'Columns after cleaning: {list(df.columns)}')

    # Debug: Show sample of cleaned data
    if len(df) > 0:
        print('\nSample of cleaned data (first row):')
        for col in df.columns:
            val = df.iloc[0][col]
            print(f'  {col}: {repr(val)} (type: {type(val).__name__})')

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as parquet
    print(f'Saving preprocessed data to {output_path}...')
    df.to_parquet(output_path, index=False, engine='pyarrow')
    print(f'Saved {len(df)} records to parquet')

    return df


def _load_csv_with_error_handling(source: str | Path) -> pd.DataFrame:
    """Load CSV file with error handling for malformed lines.

    Automatically detects separator (semicolon or comma) by checking the first line.

    Args:
        source: File path or URL

    Returns:
        DataFrame with loaded CSV data

    Raises:
        ValueError: If CSV cannot be loaded
    """

    def _detect_separator(content: str) -> str:
        """Detect CSV separator from content.

        Args:
            content: CSV content as string

        Returns:
            Detected separator (';' or ',')
        """
        first_line = content.split('\n')[0] if '\n' in content else content
        # Count semicolons and commas in header
        semicolon_count = first_line.count(';')
        comma_count = first_line.count(',')
        # Use semicolon if it appears more frequently, otherwise default to comma
        return ';' if semicolon_count > comma_count else ','

    def _read_csv_with_error_handling(
        file_or_buffer: Any, sep: str = ','
    ) -> pd.DataFrame:
        """Read CSV with error handling for malformed lines.

        Args:
            file_or_buffer: File path, URL response, or file-like object
            sep: CSV separator (default: ',')

        Returns:
            DataFrame with loaded CSV data
        """
        # Try with pandas >= 1.3.0 parameter first (on_bad_lines)
        try:
            return pd.read_csv(
                file_or_buffer, sep=sep, on_bad_lines='skip', encoding='utf-8'
            )
        except (TypeError, ValueError):
            # TypeError: parameter doesn't exist in this pandas version
            # ValueError: parameter value not accepted
            # Fallback for pandas < 1.3.0
            try:
                return pd.read_csv(
                    file_or_buffer,
                    sep=sep,
                    error_bad_lines=False,
                    warn_bad_lines=False,
                    encoding='utf-8',
                )
            except (TypeError, ValueError):
                # Last resort: try without error handling parameters
                return pd.read_csv(file_or_buffer, sep=sep, encoding='utf-8')

    try:
        # Get content to detect separator
        if isinstance(source, (str, Path)) and Path(source).exists():
            # Local file
            with open(source, encoding='utf-8') as f:
                content = f.read()
            sep = _detect_separator(content)
            return _read_csv_with_error_handling(io.StringIO(content), sep=sep)
        elif isinstance(source, str) and source.startswith('http'):
            # URL
            response = urlopen(source)
            content = response.read().decode('utf-8')
            sep = _detect_separator(content)
            return _read_csv_with_error_handling(io.StringIO(content), sep=sep)
        else:
            # Try as local path first, then URL
            try:
                with open(source, encoding='utf-8') as f:
                    content = f.read()
                sep = _detect_separator(content)
                return _read_csv_with_error_handling(io.StringIO(content), sep=sep)
            except Exception:
                response = urlopen(source)
                content = response.read().decode('utf-8')
                sep = _detect_separator(content)
                return _read_csv_with_error_handling(io.StringIO(content), sep=sep)
    except Exception as e:
        raise ValueError(f'Failed to load CSV from {source}: {str(e)}') from e


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess DataFrame.

    Args:
        df: Raw DataFrame

    Returns:
        Cleaned DataFrame
    """
    # Make a copy to avoid modifying original
    df = df.copy()

    # Standardize column names (handle case variations)
    df.columns = df.columns.str.strip()

    # Fix malformed parquet files where all columns were merged into one
    # Check if we have a single column with semicolon-separated column names
    if len(df.columns) == 1:
        col_name = df.columns[0]
        # Check if column name contains semicolons (indicating merged columns)
        if ';' in col_name:
            expected_columns = [
                'Title',
                'Abstract',
                'Topic',
                'Subtopic',
                'Authors',
                'Publication_Date',
            ]
            # Check if the column name matches expected pattern
            if all(col in col_name for col in expected_columns):
                # Split the single column into multiple columns
                print(f'Detected malformed parquet file with merged column: {col_name}')
                print('Splitting column into individual fields...')
                # Split each row value by semicolon
                split_data = df[col_name].str.split(
                    ';', expand=True, n=len(expected_columns) - 1
                )
                # Set proper column names
                split_data.columns = expected_columns
                df = split_data
                print(f'Successfully split into columns: {list(df.columns)}')

    # Clean text columns
    text_columns = ['Title', 'Abstract', 'Topic', 'Subtopic', 'Authors']
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)

    # Handle Publication_Date - convert to datetime if possible
    if 'Publication_Date' in df.columns:
        df['Publication_Date'] = pd.to_datetime(
            df['Publication_Date'],
            errors='coerce',
            format='mixed',
        )

    # Remove rows where all key text fields are empty
    key_fields = ['Title', 'Abstract']
    available_key_fields = [f for f in key_fields if f in df.columns]

    if available_key_fields:
        # Keep rows where at least one key field has content
        mask = df[available_key_fields].apply(
            lambda row: any(pd.notna(val) and str(val).strip() for val in row),
            axis=1,
        )
        df = df[mask].copy()

    # Reset index after filtering
    df = df.reset_index(drop=True)

    # Fill missing values in non-critical columns with empty string
    # (but keep NaN for critical fields to filter them out)
    optional_columns = ['Subtopic', 'Authors']
    for col in optional_columns:
        if col in df.columns:
            df[col] = df[col].fillna('')

    # Ensure all string columns are strings (not mixed types)
    # But preserve actual content - don't convert to empty strings
    for col in df.columns:
        if df[col].dtype == 'object':
            # Only convert if not already cleaned
            if col not in text_columns:
                df[col] = df[col].astype(str).replace('nan', '').replace('None', '')

    return df


def validate_preprocessed_data(df: pd.DataFrame) -> dict[str, Any]:
    """Validate preprocessed DataFrame.

    Args:
        df: Preprocessed DataFrame

    Returns:
        Dictionary with validation results
    """
    results: dict[str, Any] = {
        'total_rows': len(df),
        'valid_rows': 0,
        'missing_fields': {},
        'warnings': [],
    }

    # Check required fields
    required_fields = ['Title', 'Abstract']
    for field in required_fields:
        if field not in df.columns:
            results['warnings'].append(f'Missing required field: {field}')
        else:
            missing_count = (
                df[field].isna().sum() + (df[field].astype(str).str.strip() == '').sum()
            )
            results['missing_fields'][field] = missing_count

    # Count valid rows (have at least Title or Abstract)
    if 'Title' in df.columns and 'Abstract' in df.columns:
        valid_mask = (
            df['Title'].notna() & (df['Title'].astype(str).str.strip() != '')
        ) | (df['Abstract'].notna() & (df['Abstract'].astype(str).str.strip() != ''))
        results['valid_rows'] = valid_mask.sum()

    return results
