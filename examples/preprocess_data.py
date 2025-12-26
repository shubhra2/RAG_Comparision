"""Example script to preprocess the synthetic articles dataset."""

from rag_comparision.core.data_preprocessing import (
    preprocess_synthetic_articles_dataset,
    validate_preprocessed_data,
)


def main():
    """Preprocess the dataset and validate the results."""
    print('=' * 60)
    print('Data Preprocessing Example')
    print('=' * 60)

    # Preprocess the dataset
    # This will download the CSV, clean it, and save as parquet
    df = preprocess_synthetic_articles_dataset(
        source=None,  # Uses default URL
        output_path='data/processed/synthetic_articles.parquet',
        force_reprocess=False,  # Set to True to reprocess even if parquet exists
    )

    print(f'\nPreprocessed DataFrame shape: {df.shape}')
    print(f'Columns: {list(df.columns)}')

    # Validate the preprocessed data
    print('\n' + '=' * 60)
    print('Validation Results')
    print('=' * 60)
    validation = validate_preprocessed_data(df)
    
    print(f'Total rows: {validation["total_rows"]}')
    print(f'Valid rows: {validation["valid_rows"]}')
    print(f'Missing fields: {validation["missing_fields"]}')
    
    if validation['warnings']:
        print('\nWarnings:')
        for warning in validation['warnings']:
            print(f'  - {warning}')
    else:
        print('\nNo warnings!')

    # Show sample data
    print('\n' + '=' * 60)
    print('Sample Data (first 3 rows)')
    print('=' * 60)
    print(df.head(3).to_string())

    print('\n' + '=' * 60)
    print('Preprocessing complete!')
    print('=' * 60)
    print(f'Processed data saved to: data/processed/synthetic_articles.parquet')
    print('You can now use load_synthetic_articles_dataset() to load it quickly.')


if __name__ == '__main__':
    main()


