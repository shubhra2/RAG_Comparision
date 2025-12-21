"""Dataset information and metadata."""


def get_dataset_info() -> dict[str, dict[str, str]]:
    """Get information about datasets used in the project.

    Returns:
        Dictionary mapping dataset names to their information
    """
    return {
        'Synthetic Articles': {
            'Description': 'Synthetic research articles dataset for knowledge graph construction',
            'Size': '20 articles',
            'Format': 'CSV with Title, Abstract, Topic, Subtopic, Authors, Publication_Date',
            'Source': 'https://github.com/dcarpintero/ai-engineering/blob/main/dataset/synthetic_articles.csv',
            'Entities': 'Researcher, Article, Topic',
            'Relationships': 'Researcher --[PUBLISHED]--> Article, Article --[IN_TOPIC]--> Topic',
            'Status': 'Primary Dataset',
        }
    }
