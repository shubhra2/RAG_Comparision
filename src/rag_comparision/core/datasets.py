"""Dataset information and metadata."""


def get_dataset_info() -> dict[str, dict[str, str]]:
    """Get information about datasets used in the project.

    Returns:
        Dictionary mapping dataset names to their information
    """
    return {
        'HotpotQA': {
            'Description': 'Multi-hop question answering dataset',
            'Train Size': '~113k questions',
            'Dev Size': '~7.6k questions',
            'Status': 'Primary Dataset',
        }
    }
