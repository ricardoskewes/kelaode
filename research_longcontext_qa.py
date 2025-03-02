"""
Research script for long-context QA datasets.
"""

import requests
from bs4 import BeautifulSoup

# Function to search for long-context QA datasets
def search_longcontext_qa_datasets():
    print('Searching for long-context QA datasets...')
    
    # List of potential datasets to investigate
    datasets = [
        {'name': 'LongBench', 'description': 'Benchmark for evaluating LLMs on long-context understanding'},
        {'name': 'QMSum', 'description': 'Query-based multi-document summarization dataset with long contexts'},
        {'name': 'NarrativeQA', 'description': 'QA dataset based on books and movie scripts'},
        {'name': 'HotpotQA', 'description': 'Multi-hop QA dataset requiring reasoning across documents'},
        {'name': 'TriviaQA', 'description': 'Large-scale QA dataset with long contexts'},
        {'name': 'QuALITY', 'description': 'Multiple-choice QA dataset with long narrative contexts'},
        {'name': 'SCROLLS', 'description': 'Suite of tasks requiring long-context understanding'},
        {'name': 'LongForm', 'description': 'Dataset for long-form question answering'}
    ]
    
    print('\nPotential Long-Context QA Datasets:')
    print('=====================================')
    for i, dataset in enumerate(datasets):
        print(f"{i+1}. {dataset['name']}: {dataset['description']}")
    
    # Try to get more information about LongBench
    try:
        print('\nGetting more information about LongBench...')
        response = requests.get('https://huggingface.co/datasets/THUDM/LongBench')
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            description = soup.find('div', {'class': 'prose'})
            if description:
                print('LongBench Description:')
                print(description.get_text()[:500] + '...')
            else:
                print('Could not extract LongBench description.')
        else:
            print(f'Failed to get LongBench information. Status code: {response.status_code}')
    except Exception as e:
        print(f'Error getting LongBench information: {str(e)}')
    
    # Try to get more information about SCROLLS
    try:
        print('\nGetting more information about SCROLLS...')
        response = requests.get('https://huggingface.co/datasets/tau/scrolls')
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            description = soup.find('div', {'class': 'prose'})
            if description:
                print('SCROLLS Description:')
                print(description.get_text()[:500] + '...')
            else:
                print('Could not extract SCROLLS description.')
        else:
            print(f'Failed to get SCROLLS information. Status code: {response.status_code}')
    except Exception as e:
        print(f'Error getting SCROLLS information: {str(e)}')

if __name__ == "__main__":
    # Execute the search
    search_longcontext_qa_datasets()
