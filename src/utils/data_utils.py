import os
import requests
from pathlib import Path

def download_file(url: str, output_path: str) -> None:
    """
    Download a file from a URL to a specified path.
    
    Args:
        url (str): URL of the file to download
        output_path (str): Path where the file should be saved
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Write file
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Successfully downloaded: {output_path}")
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {str(e)}")


def download_10k_reports():
    """Download 10-K reports for Uber and Lyft."""
    # Define base directory - works in both notebooks and scripts
    try:
        # Try script-style path first
        base_dir = Path(__file__).parent
    except NameError:
        # If in Jupyter, use current working directory
        base_dir = Path.cwd()
    
    data_dir = base_dir / 'src' / 'data' / '10k'
    
    # Create directory
    os.makedirs(data_dir, exist_ok=True)
    
    # Define files to download
    files = {
        'uber_2021.pdf': 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10k/uber_2021.pdf',
        'lyft_2021.pdf': 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10k/lyft_2021.pdf'
    }
    
    # Download each file
    for filename, url in files.items():
        output_path = data_dir / filename
        download_file(url, str(output_path))
