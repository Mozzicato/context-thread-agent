"""
Notebook downloader for collecting sample notebooks
"""

import requests
from pathlib import Path
from typing import List
import time
import json


class NotebookDownloader:
    """Download sample notebooks from GitHub."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def download_all(self) -> List[str]:
        """Download notebooks from various sources."""
        downloaded = []
        
        # Download from predefined sources
        sources = [
            self._download_from_github,
        ]
        
        for source_func in sources:
            try:
                notebooks = source_func()
                downloaded.extend(notebooks)
            except Exception as e:
                print(f"Error downloading from source: {e}")
        
        return downloaded
    
    def _download_from_github(self) -> List[str]:
        """Download notebooks from GitHub repositories."""
        repos = [
            "pandas-dev/pandas",
            "matplotlib/matplotlib", 
            "scikit-learn/scikit-learn",
            "statsmodels/statsmodels"
        ]
        
        downloaded = []
        
        for repo in repos:
            try:
                print(f"Fetching from {repo}...")
                notebooks = self._search_github_notebooks(repo)
                for nb_url, nb_name in notebooks[:2]:  # Limit per repo
                    try:
                        self._download_notebook(nb_url, nb_name)
                        downloaded.append(nb_name)
                        time.sleep(1)  # Rate limiting
                    except Exception as e:
                        print(f"Failed to download {nb_name}: {e}")
            except Exception as e:
                print(f"Failed to fetch from {repo}: {e}")
        
        return downloaded
    
    def _search_github_notebooks(self, repo: str) -> List[tuple]:
        """Search for notebooks in a GitHub repo."""
        # This is a simplified version - in practice, you'd use GitHub API
        # For now, return some known notebook URLs
        
        known_notebooks = {
            "pandas-dev/pandas": [
                ("https://raw.githubusercontent.com/pandas-dev/pandas/main/doc/source/user_guide/10min.ipynb", "pandas_10min.ipynb")
            ],
            "matplotlib/matplotlib": [
                ("https://raw.githubusercontent.com/matplotlib/matplotlib/main/tutorials/introductory/sample_plots.ipynb", "matplotlib_sample.ipynb")
            ],
            "scikit-learn/scikit-learn": [
                ("https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/examples/linear_model/plot_ols.ipynb", "sklearn_ols.ipynb")
            ]
        }
        
        return known_notebooks.get(repo, [])
    
    def _download_notebook(self, url: str, filename: str):
        """Download a single notebook."""
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Validate it's a notebook
        try:
            data = response.json()
            if 'cells' not in data:
                raise ValueError("Not a valid notebook")
        except:
            raise ValueError("Invalid notebook format")
        
        output_path = self.output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=1)
        
        print(f"✓ Downloaded: {filename}")