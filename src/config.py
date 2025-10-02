# config.py
from pathlib import Path

# Project structure
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / 'data'
RESULTS_DIR = PROJECT_ROOT / 'results'

# Dataset paths
DATASETS = {
    'dataset1': DATA_DIR / 'dataset1.txt',
    'dataset2': DATA_DIR / 'dataset2.txt'
}

# Create directories if they don't exist
RESULTS_DIR.mkdir(exist_ok=True)