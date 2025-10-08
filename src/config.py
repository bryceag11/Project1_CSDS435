# config.py
from pathlib import Path

# Project structure
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
RESULTS_DIR = PROJECT_ROOT / 'results'

# Dataset paths (use actual project filenames)
DATASETS = {
    'dataset1': DATA_DIR / 'project1_dataset1.txt',  # ← Changed
    'dataset2': DATA_DIR / 'project1_dataset2.txt'   # ← Changed
}

# Create directories if they don't exist
RESULTS_DIR.mkdir(exist_ok=True)