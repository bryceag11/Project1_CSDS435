# main.py
import numpy as np
import pandas as pd
from src.config import DATASETS, RESULTS_DIR
from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.classifiers import get_classifiers
from src.network import get_neural_network
from src.evaluation import evaluate_all_models
from src.plotting import generate_all_plots, plot_algorithm_comparison, create_summary_table

def run_experiment(dataset_path, dataset_name):
    """Run all 6 classifiers on one dataset"""
    print(f"Processing: {dataset_name}")

    
    # Load and preprocess
    X, y = load_data(dataset_path)
    X_processed, y = preprocess_data(X, y)
    
    print(f"Dataset shape: {X_processed.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Get all classifiers
    # classifiers = get_classifiers()
    classifiers = {}
    classifiers['Neural Network'] = get_neural_network(input_size=X_processed.shape[1])
    
    # Evaluate all models with 10-fold CV
    results = evaluate_all_models(classifiers, X_processed, y, cv=10)
    
    # Save results
    results_df = pd.DataFrame(results)
    output_path = RESULTS_DIR / f"{dataset_name}_results.csv"
    results_df.to_csv(output_path, index=False)
    
    print(f"\nResults saved to: {output_path}")
    print("\nSummary:")
    print(results_df[['classifier', 'accuracy', 'f1']].to_string(index=False))
    
    # Generate plots
    generate_all_plots(results_df, dataset_name, RESULTS_DIR)
    
    return results_df

if __name__ == "__main__":
    # Process all datasets from config
    all_results = {}
    
    for dataset_name, dataset_path in DATASETS.items():
        if dataset_path.exists():
            results = run_experiment(dataset_path, dataset_name)
            all_results[dataset_name] = results
        else:
            print(f"Warning: {dataset_path} not found!")
    
    # Generate comparison plots
    if len(all_results) > 1:
        plots_dir = RESULTS_DIR / 'plots'
        plots_dir.mkdir(exist_ok=True)
        plot_algorithm_comparison(all_results, plots_dir)
        create_summary_table(all_results, plots_dir)

    print(f"Results and plots saved to: {RESULTS_DIR}/")
