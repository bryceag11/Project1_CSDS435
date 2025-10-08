# plotting.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix

# Set modern style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)
COLORS = sns.color_palette("Set2", 8)

def plot_performance_comparison(results_df, dataset_name, save_dir):
    """
    Create bar plot comparing all classifiers across metrics
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Performance Comparison - {dataset_name}', fontsize=16, fontweight='bold')
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        # Sort by metric value
        sorted_df = results_df.sort_values(metric, ascending=True)
        
        # Create bar plot
        bars = ax.barh(sorted_df['classifier'], sorted_df[metric], color=COLORS[:len(sorted_df)])
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f'{width:.3f}', 
                   ha='left', va='center', fontsize=9, fontweight='bold')
        
        ax.set_xlabel(metric.capitalize(), fontsize=12)
        ax.set_title(f'{metric.capitalize()}', fontsize=13, fontweight='bold')
        ax.set_xlim(0, 1.05)
        ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    save_path = save_dir / f'{dataset_name}_performance_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_metrics_heatmap(results_df, dataset_name, save_dir):
    """
    Create heatmap showing all metrics for all classifiers
    """
    # Prepare data
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    heatmap_data = results_df.set_index('classifier')[metrics]
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlGnBu', 
                cbar_kws={'label': 'Score'}, linewidths=0.5, ax=ax,
                vmin=0, vmax=1)
    
    ax.set_title(f'Performance Heatmap - {dataset_name}', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Classifiers', fontsize=12)
    
    plt.tight_layout()
    save_path = save_dir / f'{dataset_name}_heatmap.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_algorithm_comparison(all_results, save_dir):
    """
    Compare algorithm performance across both datasets
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Algorithm Performance Across Datasets', fontsize=16, fontweight='bold')
    
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        # Prepare data for grouped bar chart
        data = []
        for dataset_name, results_df in all_results.items():
            for _, row in results_df.iterrows():
                data.append({
                    'Dataset': dataset_name,
                    'Classifier': row['classifier'],
                    'Score': row[metric]
                })
        
        plot_df = pd.DataFrame(data)
        
        # Create grouped bar plot
        classifiers = plot_df['Classifier'].unique()
        x = np.arange(len(classifiers))
        width = 0.35
        
        datasets = plot_df['Dataset'].unique()
        for i, dataset in enumerate(datasets):
            dataset_data = plot_df[plot_df['Dataset'] == dataset]
            scores = [dataset_data[dataset_data['Classifier'] == clf]['Score'].values[0] 
                     for clf in classifiers]
            ax.bar(x + i*width, scores, width, label=dataset, alpha=0.8)
        
        ax.set_xlabel('Classifier', fontsize=11)
        ax.set_ylabel(metric.capitalize(), fontsize=11)
        ax.set_title(f'{metric.capitalize()} Comparison', fontsize=12, fontweight='bold')
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(classifiers, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    save_path = save_dir / 'cross_dataset_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_best_worst_comparison(results_df, dataset_name, save_dir):
    """
    Highlight best and worst performing algorithms
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get F1 scores (overall metric)
    sorted_df = results_df.sort_values('f1', ascending=False)
    
    # Create color list (green for best, red for worst, gray for others)
    colors = ['#2ecc71' if i == 0 else '#e74c3c' if i == len(sorted_df)-1 else '#95a5a6' 
              for i in range(len(sorted_df))]
    
    bars = ax.bar(sorted_df['classifier'], sorted_df['f1'], color=colors, alpha=0.8)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('F1-Score', fontsize=12)
    ax.set_title(f'F1-Score Ranking - {dataset_name}', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.1)
    plt.xticks(rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='Best'),
        Patch(facecolor='#e74c3c', label='Worst'),
        Patch(facecolor='#95a5a6', label='Others')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    save_path = save_dir / f'{dataset_name}_best_worst.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_confidence_intervals(results_df, dataset_name, save_dir):
    """
    Plot accuracy with confidence intervals (using std)
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sorted_df = results_df.sort_values('accuracy', ascending=True)
    
    # Plot with error bars
    ax.barh(sorted_df['classifier'], sorted_df['accuracy'], 
            xerr=sorted_df['accuracy_std'], 
            color=COLORS[:len(sorted_df)], alpha=0.7,
            capsize=5, error_kw={'linewidth': 2})
    
    ax.set_xlabel('Accuracy', fontsize=12)
    ax.set_title(f'Accuracy with Standard Deviation - {dataset_name}', 
                fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.05)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    save_path = save_dir / f'{dataset_name}_confidence_intervals.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def create_summary_table(all_results, save_dir):
    """
    Create a publication-ready summary table
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data
    summary_data = []
    for dataset_name, results_df in all_results.items():
        for _, row in results_df.iterrows():
            summary_data.append([
                dataset_name,
                row['classifier'],
                f"{row['accuracy']:.3f}",
                f"{row['precision']:.3f}",
                f"{row['recall']:.3f}",
                f"{row['f1']:.3f}"
            ])
    
    # Create table
    table = ax.table(cellText=summary_data,
                    colLabels=['Dataset', 'Classifier', 'Accuracy', 'Precision', 'Recall', 'F1'],
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.15, 0.25, 0.12, 0.12, 0.12, 0.12])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(6):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(summary_data) + 1):
        for j in range(6):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.title('Performance Summary - All Experiments', fontsize=14, fontweight='bold', pad=20)
    
    save_path = save_dir / 'summary_table.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def generate_all_plots(results_df, dataset_name, save_dir):
    """
    Generate all plots for a single dataset
    """
    print(f"\nGenerating plots for {dataset_name}...")
    
    # Create plots subdirectory
    plots_dir = save_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    # Generate individual plots
    plot_performance_comparison(results_df, dataset_name, plots_dir)
    plot_metrics_heatmap(results_df, dataset_name, plots_dir)
    plot_best_worst_comparison(results_df, dataset_name, plots_dir)
    plot_confidence_intervals(results_df, dataset_name, plots_dir)
    
    print(f"All plots saved to {plots_dir}/")

def plot_confusion_matrix(y_true, y_pred, classifier_name, dataset_name, save_dir):
    """
    Plot confusion matrix for a classifier
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Class 0', 'Class 1'],
                yticklabels=['Class 0', 'Class 1'],
                cbar_kws={'label': 'Count'})
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(f'Confusion Matrix - {classifier_name}\n{dataset_name}', 
                fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    save_path = save_dir / f'{dataset_name}_{classifier_name}_confusion_matrix.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()