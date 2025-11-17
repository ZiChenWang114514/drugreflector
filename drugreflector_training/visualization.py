"""
Visualization utilities for DrugReflector training.
"""

import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict


def plot_training_history(histories: List[Dict], output_dir: Path):
    """
    Plot training history for ensemble models.
    
    Parameters
    ----------
    histories : List[Dict]
        List of training histories from 3 models
    output_dir : Path
        Directory to save plots
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    metrics = [
        ('train_loss', 'Training Loss', 'Loss'),
        ('val_loss', 'Validation Loss', 'Loss'),
        ('val_recall', 'Validation Recall (Top 1%)', 'Recall'),
        ('val_top1_acc', 'Top-1 Accuracy', 'Accuracy'),
        ('val_top10_acc', 'Top-10 Accuracy', 'Accuracy'),
        ('learning_rates', 'Learning Rate', 'LR')
    ]
    
    for idx, (metric, title, ylabel) in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        
        for fold_id, history in enumerate(histories):
            if metric in history:
                epochs = range(1, len(history[metric]) + 1)
                ax.plot(epochs, history[metric], label=f'Fold {fold_id}', linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = output_dir / 'training_history.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Training history plot saved to {save_path}")
    plt.close()