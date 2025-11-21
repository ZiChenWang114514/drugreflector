#!/usr/bin/env python
"""
DrugReflector Evaluation Script

Evaluate trained DrugReflector models on test/validation data.
Supports single model and ensemble evaluation.

Usage:
    # Evaluate single model
    python eval.py --model-path models/fold0/model_fold_0.pt --data-file data.pkl --fold 0
    
    # Evaluate ensemble
    python eval.py --model-paths models/*/model_fold_*.pt --data-file data.pkl --ensemble
    
    # Evaluate on specific test set
    python eval.py --model-path model.pt --test-file test_data.pkl
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, top_k_accuracy_score
import matplotlib.pyplot as plt
import pandas as pd

# Add parent directory to path
try:
    sys.path.append(str(Path(__file__).parent.parent))
except NameError:
    import os
    sys.path.append(str(Path(os.getcwd()).parent))

from models import nnFC
from dataset import LINCSDataset
from preprocessing import clip_and_normalize_signature


def compound_level_topk_recall(labels, probs, k):
    """Compute compound-level top-k recall."""
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    n_classes = probs.shape[1]
    k = max(1, min(k, n_classes))

    topk_pred = np.argpartition(-probs, kth=k-1, axis=1)[:, :k]
    hit_per_sample = (topk_pred == labels[:, None]).any(axis=1).astype(float)

    compound_hits = {}
    compound_counts = {}
    for y, hit in zip(labels, hit_per_sample):
        compound_hits.setdefault(y, 0.0)
        compound_counts.setdefault(y, 0)
        compound_hits[y] += hit
        compound_counts[y] += 1

    compound_recalls = []
    for cid in compound_hits:
        compound_recalls.append(compound_hits[cid] / compound_counts[cid])

    return float(np.mean(compound_recalls))


class ModelEvaluator:
    """
    Evaluator for trained DrugReflector models.
    
    Parameters
    ----------
    device : str
        Device for evaluation ('auto', 'cuda', or 'cpu')
    batch_size : int
        Evaluation batch size
    num_workers : int
        DataLoader worker processes
    """
    
    def __init__(
        self,
        device: str = 'auto',
        batch_size: int = 512,
        num_workers: int = 4
    ):
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        print(f"\n{'='*80}")
        print(f"📊 Model Evaluator Initialized")
        print(f"{'='*80}")
        print(f"  Device: {self.device}")
        print(f"  Batch size: {self.batch_size}")
    
    def load_model(self, checkpoint_path: Path) -> nn.Module:
        """Load model from checkpoint."""
        print(f"\n🔄 Loading model from: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Get dimensions
        dims = checkpoint['dimensions']
        input_size = dims['input_size']
        output_size = dims['output_size']
        
        # Get model parameters
        params = checkpoint['params_init']['model_init_params']['torch_init_params']
        
        # Create model
        model = nnFC(
            input_dim=input_size,
            output_dim=output_size,
            hidden_dims=params['hidden_dims'],
            dropout_p=params['dropout_p'],
            activation=params['activation'],
            batch_norm=params['batch_norm'],
            order=params['order'],
            final_layer_bias=params['final_layer_bias']
        )
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        print(f"  ✓ Model loaded")
        print(f"  Architecture: {input_size} → {params['hidden_dims']} → {output_size}")
        print(f"  Fold ID: {checkpoint.get('fold_id', 'N/A')}")
        
        return model
    
    def evaluate_single_model(
        self,
        model: nn.Module,
        test_data: Dict,
        output_dir: Optional[Path] = None
    ) -> Dict:
        """
        Evaluate single model on test data.
        
        Parameters
        ----------
        model : nn.Module
            Model to evaluate
        test_data : Dict
            Test data dictionary
        output_dir : Path, optional
            Directory to save results
        
        Returns
        -------
        Dict
            Evaluation metrics
        """
        print(f"\n{'='*80}")
        print(f"🎯 Evaluating Model")
        print(f"{'='*80}")
        
        # Extract test data
        X = test_data['X']
        y = test_data['y']
        
        print(f"  Test samples: {len(X):,}")
        
        # Preprocess
        X_processed = clip_and_normalize_signature(X)
        
        # Create dataset and loader
        test_dataset = LINCSDataset(X_processed, y, mask=None)
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=(self.device == 'cuda')
        )
        
        # Evaluate
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch_X, batch_y in tqdm(test_loader, desc="Evaluating"):
                batch_X = batch_X.to(self.device)
                
                outputs = model(batch_X)
                probs = F.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                all_preds.append(preds.cpu().numpy())
                all_labels.append(batch_y.numpy())
                all_probs.append(probs.cpu().numpy())
        
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        all_probs = np.concatenate(all_probs)
        
        # Compute metrics
        metrics = self._compute_metrics(all_labels, all_preds, all_probs)
        
        # Print results
        self._print_metrics(metrics)
        
        # Save results if output_dir provided
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            self._save_results(metrics, all_labels, all_probs, output_dir)
        
        return metrics
    
    def evaluate_ensemble(
        self,
        model_paths: List[Path],
        test_data: Dict,
        output_dir: Optional[Path] = None
    ) -> Dict:
        """
        Evaluate ensemble of models.
        
        Parameters
        ----------
        model_paths : List[Path]
            Paths to model checkpoints
        test_data : Dict
            Test data dictionary
        output_dir : Path, optional
            Directory to save results
        
        Returns
        -------
        Dict
            Evaluation metrics for ensemble
        """
        print(f"\n{'='*80}")
        print(f"🎯 Evaluating Ensemble ({len(model_paths)} models)")
        print(f"{'='*80}")
        
        # Load all models
        models = []
        for path in model_paths:
            model = self.load_model(path)
            models.append(model)
        
        # Extract test data
        X = test_data['X']
        y = test_data['y']
        
        print(f"\n  Test samples: {len(X):,}")
        
        # Preprocess
        X_processed = clip_and_normalize_signature(X)
        
        # Create dataset and loader
        test_dataset = LINCSDataset(X_processed, y, mask=None)
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=(self.device == 'cuda')
        )
        
        # Evaluate ensemble
        all_labels = []
        all_ensemble_probs = []
        
        with torch.no_grad():
            for batch_X, batch_y in tqdm(test_loader, desc="Evaluating ensemble"):
                batch_X = batch_X.to(self.device)
                
                # Average predictions from all models
                batch_probs = []
                for model in models:
                    outputs = model(batch_X)
                    probs = F.softmax(outputs, dim=1)
                    batch_probs.append(probs.cpu().numpy())
                
                ensemble_probs = np.mean(batch_probs, axis=0)
                
                all_labels.append(batch_y.numpy())
                all_ensemble_probs.append(ensemble_probs)
        
        all_labels = np.concatenate(all_labels)
        all_ensemble_probs = np.concatenate(all_ensemble_probs)
        all_ensemble_preds = np.argmax(all_ensemble_probs, axis=1)
        
        # Compute metrics
        metrics = self._compute_metrics(all_labels, all_ensemble_preds, all_ensemble_probs)
        
        # Print results
        print(f"\n{'='*80}")
        print(f"📊 Ensemble Results")
        print(f"{'='*80}")
        self._print_metrics(metrics)
        
        # Save results if output_dir provided
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            self._save_results(metrics, all_labels, all_ensemble_probs, output_dir, 
                             prefix='ensemble_')
        
        return metrics
    
    def _compute_metrics(self, labels, preds, probs) -> Dict:
        """Compute evaluation metrics."""
        n_classes = probs.shape[1]
        
        # Top-k accuracies
        top1_acc = accuracy_score(labels, preds)
        top5_acc = top_k_accuracy_score(labels, probs, k=min(5, n_classes))
        top10_acc = top_k_accuracy_score(labels, probs, k=min(10, n_classes))
        top20_acc = top_k_accuracy_score(labels, probs, k=min(20, n_classes))
        top50_acc = top_k_accuracy_score(labels, probs, k=min(50, n_classes))
        
        # Top 1% recall (paper's main metric)
        top1_percent_k = max(1, int(0.01 * n_classes))
        recall_1pct = compound_level_topk_recall(labels, probs, top1_percent_k)
        
        # Additional top-k recalls
        recall_top5 = compound_level_topk_recall(labels, probs, 5)
        recall_top10 = compound_level_topk_recall(labels, probs, 10)
        recall_top20 = compound_level_topk_recall(labels, probs, 20)
        
        return {
            'top1_accuracy': top1_acc,
            'top5_accuracy': top5_acc,
            'top10_accuracy': top10_acc,
            'top20_accuracy': top20_acc,
            'top50_accuracy': top50_acc,
            'recall_1percent': recall_1pct,
            'recall_top5': recall_top5,
            'recall_top10': recall_top10,
            'recall_top20': recall_top20,
            'n_samples': len(labels),
            'n_classes': n_classes
        }
    
    def _print_metrics(self, metrics: Dict):
        """Print metrics in formatted table."""
        print(f"\n  Samples: {metrics['n_samples']:,}")
        print(f"  Classes: {metrics['n_classes']:,}")
        print(f"\n  📈 Accuracy Metrics:")
        print(f"    Top-1:  {metrics['top1_accuracy']:.4f}")
        print(f"    Top-5:  {metrics['top5_accuracy']:.4f}")
        print(f"    Top-10: {metrics['top10_accuracy']:.4f}")
        print(f"    Top-20: {metrics['top20_accuracy']:.4f}")
        print(f"    Top-50: {metrics['top50_accuracy']:.4f}")
        print(f"\n  🎯 Compound-Level Recall:")
        print(f"    Top 1%: {metrics['recall_1percent']:.4f} ⭐")
        print(f"    Top-5:  {metrics['recall_top5']:.4f}")
        print(f"    Top-10: {metrics['recall_top10']:.4f}")
        print(f"    Top-20: {metrics['recall_top20']:.4f}")
    
    def _save_results(
        self,
        metrics: Dict,
        labels: np.ndarray,
        probs: np.ndarray,
        output_dir: Path,
        prefix: str = ''
    ):
        """Save evaluation results."""
        # Save metrics
        metrics_path = output_dir / f'{prefix}metrics.pkl'
        with open(metrics_path, 'wb') as f:
            pickle.dump(metrics, f)
        print(f"\n💾 Metrics saved: {metrics_path}")
        
        # Save predictions
        predictions_path = output_dir / f'{prefix}predictions.npz'
        np.savez(
            predictions_path,
            labels=labels,
            probs=probs,
            preds=np.argmax(probs, axis=1)
        )
        print(f"💾 Predictions saved: {predictions_path}")
        
        # Create metrics plot
        self._plot_metrics(metrics, output_dir, prefix)
    
    def _plot_metrics(self, metrics: Dict, output_dir: Path, prefix: str = ''):
        """Plot evaluation metrics."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Top-k Accuracy
        k_values = [1, 5, 10, 20, 50]
        accuracies = [
            metrics['top1_accuracy'],
            metrics['top5_accuracy'],
            metrics['top10_accuracy'],
            metrics['top20_accuracy'],
            metrics['top50_accuracy']
        ]
        
        axes[0].plot(k_values, accuracies, marker='o', linewidth=2, 
                    markersize=8, color='#2E86AB')
        axes[0].set_xlabel('Top-K', fontsize=12)
        axes[0].set_ylabel('Accuracy', fontsize=12)
        axes[0].set_title('Top-K Accuracy', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xscale('log')
        
        # Compound-level Recall
        recall_labels = ['Top 1%', 'Top-5', 'Top-10', 'Top-20']
        recalls = [
            metrics['recall_1percent'],
            metrics['recall_top5'],
            metrics['recall_top10'],
            metrics['recall_top20']
        ]
        
        bars = axes[1].bar(recall_labels, recalls, color='#A23B72', alpha=0.8)
        axes[1].set_ylabel('Recall', fontsize=12)
        axes[1].set_title('Compound-Level Recall', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        axes[1].set_ylim([0, 1])
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=10)
        
        plt.suptitle(f'Evaluation Metrics', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        plot_path = output_dir / f'{prefix}metrics_plot.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Metrics plot saved: {plot_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained DrugReflector model(s)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model arguments
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        '--model-path',
        type=str,
        help='Path to single model checkpoint'
    )
    
    model_group.add_argument(
        '--model-paths',
        type=str,
        nargs='+',
        help='Paths to multiple model checkpoints (for ensemble)'
    )
    
    # Data arguments
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument(
        '--data-file',
        type=str,
        help='Path to training data file (will use specified fold as test)'
    )
    
    data_group.add_argument(
        '--test-file',
        type=str,
        help='Path to separate test data file'
    )
    
    parser.add_argument(
        '--fold',
        type=int,
        default=None,
        choices=[0, 1, 2],
        help='Which fold to use as test (required if using --data-file)'
    )
    
    # Output arguments
    parser.add_argument(
        '--output-dir',
        type=str,
        default='eval_results',
        help='Output directory for evaluation results'
    )
    
    # System arguments
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Evaluation device'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=512,
        help='Evaluation batch size'
    )
    
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of DataLoader workers'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.data_file and args.fold is None:
        parser.error("--fold is required when using --data-file")
    
    # Print banner
    print(f"\n{'='*80}")
    print(f"🧬 DRUGREFLECTOR EVALUATION")
    print(f"{'='*80}")
    
    # Load test data
    if args.test_file:
        print(f"\n📂 Loading test data from: {args.test_file}")
        with open(args.test_file, 'rb') as f:
            test_data = pickle.load(f)
    else:
        print(f"\n📂 Loading training data from: {args.data_file}")
        with open(args.data_file, 'rb') as f:
            full_data = pickle.load(f)
        
        # Extract test fold
        test_mask = full_data['folds'] == args.fold
        test_data = {
            'X': full_data['X'][test_mask],
            'y': full_data['y'][test_mask]
        }
        print(f"  Using fold {args.fold} as test set")
    
    # Create evaluator
    evaluator = ModelEvaluator(
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    output_dir = Path(args.output_dir)
    
    # Evaluate
    if args.model_path:
        # Single model evaluation
        model = evaluator.load_model(Path(args.model_path))
        metrics = evaluator.evaluate_single_model(model, test_data, output_dir)
    else:
        # Ensemble evaluation
        model_paths = [Path(p) for p in args.model_paths]
        metrics = evaluator.evaluate_ensemble(model_paths, test_data, output_dir)
    
    print(f"\n{'='*80}")
    print(f"✅ EVALUATION COMPLETE!")
    print(f"{'='*80}")
    print(f"  Results saved to: {output_dir}")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()