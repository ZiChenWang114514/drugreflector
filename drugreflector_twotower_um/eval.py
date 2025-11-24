#!/usr/bin/env python
"""
Two-Tower DrugReflector Evaluation Script

Evaluate trained Two-Tower DrugReflector models.
Supports single model and ensemble evaluation.

Usage:
    # Evaluate single model
    python eval.py --model-path models/twotower_fold_0.pt \
                   --data-file data.pkl --mol-embeddings mol_embeddings.pkl --fold 0
    
    # Evaluate ensemble
    python eval.py --model-paths models/twotower_fold_*.pt \
                   --data-file data.pkl --mol-embeddings mol_embeddings.pkl --ensemble
"""
import argparse
import pickle
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

from models import TwoTowerModel
from dataset import TwoTowerDataset, load_training_data_with_mol
from preprocessing import clip_and_normalize_signature, normalize_mol_embeddings


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


class TwoTowerEvaluator:
    """Evaluator for trained Two-Tower DrugReflector models."""
    
    def __init__(
        self,
        device: str = 'auto',
        batch_size: int = 512,
        num_workers: int = 4,
    ):
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        print(f"\n{'='*80}")
        print(f"📊 Two-Tower Model Evaluator Initialized")
        print(f"{'='*80}")
        print(f"  Device: {self.device}")
    
    def load_model(self, checkpoint_path: Path) -> nn.Module:
        """Load model from checkpoint."""
        print(f"\n🔄 Loading model from: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        dims = checkpoint['dimensions']
        config = checkpoint['model_config']
        
        model = TwoTowerModel(
            n_genes=dims['input_size'],
            n_compounds=dims['output_size'],
            embedding_dim=config['embedding_dim'],
            fusion_type=config['fusion_type'],
            transcript_hidden_dims=config['transcript_hidden_dims'],
            mol_hidden_dims=config['mol_hidden_dims'],
            classifier_hidden_dims=config['classifier_hidden_dims'],
            transcript_dropout=config['transcript_dropout'],
            mol_dropout=config['mol_dropout'],
            classifier_dropout=config['classifier_dropout'],
            unimol_dim=config['unimol_dim'],
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        print(f"  ✓ Model loaded (fold {checkpoint.get('fold_id', 'N/A')})")
        
        return model
    
    def evaluate_single_model(
        self,
        model: nn.Module,
        test_data: Dict,
        output_dir: Optional[Path] = None,
    ) -> Dict:
        """Evaluate single model on test data."""
        print(f"\n{'='*80}")
        print(f"🎯 Evaluating Two-Tower Model")
        print(f"{'='*80}")
        
        X = test_data['X']
        y = test_data['y']
        mol_embeddings = test_data['mol_embeddings']
        
        print(f"  Test samples: {len(X):,}")
        
        # Preprocess
        X_processed = clip_and_normalize_signature(X)
        mol_processed = normalize_mol_embeddings(mol_embeddings, method='l2')
        
        # Create dataset
        test_dataset = TwoTowerDataset(X_processed, y, mol_processed)
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=(self.device == 'cuda'),
        )
        
        # Evaluate
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch_X, batch_mol, batch_y in tqdm(test_loader, desc="Evaluating"):
                batch_X = batch_X.to(self.device)
                batch_mol = batch_mol.to(self.device)
                
                outputs = model(batch_X, batch_mol)
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
        self._print_metrics(metrics)
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            self._save_results(metrics, all_labels, all_probs, output_dir)
        
        return metrics
    
    def evaluate_ensemble(
        self,
        model_paths: List[Path],
        test_data: Dict,
        output_dir: Optional[Path] = None,
    ) -> Dict:
        """Evaluate ensemble of models."""
        print(f"\n{'='*80}")
        print(f"🎯 Evaluating Ensemble ({len(model_paths)} models)")
        print(f"{'='*80}")
        
        models = [self.load_model(path) for path in model_paths]
        
        X = test_data['X']
        y = test_data['y']
        mol_embeddings = test_data['mol_embeddings']
        
        X_processed = clip_and_normalize_signature(X)
        mol_processed = normalize_mol_embeddings(mol_embeddings, method='l2')
        
        test_dataset = TwoTowerDataset(X_processed, y, mol_processed)
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=(self.device == 'cuda'),
        )
        
        all_labels = []
        all_ensemble_probs = []
        
        with torch.no_grad():
            for batch_X, batch_mol, batch_y in tqdm(test_loader, desc="Evaluating"):
                batch_X = batch_X.to(self.device)
                batch_mol = batch_mol.to(self.device)
                
                batch_probs = []
                for model in models:
                    outputs = model(batch_X, batch_mol)
                    probs = F.softmax(outputs, dim=1)
                    batch_probs.append(probs.cpu().numpy())
                
                ensemble_probs = np.mean(batch_probs, axis=0)
                
                all_labels.append(batch_y.numpy())
                all_ensemble_probs.append(ensemble_probs)
        
        all_labels = np.concatenate(all_labels)
        all_ensemble_probs = np.concatenate(all_ensemble_probs)
        all_ensemble_preds = np.argmax(all_ensemble_probs, axis=1)
        
        metrics = self._compute_metrics(all_labels, all_ensemble_preds, all_ensemble_probs)
        
        print(f"\n📊 Ensemble Results:")
        self._print_metrics(metrics)
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            self._save_results(metrics, all_labels, all_ensemble_probs, output_dir, prefix='ensemble_')
        
        return metrics
    
    def _compute_metrics(self, labels, preds, probs) -> Dict:
        """Compute evaluation metrics."""
        n_classes = probs.shape[1]
        
        top1_acc = accuracy_score(labels, preds)
        top5_acc = top_k_accuracy_score(labels, probs, k=min(5, n_classes))
        top10_acc = top_k_accuracy_score(labels, probs, k=min(10, n_classes))
        top20_acc = top_k_accuracy_score(labels, probs, k=min(20, n_classes))
        top50_acc = top_k_accuracy_score(labels, probs, k=min(50, n_classes))
        
        top1_percent_k = max(1, int(0.01 * n_classes))
        recall_1pct = compound_level_topk_recall(labels, probs, top1_percent_k)
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
            'n_classes': n_classes,
        }
    
    def _print_metrics(self, metrics: Dict):
        """Print metrics."""
        print(f"\n  Samples: {metrics['n_samples']:,}")
        print(f"  Classes: {metrics['n_classes']:,}")
        print(f"\n  📈 Accuracy:")
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
    
    def _save_results(self, metrics, labels, probs, output_dir, prefix=''):
        """Save evaluation results."""
        metrics_path = output_dir / f'{prefix}metrics.pkl'
        with open(metrics_path, 'wb') as f:
            pickle.dump(metrics, f)
        print(f"\n💾 Metrics saved: {metrics_path}")
        
        predictions_path = output_dir / f'{prefix}predictions.npz'
        np.savez(predictions_path, labels=labels, probs=probs, preds=np.argmax(probs, axis=1))
        print(f"💾 Predictions saved: {predictions_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Two-Tower DrugReflector model(s)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument('--model-path', type=str, help='Path to single model checkpoint')
    model_group.add_argument('--model-paths', type=str, nargs='+', help='Paths to multiple model checkpoints')
    
    parser.add_argument('--data-file', type=str, required=True, help='Path to training data')
    parser.add_argument('--mol-embeddings', type=str, required=True, help='Path to molecular embeddings')
    parser.add_argument('--fold', type=int, default=None, choices=[0, 1, 2], help='Fold to use as test')
    parser.add_argument('--output-dir', type=str, default='eval_results')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--num-workers', type=int, default=4)
    
    args = parser.parse_args()
    
    if args.data_file and args.fold is None and args.model_path:
        parser.error("--fold is required when using --data-file with single model")
    
    print(f"\n{'='*80}")
    print(f"🧬 TWO-TOWER DRUGREFLECTOR EVALUATION")
    print(f"{'='*80}")
    
    # Load data
    training_data = load_training_data_with_mol(
        training_data_path=args.data_file,
        mol_embeddings_path=args.mol_embeddings,
        filter_missing_mol=True,
    )
    
    if args.fold is not None:
        test_mask = training_data['folds'] == args.fold
        test_data = {
            'X': training_data['X'][test_mask],
            'y': training_data['y'][test_mask],
            'mol_embeddings': training_data['mol_embeddings'][test_mask],
        }
    else:
        test_data = training_data
    
    evaluator = TwoTowerEvaluator(
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    
    output_dir = Path(args.output_dir)
    
    if args.model_path:
        model = evaluator.load_model(Path(args.model_path))
        metrics = evaluator.evaluate_single_model(model, test_data, output_dir)
    else:
        model_paths = [Path(p) for p in args.model_paths]
        metrics = evaluator.evaluate_ensemble(model_paths, test_data, output_dir)
    
    print(f"\n{'='*80}")
    print(f"✅ EVALUATION COMPLETE!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
