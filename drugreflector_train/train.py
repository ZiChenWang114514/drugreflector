"""
DrugReflector Training Script

Main script for training DrugReflector models on LINCS 2020 data.
Supports both single-fold and multi-fold (ensemble) training.

Usage:
    # Train single fold
    python train.py --data-file data.pkl --output-dir models/fold0 --fold 0
    
    # Train all folds (ensemble)
    python train.py --data-file data.pkl --output-dir models/ensemble --all-folds
    
    # Train specific folds
    python train.py --data-file data.pkl --output-dir models/folds_01 --folds 0 1
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

# Add parent directory to path for imports
try:
    sys.path.append(str(Path(__file__).parent.parent))
except NameError:
    import os
    sys.path.append(str(Path(os.getcwd()).parent))



def load_training_data(data_file: Path) -> dict:
    """
    Load and validate training data.
    
    Parameters
    ----------
    data_file : Path
        Path to training data pickle file
    
    Returns
    -------
    dict
        Training data dictionary
    """
    print(f"\n{'='*80}")
    print(f"📂 Loading Training Data")
    print(f"{'='*80}")
    
    if not data_file.exists():
        print(f"❌ Data file not found: {data_file}")
        sys.exit(1)
    
    print(f"  Loading from: {data_file}")
    with open(data_file, 'rb') as f:
        training_data = pickle.load(f)
    
    print(f"  ✓ Data loaded successfully")
    print(f"  Keys: {list(training_data.keys())}")
    
    # Validate required keys
    required_keys = ['X', 'y', 'folds', 'compound_names']
    missing_keys = [k for k in required_keys if k not in training_data]
    if missing_keys:
        print(f"❌ Missing required keys: {missing_keys}")
        sys.exit(1)
    
    # Print data summary
    print(f"\n📊 Data Summary:")
    print(f"  Samples: {len(training_data['X']):,}")
    print(f"  Features: {training_data['X'].shape[1]:,}")
    print(f"  Compounds: {len(training_data['compound_names']):,}")
    print(f"  Unique folds: {np.unique(training_data['folds'])}")
    
    return training_data


def print_banner():
    """Print script banner."""
    print(f"\n{'='*80}")
    print(f"🧬 DRUGREFLECTOR TRAINING")
    print(f"{'='*80}")
    print(f"  Based on Science 2025 paper")
    print(f"  Training on LINCS 2020 dataset")


def print_config(args):
    """Print training configuration."""
    print(f"\n📋 Training Configuration:")
    print(f"  Mode: {'Ensemble (all folds)' if args.all_folds else f'Single fold ({args.fold})'}")
    if args.folds:
        print(f"  Folds: {args.folds}")
    print(f"  Data file: {args.data_file}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Initial LR: {args.learning_rate}")
    print(f"  Focal γ: {args.focal_gamma}")
    print(f"  Device: {args.device}")
    print(f"  Workers: {args.num_workers}")


def main():
    parser = argparse.ArgumentParser(
        description="Train DrugReflector model(s) on LINCS 2020 data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # ===== Required Arguments =====
    parser.add_argument(
        '--data-file',
        type=str,
        required=True,
        help='Path to training data pickle file'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for models and results'
    )
    
    # ===== Fold Selection =====
    fold_group = parser.add_mutually_exclusive_group()
    fold_group.add_argument(
        '--fold',
        type=int,
        default=None,
        choices=[0, 1, 2],
        help='Train single fold (0, 1, or 2)'
    )
    
    fold_group.add_argument(
        '--all-folds',
        action='store_true',
        help='Train all folds (ensemble training)'
    )
    
    parser.add_argument(
        '--folds',
        type=int,
        nargs='+',
        default=None,
        choices=[0, 1, 2],
        help='Train specific folds (e.g., --folds 0 1)'
    )
    
    # ===== Training Hyperparameters (from SI Table S5) =====
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=256,
        help='Training batch size'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.0139,
        help='Initial learning rate (from SI Table S5)'
    )
    
    parser.add_argument(
        '--min-lr',
        type=float,
        default=0.00001,
        help='Minimum learning rate for scheduler'
    )
    
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=1e-5,
        help='L2 regularization weight decay'
    )
    
    parser.add_argument(
        '--focal-gamma',
        type=float,
        default=2.0,
        help='Focal loss gamma parameter'
    )
    
    parser.add_argument(
        '--t0',
        type=int,
        default=20,
        help='CosineAnnealingWarmRestarts T_0 parameter'
    )
    
    # ===== System Arguments =====
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Training device'
    )
    
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of DataLoader workers'
    )
    
    parser.add_argument(
        '--save-every',
        type=int,
        default=10,
        help='Save checkpoint every N epochs'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )
    
    args = parser.parse_args()
    
    # Validate fold arguments
    if not args.all_folds and args.fold is None and args.folds is None:
        parser.error("Must specify --fold, --folds, or --all-folds")
    
    # Print banner and config
    print_banner()
    print_config(args)
    
    # Load training data
    data_file = Path(args.data_file)
    training_data = load_training_data(data_file)
    
    # Create trainer
    trainer = DrugReflectorTrainer(
        device=args.device,
        initial_lr=args.learning_rate,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        t_0=args.t0,
        focal_gamma=args.focal_gamma,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        num_workers=args.num_workers,
        save_every=args.save_every,
        verbose=not args.quiet
    )
    
    output_dir = Path(args.output_dir)
    
    # Train based on mode
    if args.all_folds:
        # Train all folds
        results = trainer.train_all_folds(
            training_data=training_data,
            output_dir=output_dir,
            folds=None  # Train all folds (0, 1, 2)
        )
        
        print(f"\n{'='*80}")
        print(f"✅ ENSEMBLE TRAINING COMPLETE!")
        print(f"{'='*80}")
        print(f"\n📊 Ensemble Results:")
        for metric, value in results['ensemble_metrics'].items():
            print(f"  {metric}: {value:.4f}")
        
        print(f"\n📁 Trained Models:")
        for path in results['model_paths']:
            print(f"  • {path}")
        
        print(f"\n🎯 Next steps:")
        print(f"  1. Review ensemble comparison: {output_dir}/ensemble_comparison.png")
        print(f"  2. Load ensemble for inference:")
        print(f"     from drugreflector import DrugReflector")
        print(f"     model = DrugReflector(checkpoint_paths={results['model_paths']})")
        
    elif args.folds:
        # Train specific folds
        results = trainer.train_all_folds(
            training_data=training_data,
            output_dir=output_dir,
            folds=args.folds
        )
        
        print(f"\n{'='*80}")
        print(f"✅ MULTI-FOLD TRAINING COMPLETE!")
        print(f"{'='*80}")
        print(f"\n📊 Results:")
        for fold_id, fold_result in results['fold_results'].items():
            print(f"  Fold {fold_id}: Best recall = {fold_result['best_recall']:.4f}")
        
        print(f"\n📁 Trained Models:")
        for path in results['model_paths']:
            print(f"  • {path}")
    
    else:
        # Train single fold
        fold_id = args.fold
        result = trainer.train_single_fold(
            training_data=training_data,
            fold_id=fold_id,
            output_dir=output_dir
        )
        
        print(f"\n{'='*80}")
        print(f"✅ SINGLE FOLD TRAINING COMPLETE!")
        print(f"{'='*80}")
        print(f"  Fold: {fold_id}")
        print(f"  Best recall: {result['best_recall']:.4f}")
        print(f"  Best epoch: {result['best_epoch'] + 1}/{args.epochs}")
        print(f"  Model path: {result['model_path']}")
        
        print(f"\n🎯 Next steps:")
        print(f"  1. Review training curves: {output_dir}/training_curves_fold_{fold_id}.png")
        print(f"  2. Load model for inference:")
        print(f"     from drugreflector import DrugReflector")
        print(f"     model = DrugReflector(checkpoint_paths=['{result['model_path']}'])")
        print(f"  3. Train other folds for ensemble:")
        print(f"     python train.py --data-file {args.data_file} --output-dir {args.output_dir} --all-folds")
    
    print(f"\n{'='*80}\n")

"""
DrugReflector Trainer - Core Training Engine

Supports both single-fold and multi-fold training with paper-compliant parameters.
Based on Science 2025 paper and SI Table S5.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
from sklearn.metrics import accuracy_score, top_k_accuracy_score
import matplotlib.pyplot as plt

from drugreflector.models import nnFC
from drugreflector_training import (
    LINCSDataset,
    FocalLoss,
    clip_and_normalize_signature
)


def compound_level_topk_recall(labels, probs, k):
    """
    Compound-level top-k recall as defined in the paper:
    1) For each observation, check if true label is in top-k predictions
    2) Average hit rate within each compound
    3) Average across all compounds
    
    Parameters
    ----------
    labels : np.ndarray
        True labels (compound indices)
    probs : np.ndarray
        Predicted probabilities, shape (n_samples, n_compounds)
    k : int
        Top k to consider
    
    Returns
    -------
    float
        Compound-level top-k recall
    """
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    n_classes = probs.shape[1]
    k = max(1, min(k, n_classes))

    # Top-k predictions for each sample
    topk_pred = np.argpartition(-probs, kth=k-1, axis=1)[:, :k]
    hit_per_sample = (topk_pred == labels[:, None]).any(axis=1).astype(float)

    # Aggregate by compound
    compound_hits = {}
    compound_counts = {}
    for y, hit in zip(labels, hit_per_sample):
        compound_hits.setdefault(y, 0.0)
        compound_counts.setdefault(y, 0)
        compound_hits[y] += hit
        compound_counts[y] += 1

    # Compound-level average
    compound_recalls = []
    for cid in compound_hits:
        compound_recalls.append(compound_hits[cid] / compound_counts[cid])

    return float(np.mean(compound_recalls))


class DrugReflectorTrainer:
    """
    DrugReflector model trainer with paper-compliant hyperparameters.
    
    Hyperparameters from SI Table S5 and Page 3:
    - Architecture: input → 1024 → 2048 → output
    - Dropout: 0.64
    - Initial LR: 0.0139
    - Min LR: 0.00001
    - Weight Decay: 1e-5
    - Focal Loss γ: 2.0
    - Scheduler: CosineAnnealingWarmRestarts with T_0=20
    
    Parameters
    ----------
    device : str
        Device for training ('auto', 'cuda', or 'cpu')
    initial_lr : float
        Initial learning rate
    min_lr : float
        Minimum learning rate for scheduler
    weight_decay : float
        L2 regularization weight decay
    t_0 : int
        CosineAnnealingWarmRestarts T_0 parameter
    focal_gamma : float
        Focal loss gamma parameter
    batch_size : int
        Training batch size
    num_epochs : int
        Number of training epochs
    num_workers : int
        DataLoader worker processes
    save_every : int
        Save checkpoint every N epochs
    verbose : bool
        Print detailed training information
    """
    
    def __init__(
        self,
        device: str = 'auto',
        initial_lr: float = 0.0139,
        min_lr: float = 0.00001,
        weight_decay: float = 1e-5,
        t_0: int = 20,
        focal_gamma: float = 2.0,
        batch_size: int = 256,
        num_epochs: int = 50,
        num_workers: int = 4,
        save_every: int = 10,
        verbose: bool = True
    ):
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.weight_decay = weight_decay
        self.t_0 = t_0
        self.focal_gamma = focal_gamma
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_workers = num_workers
        self.save_every = save_every
        self.verbose = verbose
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"🚀 DrugReflector Trainer Initialized")
            print(f"{'='*80}")
            print(f"  Device: {self.device}")
            print(f"  Initial LR: {self.initial_lr}")
            print(f"  Min LR: {self.min_lr}")
            print(f"  Weight Decay: {self.weight_decay}")
            print(f"  Focal γ: {self.focal_gamma}")
            print(f"  Batch size: {self.batch_size}")
            print(f"  Epochs: {self.num_epochs}")
            print(f"  Workers: {self.num_workers}")
    
    def create_model(self, input_size: int, output_size: int) -> nn.Module:
        """
        Create nnFC model with paper-specified architecture.
        
        Parameters
        ----------
        input_size : int
            Number of input features (genes)
        output_size : int
            Number of output classes (compounds)
        
        Returns
        -------
        nn.Module
            Initialized model
        """
        model = nnFC(
            input_dim=input_size,
            output_dim=output_size,
            hidden_dims=[1024, 2048],  # SI Page 2
            dropout_p=0.64,  # SI Table S5
            activation='ReLU',
            batch_norm=True,
            order='act-drop-bn',
            final_layer_bias=True
        )
        return model
    
    def train_single_fold(
        self,
        training_data: Dict,
        fold_id: int,
        output_dir: Path
    ) -> Dict:
        """
        Train model on a single fold.
        
        Parameters
        ----------
        training_data : Dict
            Training data dictionary with keys:
            - X: expression data (n_samples, n_genes)
            - y: compound labels (n_samples,)
            - folds: fold assignments (n_samples,)
            - compound_names: list of compound names
            - gene_names: list of gene names
            - sample_meta: sample metadata DataFrame
        fold_id : int
            Which fold to use as validation (0, 1, or 2)
        output_dir : Path
            Directory to save models and results
        
        Returns
        -------
        Dict
            Training results containing:
            - model: trained model
            - history: training history
            - best_recall: best validation recall
            - best_epoch: epoch with best recall
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract data
        X = training_data['X']
        y = training_data['y']
        folds = training_data['folds']
        compound_names = training_data['compound_names']
        gene_names = training_data.get('gene_names', 
                                       [f'gene_{i}' for i in range(X.shape[1])])
        
        n_samples = len(X)
        n_genes = X.shape[1]
        n_compounds = len(compound_names)
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"📊 Training Fold {fold_id}")
            print(f"{'='*80}")
            print(f"  Total samples: {n_samples:,}")
            print(f"  Total compounds: {n_compounds:,}")
            print(f"  Gene features: {n_genes}")
        
        # Preprocess data
        if self.verbose:
            print(f"\n🔧 Preprocessing signatures...")
        X_processed = clip_and_normalize_signature(X)
        
        # Create train/val split
        val_mask = folds == fold_id
        train_mask = ~val_mask
        
        n_train = train_mask.sum()
        n_val = val_mask.sum()
        
        if self.verbose:
            print(f"\n📋 Data Split:")
            print(f"  Training samples: {n_train:,} ({n_train/n_samples*100:.1f}%)")
            print(f"  Validation samples: {n_val:,} ({n_val/n_samples*100:.1f}%)")
            
            train_compounds = training_data['sample_meta'][train_mask]['pert_id'].nunique()
            val_compounds = training_data['sample_meta'][val_mask]['pert_id'].nunique()
            print(f"  Training compounds: {train_compounds:,}")
            print(f"  Validation compounds: {val_compounds:,}")
        
        # Create datasets and loaders
        train_dataset = LINCSDataset(X_processed, y, train_mask)
        val_dataset = LINCSDataset(X_processed, y, val_mask)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=(self.device == 'cuda')
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=(self.device == 'cuda')
        )
        
        if self.verbose:
            print(f"\n  Train batches: {len(train_loader)}")
            print(f"  Val batches: {len(val_loader)}")
        
        # Create model
        if self.verbose:
            print(f"\n🏗️  Building model...")
        model = self.create_model(n_genes, n_compounds).to(self.device)
        
        if self.verbose:
            n_params = sum(p.numel() for p in model.parameters())
            print(f"  Architecture: {n_genes} → 1024 → 2048 → {n_compounds}")
            print(f"  Parameters: {n_params:,}")
        
        # Setup training
        criterion = FocalLoss(gamma=self.focal_gamma)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.initial_lr,
            weight_decay=self.weight_decay
        )
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.t_0,
            T_mult=1,
            eta_min=self.min_lr
        )
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_recall': [],
            'val_top1_acc': [],
            'val_top10_acc': [],
            'learning_rates': [],
            'epoch_times': []
        }
        
        best_recall = 0.0
        best_epoch = 0
        best_model_state = None
        
        # Training loop
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"🎯 Starting Training")
            print(f"{'='*80}")
        
        for epoch in range(self.num_epochs):
            epoch_start = time.time()
            
            # Training phase
            train_loss = self._train_epoch(model, train_loader, criterion, optimizer)
            
            # Validation phase
            val_loss, val_metrics = self._validate_epoch(model, val_loader, criterion)
            
            # Record history
            current_lr = optimizer.param_groups[0]['lr']
            epoch_time = time.time() - epoch_start
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_recall'].append(val_metrics['recall'])
            history['val_top1_acc'].append(val_metrics['top1_acc'])
            history['val_top10_acc'].append(val_metrics['top10_acc'])
            history['learning_rates'].append(current_lr)
            history['epoch_times'].append(epoch_time)
            
            # Print progress
            if self.verbose:
                print(f"\n{'='*80}")
                print(f"Epoch {epoch+1}/{self.num_epochs} ({epoch_time:.1f}s)")
                print(f"{'='*80}")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val Loss: {val_loss:.4f}")
                print(f"  Val Top 1% Recall: {val_metrics['recall']:.4f} ⭐")
                print(f"  Val Top-1 Acc: {val_metrics['top1_acc']:.4f}")
                print(f"  Val Top-10 Acc: {val_metrics['top10_acc']:.4f}")
                print(f"  Learning Rate: {current_lr:.6f}")
            
            # Save best model
            if val_metrics['recall'] > best_recall:
                best_recall = val_metrics['recall']
                best_epoch = epoch
                best_model_state = model.state_dict().copy()
                if self.verbose:
                    print(f"  ✅ New best model! (Best recall: {best_recall:.4f})")
            
            # Periodic checkpoint
            if (epoch + 1) % self.save_every == 0:
                checkpoint_path = output_dir / f"checkpoint_fold{fold_id}_epoch{epoch+1}.pt"
                self._save_checkpoint(
                    model, fold_id, epoch, history,
                    n_genes, n_compounds, gene_names, compound_names,
                    checkpoint_path
                )
                if self.verbose:
                    print(f"  💾 Checkpoint saved: {checkpoint_path.name}")
            
            # Update learning rate
            scheduler.step()
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            if self.verbose:
                print(f"\n{'='*80}")
                print(f"✅ Training Complete!")
                print(f"{'='*80}")
                print(f"  Best epoch: {best_epoch + 1}/{self.num_epochs}")
                print(f"  Best recall: {best_recall:.4f}")
                total_time = sum(history['epoch_times'])
                print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f}m)")
        
        # Save final model
        model_path = output_dir / f"model_fold_{fold_id}.pt"
        self._save_checkpoint(
            model, fold_id, self.num_epochs - 1, history,
            n_genes, n_compounds, gene_names, compound_names,
            model_path
        )
        
        if self.verbose:
            print(f"\n💾 Final model saved: {model_path}")
        
        return {
            'model': model,
            'history': history,
            'best_recall': best_recall,
            'best_epoch': best_epoch,
            'model_path': model_path
        }
    
    def train_all_folds(
        self,
        training_data: Dict,
        output_dir: Path,
        folds: Optional[List[int]] = None
    ) -> Dict:
        """
        Train models on all folds (ensemble training).
        
        Parameters
        ----------
        training_data : Dict
            Training data dictionary
        output_dir : Path
            Directory to save models and results
        folds : List[int], optional
            List of fold IDs to train. If None, trains all folds (0, 1, 2)
        
        Returns
        -------
        Dict
            Results for all folds containing:
            - fold_results: dict mapping fold_id to fold results
            - ensemble_metrics: average metrics across folds
            - model_paths: list of paths to trained models
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if folds is None:
            folds = [0, 1, 2]
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"🧬 ENSEMBLE TRAINING - {len(folds)} FOLDS")
            print(f"{'='*80}")
            print(f"  Folds to train: {folds}")
            print(f"  Output directory: {output_dir}")
        
        fold_results = {}
        model_paths = []
        
        for fold_id in folds:
            if self.verbose:
                print(f"\n\n{'#'*80}")
                print(f"# FOLD {fold_id + 1}/{len(folds)}")
                print(f"{'#'*80}\n")
            
            fold_result = self.train_single_fold(
                training_data=training_data,
                fold_id=fold_id,
                output_dir=output_dir
            )
            
            fold_results[fold_id] = fold_result
            model_paths.append(str(fold_result['model_path']))
        
        # Compute ensemble metrics
        ensemble_metrics = self._compute_ensemble_metrics(fold_results)
        
        # Save ensemble summary
        self._save_ensemble_summary(fold_results, ensemble_metrics, output_dir)
        
        # Plot ensemble comparison
        self._plot_ensemble_comparison(fold_results, output_dir)
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"✅ ENSEMBLE TRAINING COMPLETE!")
            print(f"{'='*80}")
            print(f"\n📊 Ensemble Metrics:")
            for metric, value in ensemble_metrics.items():
                print(f"  {metric}: {value:.4f}")
            print(f"\n📁 Model paths:")
            for path in model_paths:
                print(f"  • {path}")
        
        return {
            'fold_results': fold_results,
            'ensemble_metrics': ensemble_metrics,
            'model_paths': model_paths
        }
    
    def _train_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer
    ) -> float:
        """Train for one epoch."""
        model.train()
        total_loss = 0.0
        n_batches = 0
        
        pbar = tqdm(loader, desc="Training", leave=False, disable=not self.verbose)
        
        for batch_X, batch_y in pbar:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
            if self.verbose:
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / n_batches
    
    def _validate_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module
    ) -> Tuple[float, Dict]:
        """Validate for one epoch."""
        model.eval()
        total_loss = 0.0
        n_batches = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            pbar = tqdm(loader, desc="Validating", leave=False, disable=not self.verbose)
            
            for batch_X, batch_y in pbar:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                total_loss += loss.item()
                n_batches += 1
                
                probs = F.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                all_preds.append(preds.cpu().numpy())
                all_labels.append(batch_y.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
        
        avg_loss = total_loss / n_batches
        
        # Compute metrics
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        all_probs = np.concatenate(all_probs)
        
        top1_acc = accuracy_score(all_labels, all_preds)
        top10_acc = top_k_accuracy_score(all_labels, all_probs, k=10)
        
        # Top 1% recall (main metric)
        top1_percent_k = max(1, int(0.01 * all_probs.shape[1]))
        recall = compound_level_topk_recall(all_labels, all_probs, top1_percent_k)
        
        return avg_loss, {
            'recall': recall,
            'top1_acc': top1_acc,
            'top10_acc': top10_acc
        }
    
    def _save_checkpoint(
        self,
        model: nn.Module,
        fold_id: int,
        epoch: int,
        history: Dict,
        n_genes: int,
        n_compounds: int,
        gene_names: list,
        compound_names: list,
        save_path: Path
    ):
        """Save model checkpoint in DrugReflector-compatible format."""
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'fold_id': fold_id,
            'epoch': epoch,
            'history': history,
            'dimensions': {
                'input_size': n_genes,
                'output_size': n_compounds,
                'input_names': list(gene_names),
                'output_names': list(compound_names)
            },
            'params_init': {
                'model_init_params': {
                    'torch_init_params': {
                        'hidden_dims': [1024, 2048],
                        'dropout_p': 0.64,
                        'activation': 'ReLU',
                        'batch_norm': True,
                        'order': 'act-drop-bn',
                        'final_layer_bias': True
                    }
                }
            },
            'training_config': {
                'initial_lr': self.initial_lr,
                'min_lr': self.min_lr,
                'weight_decay': self.weight_decay,
                't_0': self.t_0,
                'focal_gamma': self.focal_gamma,
                'batch_size': self.batch_size,
                'num_epochs': self.num_epochs
            }
        }
        
        torch.save(checkpoint, save_path)
    
    def _compute_ensemble_metrics(self, fold_results: Dict) -> Dict:
        """Compute average metrics across folds."""
        best_recalls = [r['best_recall'] for r in fold_results.values()]
        
        # Get final metrics from histories
        final_top1_accs = []
        final_top10_accs = []
        
        for result in fold_results.values():
            history = result['history']
            best_epoch = result['best_epoch']
            final_top1_accs.append(history['val_top1_acc'][best_epoch])
            final_top10_accs.append(history['val_top10_acc'][best_epoch])
        
        return {
            'mean_recall': np.mean(best_recalls),
            'std_recall': np.std(best_recalls),
            'mean_top1_acc': np.mean(final_top1_accs),
            'std_top1_acc': np.std(final_top1_accs),
            'mean_top10_acc': np.mean(final_top10_accs),
            'std_top10_acc': np.std(final_top10_accs)
        }
    
    def _save_ensemble_summary(
        self,
        fold_results: Dict,
        ensemble_metrics: Dict,
        output_dir: Path
    ):
        """Save ensemble training summary."""
        import pickle
        
        summary = {
            'fold_results': fold_results,
            'ensemble_metrics': ensemble_metrics,
            'training_config': {
                'initial_lr': self.initial_lr,
                'min_lr': self.min_lr,
                'weight_decay': self.weight_decay,
                't_0': self.t_0,
                'focal_gamma': self.focal_gamma,
                'batch_size': self.batch_size,
                'num_epochs': self.num_epochs
            }
        }
        
        summary_path = output_dir / 'ensemble_summary.pkl'
        with open(summary_path, 'wb') as f:
            pickle.dump(summary, f)
        
        if self.verbose:
            print(f"\n💾 Ensemble summary saved: {summary_path}")
    
    def _plot_ensemble_comparison(self, fold_results: Dict, output_dir: Path):
        """Plot comparison of all folds."""
        n_folds = len(fold_results)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        colors = ['#2E86AB', '#A23B72', '#F18F01']
        
        # Plot training curves for each fold
        for idx, (fold_id, result) in enumerate(fold_results.items()):
            history = result['history']
            epochs = range(1, len(history['train_loss']) + 1)
            color = colors[idx % len(colors)]
            
            # Loss curves
            axes[0, 0].plot(epochs, history['train_loss'], 
                          label=f'Fold {fold_id} (train)', 
                          color=color, alpha=0.7, linestyle='-')
            axes[0, 0].plot(epochs, history['val_loss'], 
                          label=f'Fold {fold_id} (val)', 
                          color=color, alpha=0.9, linestyle='--')
            
            # Recall
            axes[0, 1].plot(epochs, history['val_recall'], 
                          label=f'Fold {fold_id}', color=color)
            best_epoch = result['best_epoch']
            best_recall = result['best_recall']
            axes[0, 1].scatter([best_epoch + 1], [best_recall], 
                             color=color, s=100, zorder=5, marker='*')
            
            # Top-1 Accuracy
            axes[1, 0].plot(epochs, history['val_top1_acc'], 
                          label=f'Fold {fold_id}', color=color)
            
            # Top-10 Accuracy
            axes[1, 1].plot(epochs, history['val_top10_acc'], 
                          label=f'Fold {fold_id}', color=color)
        
        # Styling
        titles = [
            'Loss Curves',
            'Validation Recall (Top 1%)',
            'Top-1 Accuracy',
            'Top-10 Accuracy'
        ]
        
        for ax, title in zip(axes.flat, titles):
            ax.set_xlabel('Epoch', fontsize=11)
            ax.set_ylabel('Value', fontsize=11)
            ax.set_title(title, fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
        
        plt.suptitle('Ensemble Training - All Folds Comparison', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        plot_path = output_dir / 'ensemble_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        
        if self.verbose:
            print(f"📊 Ensemble comparison saved: {plot_path}")
        
        plt.close()

if __name__ == "__main__":
    main()