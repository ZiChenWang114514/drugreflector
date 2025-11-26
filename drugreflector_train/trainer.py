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

from models import nnFC
from dataset import LINCSDataset
from losses import FocalLoss
from preprocessing import clip_and_normalize_signature


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
    lr_scheduler : str
        Learning rate scheduler type ('step', 'exponential', or 'cosine')
    lr_decay_rate : float
        Learning rate decay rate (for step/exponential schedulers)
    lr_decay_epochs : List[int]
        Epochs at which to decay learning rate (for step scheduler)
    min_lr : float
        Minimum learning rate
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
    plot_dir : str
        Directory to save training plots
    """
    
    def __init__(
        self,
        device: str = 'auto',
        initial_lr: float = 0.0139,
        lr_scheduler: str = 'step',  # 'step', 'exponential', or 'cosine'
        lr_decay_rate: float = 0.1,
        lr_decay_epochs: List[int] = None,  # For step scheduler
        min_lr: float = 0.00001,
        weight_decay: float = 1e-5,
        focal_gamma: float = 2.0,
        batch_size: int = 256,
        num_epochs: int = 50,
        num_workers: int = 4,
        save_every: int = 10,
        plot_dir: str = 'training_plots',  # 新增：图表输出目录
        verbose: bool = True
    ):
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        self.initial_lr = initial_lr
        self.lr_scheduler = lr_scheduler
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_epochs = lr_decay_epochs if lr_decay_epochs else [30, 40]
        self.min_lr = min_lr
        self.weight_decay = weight_decay
        self.focal_gamma = focal_gamma
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_workers = num_workers
        self.save_every = save_every
        self.plot_dir = plot_dir
        self.verbose = verbose
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"🚀 DrugReflector Trainer Initialized")
            print(f"{'='*80}")
            print(f"  Device: {self.device}")
            print(f"  Initial LR: {self.initial_lr}")
            print(f"  LR Scheduler: {self.lr_scheduler}")
            if self.lr_scheduler == 'step':
                print(f"  LR Decay Rate: {self.lr_decay_rate}")
                print(f"  LR Decay Epochs: {self.lr_decay_epochs}")
            print(f"  Min LR: {self.min_lr}")
            print(f"  Weight Decay: {self.weight_decay}")
            print(f"  Focal γ: {self.focal_gamma}")
            print(f"  Batch size: {self.batch_size}")
            print(f"  Epochs: {self.num_epochs}")
            print(f"  Plot Directory: {self.plot_dir}")
    
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
    
    def create_scheduler(self, optimizer: torch.optim.Optimizer):
        """
        Create learning rate scheduler with simple decay strategy.
        
        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            Optimizer instance
        
        Returns
        -------
        torch.optim.lr_scheduler._LRScheduler
            Learning rate scheduler
        """
        if self.lr_scheduler == 'step':
            # Step decay: multiply LR by decay_rate at specified epochs
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=self.lr_decay_epochs,
                gamma=self.lr_decay_rate
            )
        elif self.lr_scheduler == 'exponential':
            # Exponential decay: LR = initial_lr * (decay_rate ^ epoch)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=self.lr_decay_rate
            )
        elif self.lr_scheduler == 'cosine':
            # Simple cosine annealing (no warm restarts)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.num_epochs,
                eta_min=self.min_lr
            )
        else:
            raise ValueError(f"Unknown scheduler: {self.lr_scheduler}")
        
        if self.verbose:
            print(f"  📊 Using {self.lr_scheduler} LR scheduler")
        
        return scheduler

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
        scheduler = self.create_scheduler(optimizer)
        
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

        # Plot training metrics
        if self.verbose:
            print(f"\n📊 Generating training plots...")
        self._plot_training_metrics(history, fold_id, output_dir)
                
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
        
    def _plot_training_metrics(
        self, 
        history: Dict, 
        fold_id: int, 
        output_dir: Path
    ):
        """
        Plot detailed training metrics for a single fold.
        
        Parameters
        ----------
        history : Dict
            Training history
        fold_id : int
            Fold identifier
        output_dir : Path
            Output directory for plots
        """
        plot_dir = output_dir / self.plot_dir
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        epochs = np.arange(1, len(history['train_loss']) + 1)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Loss curves (combined)
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(epochs, history['train_loss'], 
                label='Train Loss', color='#2E86AB', linewidth=2, alpha=0.8)
        ax1.plot(epochs, history['val_loss'], 
                label='Val Loss', color='#A23B72', linewidth=2, alpha=0.8)
        ax1.set_xlabel('Epoch', fontsize=11)
        ax1.set_ylabel('Loss', fontsize=11)
        ax1.set_title('Training and Validation Loss', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 2. Learning Rate
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.plot(epochs, history['learning_rates'], 
                color='#F18F01', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=11)
        ax2.set_ylabel('Learning Rate', fontsize=11)
        ax2.set_title('Learning Rate Schedule', fontsize=13, fontweight='bold')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        # 3. Top 1% Recall (main metric)
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(epochs, history['val_recall'], 
                color='#06A77D', linewidth=2.5, marker='o', 
                markersize=4, markevery=max(1, len(epochs)//20))
        best_epoch = np.argmax(history['val_recall'])
        best_recall = history['val_recall'][best_epoch]
        ax3.scatter([best_epoch + 1], [best_recall], 
                color='red', s=200, zorder=5, marker='*',
                label=f'Best: {best_recall:.4f}')
        ax3.set_xlabel('Epoch', fontsize=11)
        ax3.set_ylabel('Recall', fontsize=11)
        ax3.set_title('Top 1% Recall (Main Metric)', fontsize=13, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0, 1])
        
        # 4. Top-1 Accuracy
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(epochs, history['val_top1_acc'], 
                color='#D62839', linewidth=2, marker='s',
                markersize=3, markevery=max(1, len(epochs)//20))
        ax4.set_xlabel('Epoch', fontsize=11)
        ax4.set_ylabel('Accuracy', fontsize=11)
        ax4.set_title('Top-1 Accuracy', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim([0, max(history['val_top1_acc']) * 1.1])
        
        # 5. Top-10 Accuracy
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.plot(epochs, history['val_top10_acc'], 
                color='#8338EC', linewidth=2, marker='^',
                markersize=3, markevery=max(1, len(epochs)//20))
        ax5.set_xlabel('Epoch', fontsize=11)
        ax5.set_ylabel('Accuracy', fontsize=11)
        ax5.set_title('Top-10 Accuracy', fontsize=13, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        ax5.set_ylim([0, 1])
        
        # 6. Accuracy comparison
        ax6 = fig.add_subplot(gs[2, :2])
        ax6.plot(epochs, history['val_top1_acc'], 
                label='Top-1', color='#D62839', linewidth=2, alpha=0.8)
        ax6.plot(epochs, history['val_top10_acc'], 
                label='Top-10', color='#8338EC', linewidth=2, alpha=0.8)
        ax6.plot(epochs, history['val_recall'], 
                label='Top 1% Recall', color='#06A77D', linewidth=2, alpha=0.8)
        ax6.set_xlabel('Epoch', fontsize=11)
        ax6.set_ylabel('Score', fontsize=11)
        ax6.set_title('All Metrics Comparison', fontsize=13, fontweight='bold')
        ax6.legend(fontsize=10, loc='lower right')
        ax6.grid(True, alpha=0.3)
        ax6.set_ylim([0, 1])
        
        # 7. Epoch time
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.bar(epochs, history['epoch_times'], 
            color='#FB5607', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax7.axhline(y=np.mean(history['epoch_times']), 
                color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(history["epoch_times"]):.1f}s')
        ax7.set_xlabel('Epoch', fontsize=11)
        ax7.set_ylabel('Time (seconds)', fontsize=11)
        ax7.set_title('Epoch Training Time', fontsize=13, fontweight='bold')
        ax7.legend(fontsize=9)
        ax7.grid(True, alpha=0.3, axis='y')
        
        # Overall title
        fig.suptitle(f'Training Metrics - Fold {fold_id}', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        # Save plot
        plot_path = plot_dir / f'fold_{fold_id}_metrics.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        
        if self.verbose:
            print(f"  📊 Metrics plot saved: {plot_path}")
        
        plt.close()
        
        # Also save a summary statistics file
        self._save_metrics_summary(history, fold_id, plot_dir)

    def _save_metrics_summary(
        self,
        history: Dict,
        fold_id: int,
        plot_dir: Path
    ):
        """Save metrics summary as text file."""
        summary_path = plot_dir / f'fold_{fold_id}_summary.txt'
        
        with open(summary_path, 'w') as f:
            f.write(f"{'='*60}\n")
            f.write(f"Training Summary - Fold {fold_id}\n")
            f.write(f"{'='*60}\n\n")
            
            # Best metrics
            best_recall_epoch = np.argmax(history['val_recall'])
            f.write(f"Best Metrics:\n")
            f.write(f"  Top 1% Recall: {history['val_recall'][best_recall_epoch]:.4f} (Epoch {best_recall_epoch + 1})\n")
            f.write(f"  Top-1 Acc: {history['val_top1_acc'][best_recall_epoch]:.4f}\n")
            f.write(f"  Top-10 Acc: {history['val_top10_acc'][best_recall_epoch]:.4f}\n")
            f.write(f"  Val Loss: {history['val_loss'][best_recall_epoch]:.4f}\n\n")
            
            # Final metrics
            f.write(f"Final Metrics (Epoch {len(history['train_loss'])}):\n")
            f.write(f"  Top 1% Recall: {history['val_recall'][-1]:.4f}\n")
            f.write(f"  Top-1 Acc: {history['val_top1_acc'][-1]:.4f}\n")
            f.write(f"  Top-10 Acc: {history['val_top10_acc'][-1]:.4f}\n")
            f.write(f"  Train Loss: {history['train_loss'][-1]:.4f}\n")
            f.write(f"  Val Loss: {history['val_loss'][-1]:.4f}\n\n")
            
            # Training statistics
            f.write(f"Training Statistics:\n")
            f.write(f"  Total Epochs: {len(history['train_loss'])}\n")
            f.write(f"  Total Time: {sum(history['epoch_times']):.1f}s ({sum(history['epoch_times'])/60:.1f}m)\n")
            f.write(f"  Avg Epoch Time: {np.mean(history['epoch_times']):.1f}s\n")
            f.write(f"  Initial LR: {history['learning_rates'][0]:.6f}\n")
            f.write(f"  Final LR: {history['learning_rates'][-1]:.6f}\n")
        
        if self.verbose:
            print(f"  📄 Summary saved: {summary_path}")