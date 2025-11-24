"""
Two-Tower DrugReflector Trainer - Core Training Engine

Supports:
1. Two-Tower model with transcriptome + molecular embeddings
2. Baseline transcript-only model for comparison
3. Single-fold and multi-fold (ensemble) training

Based on Science 2025 paper and SI Table S5.
"""
import time
import pickle
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

from models import TwoTowerModel, TranscriptOnlyModel
from dataset import TwoTowerDataset, TranscriptOnlyDataset
from losses import FocalLoss, TwoTowerLoss
from preprocessing import clip_and_normalize_signature, normalize_mol_embeddings


def compound_level_topk_recall(labels, probs, k):
    """
    Compound-level top-k recall as defined in the paper.
    """
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


class TwoTowerTrainer:
    """
    Two-Tower DrugReflector model trainer.
    
    Parameters
    ----------
    device : str
        Device for training ('auto', 'cuda', or 'cpu')
    embedding_dim : int
        Embedding dimension for each tower
    fusion_type : str
        Fusion strategy ('concat', 'product', 'attention', 'gated')
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
    contrastive_weight : float
        Weight for cross-modal contrastive loss (0 to disable)
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
        embedding_dim: int = 512,
        fusion_type: str = 'concat',
        initial_lr: float = 0.0139,
        min_lr: float = 0.00001,
        weight_decay: float = 1e-5,
        t_0: int = 20,
        focal_gamma: float = 2.0,
        contrastive_weight: float = 0.1,
        batch_size: int = 256,
        num_epochs: int = 50,
        num_workers: int = 4,
        save_every: int = 10,
        verbose: bool = True,
        unimol_dim: int = 512,
    ):
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        self.embedding_dim = embedding_dim
        self.fusion_type = fusion_type
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.weight_decay = weight_decay
        self.t_0 = t_0
        self.focal_gamma = focal_gamma
        self.contrastive_weight = contrastive_weight
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_workers = num_workers
        self.save_every = save_every
        self.verbose = verbose
        self.unimol_dim = unimol_dim
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"🚀 Two-Tower Trainer Initialized")
            print(f"{'='*80}")
            print(f"  Device: {self.device}")
            print(f"  Embedding dim: {self.embedding_dim}")
            print(f"  Fusion type: {self.fusion_type}")
            print(f"  Initial LR: {self.initial_lr}")
            print(f"  Focal γ: {self.focal_gamma}")
            print(f"  Contrastive weight: {self.contrastive_weight}")
            print(f"  Batch size: {self.batch_size}")
            print(f"  Epochs: {self.num_epochs}")
    
    def create_model(self, n_genes: int, n_compounds: int) -> nn.Module:
        """Create Two-Tower model."""
        model = TwoTowerModel(
            n_genes=n_genes,
            n_compounds=n_compounds,
            embedding_dim=self.embedding_dim,
            fusion_type=self.fusion_type,
            transcript_hidden_dims=[1024, 2048],
            mol_hidden_dims=[1024],
            classifier_hidden_dims=[2048, 1024],
            transcript_dropout=0.64,
            mol_dropout=0.3,
            classifier_dropout=0.3,
            unimol_dim=self.unimol_dim,
        )
        return model
    
    def train_single_fold(
        self,
        training_data: Dict,
        fold_id: int,
        output_dir: Path,
    ) -> Dict:
        """
        Train Two-Tower model on a single fold.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract data
        X = training_data['X']
        y = training_data['y']
        folds = training_data['folds']
        mol_embeddings = training_data['mol_embeddings']
        compound_names = training_data['compound_names']
        gene_names = training_data.get('gene_names', [f'gene_{i}' for i in range(X.shape[1])])
        
        n_samples = len(X)
        n_genes = X.shape[1]
        n_compounds = len(compound_names)
        mol_dim = mol_embeddings.shape[1]
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"📊 Training Two-Tower Model - Fold {fold_id}")
            print(f"{'='*80}")
            print(f"  Total samples: {n_samples:,}")
            print(f"  Total compounds: {n_compounds:,}")
            print(f"  Gene features: {n_genes}")
            print(f"  Molecular embedding dim: {mol_dim}")
        
        # Preprocess data
        if self.verbose:
            print(f"\n🔧 Preprocessing...")
        X_processed = clip_and_normalize_signature(X)
        mol_processed = normalize_mol_embeddings(mol_embeddings, method='l2')
        
        # Create train/val split
        val_mask = folds == fold_id
        train_mask = ~val_mask
        
        n_train = train_mask.sum()
        n_val = val_mask.sum()
        
        if self.verbose:
            print(f"\n📋 Data Split:")
            print(f"  Training samples: {n_train:,}")
            print(f"  Validation samples: {n_val:,}")
        
        # Create datasets
        train_dataset = TwoTowerDataset(X_processed, y, mol_processed, train_mask)
        val_dataset = TwoTowerDataset(X_processed, y, mol_processed, val_mask)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=(self.device == 'cuda'),
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=(self.device == 'cuda'),
        )
        
        # Create model
        if self.verbose:
            print(f"\n🏗️ Building Two-Tower model...")
        model = self.create_model(n_genes, n_compounds).to(self.device)
        
        if self.verbose:
            n_params = sum(p.numel() for p in model.parameters())
            print(f"  Parameters: {n_params:,}")
        
        # Setup training
        criterion = TwoTowerLoss(
            focal_gamma=self.focal_gamma,
            contrastive_weight=self.contrastive_weight,
        )
        
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.initial_lr,
            weight_decay=self.weight_decay,
        )
        
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.t_0,
            T_mult=1,
            eta_min=self.min_lr,
        )
        
        # Training history
        history = {
            'train_loss': [],
            'train_focal': [],
            'train_contrastive': [],
            'val_loss': [],
            'val_recall': [],
            'val_top1_acc': [],
            'val_top10_acc': [],
            'learning_rates': [],
            'epoch_times': [],
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
            train_metrics = self._train_epoch(model, train_loader, criterion, optimizer)
            
            # Validation phase
            val_loss, val_metrics = self._validate_epoch(model, val_loader, criterion)
            
            # Record history
            current_lr = optimizer.param_groups[0]['lr']
            epoch_time = time.time() - epoch_start
            
            history['train_loss'].append(train_metrics['total'])
            history['train_focal'].append(train_metrics['focal'])
            history['train_contrastive'].append(train_metrics['contrastive'])
            history['val_loss'].append(val_loss)
            history['val_recall'].append(val_metrics['recall'])
            history['val_top1_acc'].append(val_metrics['top1_acc'])
            history['val_top10_acc'].append(val_metrics['top10_acc'])
            history['learning_rates'].append(current_lr)
            history['epoch_times'].append(epoch_time)
            
            # Print progress
            if self.verbose:
                print(f"\nEpoch {epoch+1}/{self.num_epochs} ({epoch_time:.1f}s)")
                print(f"  Train Loss: {train_metrics['total']:.4f} "
                      f"(focal: {train_metrics['focal']:.4f}, "
                      f"contrastive: {train_metrics['contrastive']:.4f})")
                print(f"  Val Loss: {val_loss:.4f}")
                print(f"  Val Top 1% Recall: {val_metrics['recall']:.4f} ⭐")
                print(f"  Val Top-1 Acc: {val_metrics['top1_acc']:.4f}")
                print(f"  Val Top-10 Acc: {val_metrics['top10_acc']:.4f}")
                print(f"  LR: {current_lr:.6f}")
            
            # Save best model
            if val_metrics['recall'] > best_recall:
                best_recall = val_metrics['recall']
                best_epoch = epoch
                best_model_state = model.state_dict().copy()
                if self.verbose:
                    print(f"  ✅ New best model!")
            
            # Periodic checkpoint
            if (epoch + 1) % self.save_every == 0:
                checkpoint_path = output_dir / f"checkpoint_fold{fold_id}_epoch{epoch+1}.pt"
                self._save_checkpoint(
                    model, fold_id, epoch, history,
                    n_genes, n_compounds, mol_dim,
                    gene_names, compound_names,
                    checkpoint_path,
                )
            
            # Update learning rate
            scheduler.step()
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            if self.verbose:
                print(f"\n{'='*80}")
                print(f"✅ Training Complete!")
                print(f"{'='*80}")
                print(f"  Best epoch: {best_epoch + 1}")
                print(f"  Best recall: {best_recall:.4f}")
        
        # Save final model
        model_path = output_dir / f"twotower_fold_{fold_id}.pt"
        self._save_checkpoint(
            model, fold_id, self.num_epochs - 1, history,
            n_genes, n_compounds, mol_dim,
            gene_names, compound_names,
            model_path,
        )
        
        # Plot training curves
        self._plot_training_curves(history, output_dir, fold_id)
        
        return {
            'model': model,
            'history': history,
            'best_recall': best_recall,
            'best_epoch': best_epoch,
            'model_path': model_path,
        }
    
    def train_all_folds(
        self,
        training_data: Dict,
        output_dir: Path,
        folds: Optional[List[int]] = None,
    ) -> Dict:
        """Train models on all folds (ensemble training)."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if folds is None:
            folds = [0, 1, 2]
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"🧬 ENSEMBLE TRAINING - {len(folds)} FOLDS")
            print(f"{'='*80}")
        
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
                output_dir=output_dir,
            )
            
            fold_results[fold_id] = fold_result
            model_paths.append(str(fold_result['model_path']))
        
        # Compute ensemble metrics
        ensemble_metrics = self._compute_ensemble_metrics(fold_results)
        
        # Save ensemble summary
        self._save_ensemble_summary(fold_results, ensemble_metrics, output_dir)
        
        # Plot ensemble comparison
        self._plot_ensemble_comparison(fold_results, output_dir)
        
        return {
            'fold_results': fold_results,
            'ensemble_metrics': ensemble_metrics,
            'model_paths': model_paths,
        }
    
    def _train_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        criterion: TwoTowerLoss,
        optimizer: torch.optim.Optimizer,
    ) -> Dict:
        """Train for one epoch."""
        model.train()
        total_loss = 0.0
        total_focal = 0.0
        total_contrastive = 0.0
        n_batches = 0
        
        pbar = tqdm(loader, desc="Training", leave=False, disable=not self.verbose)
        
        for batch_X, batch_mol, batch_y in pbar:
            batch_X = batch_X.to(self.device)
            batch_mol = batch_mol.to(self.device)
            batch_y = batch_y.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(batch_X, batch_mol)
            
            # Get embeddings for contrastive loss
            h_transcript, h_mol, _ = model.get_embeddings(batch_X, batch_mol)
            
            # Compute loss
            losses = criterion(logits, batch_y, h_transcript, h_mol)
            losses['total'].backward()
            optimizer.step()
            
            total_loss += losses['total'].item()
            total_focal += losses['focal'].item()
            total_contrastive += losses['contrastive'].item()
            n_batches += 1
            
            if self.verbose:
                pbar.set_postfix({'loss': f'{losses["total"].item():.4f}'})
        
        return {
            'total': total_loss / n_batches,
            'focal': total_focal / n_batches,
            'contrastive': total_contrastive / n_batches,
        }
    
    def _validate_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        criterion: TwoTowerLoss,
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
            
            for batch_X, batch_mol, batch_y in pbar:
                batch_X = batch_X.to(self.device)
                batch_mol = batch_mol.to(self.device)
                batch_y = batch_y.to(self.device)
                
                logits = model(batch_X, batch_mol)
                losses = criterion(logits, batch_y)
                total_loss += losses['total'].item()
                n_batches += 1
                
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                all_preds.append(preds.cpu().numpy())
                all_labels.append(batch_y.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
        
        avg_loss = total_loss / n_batches
        
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        all_probs = np.concatenate(all_probs)
        
        top1_acc = accuracy_score(all_labels, all_preds)
        top10_acc = top_k_accuracy_score(all_labels, all_probs, k=min(10, all_probs.shape[1]))
        
        top1_percent_k = max(1, int(0.01 * all_probs.shape[1]))
        recall = compound_level_topk_recall(all_labels, all_probs, top1_percent_k)
        
        return avg_loss, {
            'recall': recall,
            'top1_acc': top1_acc,
            'top10_acc': top10_acc,
        }
    
    def _save_checkpoint(
        self,
        model: nn.Module,
        fold_id: int,
        epoch: int,
        history: Dict,
        n_genes: int,
        n_compounds: int,
        mol_dim: int,
        gene_names: list,
        compound_names: list,
        save_path: Path,
    ):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'fold_id': fold_id,
            'epoch': epoch,
            'history': history,
            'dimensions': {
                'input_size': n_genes,
                'output_size': n_compounds,
                'mol_embedding_dim': mol_dim,
                'input_names': list(gene_names),
                'output_names': list(compound_names),
            },
            'model_config': {
                'embedding_dim': self.embedding_dim,
                'fusion_type': self.fusion_type,
                'transcript_hidden_dims': [1024, 2048],
                'mol_hidden_dims': [1024],
                'classifier_hidden_dims': [2048, 1024],
                'transcript_dropout': 0.64,
                'mol_dropout': 0.3,
                'classifier_dropout': 0.3,
                'unimol_dim': self.unimol_dim,
            },
            'training_config': {
                'initial_lr': self.initial_lr,
                'min_lr': self.min_lr,
                'weight_decay': self.weight_decay,
                't_0': self.t_0,
                'focal_gamma': self.focal_gamma,
                'contrastive_weight': self.contrastive_weight,
                'batch_size': self.batch_size,
                'num_epochs': self.num_epochs,
            },
        }
        
        torch.save(checkpoint, save_path)
    
    def _compute_ensemble_metrics(self, fold_results: Dict) -> Dict:
        """Compute average metrics across folds."""
        best_recalls = [r['best_recall'] for r in fold_results.values()]
        
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
            'std_top10_acc': np.std(final_top10_accs),
        }
    
    def _save_ensemble_summary(
        self,
        fold_results: Dict,
        ensemble_metrics: Dict,
        output_dir: Path,
    ):
        """Save ensemble training summary."""
        summary = {
            'fold_results': {k: {kk: vv for kk, vv in v.items() if kk != 'model'} 
                           for k, v in fold_results.items()},
            'ensemble_metrics': ensemble_metrics,
            'training_config': {
                'embedding_dim': self.embedding_dim,
                'fusion_type': self.fusion_type,
                'initial_lr': self.initial_lr,
                'focal_gamma': self.focal_gamma,
                'contrastive_weight': self.contrastive_weight,
                'batch_size': self.batch_size,
                'num_epochs': self.num_epochs,
            },
        }
        
        summary_path = output_dir / 'ensemble_summary.pkl'
        with open(summary_path, 'wb') as f:
            pickle.dump(summary, f)
        
        if self.verbose:
            print(f"\n💾 Ensemble summary saved: {summary_path}")
    
    def _plot_training_curves(self, history: Dict, output_dir: Path, fold_id: int):
        """Plot training curves for a single fold."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss
        axes[0, 0].plot(epochs, history['train_loss'], label='Train', color='#2E86AB')
        axes[0, 0].plot(epochs, history['val_loss'], label='Val', color='#A23B72')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Recall
        axes[0, 1].plot(epochs, history['val_recall'], color='#F18F01')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Recall')
        axes[0, 1].set_title('Validation Top 1% Recall')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Accuracy
        axes[1, 0].plot(epochs, history['val_top1_acc'], label='Top-1', color='#2E86AB')
        axes[1, 0].plot(epochs, history['val_top10_acc'], label='Top-10', color='#A23B72')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_title('Validation Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Loss components
        axes[1, 1].plot(epochs, history['train_focal'], label='Focal', color='#2E86AB')
        axes[1, 1].plot(epochs, history['train_contrastive'], label='Contrastive', color='#A23B72')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('Loss Components')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Two-Tower Training - Fold {fold_id}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        plot_path = output_dir / f'training_curves_fold_{fold_id}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        if self.verbose:
            print(f"📊 Training curves saved: {plot_path}")
    
    def _plot_ensemble_comparison(self, fold_results: Dict, output_dir: Path):
        """Plot comparison of all folds."""
        n_folds = len(fold_results)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        colors = ['#2E86AB', '#A23B72', '#F18F01']
        
        for idx, (fold_id, result) in enumerate(fold_results.items()):
            history = result['history']
            epochs = range(1, len(history['train_loss']) + 1)
            color = colors[idx % len(colors)]
            
            axes[0, 0].plot(epochs, history['train_loss'], 
                          label=f'Fold {fold_id} (train)', 
                          color=color, alpha=0.7, linestyle='-')
            axes[0, 0].plot(epochs, history['val_loss'], 
                          label=f'Fold {fold_id} (val)', 
                          color=color, alpha=0.9, linestyle='--')
            
            axes[0, 1].plot(epochs, history['val_recall'], 
                          label=f'Fold {fold_id}', color=color)
            best_epoch = result['best_epoch']
            best_recall = result['best_recall']
            axes[0, 1].scatter([best_epoch + 1], [best_recall], 
                             color=color, s=100, zorder=5, marker='*')
            
            axes[1, 0].plot(epochs, history['val_top1_acc'], 
                          label=f'Fold {fold_id}', color=color)
            
            axes[1, 1].plot(epochs, history['val_top10_acc'], 
                          label=f'Fold {fold_id}', color=color)
        
        titles = ['Loss Curves', 'Validation Recall (Top 1%)', 
                  'Top-1 Accuracy', 'Top-10 Accuracy']
        
        for ax, title in zip(axes.flat, titles):
            ax.set_xlabel('Epoch')
            ax.set_title(title, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.suptitle('Two-Tower Ensemble Training', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        plot_path = output_dir / 'ensemble_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        if self.verbose:
            print(f"📊 Ensemble comparison saved: {plot_path}")
