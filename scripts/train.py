"""
DrugReflector Training Script - Single Fold Quick Test

This script trains a single DrugReflector model on one fold for quick testing.
Based on the training module from drugreflector_training/ but simplified for rapid iteration.

Usage:
    python train.py --data-file processed_data/training_data_lincs2020_final.pkl --output-dir models/test_fold0 --fold 0
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
from sklearn.metrics import accuracy_score, top_k_accuracy_score
import matplotlib.pyplot as plt

# Add parent directory to path
try:
    sys.path.append(str(Path(__file__).parent.parent))
except NameError:
    import os
    sys.path.append(str(Path(os.getcwd()).parent))

from drugreflector.models import nnFC
from drugreflector_training import (
    LINCSDataset,
    FocalLoss,
    clip_and_normalize_signature
)

def compound_level_topk_recall(labels, probs, k):
    """
    严格按照论文定义计算 top k recall：
    1) 对每个 observation 判断是否命中 top k；
    2) 在每个 compound 内部对上述 0/1 指标平均；
    3) 再在所有 compounds 上平均。
    """
    labels = np.asarray(labels)
    probs  = np.asarray(probs)
    n_classes = probs.shape[1]
    k = max(1, min(k, n_classes))

    # 每个样本的 top-k 命中情况
    topk_pred = np.argpartition(-probs, kth=k-1, axis=1)[:, :k]   # (n_samples, k)
    hit_per_sample = (topk_pred == labels[:, None]).any(axis=1).astype(float)

    # 按化合物聚合
    compound_hits = {}
    compound_counts = {}
    for y, hit in zip(labels, hit_per_sample):
        compound_hits.setdefault(y, 0.0)
        compound_counts.setdefault(y, 0)
        compound_hits[y] += hit
        compound_counts[y] += 1

    # 化合物级别平均
    compound_recalls = []
    for cid in compound_hits:
        compound_recalls.append(compound_hits[cid] / compound_counts[cid])

    return float(np.mean(compound_recalls))

class SingleFoldTrainer:
    """
    Simplified trainer for single fold testing.
    
    Parameters from SI Table S5 and Page 3:
    - Initial LR: 0.0139
    - Min LR: 0.00001
    - Weight Decay: 1e-5
    - Dropout: 0.64
    - Focal γ: 2.0
    - T_0: 20
    """
    
    def __init__(
        self,
        device='auto',
        initial_lr=0.0139,
        min_lr=0.00001,
        weight_decay=1e-5,
        t_0=20,
        focal_gamma=2.0,
        batch_size=256,
        num_epochs=50,
        num_workers=4,
        save_every=10
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
        
        print(f"\n{'='*80}")
        print(f"🚀 Single Fold Trainer Initialized")
        print(f"{'='*80}")
        print(f"  Device: {self.device}")
        print(f"  Initial LR: {self.initial_lr}")
        print(f"  Focal γ: {self.focal_gamma}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Epochs: {self.num_epochs}")
        print(f"  Workers: {self.num_workers}")
        
    def create_model(self, input_size: int, output_size: int) -> nn.Module:
        """Create nnFC model with paper-specified architecture."""
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
    
    def train(
        self,
        training_data: Dict,
        fold_id: int,
        output_dir: Path
    ) -> Dict:
        """
        Train model on specified fold.
        
        Parameters
        ----------
        training_data : Dict
            Training data with keys: X, y, folds, compound_names, gene_names, metadata
        fold_id : int
            Which fold to use as validation (0, 1, or 2)
        output_dir : Path
            Directory to save models and results
        
        Returns
        -------
        Dict
            Training history and results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract data
        X = training_data['X']
        y = training_data['y']
        folds = training_data['folds']
        compound_names = training_data['compound_names']
        gene_names = training_data.get('gene_names', [f'gene_{i}' for i in range(X.shape[1])])
        
        n_samples = len(X)
        n_genes = X.shape[1]
        n_compounds = len(compound_names)
        
        print(f"\n{'='*80}")
        print(f"📊 Dataset Summary")
        print(f"{'='*80}")
        print(f"  Total samples: {n_samples:,}")
        print(f"  Total compounds: {n_compounds:,}")
        print(f"  Gene features: {n_genes}")
        print(f"  Training fold: {fold_id}")
        
        # Preprocess data (clip to [-2, 2] with std=1)
        print(f"\n🔧 Preprocessing signatures...")
        X_processed = clip_and_normalize_signature(X)
        
        # Create train/val split
        val_mask = folds == fold_id
        train_mask = ~val_mask
        
        n_train = train_mask.sum()
        n_val = val_mask.sum()
        
        print(f"\n📋 Data Split:")
        print(f"  Training samples: {n_train:,} ({n_train/n_samples*100:.1f}%)")
        print(f"  Validation samples: {n_val:,} ({n_val/n_samples*100:.1f}%)")
        
        # Training compounds
        train_compounds = training_data['sample_meta'][train_mask]['pert_id'].nunique()
        val_compounds = training_data['sample_meta'][val_mask]['pert_id'].nunique()
        print(f"  Training compounds: {train_compounds:,}")
        print(f"  Validation compounds: {val_compounds:,}")
        
        # Create datasets
        train_dataset = LINCSDataset(X_processed, y, train_mask)
        val_dataset = LINCSDataset(X_processed, y, val_mask)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True if self.device == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True if self.device == 'cuda' else False
        )
        
        print(f"\n  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        
        # Create model
        print(f"\n🏗️  Building model...")
        model = self.create_model(n_genes, n_compounds).to(self.device)
        
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Architecture: {n_genes} → 1024 → 2048 → {n_compounds}")
        print(f"  Parameters: {n_params:,}")
        
        # Loss, optimizer, scheduler
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
        print(f"\n{'='*80}")
        print(f"🎯 Starting Training")
        print(f"{'='*80}")
        
        import time
        
        for epoch in range(self.num_epochs):
            epoch_start = time.time()
            
            # ===== Training =====
            model.train()
            train_loss = 0.0
            train_batches = 0
            
            pbar = tqdm(
                train_loader, 
                desc=f"Epoch {epoch+1}/{self.num_epochs}",
                leave=False
            )
            
            for batch_X, batch_y in pbar:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_batches += 1
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
                })
            
            avg_train_loss = train_loss / train_batches
            
            # ===== Validation =====
            model.eval()
            val_loss = 0.0
            val_batches = 0
            all_preds = []
            all_labels = []
            all_probs = []
            
            with torch.no_grad():
                for batch_X, batch_y in tqdm(val_loader, desc="Validating", leave=False):
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    val_batches += 1
                    
                    probs = F.softmax(outputs, dim=1)
                    preds = torch.argmax(probs, dim=1)
                    
                    all_preds.append(preds.cpu().numpy())
                    all_labels.append(batch_y.cpu().numpy())
                    all_probs.append(probs.cpu().numpy())
            
            avg_val_loss = val_loss / val_batches
            
            # Compute metrics
            all_preds = np.concatenate(all_preds)
            all_labels = np.concatenate(all_labels)
            all_probs = np.concatenate(all_probs)
            
            top1_acc = accuracy_score(all_labels, all_preds)
            top10_acc = top_k_accuracy_score(all_labels, all_probs, k=10)
            
            # Top 1% recall (main metric from paper)
            top1_percent_k = max(1, int(0.01 * all_probs.shape[1]))
            recall = compound_level_topk_recall(all_labels, all_probs, top1_percent_k)
            
            # Record history
            current_lr = optimizer.param_groups[0]['lr']
            epoch_time = time.time() - epoch_start
            
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['val_recall'].append(recall)
            history['val_top1_acc'].append(top1_acc)
            history['val_top10_acc'].append(top10_acc)
            history['learning_rates'].append(current_lr)
            history['epoch_times'].append(epoch_time)
            
            # Print progress
            print(f"\n{'='*80}")
            print(f"Epoch {epoch+1}/{self.num_epochs} ({epoch_time:.1f}s)")
            print(f"{'='*80}")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Loss: {avg_val_loss:.4f}")
            print(f"  Val Top 1% Recall: {recall:.4f} ⭐")
            print(f"  Val Top-1 Acc: {top1_acc:.4f}")
            print(f"  Val Top-10 Acc: {top10_acc:.4f}")
            print(f"  Learning Rate: {current_lr:.6f}")
            
            # Save best model
            if recall > best_recall:
                best_recall = recall
                best_epoch = epoch
                best_model_state = model.state_dict().copy()
                print(f"  ✅ New best model! (Best recall: {best_recall:.4f})")
            
            # Periodic checkpoint save
            if (epoch + 1) % self.save_every == 0:
                checkpoint_path = output_dir / f"checkpoint_epoch_{epoch+1}.pt"
                self._save_checkpoint(
                    model, fold_id, epoch, history,
                    n_genes, n_compounds, gene_names, compound_names,
                    checkpoint_path
                )
                print(f"  💾 Checkpoint saved: {checkpoint_path.name}")
            
            # Learning rate step
            scheduler.step()
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"\n{'='*80}")
            print(f"✅ Training Complete!")
            print(f"{'='*80}")
            print(f"  Best epoch: {best_epoch + 1}/{self.num_epochs}")
            print(f"  Best recall: {best_recall:.4f}")
            print(f"  Total time: {sum(history['epoch_times']):.1f}s ({sum(history['epoch_times'])/60:.1f}m)")
        
        # Save final model
        model_path = output_dir / f"model_fold_{fold_id}.pt"
        self._save_checkpoint(
            model, fold_id, self.num_epochs - 1, history,
            n_genes, n_compounds, gene_names, compound_names,
            model_path
        )
        print(f"\n💾 Final model saved: {model_path}")
        
        # Save training history
        history_path = output_dir / f"training_history_fold_{fold_id}.pkl"
        with open(history_path, 'wb') as f:
            pickle.dump(history, f)
        print(f"💾 Training history saved: {history_path}")
        
        # Plot training curves
        self._plot_training_curves(history, fold_id, output_dir)
        
        return {
            'model': model,
            'history': history,
            'best_recall': best_recall,
            'best_epoch': best_epoch
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
        """Save model checkpoint in format compatible with DrugReflector."""
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
    
    def _plot_training_curves(
        self,
        history: Dict,
        fold_id: int,
        output_dir: Path
    ):
        """Plot training curves."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        metrics = [
            ('train_loss', 'Training Loss', 'Loss'),
            ('val_loss', 'Validation Loss', 'Loss'),
            ('val_recall', 'Validation Recall (Top 1%)', 'Recall'),
            ('val_top1_acc', 'Top-1 Accuracy', 'Accuracy'),
            ('val_top10_acc', 'Top-10 Accuracy', 'Accuracy'),
            ('learning_rates', 'Learning Rate Schedule', 'LR')
        ]
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        for idx, (metric, title, ylabel) in enumerate(metrics):
            ax = axes[idx // 3, idx % 3]
            
            if metric in history:
                ax.plot(epochs, history[metric], linewidth=2, color='#2E86AB')
                
                # Mark best epoch for recall
                if metric == 'val_recall':
                    best_idx = np.argmax(history[metric])
                    best_val = history[metric][best_idx]
                    ax.scatter([best_idx + 1], [best_val], 
                             color='red', s=100, zorder=5, marker='*')
                    ax.annotate(f'Best: {best_val:.4f}',
                              xy=(best_idx + 1, best_val),
                              xytext=(10, 10), textcoords='offset points',
                              bbox=dict(boxstyle='round', fc='yellow', alpha=0.7),
                              fontsize=9)
            
            ax.set_xlabel('Epoch', fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(title, fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Training History - Fold {fold_id}', 
                    fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        plot_path = output_dir / f'training_curves_fold_{fold_id}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Training curves saved: {plot_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Train single DrugReflector model on one fold (quick test)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
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
    
    parser.add_argument(
        '--fold',
        type=int,
        default=0,
        choices=[0, 1, 2],
        help='Which fold to use as validation (0, 1, or 2)'
    )
    
    # Training hyperparameters
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
        help='Minimum learning rate'
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
        help='CosineAnnealing T_0 parameter'
    )
    
    # System arguments
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
    
    args = parser.parse_args()
    
    # Banner
    print(f"\n{'='*80}")
    print(f"🧬 DRUGREFLECTOR - SINGLE FOLD TRAINING")
    print(f"{'='*80}")
    print(f"\n📋 Configuration:")
    print(f"  Data file: {args.data_file}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Fold: {args.fold}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Initial LR: {args.learning_rate}")
    print(f"  Focal γ: {args.focal_gamma}")
    print(f"  Device: {args.device}")
    
    # Load data
    print(f"\n{'='*80}")
    print(f"📂 Loading Training Data")
    print(f"{'='*80}")
    
    data_file = Path(args.data_file)
    if not data_file.exists():
        print(f"❌ Data file not found: {data_file}")
        sys.exit(1)
    
    print(f"  Loading from: {data_file}")
    with open(data_file, 'rb') as f:
        training_data = pickle.load(f)
    
    print(f"  ✓ Data loaded successfully")
    print(f"  Keys: {list(training_data.keys())}")
    
    # Validate data format
    required_keys = ['X', 'y', 'folds', 'compound_names']
    missing_keys = [k for k in required_keys if k not in training_data]
    if missing_keys:
        print(f"❌ Missing required keys: {missing_keys}")
        sys.exit(1)
    
    # Create trainer
    trainer = SingleFoldTrainer(
        device=args.device,
        initial_lr=args.learning_rate,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        t_0=args.t0,
        focal_gamma=args.focal_gamma,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        num_workers=args.num_workers,
        save_every=args.save_every
    )
    
    # Train
    output_dir = Path(args.output_dir)
    results = trainer.train(
        training_data=training_data,
        fold_id=args.fold,
        output_dir=output_dir
    )
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"✅ TRAINING COMPLETE!")
    print(f"{'='*80}")
    print(f"  Best recall: {results['best_recall']:.4f}")
    print(f"  Best epoch: {results['best_epoch'] + 1}/{args.epochs}")
    print(f"  Output directory: {output_dir}")
    print(f"\n📁 Generated files:")
    print(f"  • model_fold_{args.fold}.pt - Final trained model")
    print(f"  • training_history_fold_{args.fold}.pkl - Training metrics")
    print(f"  • training_curves_fold_{args.fold}.png - Visualization")
    print(f"  • checkpoint_epoch_*.pt - Periodic checkpoints")
    
    print(f"\n🎯 Next steps:")
    print(f"  1. Review training curves: training_curves_fold_{args.fold}.png")
    print(f"  2. Load model for inference:")
    print(f"     from drugreflector import DrugReflector")
    print(f"     model = DrugReflector(checkpoint_paths=['{output_dir}/model_fold_{args.fold}.pt'])")
    print(f"  3. Make predictions:")
    print(f"     predictions = model.predict(vscores, n_top=100)")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()