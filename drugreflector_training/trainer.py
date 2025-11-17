"""
Trainer for DrugReflector ensemble models.

Implements the 3-fold ensemble training strategy described in Science 2025 SI.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
from sklearn.metrics import accuracy_score, top_k_accuracy_score
import pickle

# Import from drugreflector package
import sys
sys.path.append('..')
from drugreflector.models import nnFC

from .dataset import LINCSDataset
from .losses import FocalLoss
from .preprocessing import clip_and_normalize_signature


class DrugReflectorTrainer:
    """
    Trainer for DrugReflector 3-fold ensemble.
    
    Implements the training procedure from Science 2025 SI Pages 2-4.
    
    Parameters
    ----------
    device : str, default='auto'
        Computation device
    initial_lr : float, default=0.0139
        Initial learning rate (from SI Table S5)
    min_lr : float, default=0.00001
        Minimum learning rate (from SI Page 3)
    weight_decay : float, default=1e-5
        L2 regularization (from SI Table S5)
    t_0 : int, default=20
        Epochs before first restart (from SI Table S5)
    t_mult : int, default=1
        Restart period multiplier
    focal_gamma : float, default=2.0
        Focal loss focusing parameter (from SI Page 3)
    batch_size : int, default=256
        Training batch size
    num_epochs : int, default=50
        Total training epochs (from SI Page 3)
    """
    
    def __init__(
        self,
        device='auto',
        initial_lr=0.0139,
        min_lr=0.00001,
        weight_decay=1e-5,
        t_0=20,
        t_mult=1,
        focal_gamma=2.0,
        batch_size=256,
        num_epochs=50
    ):
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.weight_decay = weight_decay
        self.t_0 = t_0
        self.t_mult = t_mult
        self.focal_gamma = focal_gamma
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        
        print(f"\n🚀 DrugReflector Trainer initialized")
        print(f"   Device: {self.device}")
        print(f"   Initial LR: {self.initial_lr}")
        print(f"   Focal γ: {self.focal_gamma}")
        print(f"   Batch size: {self.batch_size}")
        print(f"   Epochs: {self.num_epochs}")
    
    def _create_model(self, input_size: int, output_size: int) -> nn.Module:
        """
        Create model using the existing nnFC architecture.
        
        Architecture matches SI Page 2:
        - Input: 978 (landmark genes)
        - Hidden 1: 1024 nodes
        - Hidden 2: 2048 nodes  
        - Output: 9597 (compounds)
        - Dropout: 0.64 (from SI Table S5)
        """
        model = nnFC(
            input_dim=input_size,
            output_dim=output_size,
            hidden_dims=[1024, 2048],  # SI Page 2
            dropout_p=0.64,  # SI Table S5
            activation='ReLU',
            batch_norm=True,
            order='act-drop-bn',  # Standard order
            final_layer_bias=True
        )
        
        return model
    
    def train_single_model(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        fold_id: int
    ) -> Dict:
        """
        Train a single model on one fold.
        
        Parameters
        ----------
        model : nn.Module
            Model to train
        train_loader : DataLoader
            Training data loader
        val_loader : DataLoader
            Validation data loader
        fold_id : int
            Fold identifier (0, 1, or 2)
        
        Returns
        -------
        Dict
            Training history
        """
        print(f"\n{'='*80}")
        print(f"Training Model Fold {fold_id}")
        print(f"{'='*80}")
        
        # Loss and optimizer
        criterion = FocalLoss(gamma=self.focal_gamma)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.initial_lr,
            weight_decay=self.weight_decay
        )
        
        # Cosine Annealing with Warm Restarts (SI Page 3-4)
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.t_0,
            T_mult=self.t_mult,
            eta_min=self.min_lr
        )
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_recall': [],
            'val_top1_acc': [],
            'val_top10_acc': [],
            'learning_rates': []
        }
        
        best_recall = 0.0
        best_epoch = 0
        best_model_state = None
        
        # Training loop
        for epoch in range(self.num_epochs):
            # ===== Training =====
            model.train()
            train_loss = 0.0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
            for batch_X, batch_y in pbar:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_train_loss = train_loss / len(train_loader)
            
            # ===== Validation =====
            model.eval()
            val_loss = 0.0
            all_preds = []
            all_labels = []
            all_probs = []
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    
                    probs = F.softmax(outputs, dim=1)
                    preds = torch.argmax(probs, dim=1)
                    
                    all_preds.append(preds.cpu().numpy())
                    all_labels.append(batch_y.cpu().numpy())
                    all_probs.append(probs.cpu().numpy())
            
            avg_val_loss = val_loss / len(val_loader)
            
            # Compute metrics
            all_preds = np.concatenate(all_preds)
            all_labels = np.concatenate(all_labels)
            all_probs = np.concatenate(all_probs)
            
            top1_acc = accuracy_score(all_labels, all_preds)
            top10_acc = top_k_accuracy_score(all_labels, all_probs, k=10)
            
            # Top 1% recall (main metric)
            top1_percent_k = max(1, int(0.01 * all_probs.shape[1]))
            recall = top_k_accuracy_score(all_labels, all_probs, k=top1_percent_k)
            
            # Record history
            current_lr = optimizer.param_groups[0]['lr']
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['val_recall'].append(recall)
            history['val_top1_acc'].append(top1_acc)
            history['val_top10_acc'].append(top10_acc)
            history['learning_rates'].append(current_lr)
            
            # Print progress
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Loss: {avg_val_loss:.4f}")
            print(f"  Val Recall (top 1%): {recall:.4f}")
            print(f"  Val Top-1 Acc: {top1_acc:.4f}")
            print(f"  Val Top-10 Acc: {top10_acc:.4f}")
            print(f"  Learning Rate: {current_lr:.6f}")
            
            # Save best model
            if recall > best_recall:
                best_recall = recall
                best_epoch = epoch
                best_model_state = model.state_dict().copy()
                print(f"  ✓ New best model! (Recall: {best_recall:.4f})")
            
            # Learning rate step
            scheduler.step()
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"\n✓ Loaded best model from epoch {best_epoch+1} "
                  f"(Recall: {best_recall:.4f})")
        
        return history
    
    def train_ensemble(
        self,
        training_data: Dict,
        output_dir: Path
    ) -> tuple:
        """
        Train 3-fold ensemble.
        
        From SI Page 2-3:
        "The training data was divided randomly into three folds, with perturbation 
        replicates balanced across the folds. Models were independently trained on 
        two of three folds."
        
        Parameters
        ----------
        training_data : Dict
            Dictionary with keys: 'X', 'y', 'folds', 'compound_names'
        output_dir : Path
            Directory to save models
        
        Returns
        -------
        tuple
            (models, histories)
        """
        X = training_data['X']
        y = training_data['y']
        folds = training_data['folds']
        n_compounds = len(training_data['compound_names'])
        
        # Preprocess data
        X_processed = clip_and_normalize_signature(X)
        
        print(f"\n{'='*80}")
        print(f"🎯 Training 3-Fold Ensemble")
        print(f"{'='*80}")
        print(f"  Total samples: {len(X):,}")
        print(f"  Total compounds: {n_compounds:,}")
        print(f"  Input features: {X.shape[1]}")
        
        models = []
        histories = []
        
        # Train 3 models
        for fold_id in range(3):
            print(f"\n{'='*80}")
            print(f"Training Fold {fold_id} Model")
            print(f"{'='*80}")
            
            # Prepare data splits
            val_mask = folds == fold_id
            train_mask = ~val_mask
            
            print(f"  Training samples: {train_mask.sum():,}")
            print(f"  Validation samples: {val_mask.sum():,}")
            
            # Create datasets
            train_dataset = LINCSDataset(X_processed, y, train_mask)
            val_dataset = LINCSDataset(X_processed, y, val_mask)
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True if self.device == 'cuda' else False
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True if self.device == 'cuda' else False
            )
            
            # Create model
            model = self._create_model(
                input_size=X.shape[1],
                output_size=n_compounds
            ).to(self.device)
            
            n_params = sum(p.numel() for p in model.parameters())
            print(f"\n  Model architecture:")
            print(f"    Input: {X.shape[1]} → Hidden1: 1024 → Hidden2: 2048 → Output: {n_compounds}")
            print(f"    Parameters: {n_params:,}")
            
            # Train model
            history = self.train_single_model(
                model,
                train_loader,
                val_loader,
                fold_id
            )
            
            models.append(model)
            histories.append(history)
            
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'fold_id': fold_id,
                'history': history,
                'dimensions': {
                    'input_size': X.shape[1],
                    'output_size': n_compounds,
                    'input_names': list(training_data.get('gene_names', [f'gene_{i}' for i in range(X.shape[1])])),  # 确保是list
                    'output_names': list(training_data['compound_names'])  # 确保是list
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
                # 添加元数据信息（可选但推荐）
                'preprocessing_info': {
                    'dataset': 'LINCS2020',
                    'n_samples': len(X),
                    'n_genes': X.shape[1],
                    'n_compounds': n_compounds
                }
            }
            model_path = output_dir / f"model_fold_{fold_id}.pt"
            torch.save(checkpoint, model_path)
            print(f"\n  ✓ Model saved to {model_path}")
        
        # Save ensemble history
        history_path = output_dir / "ensemble_history.pkl"
        with open(history_path, 'wb') as f:
            pickle.dump(histories, f)
        
        print(f"\n{'='*80}")
        print(f"✅ Ensemble Training Complete!")
        print(f"{'='*80}")
        print(f"  Models saved to: {output_dir}")
        
        return models, histories