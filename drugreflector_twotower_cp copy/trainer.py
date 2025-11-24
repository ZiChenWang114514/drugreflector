"""
Two-Tower DrugReflector Trainer
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

from models import TwoTowerModel
from dataset import TwoTowerDataset, collate_two_tower
from preprocessing import clip_and_normalize_signature


def compound_level_topk_recall(labels, probs, k):
    """
    Compute compound-level top-k recall.
    
    修改：确保k不超过实际类别数
    """
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    n_classes = probs.shape[1]
    k = max(1, min(k, n_classes))  # 确保k有效

    topk_pred = np.argpartition(-probs, kth=k-1, axis=1)[:, :k]
    hit_per_sample = (topk_pred == labels[:, None]).any(axis=1).astype(float)

    compound_hits = {}
    compound_counts = {}
    for y, hit in zip(labels, hit_per_sample):
        compound_hits.setdefault(y, 0.0)
        compound_counts.setdefault(y, 0)
        compound_hits[y] += hit
        compound_counts[y] += 1

    compound_recalls = [compound_hits[cid] / compound_counts[cid] 
                       for cid in compound_hits]
    return float(np.mean(compound_recalls))


class FocalLoss(nn.Module):
    """Focal Loss for class imbalance."""
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        return focal_loss.mean()


class TwoTowerTrainer:
    """
    Trainer for Two-Tower DrugReflector.
    
    Parameters match original trainer for consistency.
    """
    
    def __init__(
        self,
        device: str = 'auto',
        chem_hidden_dim: int = 512,
        transcript_hidden_dims: List[int] = [1024, 2048],
        fusion_method: str = 'concat',
        mpnn_depth: int = 3,
        mpnn_dropout: float = 0.0,
        use_3d: bool = False,
        d_coord: int = 16,
        conformer_method: str = 'ETKDG',
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
        
        self.chem_hidden_dim = chem_hidden_dim
        self.transcript_hidden_dims = transcript_hidden_dims
        self.fusion_method = fusion_method
        self.mpnn_depth = mpnn_depth
        self.mpnn_dropout = mpnn_dropout
        self.use_3d = use_3d
        self.d_coord = d_coord
        self.conformer_method = conformer_method
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
            print(f"������ Two-Tower Trainer Initialized")
            print(f"{'='*80}")
            print(f"  Device: {self.device}")
            print(f"  Fusion: {self.fusion_method}")
            print(f"  Chem dim: {self.chem_hidden_dim}")
            print(f"  MPNN depth: {self.mpnn_depth}")
            print(f"  Use 3D: {self.use_3d}")  # ������ 新增
            if self.use_3d:
                print(f"  3D coord dim: {self.d_coord}")
                print(f"  Conformer method: {self.conformer_method}")
            
    def create_model(self, input_size: int, output_size: int) -> nn.Module:
        """Create Two-Tower model."""
        model = TwoTowerModel(
            n_genes=input_size,
            n_compounds=output_size,
            chem_hidden_dim=self.chem_hidden_dim,
            transcript_hidden_dims=self.transcript_hidden_dims,
            fusion_method=self.fusion_method,
            mpnn_depth=self.mpnn_depth,
            mpnn_dropout=self.mpnn_dropout,
            use_3d=self.use_3d,
            d_coord=self.d_coord,
            conformer_method=self.conformer_method
        )
        return model
    
    def train_single_fold(
        self,
        training_data: Dict,
        fold_id: int,
        output_dir: Path
    ) -> Dict:
        """Train model on single fold."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract data
        X = training_data['X']
        y = training_data['y']
        folds = training_data['folds']
        compound_names = training_data['compound_names']
        smiles_dict = training_data['smiles_dict']
        
        # Get compound IDs for each sample
        sample_meta = training_data['sample_meta']
        compound_ids = sample_meta['pert_id'].values
        
        n_samples = len(X)
        n_genes = X.shape[1]
        n_compounds = len(compound_names)
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"������ Training Fold {fold_id}")
            print(f"{'='*80}")
            print(f"  Total samples: {n_samples:,}")
            print(f"  Compounds: {n_compounds:,}")
            print(f"  Genes: {n_genes}")
        
        # Preprocess
        X_processed = clip_and_normalize_signature(X)
        
        # Create train/val split
        val_mask = folds == fold_id
        train_mask = ~val_mask
        
        # Create datasets
        train_dataset = TwoTowerDataset(
            X_processed, y, compound_ids, smiles_dict, train_mask,
            use_3d=self.use_3d, conformer_method=self.conformer_method
        )
        val_dataset = TwoTowerDataset(
            X_processed, y, compound_ids, smiles_dict, val_mask,
            use_3d=self.use_3d, conformer_method=self.conformer_method
        )

        # 统计训练集和验证集的类别分布
        train_compounds = np.unique(y[train_mask])
        val_compounds = np.unique(y[val_mask])
        
        if self.verbose:
            print(f"\n Data Split:")
            print(f"  Train: {len(train_dataset):,} samples")
            print(f"    - {len(train_compounds)} unique compounds")
            print(f"  Val: {len(val_dataset):,} samples")
            print(f"    - {len(val_compounds)} unique compounds")
            
            # 检查是否有类别重叠
            overlap = set(train_compounds) & set(val_compounds)
            if len(overlap) > 0:
                print(f"    !  {len(overlap)} compounds appear in both train and val")
            else:
                print(f"    ✓ No compound overlap (scaffold split detected)")
                
        # Create loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_two_tower,
            pin_memory=(self.device == 'cuda')
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_two_tower,
            pin_memory=(self.device == 'cuda')
        )
        
        # Create model
        model = self.create_model(n_genes, n_compounds).to(self.device)
        
        if self.verbose:
            n_params = sum(p.numel() for p in model.parameters())
            print(f"\n������️  Model:")
            print(f"  Parameters: {n_params:,}")
        
        # Setup training
        criterion = FocalLoss(gamma=self.focal_gamma)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.initial_lr,
            weight_decay=self.weight_decay
        )
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=self.t_0, eta_min=self.min_lr
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
            epoch_start = time.time()
            
            # Train
            train_loss = self._train_epoch(
                model, train_loader, criterion, optimizer
            )
            
            # Validate
            val_loss, val_metrics = self._validate_epoch(
                model, val_loader, criterion
            )
            
            # Record
            current_lr = optimizer.param_groups[0]['lr']
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_recall'].append(val_metrics['recall'])
            history['val_top1_acc'].append(val_metrics['top1_acc'])
            history['val_top10_acc'].append(val_metrics['top10_acc'])
            history['learning_rates'].append(current_lr)
            
            if self.verbose:
                print(f"\nEpoch {epoch+1}/{self.num_epochs} ({time.time()-epoch_start:.1f}s)")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val Loss: {val_loss:.4f}")
                print(f"  Val Recall: {val_metrics['recall']:.4f} ⭐")
                print(f"  Val Top-1: {val_metrics['top1_acc']:.4f}")
                print(f"  LR: {current_lr:.6f}")
            
            # Save best
            if val_metrics['recall'] > best_recall:
                best_recall = val_metrics['recall']
                best_epoch = epoch
                best_model_state = model.state_dict().copy()
                if self.verbose:
                    print(f"  ✅ New best!")
            
            scheduler.step()
        
        # Load best
        if best_model_state:
            model.load_state_dict(best_model_state)
        
        # Save final model
        model_path = output_dir / f"model_fold_{fold_id}.pt"
        self._save_checkpoint(
            model, fold_id, history, n_genes, n_compounds,
            compound_names, model_path,
            scaffold_split=training_data.get('scaffold_split', False)
        )
        
        if self.verbose:
            print(f"\n✅ Training Complete!")
            print(f"  Best epoch: {best_epoch+1}")
            print(f"  Best recall: {best_recall:.4f}")
            print(f"  Model: {model_path}")
        
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
        """Train on all folds."""
        if folds is None:
            folds = [0, 1, 2]
        
        fold_results = {}
        model_paths = []
        
        for fold_id in folds:
            result = self.train_single_fold(
                training_data, fold_id, output_dir
            )
            fold_results[fold_id] = result
            model_paths.append(str(result['model_path']))
        
        # Ensemble metrics
        ensemble_metrics = {
            'mean_recall': np.mean([r['best_recall'] for r in fold_results.values()]),
            'std_recall': np.std([r['best_recall'] for r in fold_results.values()])
        }
        
        return {
            'fold_results': fold_results,
            'ensemble_metrics': ensemble_metrics,
            'model_paths': model_paths
        }
    
    def _train_epoch(self, model, loader, criterion, optimizer):
        """Train one epoch with robust error handling."""
        model.train()
        total_loss = 0.0
        n_batches = 0
        n_skipped = 0
        
        for batch_idx, batch_data in enumerate(tqdm(loader, desc="Training", 
                                                    leave=False, 
                                                    disable=not self.verbose)):
            try:
                if len(batch_data) == 4:
                    x_t, bmg, coords_3d, y = batch_data
                else:
                    x_t, bmg, y = batch_data
                    coords_3d = None
                    
                # 关键修复：在移动到GPU之前先验证
                if bmg is None or bmg.V is None:
                    print(f"  Batch {batch_idx}: bmg invalid before GPU transfer")
                    n_skipped += 1
                    continue
                
                # 保存原始形状用于验证
                original_V_shape = bmg.V.shape
                original_E_shape = bmg.E.shape
                
                # Move to device
                x_t = x_t.to(self.device)
                y = y.to(self.device)
                
                # Move 3D coords to device
                if coords_3d is not None:
                    coords_3d = coords_3d.to(self.device)
                    
                # 关键：正确的 GPU 传输方法
                if self.device == 'cuda':
                    try:
                        # 不使用 bmg = bmg.to(device)，会导致问题
                        # 正确方法：逐个属性 in-place 转换
                        
                        if hasattr(bmg, 'V') and bmg.V is not None:
                            bmg.V = bmg.V.to(self.device, non_blocking=True)
                        
                        if hasattr(bmg, 'E') and bmg.E is not None:
                            bmg.E = bmg.E.to(self.device, non_blocking=True)
                        
                        if hasattr(bmg, 'edge_index') and bmg.edge_index is not None:
                            bmg.edge_index = bmg.edge_index.to(self.device, non_blocking=True)
                        
                        if hasattr(bmg, 'batch') and bmg.batch is not None:
                            bmg.batch = bmg.batch.to(self.device, non_blocking=True)
                        
                        if hasattr(bmg, 'rev_edge_index') and bmg.rev_edge_index is not None:
                            bmg.rev_edge_index = bmg.rev_edge_index.to(self.device, non_blocking=True)
                        
                        # ������ 验证传输后数据完整性
                        if bmg.V is None:
                            print(f"⚠️  Batch {batch_idx}: bmg.V became None after GPU transfer!")
                            print(f"    Original shape was: {original_V_shape}")
                            n_skipped += 1
                            continue
                        
                        if bmg.V.shape != original_V_shape:
                            print(f"⚠️  Batch {batch_idx}: Shape changed after GPU transfer!")
                            print(f"    Before: {original_V_shape}, After: {bmg.V.shape}")
                            n_skipped += 1
                            continue
                        
                    except Exception as e:
                        print(f"  Batch {batch_idx}: GPU transfer failed: {e}")
                        n_skipped += 1
                        continue
                
                # Forward pass
                optimizer.zero_grad()
                
                try:
                    outputs = model(x_t, bmg, coords_3d)
                except Exception as e:
                    print(f"⚠️  Batch {batch_idx}: Forward pass failed: {e}")
                    n_skipped += 1
                    continue
                
                # Backward pass
                loss = criterion(outputs, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
                
            except Exception as e:
                print(f"⚠️  Batch {batch_idx}: Unexpected error: {e}")
                n_skipped += 1
                continue
        
        if n_batches == 0:
            raise RuntimeError(
                f"No valid batches processed! Skipped: {n_skipped}/{n_skipped + n_batches}"
            )
        
        if n_skipped > 0:
            print(f"  ⚠️  Skipped {n_skipped} batches this epoch")
        
        return total_loss / n_batches


    def _validate_epoch(self, model, loader, criterion):
        """Validate one epoch."""
        model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        n_skipped = 0
            
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm(loader, desc="Validating", 
                                                        leave=False,
                                                        disable=not self.verbose)):
                try:
                    if len(batch_data) == 4:
                        x_t, bmg, coords_3d, y = batch_data
                    else:
                        x_t, bmg, y = batch_data
                        coords_3d = None
                    
                    # 验证
                    if bmg is None or bmg.V is None:
                        print(f"  Batch {batch_idx}: bmg invalid before GPU transfer")
                        n_skipped += 1
                        continue
                    
                    # Move to device
                    x_t = x_t.to(self.device)
                    y = y.to(self.device)
                    
                    if coords_3d is not None:
                        coords_3d = coords_3d.to(self.device)
                    
                    # GPU 传输（同训练）
                    if self.device == 'cuda':
                        if hasattr(bmg, 'V') and bmg.V is not None:
                            bmg.V = bmg.V.to(self.device, non_blocking=True)
                        if hasattr(bmg, 'E') and bmg.E is not None:
                            bmg.E = bmg.E.to(self.device, non_blocking=True)
                        if hasattr(bmg, 'edge_index') and bmg.edge_index is not None:
                            bmg.edge_index = bmg.edge_index.to(self.device, non_blocking=True)
                        if hasattr(bmg, 'batch') and bmg.batch is not None:
                            bmg.batch = bmg.batch.to(self.device, non_blocking=True)
                        if hasattr(bmg, 'rev_edge_index') and bmg.rev_edge_index is not None:
                            bmg.rev_edge_index = bmg.rev_edge_index.to(self.device, non_blocking=True)
                        
                        if bmg.V is None:
                            n_skipped += 1
                            continue
                    
                    outputs = model(x_t, bmg, coords_3d)
                    loss = criterion(outputs, y)
                    total_loss += loss.item()
                    
                    probs = F.softmax(outputs, dim=1)
                    preds = torch.argmax(probs, dim=1)
                    
                    all_preds.append(preds.cpu().numpy())
                    all_labels.append(y.cpu().numpy())
                    all_probs.append(probs.cpu().numpy())
                    
                except Exception as e:
                    print(f"⚠️  Validation batch {batch_idx} failed: {e}")
                    n_skipped += 1
                    continue
        
        if len(all_preds) == 0:
            raise RuntimeError("No valid validation batches!")
        
        if n_skipped > 0:
            print(f"  ⚠️  Skipped {n_skipped} validation batches")
        
        avg_loss = total_loss / len(all_preds)
        
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        all_probs = np.concatenate(all_probs)
        
        # 获取验证集中实际出现的类别
        unique_labels = np.unique(all_labels)
        n_classes_in_val = len(unique_labels)
        n_classes_total = all_probs.shape[1]
        
        # Top-1 accuracy (不受影响)
        top1_acc = accuracy_score(all_labels, all_preds)
        
        # Top-10 accuracy：需要考虑验证集类别数
        # 如果验证集类别数 < 10，则使用实际类别数
        k_for_top10 = min(10, n_classes_in_val)
        
        if n_classes_in_val < n_classes_total:
            # Scaffold split情况：验证集类别少于总类别
            # 方法1：只考虑验证集中出现的类别的概率
            # 创建label到索引的映射
            label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
            
            # 提取验证集类别对应的概率
            probs_subset = all_probs[:, unique_labels]
            
            # 使用子集概率计算top-k accuracy
            top10_acc = top_k_accuracy_score(
                all_labels, 
                probs_subset, 
                k=k_for_top10,
                labels=unique_labels  # 明确指定类别
            )
            
            if self.verbose and n_classes_in_val < n_classes_total:
                print(f"  !  Validation set contains {n_classes_in_val}/{n_classes_total} classes")
        else:
            # 原始情况：验证集包含所有类别
            top10_acc = top_k_accuracy_score(all_labels, all_probs, k=k_for_top10)
        
        # Recall计算：使用验证集实际类别数
        top1_percent_k = max(1, int(0.01 * n_classes_in_val))
        
        # 使用子集概率计算recall
        if n_classes_in_val < n_classes_total:
            probs_for_recall = all_probs[:, unique_labels]
        else:
            probs_for_recall = all_probs
        
        recall = compound_level_topk_recall(all_labels, probs_for_recall, top1_percent_k)
        
        return avg_loss, {
            'recall': recall,
            'top1_acc': top1_acc,
            'top10_acc': top10_acc,
            'n_classes_val': n_classes_in_val,  # 额外返回验证集类别数
            'k_used': k_for_top10  # 实际使用的k值
        }
    
    
    def _save_checkpoint(
        self,
        model: nn.Module,
        fold_id: int,
        history: Dict,
        n_genes: int,
        n_compounds: int,
        compound_names: list,
        save_path: Path,
        scaffold_split: bool = False
    ):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'fold_id': fold_id,
            'history': history,
            'scaffold_split': scaffold_split,
            'dimensions': {
                'input_size': n_genes,
                'output_size': n_compounds,
                'output_names': list(compound_names)
            },
            'architecture': {
                'chem_hidden_dim': self.chem_hidden_dim,
                'transcript_hidden_dims': self.transcript_hidden_dims,
                'fusion_method': self.fusion_method,
                'mpnn_depth': self.mpnn_depth,
                'mpnn_dropout': self.mpnn_dropout
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