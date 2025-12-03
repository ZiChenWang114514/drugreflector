"""
HCA-DR Trainer
三阶段训练逻辑实现：
- 阶段1：全局模型预训练
- 阶段2：FiLM分支训练
- 阶段3：端到端微调
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
from tqdm import tqdm
import json
from collections import defaultdict

from model import HCADR, HCADROutput
from losses import Stage1Loss, Stage2Loss, Stage3Loss, HCADRLoss
from config import HCADRConfig


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class HCADRTrainer:
    """
    HCA-DR训练器
    
    实现三阶段训练策略
    """
    
    def __init__(self,
                 model: HCADR,
                 config: HCADRConfig,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: str = 'cuda'):
        """
        初始化训练器
        
        参数：
            model: HCA-DR模型
            config: 配置对象
            train_loader: 训练DataLoader
            val_loader: 验证DataLoader
            device: 计算设备
        """
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # 损失函数
        self.stage1_loss = Stage1Loss(gamma=config.training.focal_gamma)
        self.stage2_loss = Stage2Loss(
            gamma=config.training.focal_gamma,
            temperature=config.training.contrast_temperature,
            lambda_contrast=config.training.lambda_contrast,
            lambda_alpha=config.training.lambda_alpha_penalty
        )
        self.stage3_loss = Stage3Loss(
            gamma=config.training.focal_gamma,
            temperature=config.training.contrast_temperature,
            lambda_contrast=config.training.stage3_lambda_contrast,
            lambda_global=config.training.stage3_lambda_global,
            lambda_alpha=config.training.stage3_lambda_alpha_penalty
        )
        
        # 训练历史
        self.history = defaultdict(list)
        self.current_stage = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_model_state = None
        
        # 输出路径
        self.output_dir = Path(config.data.output_dir)
        self.checkpoint_dir = self.output_dir / config.data.checkpoint_dir
        self.log_dir = self.output_dir / config.data.log_dir
        
        print(f"✓ Trainer initialized")
        print(f"  Device: {device}")
        print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def _create_optimizer(self, stage: int) -> optim.Optimizer:
        """创建优化器"""
        cfg = self.config.training
        
        if stage == 1:
            params = self.model.get_trainable_params(stage=1)
            lr = cfg.stage1_lr
        elif stage == 2:
            params = self.model.get_trainable_params(stage=2)
            lr = cfg.stage2_lr
        else:
            params = self.model.get_trainable_params(stage=3)
            lr = cfg.stage3_lr
        
        optimizer = optim.AdamW(
            params,
            lr=lr,
            betas=(cfg.beta1, cfg.beta2),
            eps=cfg.eps,
            weight_decay=cfg.weight_decay
        )
        
        return optimizer
    
    def _create_scheduler(self, optimizer: optim.Optimizer, stage: int, n_epochs: int):
        """创建学习率调度器"""
        cfg = self.config.training
        
        if stage == 1:
            # Warmup + 恒定
            def warmup_fn(epoch):
                if epoch < cfg.warmup_epochs:
                    return (epoch + 1) / cfg.warmup_epochs
                return 1.0
            scheduler = LambdaLR(optimizer, warmup_fn)
        
        elif stage == 2:
            # Cosine衰减
            scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=0)
        
        else:
            # 恒定学习率
            scheduler = LambdaLR(optimizer, lambda epoch: 1.0)
        
        return scheduler
    
    def train_stage1(self, n_epochs: Optional[int] = None) -> Dict:
        """
        阶段1：全局模型预训练
        
        目标：复现DrugReflector，建立强大的全局基线
        
        训练：
        - 全局扰动编码器
        - 分类头
        
        冻结：
        - 上下文编码器
        - FiLM模块
        """
        print("\n" + "=" * 80)
        print("📊 STAGE 1: Global Model Pre-training")
        print("=" * 80)
        
        self.current_stage = 1
        n_epochs = n_epochs or self.config.training.stage1_epochs
        
        # 冻结上下文和FiLM模块
        self.model.freeze_context_and_film()
        
        # 创建优化器和调度器
        optimizer = self._create_optimizer(stage=1)
        scheduler = self._create_scheduler(optimizer, stage=1, n_epochs=n_epochs)
        
        # 早停
        early_stopping = EarlyStopping(
            patience=self.config.training.patience,
            min_delta=self.config.training.min_delta
        )
        
        for epoch in range(n_epochs):
            self.current_epoch = epoch
            
            # 训练
            train_metrics = self._train_epoch_stage1(optimizer)
            
            # 验证
            val_metrics = self._validate_stage1()
            
            # 更新学习率
            scheduler.step()
            
            # 记录
            self._log_epoch(epoch, n_epochs, train_metrics, val_metrics, stage=1)
            
            # 保存最优模型
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                self._save_checkpoint(epoch, stage=1, is_best=True)
            
            # 早停检查
            if early_stopping(val_metrics['val_loss']):
                print(f"\n   Early stopping triggered at epoch {epoch+1}")
                break
        
        # 恢复最优模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        return {'stage': 1, 'final_val_loss': self.best_val_loss}
    
    def train_stage2(self, n_epochs: Optional[int] = None) -> Dict:
        """
        阶段2：FiLM分支训练
        
        目标：让FiLM学习全局模型的残差（细胞系特异性信息）
        
        训练：
        - 上下文编码器
        - FiLM模块
        - 分类头
        
        冻结：
        - 全局扰动编码器
        """
        print("\n" + "=" * 80)
        print("📊 STAGE 2: FiLM Branch Training")
        print("=" * 80)
        
        self.current_stage = 2
        n_epochs = n_epochs or self.config.training.stage2_epochs
        
        # 冻结全局编码器，解冻上下文和FiLM
        self.model.freeze_global_encoder()
        self.model.unfreeze_context_and_film()
        
        # 重置最优损失
        self.best_val_loss = float('inf')
        
        # 创建优化器和调度器
        optimizer = self._create_optimizer(stage=2)
        scheduler = self._create_scheduler(optimizer, stage=2, n_epochs=n_epochs)
        
        # 早停
        early_stopping = EarlyStopping(
            patience=self.config.training.patience,
            min_delta=self.config.training.min_delta
        )
        
        for epoch in range(n_epochs):
            self.current_epoch = epoch
            
            # 训练
            train_metrics = self._train_epoch_stage2(optimizer)
            
            # 验证
            val_metrics = self._validate_stage2()
            
            # 更新学习率
            scheduler.step()
            
            # 记录
            self._log_epoch(epoch, n_epochs, train_metrics, val_metrics, stage=2)
            
            # 保存最优模型
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                self._save_checkpoint(epoch, stage=2, is_best=True)
            
            # 早停检查
            if early_stopping(val_metrics['val_loss']):
                print(f"\n   Early stopping triggered at epoch {epoch+1}")
                break
        
        # 恢复最优模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        return {'stage': 2, 'final_val_loss': self.best_val_loss}
    
    def train_stage3(self, n_epochs: Optional[int] = None) -> Dict:
        """
        阶段3：端到端微调
        
        目标：联合优化所有参数
        
        训练：
        - 所有参数
        """
        print("\n" + "=" * 80)
        print("📊 STAGE 3: End-to-End Fine-tuning")
        print("=" * 80)
        
        self.current_stage = 3
        n_epochs = n_epochs or self.config.training.stage3_epochs
        
        # 解冻所有参数
        self.model.unfreeze_global_encoder()
        self.model.unfreeze_context_and_film()
        
        # 重置最优损失
        self.best_val_loss = float('inf')
        
        # 创建优化器和调度器
        optimizer = self._create_optimizer(stage=3)
        scheduler = self._create_scheduler(optimizer, stage=3, n_epochs=n_epochs)
        
        # 早停
        early_stopping = EarlyStopping(
            patience=self.config.training.patience,
            min_delta=self.config.training.min_delta
        )
        
        for epoch in range(n_epochs):
            self.current_epoch = epoch
            
            # 训练
            train_metrics = self._train_epoch_stage3(optimizer)
            
            # 验证
            val_metrics = self._validate_stage3()
            
            # 更新学习率
            scheduler.step()
            
            # 记录
            self._log_epoch(epoch, n_epochs, train_metrics, val_metrics, stage=3)
            
            # 保存最优模型
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                self._save_checkpoint(epoch, stage=3, is_best=True)
            
            # 早停检查
            if early_stopping(val_metrics['val_loss']):
                print(f"\n   Early stopping triggered at epoch {epoch+1}")
                break
        
        # 恢复最优模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        return {'stage': 3, 'final_val_loss': self.best_val_loss}
    
    def _train_epoch_stage1(self, optimizer: optim.Optimizer) -> Dict:
        """阶段1单个epoch训练"""
        self.model.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc=f"Stage 1 Training", leave=False)
        
        for batch in pbar:
            # 移动数据到设备
            x_pert = batch['x_pert'].to(self.device)
            y = batch['y'].to(self.device)
            
            # 前向传播（仅使用全局模型）
            optimizer.zero_grad()
            logits = self.model.forward_global_only(x_pert)
            
            # 计算损失
            loss, loss_dict = self.stage1_loss(logits, y)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 统计
            total_loss += loss.item() * len(y)
            pred = logits.argmax(dim=1)
            total_correct += (pred == y).sum().item()
            total_samples += len(y)
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{total_correct/total_samples*100:.2f}%"
            })
        
        return {
            'train_loss': total_loss / total_samples,
            'train_acc': total_correct / total_samples
        }
    
    def _train_epoch_stage2(self, optimizer: optim.Optimizer) -> Dict:
        """阶段2单个epoch训练"""
        self.model.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        alpha_sum = 0.0
        dropout_alpha_sum = 0.0
        dropout_count = 0
        
        pbar = tqdm(self.train_loader, desc=f"Stage 2 Training", leave=False)
        
        for batch in pbar:
            # 移动数据到设备
            x_pert = batch['x_pert'].to(self.device)
            x_ctx = batch['x_ctx'].to(self.device)
            y = batch['y'].to(self.device)
            cell_ids = batch['cell_id'].to(self.device)
            is_ctx_dropout = batch['is_ctx_dropout'].to(self.device)
            
            # 前向传播
            optimizer.zero_grad()
            output = self.model(x_pert, x_ctx)
            
            # 计算损失
            loss, loss_dict = self.stage2_loss(
                output.logits, output.h_ctx, output.alpha,
                y, cell_ids, is_ctx_dropout
            )
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 统计
            total_loss += loss.item() * len(y)
            pred = output.logits.argmax(dim=1)
            total_correct += (pred == y).sum().item()
            total_samples += len(y)
            
            # Alpha统计
            alpha_sum += output.alpha.sum().item()
            dropout_mask = is_ctx_dropout > 0.5
            if dropout_mask.sum() > 0:
                dropout_alpha_sum += output.alpha[dropout_mask].sum().item()
                dropout_count += dropout_mask.sum().item()
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'α': f"{output.alpha.mean().item():.3f}"
            })
        
        return {
            'train_loss': total_loss / total_samples,
            'train_acc': total_correct / total_samples,
            'mean_alpha': alpha_sum / total_samples,
            'dropout_alpha': dropout_alpha_sum / max(dropout_count, 1)
        }
    
    def _train_epoch_stage3(self, optimizer: optim.Optimizer) -> Dict:
        """阶段3单个epoch训练"""
        self.model.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        alpha_sum = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Stage 3 Training", leave=False)
        
        for batch in pbar:
            # 移动数据到设备
            x_pert = batch['x_pert'].to(self.device)
            x_ctx = batch['x_ctx'].to(self.device)
            y = batch['y'].to(self.device)
            cell_ids = batch['cell_id'].to(self.device)
            is_ctx_dropout = batch['is_ctx_dropout'].to(self.device)
            
            # 前向传播
            optimizer.zero_grad()
            output = self.model(x_pert, x_ctx)
            
            # 计算损失
            loss, loss_dict = self.stage3_loss(
                output.logits, output.logits_global, output.h_ctx, output.alpha,
                y, cell_ids, is_ctx_dropout
            )
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 统计
            total_loss += loss.item() * len(y)
            pred = output.logits.argmax(dim=1)
            total_correct += (pred == y).sum().item()
            total_samples += len(y)
            alpha_sum += output.alpha.sum().item()
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{total_correct/total_samples*100:.2f}%"
            })
        
        return {
            'train_loss': total_loss / total_samples,
            'train_acc': total_correct / total_samples,
            'mean_alpha': alpha_sum / total_samples
        }
    
    @torch.no_grad()
    def _validate_stage1(self) -> Dict:
        """阶段1验证"""
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch in self.val_loader:
            x_pert = batch['x_pert'].to(self.device)
            y = batch['y'].to(self.device)
            
            logits = self.model.forward_global_only(x_pert)
            loss, _ = self.stage1_loss(logits, y)
            
            total_loss += loss.item() * len(y)
            pred = logits.argmax(dim=1)
            total_correct += (pred == y).sum().item()
            total_samples += len(y)
        
        return {
            'val_loss': total_loss / total_samples,
            'val_acc': total_correct / total_samples
        }
    
    @torch.no_grad()
    def _validate_stage2(self) -> Dict:
        """阶段2验证"""
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        alpha_sum = 0.0
        
        for batch in self.val_loader:
            x_pert = batch['x_pert'].to(self.device)
            x_ctx = batch['x_ctx'].to(self.device)
            y = batch['y'].to(self.device)
            cell_ids = batch['cell_id'].to(self.device)
            is_ctx_dropout = batch['is_ctx_dropout'].to(self.device)
            
            output = self.model(x_pert, x_ctx)
            loss, _ = self.stage2_loss(
                output.logits, output.h_ctx, output.alpha,
                y, cell_ids, is_ctx_dropout
            )
            
            total_loss += loss.item() * len(y)
            pred = output.logits.argmax(dim=1)
            total_correct += (pred == y).sum().item()
            total_samples += len(y)
            alpha_sum += output.alpha.sum().item()
        
        return {
            'val_loss': total_loss / total_samples,
            'val_acc': total_correct / total_samples,
            'mean_alpha': alpha_sum / total_samples
        }
    
    @torch.no_grad()
    def _validate_stage3(self) -> Dict:
        """阶段3验证"""
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        alpha_sum = 0.0
        
        for batch in self.val_loader:
            x_pert = batch['x_pert'].to(self.device)
            x_ctx = batch['x_ctx'].to(self.device)
            y = batch['y'].to(self.device)
            cell_ids = batch['cell_id'].to(self.device)
            is_ctx_dropout = batch['is_ctx_dropout'].to(self.device)
            
            output = self.model(x_pert, x_ctx)
            loss, _ = self.stage3_loss(
                output.logits, output.logits_global, output.h_ctx, output.alpha,
                y, cell_ids, is_ctx_dropout
            )
            
            total_loss += loss.item() * len(y)
            pred = output.logits.argmax(dim=1)
            total_correct += (pred == y).sum().item()
            total_samples += len(y)
            alpha_sum += output.alpha.sum().item()
        
        return {
            'val_loss': total_loss / total_samples,
            'val_acc': total_correct / total_samples,
            'mean_alpha': alpha_sum / total_samples
        }
    
    def _log_epoch(self, epoch: int, n_epochs: int, 
                   train_metrics: Dict, val_metrics: Dict, stage: int):
        """记录并打印epoch结果"""
        # 保存到历史
        for key, value in train_metrics.items():
            self.history[f'stage{stage}_{key}'].append(value)
        for key, value in val_metrics.items():
            self.history[f'stage{stage}_{key}'].append(value)
        
        # 打印
        print(f"\n   Epoch {epoch+1}/{n_epochs}")
        print(f"   Train Loss: {train_metrics['train_loss']:.4f}, Acc: {train_metrics['train_acc']*100:.2f}%")
        print(f"   Val Loss: {val_metrics['val_loss']:.4f}, Acc: {val_metrics['val_acc']*100:.2f}%")
        
        if 'mean_alpha' in train_metrics:
            print(f"   Mean Alpha (train): {train_metrics['mean_alpha']:.4f}")
        if 'mean_alpha' in val_metrics:
            print(f"   Mean Alpha (val): {val_metrics['mean_alpha']:.4f}")
        if 'dropout_alpha' in train_metrics:
            print(f"   Dropout Alpha: {train_metrics['dropout_alpha']:.4f}")
    
    def _save_checkpoint(self, epoch: int, stage: int, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'stage': stage,
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'history': dict(self.history),
            'best_val_loss': self.best_val_loss
        }
        
        # 保存最新
        path = self.checkpoint_dir / f'stage{stage}_latest.pt'
        torch.save(checkpoint, path)
        
        # 保存最优
        if is_best:
            path = self.checkpoint_dir / f'stage{stage}_best.pt'
            torch.save(checkpoint, path)
            print(f"   ✓ Saved best model (val_loss: {self.best_val_loss:.4f})")
    
    def train_all_stages(self) -> Dict:
        """
        执行完整的三阶段训练
        """
        print("\n" + "=" * 80)
        print("🚀 Starting HCA-DR Three-Stage Training")
        print("=" * 80)
        
        start_time = time.time()
        
        # 阶段1
        result1 = self.train_stage1()
        
        # 阶段2
        result2 = self.train_stage2()
        
        # 阶段3
        result3 = self.train_stage3()
        
        total_time = time.time() - start_time
        
        # 保存最终模型
        final_path = self.checkpoint_dir / 'final_model.pt'
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'history': dict(self.history),
            'training_time': total_time
        }, final_path)
        
        # 保存训练历史
        history_path = self.log_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            # 转换numpy类型为python类型
            history_dict = {}
            for k, v in self.history.items():
                history_dict[k] = [float(x) if isinstance(x, (np.floating, float)) else x for x in v]
            json.dump(history_dict, f, indent=2)
        
        print("\n" + "=" * 80)
        print("✅ Training Complete!")
        print("=" * 80)
        print(f"   Total time: {total_time/3600:.2f} hours")
        print(f"   Final model saved to: {final_path}")
        print(f"   Training history saved to: {history_path}")
        
        return {
            'stage1': result1,
            'stage2': result2,
            'stage3': result3,
            'total_time': total_time
        }
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = defaultdict(list, checkpoint.get('history', {}))
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"✓ Loaded checkpoint from: {checkpoint_path}")


if __name__ == "__main__":
    print("HCA-DR Trainer module loaded successfully")