"""
HCA-DR Loss Functions
损失函数定义：
1. Focal Loss - 主分类任务
2. Supervised Contrastive Loss - 细胞系表示学习
3. Global Regularization Loss - 全局模型正则化
4. Alpha Penalty - Context Dropout惩罚
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class FocalLoss(nn.Module):
    """
    Focal Loss用于处理类别不平衡
    
    公式：FL(p_t) = -(1 - p_t)^γ * log(p_t)
    
    其中：
    - p_t = p if y=1 else 1-p
    - γ (gamma) 控制困难样本的权重
    """
    
    def __init__(self, 
                 gamma: float = 2.0,
                 weight: Optional[torch.Tensor] = None,
                 reduction: str = 'mean'):
        """
        初始化Focal Loss
        
        参数：
            gamma: 聚焦参数（默认2.0）
            weight: 类别权重
            reduction: 'mean', 'sum', 或 'none'
        """
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
    
    def forward(self, 
                logits: torch.Tensor, 
                targets: torch.Tensor) -> torch.Tensor:
        """
        计算Focal Loss
        
        参数：
            logits: 预测logits (B, C)
            targets: 目标标签 (B,)
        
        返回：
            loss: Focal Loss值
        """
        # 计算交叉熵（不做reduction）
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        
        # 计算p_t
        p = F.softmax(logits, dim=1)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # 计算focal weight
        focal_weight = (1 - p_t) ** self.gamma
        
        # Focal Loss
        focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class SupervisedContrastiveLoss(nn.Module):
    """
    监督对比学习损失
    
    使用细胞系标签定义正负样本对
    
    公式：
    L_contrast = -1/|P(i)| * Σ_{p∈P(i)} log(exp(s_ip/τ) / Σ_{a∈P(i)∪N(i)} exp(s_ia/τ))
    
    其中：
    - P(i): 与i同细胞系的正样本集
    - N(i): 与i不同细胞系的负样本集
    - s_ij: 样本i和j的相似度
    - τ: 温度参数
    """
    
    def __init__(self, 
                 temperature: float = 0.1,
                 base_temperature: float = 0.07):
        """
        初始化监督对比损失
        
        参数：
            temperature: 温度参数
            base_temperature: 基础温度（用于缩放）
        """
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
    
    def forward(self,
                features: torch.Tensor,
                cell_ids: torch.Tensor) -> torch.Tensor:
        """
        计算监督对比损失
        
        参数：
            features: 归一化的特征向量 (B, D)
            cell_ids: 细胞系ID (B,)
        
        返回：
            loss: 对比损失值
        """
        device = features.device
        batch_size = features.shape[0]
        
        if batch_size <= 1:
            return torch.tensor(0.0, device=device)
        
        # L2归一化
        features = F.normalize(features, p=2, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # 创建正样本mask（同细胞系）
        cell_ids = cell_ids.contiguous().view(-1, 1)
        mask = torch.eq(cell_ids, cell_ids.T).float().to(device)
        
        # 移除对角线（自身）
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # 计算log-softmax（对每个样本，在所有非自身样本上）
        # 使用log-sum-exp技巧提高数值稳定性
        max_sim = similarity_matrix.max(dim=1, keepdim=True)[0]
        exp_sim = torch.exp(similarity_matrix - max_sim) * logits_mask
        log_prob = similarity_matrix - max_sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        
        # 只保留正样本的log_prob
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        
        # 忽略没有正样本的样本
        valid_mask = mask.sum(dim=1) > 0
        
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=device)
        
        # 计算损失
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss[valid_mask].mean()
        
        return loss


class AlphaPenalty(nn.Module):
    """
    Alpha Penalty损失
    
    当输入为dummy context (is_ctx_dropout=1) 时，惩罚非零的alpha值
    
    公式：L_α = d * α²
    
    目标：学会在OOD情况下回退到全局模型（α→0）
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self,
                alpha: torch.Tensor,
                is_ctx_dropout: torch.Tensor) -> torch.Tensor:
        """
        计算Alpha Penalty
        
        参数：
            alpha: 混合权重 (B, 1)
            is_ctx_dropout: dropout标志 (B,)
        
        返回：
            loss: alpha penalty值
        """
        # alpha: (B, 1), is_ctx_dropout: (B,)
        alpha = alpha.squeeze(-1)  # (B,)
        
        # 只惩罚dropout情况下的alpha
        penalty = is_ctx_dropout * (alpha ** 2)
        
        return penalty.mean()


class HCADRLoss(nn.Module):
    """
    HCA-DR完整损失函数
    
    组合所有损失：
    L_total = L_drug + λ₁*L_contrast + λ₂*L_global + λ₃*L_α
    """
    
    def __init__(self,
                 gamma: float = 2.0,
                 temperature: float = 0.1,
                 lambda_contrast: float = 0.1,
                 lambda_global: float = 0.3,
                 lambda_alpha: float = 0.5):
        """
        初始化HCA-DR损失
        
        参数：
            gamma: Focal Loss的gamma参数
            temperature: 对比学习温度
            lambda_contrast: 对比学习损失权重
            lambda_global: 全局正则化损失权重
            lambda_alpha: Alpha Penalty权重
        """
        super().__init__()
        
        self.focal_loss = FocalLoss(gamma=gamma)
        self.contrast_loss = SupervisedContrastiveLoss(temperature=temperature)
        self.alpha_penalty = AlphaPenalty()
        
        self.lambda_contrast = lambda_contrast
        self.lambda_global = lambda_global
        self.lambda_alpha = lambda_alpha
    
    def forward(self,
                logits: torch.Tensor,
                logits_global: torch.Tensor,
                h_ctx: torch.Tensor,
                alpha: torch.Tensor,
                targets: torch.Tensor,
                cell_ids: torch.Tensor,
                is_ctx_dropout: torch.Tensor,
                compute_all: bool = True) -> Tuple[torch.Tensor, dict]:
        """
        计算总损失
        
        参数：
            logits: 最终预测logits (B, C)
            logits_global: 全局模型logits (B, C)
            h_ctx: 上下文特征 (B, D)
            alpha: 混合权重 (B, 1)
            targets: 目标标签 (B,)
            cell_ids: 细胞系ID (B,)
            is_ctx_dropout: dropout标志 (B,)
            compute_all: 是否计算所有损失项
        
        返回：
            total_loss: 总损失
            loss_dict: 各项损失的字典
        """
        # 主分类损失
        loss_drug = self.focal_loss(logits, targets)
        
        loss_dict = {
            'loss_drug': loss_drug.item(),
        }
        
        total_loss = loss_drug
        
        if compute_all:
            # 对比学习损失
            loss_contrast = self.contrast_loss(h_ctx, cell_ids)
            total_loss = total_loss + self.lambda_contrast * loss_contrast
            loss_dict['loss_contrast'] = loss_contrast.item()
            
            # 全局正则化损失
            loss_global = self.focal_loss(logits_global, targets)
            total_loss = total_loss + self.lambda_global * loss_global
            loss_dict['loss_global'] = loss_global.item()
            
            # Alpha Penalty
            loss_alpha = self.alpha_penalty(alpha, is_ctx_dropout)
            total_loss = total_loss + self.lambda_alpha * loss_alpha
            loss_dict['loss_alpha'] = loss_alpha.item()
        
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict
    
    def update_weights(self,
                       lambda_contrast: float,
                       lambda_global: float,
                       lambda_alpha: float):
        """更新损失权重（用于阶段切换）"""
        self.lambda_contrast = lambda_contrast
        self.lambda_global = lambda_global
        self.lambda_alpha = lambda_alpha


class Stage1Loss(nn.Module):
    """
    阶段1损失：仅全局模型训练
    
    L_stage1 = L_global (Focal Loss on global predictions)
    """
    
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.focal_loss = FocalLoss(gamma=gamma)
    
    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        计算阶段1损失
        
        参数：
            logits: 全局模型预测 (B, C)
            targets: 目标标签 (B,)
        
        返回：
            loss: 损失值
            loss_dict: 损失字典
        """
        loss = self.focal_loss(logits, targets)
        
        return loss, {'loss_global': loss.item(), 'total_loss': loss.item()}


class Stage2Loss(nn.Module):
    """
    阶段2损失：FiLM分支训练
    
    L_stage2 = L_drug + λ₁*L_contrast + λ₃*L_α
    """
    
    def __init__(self,
                 gamma: float = 2.0,
                 temperature: float = 0.1,
                 lambda_contrast: float = 0.1,
                 lambda_alpha: float = 0.5):
        super().__init__()
        
        self.focal_loss = FocalLoss(gamma=gamma)
        self.contrast_loss = SupervisedContrastiveLoss(temperature=temperature)
        self.alpha_penalty = AlphaPenalty()
        
        self.lambda_contrast = lambda_contrast
        self.lambda_alpha = lambda_alpha
    
    def forward(self,
                logits: torch.Tensor,
                h_ctx: torch.Tensor,
                alpha: torch.Tensor,
                targets: torch.Tensor,
                cell_ids: torch.Tensor,
                is_ctx_dropout: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        计算阶段2损失
        """
        # 主分类损失
        loss_drug = self.focal_loss(logits, targets)
        
        # 对比学习损失
        loss_contrast = self.contrast_loss(h_ctx, cell_ids)
        
        # Alpha Penalty
        loss_alpha = self.alpha_penalty(alpha, is_ctx_dropout)
        
        # 总损失
        total_loss = loss_drug + self.lambda_contrast * loss_contrast + self.lambda_alpha * loss_alpha
        
        return total_loss, {
            'loss_drug': loss_drug.item(),
            'loss_contrast': loss_contrast.item(),
            'loss_alpha': loss_alpha.item(),
            'total_loss': total_loss.item()
        }


class Stage3Loss(HCADRLoss):
    """
    阶段3损失：端到端微调
    
    使用调整后的权重
    """
    
    def __init__(self,
                 gamma: float = 2.0,
                 temperature: float = 0.1,
                 lambda_contrast: float = 0.05,
                 lambda_global: float = 0.1,
                 lambda_alpha: float = 0.3):
        super().__init__(
            gamma=gamma,
            temperature=temperature,
            lambda_contrast=lambda_contrast,
            lambda_global=lambda_global,
            lambda_alpha=lambda_alpha
        )


if __name__ == "__main__":
    # 测试损失函数
    print("Testing HCA-DR Loss Functions...")
    
    batch_size = 32
    n_classes = 100
    hidden_dim = 256
    
    # 模拟数据
    logits = torch.randn(batch_size, n_classes)
    logits_global = torch.randn(batch_size, n_classes)
    h_ctx = torch.randn(batch_size, hidden_dim)
    alpha = torch.sigmoid(torch.randn(batch_size, 1))
    targets = torch.randint(0, n_classes, (batch_size,))
    cell_ids = torch.randint(0, 10, (batch_size,))
    is_ctx_dropout = torch.bernoulli(torch.full((batch_size,), 0.15))
    
    # 测试Focal Loss
    focal_loss = FocalLoss(gamma=2.0)
    loss_focal = focal_loss(logits, targets)
    print(f"\n✓ Focal Loss: {loss_focal.item():.4f}")
    
    # 测试Supervised Contrastive Loss
    contrast_loss = SupervisedContrastiveLoss(temperature=0.1)
    loss_contrast = contrast_loss(h_ctx, cell_ids)
    print(f"✓ Contrastive Loss: {loss_contrast.item():.4f}")
    
    # 测试Alpha Penalty
    alpha_penalty = AlphaPenalty()
    loss_alpha = alpha_penalty(alpha, is_ctx_dropout)
    print(f"✓ Alpha Penalty: {loss_alpha.item():.4f}")
    
    # 测试完整损失
    hca_loss = HCADRLoss()
    total_loss, loss_dict = hca_loss(
        logits, logits_global, h_ctx, alpha,
        targets, cell_ids, is_ctx_dropout
    )
    print(f"\n✓ Total Loss: {total_loss.item():.4f}")
    print(f"  Loss breakdown: {loss_dict}")
    
    # 测试阶段1损失
    stage1_loss = Stage1Loss()
    loss1, dict1 = stage1_loss(logits_global, targets)
    print(f"\n✓ Stage 1 Loss: {loss1.item():.4f}")
    
    # 测试阶段2损失
    stage2_loss = Stage2Loss()
    loss2, dict2 = stage2_loss(logits, h_ctx, alpha, targets, cell_ids, is_ctx_dropout)
    print(f"✓ Stage 2 Loss: {loss2.item():.4f}")
    
    print("\n✓ All loss functions working correctly!")