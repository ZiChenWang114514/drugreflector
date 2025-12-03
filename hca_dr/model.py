"""
HCA-DR Model Architecture
Hierarchical Cell-Aware DrugReflector模型定义

包含四个核心模块：
1. 全局扰动编码器 (Global Perturbation Encoder)
2. 细胞系上下文编码器 (Cell Line Context Encoder)
3. 自适应FiLM调制 (Adaptive FiLM Modulation)
4. 分类头 (Classification Head)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class HCADROutput:
    """HCA-DR模型输出"""
    logits: torch.Tensor              # 最终预测logits (B, n_compounds)
    logits_global: torch.Tensor       # 全局模型logits (B, n_compounds)
    h_global: torch.Tensor            # 全局特征 (B, hidden_dim)
    h_ctx: torch.Tensor               # 上下文特征 (B, context_dim)
    h_adapted: torch.Tensor           # 调制后特征 (B, hidden_dim)
    alpha: torch.Tensor               # 混合权重 (B, 1)
    gamma: torch.Tensor               # FiLM scale参数 (B, hidden_dim)
    beta: torch.Tensor                # FiLM shift参数 (B, hidden_dim)


class GlobalPerturbationEncoder(nn.Module):
    """
    Module 1: 全局扰动编码器
    
    将978维的扰动签名编码为高维特征表示
    架构：978 -> 1024 -> 2048
    """
    
    def __init__(self, 
                 input_dim: int = 978,
                 hidden_dims: list = [1024, 2048],
                 dropout: float = 0.64):
        super().__init__()
        
        layers = []
        in_dim = input_dim
        
        for i, out_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            in_dim = out_dim
        
        self.encoder = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]
    
    def forward(self, x_pert: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数：
            x_pert: 扰动签名 (B, 978)
        
        返回：
            h_global: 全局特征 (B, 2048)
        """
        return self.encoder(x_pert)


class CellContextEncoder(nn.Module):
    """
    Module 2: 细胞系上下文编码器
    
    将978维的细胞系上下文编码为紧凑表示
    架构：978 -> 256
    """
    
    def __init__(self,
                 input_dim: int = 978,
                 hidden_dim: int = 256):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.output_dim = hidden_dim
    
    def forward(self, x_ctx: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数：
            x_ctx: 细胞系上下文 (B, 978)
        
        返回：
            h_ctx: 上下文特征 (B, 256)
        """
        return self.encoder(x_ctx)


class AdaptiveFiLMModule(nn.Module):
    """
    Module 3: 自适应FiLM调制
    
    Feature-wise Linear Modulation with adaptive mixing
    
    数学表示：
    [γ; β] = W_film * h_ctx + b_film
    h_modulated = γ ⊙ h_global + β
    α = σ(w_α^T * h_ctx + b_α)
    h_adapted = (1 - α) * h_global + α * h_modulated
    """
    
    def __init__(self,
                 context_dim: int = 256,
                 feature_dim: int = 2048):
        super().__init__()
        
        self.feature_dim = feature_dim
        
        # FiLM参数生成器: context -> (gamma, beta)
        self.film_generator = nn.Linear(context_dim, feature_dim * 2)
        
        # 自适应混合权重生成器: context -> alpha
        self.alpha_generator = nn.Linear(context_dim, 1)
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重，使gamma初始为1，beta初始为0"""
        # FiLM generator
        nn.init.zeros_(self.film_generator.weight)
        nn.init.zeros_(self.film_generator.bias)
        # 设置gamma的bias为1（对应恒等变换）
        self.film_generator.bias.data[:self.feature_dim] = 1.0
        
        # Alpha generator - 初始化使alpha接近0.5
        nn.init.zeros_(self.alpha_generator.weight)
        nn.init.zeros_(self.alpha_generator.bias)
    
    def forward(self, 
                h_global: torch.Tensor,
                h_ctx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        参数：
            h_global: 全局特征 (B, 2048)
            h_ctx: 上下文特征 (B, 256)
        
        返回：
            h_adapted: 调制后特征 (B, 2048)
            alpha: 混合权重 (B, 1)
            gamma: scale参数 (B, 2048)
            beta: shift参数 (B, 2048)
        """
        # 生成FiLM参数
        film_params = self.film_generator(h_ctx)  # (B, 4096)
        gamma = film_params[:, :self.feature_dim]  # (B, 2048)
        beta = film_params[:, self.feature_dim:]   # (B, 2048)
        
        # FiLM变换
        h_modulated = gamma * h_global + beta  # (B, 2048)
        
        # 生成自适应混合权重
        alpha = torch.sigmoid(self.alpha_generator(h_ctx))  # (B, 1)
        
        # 自适应混合
        h_adapted = (1 - alpha) * h_global + alpha * h_modulated  # (B, 2048)
        
        return h_adapted, alpha, gamma, beta


class ClassificationHead(nn.Module):
    """
    Module 4: 分类头
    
    将特征映射到药物类别
    """
    
    def __init__(self,
                 input_dim: int = 2048,
                 n_classes: int = 6971):
        super().__init__()
        
        self.classifier = nn.Linear(input_dim, n_classes)
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数：
            h: 特征向量 (B, 2048)
        
        返回：
            logits: 分类logits (B, n_classes)
        """
        return self.classifier(h)


class HCADR(nn.Module):
    """
    HCA-DR: Hierarchical Cell-Aware DrugReflector
    
    完整模型架构，整合四个模块
    """
    
    def __init__(self,
                 n_genes: int = 978,
                 n_compounds: int = 6971,
                 n_cell_lines: int = 114,
                 encoder_hidden_dims: list = [1024, 2048],
                 encoder_dropout: float = 0.64,
                 context_hidden_dim: int = 256):
        super().__init__()
        
        self.n_genes = n_genes
        self.n_compounds = n_compounds
        self.n_cell_lines = n_cell_lines
        
        # Module 1: 全局扰动编码器
        self.global_encoder = GlobalPerturbationEncoder(
            input_dim=n_genes,
            hidden_dims=encoder_hidden_dims,
            dropout=encoder_dropout
        )
        
        # Module 2: 细胞系上下文编码器
        self.context_encoder = CellContextEncoder(
            input_dim=n_genes,
            hidden_dim=context_hidden_dim
        )
        
        # Module 3: 自适应FiLM调制
        self.film_module = AdaptiveFiLMModule(
            context_dim=context_hidden_dim,
            feature_dim=self.global_encoder.output_dim
        )
        
        # Module 4: 分类头
        self.classifier = ClassificationHead(
            input_dim=self.global_encoder.output_dim,
            n_classes=n_compounds
        )
        
        # 保存模块输出维度
        self.feature_dim = self.global_encoder.output_dim
        self.context_dim = context_hidden_dim
    
    def forward(self,
                x_pert: torch.Tensor,
                x_ctx: torch.Tensor,
                return_all: bool = True) -> HCADROutput:
        """
        前向传播
        
        参数：
            x_pert: 扰动签名 (B, 978)
            x_ctx: 细胞系上下文 (B, 978)
            return_all: 是否返回所有中间结果
        
        返回：
            HCADROutput对象
        """
        # Module 1: 全局编码
        h_global = self.global_encoder(x_pert)  # (B, 2048)
        
        # Module 2: 上下文编码
        h_ctx = self.context_encoder(x_ctx)  # (B, 256)
        
        # Module 3: FiLM调制
        h_adapted, alpha, gamma, beta = self.film_module(h_global, h_ctx)
        
        # Module 4: 分类
        logits = self.classifier(h_adapted)  # (B, n_compounds)
        logits_global = self.classifier(h_global)  # 全局模型预测
        
        return HCADROutput(
            logits=logits,
            logits_global=logits_global,
            h_global=h_global,
            h_ctx=h_ctx,
            h_adapted=h_adapted,
            alpha=alpha,
            gamma=gamma,
            beta=beta
        )
    
    def forward_global_only(self, x_pert: torch.Tensor) -> torch.Tensor:
        """
        仅使用全局模型的前向传播（阶段1训练用）
        
        参数：
            x_pert: 扰动签名 (B, 978)
        
        返回：
            logits: 分类logits (B, n_compounds)
        """
        h_global = self.global_encoder(x_pert)
        logits = self.classifier(h_global)
        return logits
    
    def freeze_global_encoder(self):
        """冻结全局编码器参数（阶段2用）"""
        for param in self.global_encoder.parameters():
            param.requires_grad = False
        print("✓ Global encoder frozen")
    
    def unfreeze_global_encoder(self):
        """解冻全局编码器参数（阶段3用）"""
        for param in self.global_encoder.parameters():
            param.requires_grad = True
        print("✓ Global encoder unfrozen")
    
    def freeze_context_and_film(self):
        """冻结上下文编码器和FiLM模块（阶段1用）"""
        for param in self.context_encoder.parameters():
            param.requires_grad = False
        for param in self.film_module.parameters():
            param.requires_grad = False
        print("✓ Context encoder and FiLM module frozen")
    
    def unfreeze_context_and_film(self):
        """解冻上下文编码器和FiLM模块"""
        for param in self.context_encoder.parameters():
            param.requires_grad = True
        for param in self.film_module.parameters():
            param.requires_grad = True
        print("✓ Context encoder and FiLM module unfrozen")
    
    def get_trainable_params(self, stage: int) -> list:
        """
        获取指定阶段的可训练参数
        
        参数：
            stage: 训练阶段 (1, 2, 3)
        
        返回：
            参数列表
        """
        if stage == 1:
            # 阶段1：只训练全局编码器和分类头
            return list(self.global_encoder.parameters()) + \
                   list(self.classifier.parameters())
        elif stage == 2:
            # 阶段2：训练上下文编码器、FiLM模块和分类头
            return list(self.context_encoder.parameters()) + \
                   list(self.film_module.parameters()) + \
                   list(self.classifier.parameters())
        else:
            # 阶段3：所有参数
            return list(self.parameters())
    
    def count_parameters(self) -> Dict[str, int]:
        """统计各模块参数量"""
        counts = {
            'global_encoder': sum(p.numel() for p in self.global_encoder.parameters()),
            'context_encoder': sum(p.numel() for p in self.context_encoder.parameters()),
            'film_module': sum(p.numel() for p in self.film_module.parameters()),
            'classifier': sum(p.numel() for p in self.classifier.parameters()),
        }
        counts['total'] = sum(counts.values())
        counts['trainable'] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return counts


def build_model(config) -> HCADR:
    """
    根据配置构建模型
    
    参数：
        config: HCADRConfig对象
    
    返回：
        HCADR模型实例
    """
    model = HCADR(
        n_genes=config.model.n_genes,
        n_compounds=config.model.n_compounds,
        n_cell_lines=config.model.n_cell_lines,
        encoder_hidden_dims=config.model.encoder_hidden_dims,
        encoder_dropout=config.model.encoder_dropout,
        context_hidden_dim=config.model.context_hidden_dim
    )
    
    return model


if __name__ == "__main__":
    # 测试模型
    print("Testing HCA-DR Model...")
    
    model = HCADR(
        n_genes=978,
        n_compounds=6971,
        n_cell_lines=114
    )
    
    # 测试输入
    batch_size = 32
    x_pert = torch.randn(batch_size, 978)
    x_ctx = torch.randn(batch_size, 978)
    
    # 前向传播
    output = model(x_pert, x_ctx)
    
    print(f"\n✓ Model output shapes:")
    print(f"  logits: {output.logits.shape}")
    print(f"  logits_global: {output.logits_global.shape}")
    print(f"  h_global: {output.h_global.shape}")
    print(f"  h_ctx: {output.h_ctx.shape}")
    print(f"  h_adapted: {output.h_adapted.shape}")
    print(f"  alpha: {output.alpha.shape}")
    print(f"  gamma: {output.gamma.shape}")
    print(f"  beta: {output.beta.shape}")
    
    print(f"\n✓ Parameter counts:")
    for name, count in model.count_parameters().items():
        print(f"  {name}: {count:,}")
    
    print(f"\n✓ Alpha statistics:")
    print(f"  mean: {output.alpha.mean().item():.4f}")
    print(f"  std: {output.alpha.std().item():.4f}")