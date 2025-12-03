"""
HCA-DR: Hierarchical Cell-Aware DrugReflector

一个基于细胞系感知的药物反应预测模型，使用FiLM调制实现细胞系特异性建模。

模块：
- model.py: HCA-DR模型架构
- dataset.py: 数据集定义
- losses.py: 损失函数
- trainer.py: 三阶段训练逻辑
- train.py: 训练脚本
- eval.py: 评估脚本
- config.py: 配置定义

使用示例：
    from hca_dr import HCADR, HCADRConfig, HCADRTrainer
    
    # 创建配置
    config = HCADRConfig()
    
    # 创建模型
    model = HCADR(
        n_genes=978,
        n_compounds=6971,
        n_cell_lines=114
    )
    
    # 训练
    trainer = HCADRTrainer(model, config, train_loader, val_loader)
    trainer.train_all_stages()
"""

from .model import HCADR, HCADROutput, build_model
from .config import HCADRConfig, ModelConfig, TrainingConfig, DataConfig, EvalConfig, get_default_config
from .dataset import HCADRDataset, LOCODataset, OODDataset, load_data, create_dataloaders
from .losses import FocalLoss, SupervisedContrastiveLoss, AlphaPenalty, HCADRLoss
from .trainer import HCADRTrainer, EarlyStopping
from .eval import HCADREvaluator, load_model_from_checkpoint

__version__ = "1.0.0"
__author__ = "Zi-Chen"

__all__ = [
    # Model
    'HCADR',
    'HCADROutput',
    'build_model',
    
    # Config
    'HCADRConfig',
    'ModelConfig',
    'TrainingConfig',
    'DataConfig',
    'EvalConfig',
    'get_default_config',
    
    # Dataset
    'HCADRDataset',
    'LOCODataset',
    'OODDataset',
    'load_data',
    'create_dataloaders',
    
    # Losses
    'FocalLoss',
    'SupervisedContrastiveLoss',
    'AlphaPenalty',
    'HCADRLoss',
    
    # Trainer
    'HCADRTrainer',
    'EarlyStopping',
    
    # Evaluation
    'HCADREvaluator',
    'load_model_from_checkpoint',
]