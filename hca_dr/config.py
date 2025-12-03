"""
HCA-DR Configuration
配置文件：定义所有超参数和训练设置
"""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class ModelConfig:
    """模型架构配置"""
    # 输入维度
    n_genes: int = 978
    n_compounds: int = 6971  # 将在运行时更新
    n_cell_lines: int = 114  # 将在运行时更新
    
    # 全局扰动编码器 (Module 1)
    encoder_hidden_dims: List[int] = field(default_factory=lambda: [1024, 2048])
    encoder_dropout: float = 0.64
    
    # 细胞系上下文编码器 (Module 2)
    context_hidden_dim: int = 256
    
    # FiLM模块 (Module 3)
    film_hidden_dim: int = 2048  # 与encoder最后一层一致
    
    # 分类头 (Module 4)
    classifier_input_dim: int = 2048


@dataclass
class TrainingConfig:
    """训练配置"""
    # 基础训练参数
    batch_size: int = 256
    num_workers: int = 4
    
    # 三阶段训练的epoch数
    stage1_epochs: int = 20  # 全局模型预训练
    stage2_epochs: int = 20  # FiLM分支训练
    stage3_epochs: int = 10  # 端到端微调
    
    # 学习率
    stage1_lr: float = 0.0139
    stage2_lr: float = 0.01
    stage3_lr: float = 0.001
    
    # 优化器参数
    weight_decay: float = 1e-5
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    
    # 学习率调度
    warmup_epochs: int = 5
    
    # Context Dropout
    context_dropout_prob: float = 0.15
    
    # 损失函数权重
    lambda_contrast: float = 0.1      # 对比学习权重
    lambda_global: float = 0.3        # 全局正则化权重
    lambda_alpha_penalty: float = 0.5 # α-penalty权重
    
    # 阶段3的调整权重
    stage3_lambda_contrast: float = 0.05
    stage3_lambda_global: float = 0.1
    stage3_lambda_alpha_penalty: float = 0.3
    
    # Focal Loss参数
    focal_gamma: float = 2.0
    
    # 对比学习参数
    contrast_temperature: float = 0.1
    
    # 分层采样
    stratified_alpha: float = 0.7
    
    # 早停
    patience: int = 10
    min_delta: float = 1e-4
    
    # 设备
    device: str = "cuda"
    
    # 随机种子
    seed: int = 42


@dataclass
class DataConfig:
    """数据配置"""
    # 数据路径
    data_dir: str = "E:/科研/Models/drugreflector/processed_data"
    data_file: str = "hca_dr_training_data.pkl"
    
    # 交叉验证
    n_folds: int = 3
    train_folds: List[int] = field(default_factory=lambda: [0, 1])
    val_folds: List[int] = field(default_factory=lambda: [2])
    
    # 输出路径
    output_dir: str = "E:/科研/Models/drugreflector/outputs/hca_dr"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"


@dataclass
class EvalConfig:
    """评估配置"""
    # Top-k Recall
    top_k_percentages: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.1])
    
    # LOCO评估
    loco_n_cells: int = 10  # 随机选择的细胞系数量
    
    # α值分析阈值
    alpha_ood_threshold: float = 0.3
    
    # OOD Fallback Quality阈值
    ood_fallback_quality_threshold: float = 0.7


@dataclass
class HCADRConfig:
    """HCA-DR完整配置"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    
    # 实验名称
    experiment_name: str = "hca_dr_v1"
    
    def __post_init__(self):
        """创建必要的目录"""
        output_path = Path(self.data.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / self.data.checkpoint_dir).mkdir(exist_ok=True)
        (output_path / self.data.log_dir).mkdir(exist_ok=True)
    
    def update_from_data(self, n_compounds: int, n_cell_lines: int):
        """根据数据更新配置"""
        self.model.n_compounds = n_compounds
        self.model.n_cell_lines = n_cell_lines


def get_default_config() -> HCADRConfig:
    """获取默认配置"""
    return HCADRConfig()


if __name__ == "__main__":
    config = get_default_config()
    print("HCA-DR Configuration:")
    print(f"  Model: {config.model}")
    print(f"  Training: {config.training}")
    print(f"  Data: {config.data}")