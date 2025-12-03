"""
HCA-DR Training Script
主训练脚本

用法：
    python train.py --data_path <path_to_data> --output_dir <output_dir>
"""

import argparse
import torch
import numpy as np
import random
from pathlib import Path
import sys

from config import HCADRConfig, get_default_config
from model import HCADR, build_model
from dataset import load_data, create_dataloaders
from trainer import HCADRTrainer


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='HCA-DR Training')
    
    # 数据路径
    parser.add_argument('--data_path', type=str, 
                        default='E:/科研/Models/drugreflector/processed_data/hca_dr_training_data.pkl',
                        help='Path to HCA-DR training data')
    parser.add_argument('--output_dir', type=str,
                        default='E:/科研/Models/drugreflector/outputs/hca_dr',
                        help='Output directory')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--stage1_epochs', type=int, default=20,
                        help='Stage 1 epochs')
    parser.add_argument('--stage2_epochs', type=int, default=20,
                        help='Stage 2 epochs')
    parser.add_argument('--stage3_epochs', type=int, default=10,
                        help='Stage 3 epochs')
    
    # 学习率
    parser.add_argument('--stage1_lr', type=float, default=0.0139,
                        help='Stage 1 learning rate')
    parser.add_argument('--stage2_lr', type=float, default=0.01,
                        help='Stage 2 learning rate')
    parser.add_argument('--stage3_lr', type=float, default=0.001,
                        help='Stage 3 learning rate')
    
    # 模型参数
    parser.add_argument('--encoder_dropout', type=float, default=0.64,
                        help='Encoder dropout rate')
    parser.add_argument('--context_dropout', type=float, default=0.15,
                        help='Context dropout probability')
    
    # 损失权重
    parser.add_argument('--lambda_contrast', type=float, default=0.1,
                        help='Contrastive loss weight')
    parser.add_argument('--lambda_global', type=float, default=0.3,
                        help='Global regularization weight')
    parser.add_argument('--lambda_alpha', type=float, default=0.5,
                        help='Alpha penalty weight')
    
    # 其他
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # 交叉验证
    parser.add_argument('--train_folds', type=int, nargs='+', default=[0, 1],
                        help='Training folds')
    parser.add_argument('--val_folds', type=int, nargs='+', default=[2],
                        help='Validation folds')
    
    # 恢复训练
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint for resuming training')
    
    return parser.parse_args()


def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    print("=" * 80)
    print("🧬 HCA-DR (Hierarchical Cell-Aware DrugReflector) Training")
    print("=" * 80)
    
    # 设置随机种子
    set_seed(args.seed)
    print(f"✓ Random seed: {args.seed}")
    
    # 设置设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        args.device = 'cpu'
    device = torch.device(args.device)
    print(f"✓ Device: {device}")
    
    if args.device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # 创建配置
    config = get_default_config()
    
    # 更新配置
    config.training.batch_size = args.batch_size
    config.training.stage1_epochs = args.stage1_epochs
    config.training.stage2_epochs = args.stage2_epochs
    config.training.stage3_epochs = args.stage3_epochs
    config.training.stage1_lr = args.stage1_lr
    config.training.stage2_lr = args.stage2_lr
    config.training.stage3_lr = args.stage3_lr
    config.training.context_dropout_prob = args.context_dropout
    config.training.lambda_contrast = args.lambda_contrast
    config.training.lambda_global = args.lambda_global
    config.training.lambda_alpha_penalty = args.lambda_alpha
    config.training.device = args.device
    config.training.seed = args.seed
    
    config.model.encoder_dropout = args.encoder_dropout
    
    config.data.output_dir = args.output_dir
    config.data.train_folds = args.train_folds
    config.data.val_folds = args.val_folds
    
    # 创建输出目录
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / 'checkpoints').mkdir(exist_ok=True)
    (output_path / 'logs').mkdir(exist_ok=True)
    
    # 加载数据
    print("\n" + "=" * 80)
    print("📂 Loading Data")
    print("=" * 80)
    
    data = load_data(args.data_path)
    
    # 更新配置中的数据维度
    config.update_from_data(
        n_compounds=data['n_compounds'],
        n_cell_lines=data['n_cell_lines']
    )
    
    # 创建DataLoader
    print("\n" + "=" * 80)
    print("📊 Creating DataLoaders")
    print("=" * 80)
    
    train_loader, val_loader = create_dataloaders(
        data=data,
        train_folds=args.train_folds,
        val_folds=args.val_folds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        context_dropout_prob=args.context_dropout,
        use_stratified_sampling=True
    )
    
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")
    
    # 构建模型
    print("\n" + "=" * 80)
    print("🏗️  Building Model")
    print("=" * 80)
    
    model = build_model(config)
    
    # 打印模型信息
    param_counts = model.count_parameters()
    print(f"\n✓ Model Architecture:")
    print(f"  Global Encoder: {param_counts['global_encoder']:,} params")
    print(f"  Context Encoder: {param_counts['context_encoder']:,} params")
    print(f"  FiLM Module: {param_counts['film_module']:,} params")
    print(f"  Classifier: {param_counts['classifier']:,} params")
    print(f"  Total: {param_counts['total']:,} params")
    
    # 创建训练器
    print("\n" + "=" * 80)
    print("🎯 Initializing Trainer")
    print("=" * 80)
    
    trainer = HCADRTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device
    )
    
    # 恢复训练（如果指定）
    if args.resume:
        print(f"\n📥 Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # 开始训练
    results = trainer.train_all_stages()
    
    # 打印最终结果
    print("\n" + "=" * 80)
    print("📊 Training Results Summary")
    print("=" * 80)
    print(f"Stage 1 Final Val Loss: {results['stage1']['final_val_loss']:.4f}")
    print(f"Stage 2 Final Val Loss: {results['stage2']['final_val_loss']:.4f}")
    print(f"Stage 3 Final Val Loss: {results['stage3']['final_val_loss']:.4f}")
    print(f"Total Training Time: {results['total_time']/3600:.2f} hours")
    
    print("\n✅ Training completed successfully!")
    print(f"📁 Results saved to: {args.output_dir}")
    
    return results


if __name__ == "__main__":
    main()