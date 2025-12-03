"""
HCA-DR Evaluation Script
全面评估脚本，包括：
1. Top-k% Recall
2. LOCO (Leave-One-Cell-Out) 评估
3. OOD (Out-of-Distribution) 评估
4. Alpha值分析

用法：
    python eval.py --checkpoint <path_to_checkpoint> --data_path <path_to_data>
"""

import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import seaborn as sns

from model import HCADR, HCADROutput
from dataset import (
    load_data, HCADRDataset, LOCODataset, OODDataset, collate_fn
)
from torch.utils.data import DataLoader
from config import HCADRConfig


class HCADREvaluator:
    """
    HCA-DR评估器
    
    实现多种评估指标
    """
    
    def __init__(self,
                 model: HCADR,
                 device: str = 'cuda'):
        """
        初始化评估器
        
        参数：
            model: 训练好的HCA-DR模型
            device: 计算设备
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
    
    @torch.no_grad()
    def compute_top_k_recall(self,
                             dataloader: DataLoader,
                             k_percentages: List[float] = [0.01, 0.05, 0.1]) -> Dict:
        """
        计算Top-k% Recall
        
        对于每个化合物，计算其样本被正确预测在top-k%的比例
        
        参数：
            dataloader: 数据加载器
            k_percentages: k的百分比列表
        
        返回：
            包含各k值recall的字典
        """
        print("\n📊 Computing Top-k% Recall...")
        
        all_logits = []
        all_labels = []
        all_compounds = []
        
        for batch in tqdm(dataloader, desc="   Evaluating"):
            x_pert = batch['x_pert'].to(self.device)
            x_ctx = batch['x_ctx'].to(self.device)
            y = batch['y']
            
            output = self.model(x_pert, x_ctx)
            
            all_logits.append(output.logits.cpu())
            all_labels.append(y)
            all_compounds.append(y)
        
        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)
        
        n_classes = logits.shape[1]
        results = {}
        
        for k_pct in k_percentages:
            k = max(1, int(k_pct * n_classes))
            
            # 获取top-k预测
            _, top_k_pred = torch.topk(logits, k, dim=1)
            
            # 计算每个样本是否命中
            hits = (top_k_pred == labels.unsqueeze(1)).any(dim=1).float()
            
            # 按化合物分组计算recall
            compound_recalls = defaultdict(list)
            for i, (label, hit) in enumerate(zip(labels.numpy(), hits.numpy())):
                compound_recalls[label].append(hit)
            
            # 计算平均recall
            per_compound_recall = {
                c: np.mean(hits) for c, hits in compound_recalls.items()
            }
            mean_recall = np.mean(list(per_compound_recall.values()))
            
            results[f'recall@{k_pct*100:.0f}%'] = mean_recall
            results[f'k@{k_pct*100:.0f}%'] = k
            
            print(f"   Top-{k_pct*100:.0f}% (k={k}): Recall = {mean_recall:.4f}")
        
        # 整体准确率
        pred = logits.argmax(dim=1)
        accuracy = (pred == labels).float().mean().item()
        results['accuracy'] = accuracy
        print(f"   Accuracy: {accuracy:.4f}")
        
        return results
    
    @torch.no_grad()
    def analyze_alpha_values(self,
                             dataloader: DataLoader,
                             ood_dataloader: Optional[DataLoader] = None) -> Dict:
        """
        分析Alpha值分布
        
        验证模型是否学会了在OOD情况下回退到全局模型
        
        参数：
            dataloader: 正常数据加载器
            ood_dataloader: OOD数据加载器（上下文全为0）
        
        返回：
            Alpha分析结果
        """
        print("\n📊 Analyzing Alpha Values...")
        
        # 收集正常样本的alpha
        seen_alphas = []
        for batch in tqdm(dataloader, desc="   Normal samples"):
            x_pert = batch['x_pert'].to(self.device)
            x_ctx = batch['x_ctx'].to(self.device)
            
            output = self.model(x_pert, x_ctx)
            seen_alphas.append(output.alpha.cpu().numpy())
        
        seen_alphas = np.concatenate(seen_alphas, axis=0).flatten()
        
        results = {
            'seen_alpha_mean': float(np.mean(seen_alphas)),
            'seen_alpha_std': float(np.std(seen_alphas)),
            'seen_alpha_median': float(np.median(seen_alphas)),
            'seen_alpha_min': float(np.min(seen_alphas)),
            'seen_alpha_max': float(np.max(seen_alphas)),
        }
        
        print(f"   Seen samples α: mean={results['seen_alpha_mean']:.4f}, "
              f"std={results['seen_alpha_std']:.4f}")
        
        # 如果提供了OOD数据
        if ood_dataloader is not None:
            ood_alphas = []
            for batch in tqdm(ood_dataloader, desc="   OOD samples"):
                x_pert = batch['x_pert'].to(self.device)
                x_ctx = batch['x_ctx'].to(self.device)  # 全为0
                
                output = self.model(x_pert, x_ctx)
                ood_alphas.append(output.alpha.cpu().numpy())
            
            ood_alphas = np.concatenate(ood_alphas, axis=0).flatten()
            
            results['ood_alpha_mean'] = float(np.mean(ood_alphas))
            results['ood_alpha_std'] = float(np.std(ood_alphas))
            results['ood_alpha_median'] = float(np.median(ood_alphas))
            
            # OOD Fallback Quality
            fallback_quality = (results['seen_alpha_mean'] - results['ood_alpha_mean']) / \
                               max(results['seen_alpha_mean'], 1e-6)
            results['ood_fallback_quality'] = fallback_quality
            
            print(f"   OOD samples α: mean={results['ood_alpha_mean']:.4f}, "
                  f"std={results['ood_alpha_std']:.4f}")
            print(f"   OOD Fallback Quality: {fallback_quality:.4f}")
        
        return results
    
    @torch.no_grad()
    def loco_evaluation(self,
                        data: Dict,
                        cell_ids_to_test: Optional[List[int]] = None,
                        n_cells: int = 10,
                        batch_size: int = 256) -> Dict:
        """
        Leave-One-Cell-Out (LOCO) 评估
        
        对于每个被留出的细胞系，测试模型的泛化能力
        
        参数：
            data: HCA-DR数据字典
            cell_ids_to_test: 要测试的细胞系ID列表
            n_cells: 如果未指定cell_ids_to_test，随机选择的细胞系数量
            batch_size: 批大小
        
        返回：
            LOCO评估结果
        """
        print("\n📊 LOCO (Leave-One-Cell-Out) Evaluation...")
        
        # 获取所有细胞系
        unique_cells = np.unique(data['cell_ids'])
        unique_cells = unique_cells[unique_cells >= 0]  # 排除-1
        
        if cell_ids_to_test is None:
            if len(unique_cells) > n_cells:
                cell_ids_to_test = np.random.choice(unique_cells, n_cells, replace=False)
            else:
                cell_ids_to_test = unique_cells
        
        print(f"   Testing {len(cell_ids_to_test)} cell lines")
        
        loco_results = {}
        all_recalls = []
        all_alphas = []
        
        for cell_id in tqdm(cell_ids_to_test, desc="   LOCO"):
            # 创建测试数据集（只包含该细胞系）
            test_dataset = LOCODataset(data, held_out_cell=int(cell_id), mode="test")
            
            if len(test_dataset) < 10:
                print(f"      Cell {cell_id}: too few samples ({len(test_dataset)}), skipping")
                continue
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn
            )
            
            # 评估
            correct = 0
            total = 0
            alphas = []
            
            for batch in test_loader:
                x_pert = batch['x_pert'].to(self.device)
                x_ctx = batch['x_ctx'].to(self.device)
                y = batch['y'].to(self.device)
                
                output = self.model(x_pert, x_ctx)
                pred = output.logits.argmax(dim=1)
                
                correct += (pred == y).sum().item()
                total += len(y)
                alphas.append(output.alpha.cpu().numpy())
            
            accuracy = correct / total
            mean_alpha = np.mean(np.concatenate(alphas))
            
            loco_results[int(cell_id)] = {
                'accuracy': accuracy,
                'n_samples': total,
                'mean_alpha': float(mean_alpha)
            }
            
            all_recalls.append(accuracy)
            all_alphas.append(mean_alpha)
        
        # 汇总
        results = {
            'per_cell_results': loco_results,
            'mean_accuracy': float(np.mean(all_recalls)),
            'std_accuracy': float(np.std(all_recalls)),
            'mean_alpha': float(np.mean(all_alphas)),
            'n_cells_tested': len(cell_ids_to_test)
        }
        
        print(f"\n   LOCO Results:")
        print(f"      Mean Accuracy: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
        print(f"      Mean Alpha: {results['mean_alpha']:.4f}")
        
        return results
    
    @torch.no_grad()
    def ood_evaluation(self,
                       data: Dict,
                       fold_ids: List[int],
                       batch_size: int = 256) -> Dict:
        """
        OOD (Out-of-Distribution) 评估
        
        将所有上下文设为零向量，测试模型的回退能力
        
        参数：
            data: HCA-DR数据字典
            fold_ids: 使用的fold列表
            batch_size: 批大小
        
        返回：
            OOD评估结果
        """
        print("\n📊 OOD (Zero-Context) Evaluation...")
        
        # 创建OOD数据集
        ood_dataset = OODDataset(data, fold_ids=fold_ids)
        ood_loader = DataLoader(
            ood_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )
        
        # 同时创建正常数据集进行对比
        normal_dataset = HCADRDataset(
            data, fold_ids=fold_ids, mode="val", context_dropout_prob=0.0
        )
        normal_loader = DataLoader(
            normal_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )
        
        # 评估正常样本
        print("   Evaluating normal samples...")
        normal_correct = 0
        normal_total = 0
        normal_alphas = []
        
        for batch in tqdm(normal_loader, desc="      Normal"):
            x_pert = batch['x_pert'].to(self.device)
            x_ctx = batch['x_ctx'].to(self.device)
            y = batch['y'].to(self.device)
            
            output = self.model(x_pert, x_ctx)
            pred = output.logits.argmax(dim=1)
            
            normal_correct += (pred == y).sum().item()
            normal_total += len(y)
            normal_alphas.append(output.alpha.cpu().numpy())
        
        normal_accuracy = normal_correct / normal_total
        normal_alpha_mean = np.mean(np.concatenate(normal_alphas))
        
        # 评估OOD样本
        print("   Evaluating OOD samples (zero context)...")
        ood_correct = 0
        ood_total = 0
        ood_alphas = []
        
        for batch in tqdm(ood_loader, desc="      OOD"):
            x_pert = batch['x_pert'].to(self.device)
            x_ctx = batch['x_ctx'].to(self.device)  # 全为0
            y = batch['y'].to(self.device)
            
            output = self.model(x_pert, x_ctx)
            pred = output.logits.argmax(dim=1)
            
            ood_correct += (pred == y).sum().item()
            ood_total += len(y)
            ood_alphas.append(output.alpha.cpu().numpy())
        
        ood_accuracy = ood_correct / ood_total
        ood_alpha_mean = np.mean(np.concatenate(ood_alphas))
        
        # 计算性能下降
        accuracy_drop = normal_accuracy - ood_accuracy
        alpha_drop = normal_alpha_mean - ood_alpha_mean
        
        results = {
            'normal_accuracy': normal_accuracy,
            'normal_alpha': float(normal_alpha_mean),
            'ood_accuracy': ood_accuracy,
            'ood_alpha': float(ood_alpha_mean),
            'accuracy_drop': accuracy_drop,
            'alpha_drop': float(alpha_drop),
            'ood_fallback_quality': float(alpha_drop / max(normal_alpha_mean, 1e-6))
        }
        
        print(f"\n   OOD Results:")
        print(f"      Normal: Accuracy={normal_accuracy:.4f}, α={normal_alpha_mean:.4f}")
        print(f"      OOD: Accuracy={ood_accuracy:.4f}, α={ood_alpha_mean:.4f}")
        print(f"      Accuracy Drop: {accuracy_drop:.4f}")
        print(f"      Alpha Drop: {alpha_drop:.4f}")
        print(f"      OOD Fallback Quality: {results['ood_fallback_quality']:.4f}")
        
        return results
    
    def full_evaluation(self,
                        data: Dict,
                        val_folds: List[int] = [2],
                        batch_size: int = 256,
                        output_dir: Optional[str] = None) -> Dict:
        """
        完整评估
        
        运行所有评估指标
        """
        print("\n" + "=" * 80)
        print("🎯 Full HCA-DR Evaluation")
        print("=" * 80)
        
        # 创建验证数据集
        val_dataset = HCADRDataset(
            data, fold_ids=val_folds, mode="val", context_dropout_prob=0.0
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )
        
        results = {}
        
        # 1. Top-k Recall
        results['top_k_recall'] = self.compute_top_k_recall(
            val_loader, 
            k_percentages=[0.01, 0.05, 0.1]
        )
        
        # 2. Alpha分析
        ood_dataset = OODDataset(data, fold_ids=val_folds)
        ood_loader = DataLoader(
            ood_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )
        results['alpha_analysis'] = self.analyze_alpha_values(val_loader, ood_loader)
        
        # 3. LOCO评估
        results['loco'] = self.loco_evaluation(data, n_cells=10, batch_size=batch_size)
        
        # 4. OOD评估
        results['ood'] = self.ood_evaluation(data, val_folds, batch_size)
        
        # 保存结果
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 保存JSON
            results_path = output_path / 'evaluation_results.json'
            
            # 转换numpy类型
            def convert_to_python(obj):
                if isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, dict):
                    return {k: convert_to_python(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_python(v) for v in obj]
                return obj
            
            results_serializable = convert_to_python(results)
            
            with open(results_path, 'w') as f:
                json.dump(results_serializable, f, indent=2)
            
            print(f"\n✓ Results saved to: {results_path}")
        
        # 打印摘要
        print("\n" + "=" * 80)
        print("📊 Evaluation Summary")
        print("=" * 80)
        print(f"Top-1% Recall: {results['top_k_recall']['recall@1%']:.4f}")
        print(f"Top-5% Recall: {results['top_k_recall']['recall@5%']:.4f}")
        print(f"Accuracy: {results['top_k_recall']['accuracy']:.4f}")
        print(f"LOCO Accuracy: {results['loco']['mean_accuracy']:.4f}")
        print(f"OOD Accuracy: {results['ood']['ood_accuracy']:.4f}")
        print(f"OOD Fallback Quality: {results['ood']['ood_fallback_quality']:.4f}")
        
        return results


def load_model_from_checkpoint(checkpoint_path: str, device: str = 'cuda') -> HCADR:
    """从检查点加载模型"""
    print(f"📥 Loading model from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config', None)
    
    if config is None:
        # 使用默认配置
        from config import get_default_config
        config = get_default_config()
    
    # 从state_dict推断n_compounds
    state_dict = checkpoint['model_state_dict']
    n_compounds = state_dict['classifier.classifier.weight'].shape[0]
    config.model.n_compounds = n_compounds
    
    # 构建模型
    model = HCADR(
        n_genes=config.model.n_genes,
        n_compounds=config.model.n_compounds,
        n_cell_lines=config.model.n_cell_lines,
        encoder_hidden_dims=config.model.encoder_hidden_dims,
        encoder_dropout=config.model.encoder_dropout,
        context_hidden_dim=config.model.context_hidden_dim
    )
    
    model.load_state_dict(state_dict)
    print(f"✓ Model loaded successfully")
    
    return model


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='HCA-DR Evaluation')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to HCA-DR data')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--val_folds', type=int, nargs='+', default=[2],
                        help='Validation folds')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device')
    parser.add_argument('--loco_n_cells', type=int, default=10,
                        help='Number of cells for LOCO evaluation')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    print("=" * 80)
    print("🎯 HCA-DR Model Evaluation")
    print("=" * 80)
    
    # 设置设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        args.device = 'cpu'
    device = torch.device(args.device)
    
    # 加载模型
    model = load_model_from_checkpoint(args.checkpoint, args.device)
    
    # 加载数据
    data = load_data(args.data_path)
    
    # 创建评估器
    evaluator = HCADREvaluator(model, device=args.device)
    
    # 运行完整评估
    results = evaluator.full_evaluation(
        data=data,
        val_folds=args.val_folds,
        batch_size=args.batch_size,
        output_dir=args.output_dir
    )
    
    print("\n✅ Evaluation completed!")
    
    return results


if __name__ == "__main__":
    main()