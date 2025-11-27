#!/usr/bin/env python
"""
汇总单折超参数优化结果
"""
import pickle
import pandas as pd
from pathlib import Path
import sys
import argparse
import numpy as np
import torch


def parse_config_from_path(config_path):
    """从配置文件解析超参数"""
    config = {}
    with open(config_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if '学习率:' in line and '最小学习率' not in line and '衰减' not in line:
                config['learning_rate'] = float(line.split(':')[1].strip())
            elif '学习率调度器:' in line:
                config['lr_scheduler'] = line.split(':')[1].strip()
            elif '权重衰减:' in line:
                config['weight_decay'] = float(line.split(':')[1].strip())
            elif 'Dropout:' in line:
                config['dropout'] = float(line.split(':')[1].strip())
            elif 'Focal Gamma:' in line:
                config['focal_gamma'] = float(line.split(':')[1].strip())
            elif '训练 Fold:' in line:
                config['fold_id'] = int(line.split(':')[1].strip())
    return config


def extract_metrics_from_checkpoint(checkpoint_path, fold_id):
    """从单折模型 checkpoint 提取指标"""
    try:
        # 尝试从 metrics.pkl 读取（优先）
        metrics_file = checkpoint_path.parent / 'metrics.pkl'
        if metrics_file.exists():
            with open(metrics_file, 'rb') as f:
                metrics = pickle.load(f)
            return metrics
        
        # 否则从模型 checkpoint 读取
        model_file = checkpoint_path.parent / f'model_fold_{fold_id}.pt'
        if not model_file.exists():
            print(f"警告: 模型文件不存在 {model_file}")
            return None
        
        checkpoint = torch.load(model_file, map_location='cpu')
        history = checkpoint['history']
        
        # 找到最佳 recall 的 epoch
        best_recall_idx = max(range(len(history['val_recall'])), 
                             key=lambda i: history['val_recall'][i])
        
        return {
            'best_recall': history['val_recall'][best_recall_idx],
            'best_top1_acc': history['val_top1_acc'][best_recall_idx],
            'best_top10_acc': history['val_top10_acc'][best_recall_idx],
            'best_epoch': best_recall_idx,
            'fold_id': fold_id
        }
        
    except Exception as e:
        print(f"警告: 无法读取 {checkpoint_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="汇总单折超参数优化结果")
    parser.add_argument(
        '--search-dir',
        type=str,
        default='../models/hyperparam_search_1fold',
        help='超参数搜索结果目录'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='hyperparam_results_1fold_summary.csv',
        help='输出CSV文件路径'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=20,
        help='显示前 N 个最佳配置'
    )
    
    args = parser.parse_args()
    
    search_dir = Path(args.search_dir)
    
    if not search_dir.exists():
        print(f"错误: 目录不存在 {search_dir}")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print(f"单折超参数优化结果汇总")
    print(f"{'='*80}")
    print(f"搜索目录: {search_dir}")
    print(f"输出文件: {args.output}")
    
    # 收集所有配置结果
    results = []
    config_dirs = sorted([d for d in search_dir.iterdir() 
                         if d.is_dir() and d.name.startswith('config_')])
    
    print(f"\n找到 {len(config_dirs)} 个配置目录")
    print(f"\n正在提取结果...")
    
    for config_dir in config_dirs:
        config_file = config_dir / 'config.txt'
        
        if not config_file.exists():
            print(f"⚠ 跳过 {config_dir.name}: 缺少 config.txt")
            continue
        
        # 解析配置
        config = parse_config_from_path(config_file)
        
        if 'fold_id' not in config:
            print(f"⚠ 跳过 {config_dir.name}: 无法解析 fold_id")
            continue
        
        # 提取指标
        metrics = extract_metrics_from_checkpoint(config_dir, config['fold_id'])
        
        if metrics is None:
            continue
        
        # 合并结果
        result = {**config, **metrics}
        result['config_dir'] = config_dir.name
        results.append(result)
        
        print(f"✓ {config_dir.name}: Recall={metrics['best_recall']:.4f}")
    
    if not results:
        print("\n错误: 未找到有效结果")
        sys.exit(1)
    
    # 创建 DataFrame
    df = pd.DataFrame(results)
    
    # 按 best_recall 排序
    df = df.sort_values('best_recall', ascending=False)
    
    # 保存到 CSV
    df.to_csv(args.output, index=False)
    
    print(f"\n{'='*80}")
    print(f"结果汇总完成!")
    print(f"{'='*80}")
    print(f"总配置数: {len(df)}")
    print(f"结果已保存到: {args.output}")
    
    # 显示 Top N
    print(f"\n{'='*80}")
    print(f"Top {args.top_n} 配置 (按 Top 1% Recall 排序)")
    print(f"{'='*80}\n")
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    
    top_n = df.head(args.top_n)[['learning_rate', 'lr_scheduler', 'weight_decay', 
                                   'dropout', 'focal_gamma', 'best_recall',
                                   'best_top1_acc', 'best_top10_acc', 'best_epoch']]
    
    print(top_n.to_string(index=False))
    
    # 统计分析
    print(f"\n{'='*80}")
    print(f"统计分析")
    print(f"{'='*80}\n")
    
    print("最佳配置:")
    best_row = df.iloc[0]
    print(f"  学习率: {best_row['learning_rate']}")
    print(f"  调度器: {best_row['lr_scheduler']}")
    print(f"  权重衰减: {best_row['weight_decay']}")
    print(f"  Dropout: {best_row['dropout']}")
    print(f"  Focal Gamma: {best_row['focal_gamma']}")
    print(f"  Top 1% Recall: {best_row['best_recall']:.4f}")
    print(f"  Top-1 Acc: {best_row['best_top1_acc']:.4f}")
    print(f"  Top-10 Acc: {best_row['best_top10_acc']:.4f}")
    print(f"  最佳 Epoch: {best_row['best_epoch'] + 1}")
    
    print(f"\n按调度器类型分组:")
    for scheduler in df['lr_scheduler'].unique():
        scheduler_df = df[df['lr_scheduler'] == scheduler]
        best_for_scheduler = scheduler_df.iloc[0]
        print(f"  {scheduler}: 最佳 Recall = {best_for_scheduler['best_recall']:.4f} " + 
              f"(LR={best_for_scheduler['learning_rate']}, " +
              f"WD={best_for_scheduler['weight_decay']}, " +
              f"Dropout={best_for_scheduler['dropout']})")
    
    print(f"\n按 Dropout 分组 (平均 Recall):")
    dropout_groups = df.groupby('dropout')['best_recall'].agg(['mean', 'std', 'count'])
    dropout_groups = dropout_groups.sort_values('mean', ascending=False)
    print(dropout_groups.to_string())
    
    print(f"\n按学习率分组 (平均 Recall):")
    lr_groups = df.groupby('learning_rate')['best_recall'].agg(['mean', 'std', 'count'])
    lr_groups = lr_groups.sort_values('mean', ascending=False)
    print(lr_groups.to_string())
    
    print(f"\n按权重衰减分组 (平均 Recall):")
    wd_groups = df.groupby('weight_decay')['best_recall'].agg(['mean', 'std', 'count'])
    wd_groups = wd_groups.sort_values('mean', ascending=False)
    print(wd_groups.to_string())
    
    # 生成推荐配置
    print(f"\n{'='*80}")
    print(f"推荐配置 (基于 Top 10 平均)")
    print(f"{'='*80}")
    
    top10 = df.head(10)
    
    # 找出 Top 10 中最常见的超参数值
    print(f"\n最优超参数倾向:")
    print(f"  学习率 (众数): {top10['learning_rate'].mode().values[0]}")
    print(f"  调度器 (众数): {top10['lr_scheduler'].mode().values[0]}")
    print(f"  权重衰减 (众数): {top10['weight_decay'].mode().values[0]}")
    print(f"  Dropout (众数): {top10['dropout'].mode().values[0]}")
    
    print(f"\n  平均 Recall (Top 10): {top10['best_recall'].mean():.4f} ± {top10['best_recall'].std():.4f}")
    
    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    main()