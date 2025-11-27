#!/usr/bin/env python
"""
汇总超参数优化结果
"""
import pickle
import pandas as pd
from pathlib import Path
import sys
import argparse
import numpy as np


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
            elif 'Focal Gamma:' in line:
                config['focal_gamma'] = float(line.split(':')[1].strip())
    return config


def extract_metrics_from_summary(summary_path):
    """从 ensemble_summary.pkl 提取指标"""
    try:
        with open(summary_path, 'rb') as f:
            summary = pickle.load(f)
        
        metrics = summary['ensemble_metrics']
        return {
            'mean_recall': metrics['mean_recall'],
            'std_recall': metrics['std_recall'],
            'mean_top1_acc': metrics['mean_top1_acc'],
            'std_top1_acc': metrics['std_top1_acc'],
            'mean_top10_acc': metrics['mean_top10_acc'],
            'std_top10_acc': metrics['std_top10_acc']
        }
    except Exception as e:
        print(f"警告: 无法读取 {summary_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="汇总超参数优化结果")
    parser.add_argument(
        '--search-dir',
        type=str,
        default='../models/hyperparam_search',
        help='超参数搜索结果目录'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='hyperparam_results_summary.csv',
        help='输出CSV文件路径'
    )
    
    args = parser.parse_args()
    
    search_dir = Path(args.search_dir)
    
    if not search_dir.exists():
        print(f"错误: 目录不存在 {search_dir}")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print(f"超参数优化结果汇总")
    print(f"{'='*80}")
    print(f"搜索目录: {search_dir}")
    print(f"输出文件: {args.output}")
    
    # 收集所有配置结果
    results = []
    config_dirs = sorted([d for d in search_dir.iterdir() if d.is_dir() and d.name.startswith('config_')])
    
    print(f"\n找到 {len(config_dirs)} 个配置目录")
    print(f"\n正在提取结果...")
    
    for config_dir in config_dirs:
        config_file = config_dir / 'config.txt'
        summary_file = config_dir / 'ensemble_summary.pkl'
        
        if not config_file.exists():
            print(f"⚠ 跳过 {config_dir.name}: 缺少 config.txt")
            continue
        
        if not summary_file.exists():
            print(f"⚠ 跳过 {config_dir.name}: 缺少 ensemble_summary.pkl")
            continue
        
        # 解析配置
        config = parse_config_from_path(config_file)
        
        # 提取指标
        metrics = extract_metrics_from_summary(summary_file)
        
        if metrics is None:
            continue
        
        # 合并结果
        result = {**config, **metrics}
        result['config_dir'] = config_dir.name
        results.append(result)
        
        print(f"✓ {config_dir.name}: Recall={metrics['mean_recall']:.4f}")
    
    if not results:
        print("\n错误: 未找到有效结果")
        sys.exit(1)
    
    # 创建 DataFrame
    df = pd.DataFrame(results)
    
    # 按 mean_recall 排序
    df = df.sort_values('mean_recall', ascending=False)
    
    # 保存到 CSV
    df.to_csv(args.output, index=False)
    
    print(f"\n{'='*80}")
    print(f"结果汇总完成!")
    print(f"{'='*80}")
    print(f"总配置数: {len(df)}")
    print(f"结果已保存到: {args.output}")
    
    # 显示 Top 10
    print(f"\n{'='*80}")
    print(f"Top 10 配置 (按 Top 1% Recall 排序)")
    print(f"{'='*80}\n")
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    
    top10 = df.head(10)[['learning_rate', 'lr_scheduler', 'weight_decay', 
                         'focal_gamma', 'mean_recall', 'std_recall',
                         'mean_top1_acc', 'mean_top10_acc']]
    
    print(top10.to_string(index=False))
    
    # 统计分析
    print(f"\n{'='*80}")
    print(f"统计分析")
    print(f"{'='*80}\n")
    
    print("最佳配置:")
    best_row = df.iloc[0]
    print(f"  学习率: {best_row['learning_rate']}")
    print(f"  调度器: {best_row['lr_scheduler']}")
    print(f"  权重衰减: {best_row['weight_decay']}")
    print(f"  Focal Gamma: {best_row['focal_gamma']}")
    print(f"  Top 1% Recall: {best_row['mean_recall']:.4f} ± {best_row['std_recall']:.4f}")
    print(f"  Top-1 Acc: {best_row['mean_top1_acc']:.4f}")
    print(f"  Top-10 Acc: {best_row['mean_top10_acc']:.4f}")
    
    print(f"\n按调度器类型分组:")
    for scheduler in df['lr_scheduler'].unique():
        scheduler_df = df[df['lr_scheduler'] == scheduler]
        best_for_scheduler = scheduler_df.iloc[0]
        print(f"  {scheduler}: 最佳 Recall = {best_for_scheduler['mean_recall']:.4f} " + 
              f"(LR={best_for_scheduler['learning_rate']}, WD={best_for_scheduler['weight_decay']})")
    
    print(f"\n按学习率分组 (平均 Recall):")
    lr_groups = df.groupby('learning_rate')['mean_recall'].agg(['mean', 'std', 'count'])
    lr_groups = lr_groups.sort_values('mean', ascending=False)
    print(lr_groups.to_string())
    
    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    main()