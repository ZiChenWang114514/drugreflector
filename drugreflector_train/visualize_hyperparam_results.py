#!/usr/bin/env python
"""
可视化超参数优化结果
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='hyperparam_plots')
    args = parser.parse_args()
    
    df = pd.read_csv(args.csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置绘图风格
    sns.set_style("whitegrid")
    
    # 1. 学习率 vs Recall
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, scheduler in enumerate(df['lr_scheduler'].unique()):
        df_sched = df[df['lr_scheduler'] == scheduler]
        axes[idx].scatter(df_sched['learning_rate'], df_sched['mean_recall'], s=100, alpha=0.6)
        axes[idx].set_xlabel('Learning Rate')
        axes[idx].set_ylabel('Top 1% Recall')
        axes[idx].set_title(f'Scheduler: {scheduler}')
        axes[idx].set_xscale('log')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'lr_vs_recall.png', dpi=300)
    print(f"✓ 已保存: {output_dir / 'lr_vs_recall.png'}")
    
    # 2. 热图：LR vs Weight Decay
    for scheduler in df['lr_scheduler'].unique():
        df_sched = df[df['lr_scheduler'] == scheduler]
        pivot = df_sched.pivot_table(
            values='mean_recall',
            index='weight_decay',
            columns='learning_rate',
            aggfunc='mean'
        )
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot, annot=True, fmt='.4f', cmap='YlOrRd')
        plt.title(f'Top 1% Recall Heatmap - {scheduler}')
        plt.tight_layout()
        plt.savefig(output_dir / f'heatmap_{scheduler}.png', dpi=300)
        print(f"✓ 已保存: {output_dir / f'heatmap_{scheduler}.png'}")
    
    # 3. 箱型图：调度器比较
    plt.figure(figsize=(10, 6))
    df.boxplot(column='mean_recall', by='lr_scheduler', ax=plt.gca())
    plt.xlabel('LR Scheduler')
    plt.ylabel('Top 1% Recall')
    plt.title('Recall Distribution by Scheduler')
    plt.suptitle('')
    plt.tight_layout()
    plt.savefig(output_dir / 'scheduler_comparison.png', dpi=300)
    print(f"✓ 已保存: {output_dir / 'scheduler_comparison.png'}")
    
    print(f"\n所有图表已保存到: {output_dir}")

if __name__ == '__main__':
    main()