"""
HCA-DR Dataset
数据集定义，支持Context Dropout和分层采样
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import random


class HCADRDataset(Dataset):
    """
    HCA-DR数据集
    
    特性：
    1. 支持Context Dropout（训练时以概率p_drop将上下文置零）
    2. 返回dropout标志用于α-penalty计算
    3. 支持按fold划分训练/验证集
    """
    
    def __init__(self,
                 data: Dict,
                 fold_ids: List[int],
                 mode: str = "train",
                 context_dropout_prob: float = 0.15):
        """
        初始化数据集
        
        参数：
            data: HCA-DR数据字典
            fold_ids: 使用的fold列表
            mode: "train" 或 "val"/"test"
            context_dropout_prob: 上下文dropout概率（仅训练时生效）
        """
        self.mode = mode
        self.context_dropout_prob = context_dropout_prob if mode == "train" else 0.0
        
        # 根据fold筛选数据
        mask = np.isin(data['folds'], fold_ids)
        
        self.X_pert = torch.FloatTensor(data['X_pert'][mask])
        self.X_ctx = torch.FloatTensor(data['X_ctx'][mask])
        self.y = torch.LongTensor(data['y'][mask])
        self.cell_ids = torch.LongTensor(data['cell_ids'][mask])
        
        # 保存原始索引（用于调试）
        self.original_indices = np.where(mask)[0]
        
        # 元数据
        self.n_samples = len(self.y)
        self.n_genes = self.X_pert.shape[1]
        
        # 获取唯一的化合物和细胞系
        self.unique_compounds = torch.unique(self.y).numpy()
        self.unique_cell_lines = torch.unique(self.cell_ids).numpy()
        self.n_compounds = len(self.unique_compounds)
        self.n_cell_lines = len(self.unique_cell_lines)
        
        # 构建化合物->样本索引映射（用于分层采样）
        self.compound_to_indices = defaultdict(list)
        for idx, compound in enumerate(self.y.numpy()):
            self.compound_to_indices[compound].append(idx)
        
        # 构建细胞系->样本索引映射（用于对比学习）
        self.cell_to_indices = defaultdict(list)
        for idx, cell in enumerate(self.cell_ids.numpy()):
            self.cell_to_indices[cell].append(idx)
        
        print(f"✓ Dataset created ({mode}): {self.n_samples:,} samples, "
              f"{self.n_compounds} compounds, {self.n_cell_lines} cell lines")
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本
        
        返回：
            x_pert: 扰动签名 (978,)
            x_ctx: 细胞系上下文 (978,)，可能被dropout
            y: 药物标签
            cell_id: 细胞系ID
            is_ctx_dropout: 是否进行了context dropout (0或1)
        """
        x_pert = self.X_pert[idx]
        x_ctx = self.X_ctx[idx].clone()  # 克隆以避免修改原数据
        y = self.y[idx]
        cell_id = self.cell_ids[idx]
        
        # Context Dropout
        is_ctx_dropout = 0
        if self.mode == "train" and random.random() < self.context_dropout_prob:
            x_ctx = torch.zeros_like(x_ctx)
            is_ctx_dropout = 1
        
        return {
            'x_pert': x_pert,
            'x_ctx': x_ctx,
            'y': y,
            'cell_id': cell_id,
            'is_ctx_dropout': torch.tensor(is_ctx_dropout, dtype=torch.float32)
        }
    
    def get_sample_weights(self, alpha: float = 0.7) -> torch.Tensor:
        """
        计算分层采样权重
        
        使用加权采样确保每个细胞系和化合物都有代表性
        
        参数：
            alpha: 平衡因子
        
        返回：
            采样权重
        """
        # 计算每个化合物的频率
        compound_counts = torch.bincount(self.y)
        compound_weights = 1.0 / (compound_counts[self.y].float() ** alpha)
        
        # 归一化
        compound_weights = compound_weights / compound_weights.sum() * len(self.y)
        
        return compound_weights


class StratifiedBatchSampler(Sampler):
    """
    分层批采样器
    
    确保每个batch中：
    1. 每个细胞系至少有min_samples_per_cell个样本
    2. 尽可能平衡化合物分布
    """
    
    def __init__(self,
                 dataset: HCADRDataset,
                 batch_size: int,
                 min_samples_per_cell: int = 1,
                 drop_last: bool = False):
        """
        初始化采样器
        
        参数：
            dataset: HCADRDataset实例
            batch_size: 批大小
            min_samples_per_cell: 每个batch中每个细胞系的最小样本数
            drop_last: 是否丢弃最后不完整的batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.min_samples_per_cell = min_samples_per_cell
        self.drop_last = drop_last
        
        # 预计算
        self.cell_to_indices = dataset.cell_to_indices
        self.n_cells = len(self.cell_to_indices)
        self.n_samples = len(dataset)
    
    def __iter__(self):
        """生成batch索引"""
        # 打乱每个细胞系的样本顺序
        cell_indices = {
            cell: list(indices) 
            for cell, indices in self.cell_to_indices.items()
        }
        for indices in cell_indices.values():
            random.shuffle(indices)
        
        # 创建batch
        batches = []
        current_batch = []
        cell_pointers = {cell: 0 for cell in cell_indices}
        
        while True:
            # 为每个细胞系添加样本
            for cell in cell_indices:
                pointer = cell_pointers[cell]
                indices = cell_indices[cell]
                
                if pointer < len(indices):
                    # 添加min_samples_per_cell个样本
                    for _ in range(self.min_samples_per_cell):
                        if pointer < len(indices) and len(current_batch) < self.batch_size:
                            current_batch.append(indices[pointer])
                            pointer += 1
                    cell_pointers[cell] = pointer
            
            # 如果还没填满batch，随机添加样本
            all_remaining = []
            for cell, indices in cell_indices.items():
                pointer = cell_pointers[cell]
                all_remaining.extend(indices[pointer:])
            
            random.shuffle(all_remaining)
            
            while len(current_batch) < self.batch_size and all_remaining:
                idx = all_remaining.pop()
                if idx not in current_batch:
                    current_batch.append(idx)
            
            # 完成一个batch
            if len(current_batch) == self.batch_size:
                batches.append(current_batch)
                current_batch = []
            
            # 检查是否所有样本都被使用
            total_used = sum(cell_pointers.values())
            if total_used >= self.n_samples * 0.99:  # 允许1%的误差
                break
        
        # 处理最后的batch
        if current_batch and not self.drop_last:
            batches.append(current_batch)
        
        # 打乱batch顺序
        random.shuffle(batches)
        
        for batch in batches:
            yield batch
    
    def __len__(self) -> int:
        if self.drop_last:
            return self.n_samples // self.batch_size
        else:
            return (self.n_samples + self.batch_size - 1) // self.batch_size


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    批处理整理函数
    
    参数：
        batch: 样本列表
    
    返回：
        整理后的批数据
    """
    return {
        'x_pert': torch.stack([item['x_pert'] for item in batch]),
        'x_ctx': torch.stack([item['x_ctx'] for item in batch]),
        'y': torch.stack([item['y'] for item in batch]),
        'cell_id': torch.stack([item['cell_id'] for item in batch]),
        'is_ctx_dropout': torch.stack([item['is_ctx_dropout'] for item in batch])
    }


def load_data(data_path: str) -> Dict:
    """
    加载HCA-DR数据
    
    参数：
        data_path: 数据文件路径
    
    返回：
        数据字典
    """
    print(f"📖 Loading data from: {data_path}")
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"   ✓ Samples: {data['n_samples']:,}")
    print(f"   ✓ Compounds: {data['n_compounds']:,}")
    print(f"   ✓ Cell lines: {data['n_cell_lines']}")
    print(f"   ✓ Genes: {data['n_genes']}")
    
    return data


def create_dataloaders(data: Dict,
                       train_folds: List[int],
                       val_folds: List[int],
                       batch_size: int = 256,
                       num_workers: int = 4,
                       context_dropout_prob: float = 0.15,
                       use_stratified_sampling: bool = True) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练和验证DataLoader
    
    参数：
        data: HCA-DR数据字典
        train_folds: 训练fold列表
        val_folds: 验证fold列表
        batch_size: 批大小
        num_workers: 数据加载线程数
        context_dropout_prob: Context Dropout概率
        use_stratified_sampling: 是否使用分层采样
    
    返回：
        train_loader, val_loader
    """
    # 创建数据集
    train_dataset = HCADRDataset(
        data=data,
        fold_ids=train_folds,
        mode="train",
        context_dropout_prob=context_dropout_prob
    )
    
    val_dataset = HCADRDataset(
        data=data,
        fold_ids=val_folds,
        mode="val",
        context_dropout_prob=0.0  # 验证时不做dropout
    )
    
    # 创建DataLoader
    if use_stratified_sampling:
        train_sampler = StratifiedBatchSampler(
            train_dataset,
            batch_size=batch_size,
            min_samples_per_cell=1,
            drop_last=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader


class LOCODataset(Dataset):
    """
    Leave-One-Cell-Out (LOCO) 评估数据集
    
    用于评估模型在未见细胞系上的泛化能力
    """
    
    def __init__(self,
                 data: Dict,
                 held_out_cell: int,
                 mode: str = "test"):
        """
        初始化LOCO数据集
        
        参数：
            data: HCA-DR数据字典
            held_out_cell: 留出的细胞系ID
            mode: "train" (排除该细胞系) 或 "test" (只包含该细胞系)
        """
        self.mode = mode
        self.held_out_cell = held_out_cell
        
        # 根据细胞系筛选
        if mode == "train":
            mask = data['cell_ids'] != held_out_cell
        else:
            mask = data['cell_ids'] == held_out_cell
        
        self.X_pert = torch.FloatTensor(data['X_pert'][mask])
        self.X_ctx = torch.FloatTensor(data['X_ctx'][mask])
        self.y = torch.LongTensor(data['y'][mask])
        self.cell_ids = torch.LongTensor(data['cell_ids'][mask])
        
        self.n_samples = len(self.y)
        
        print(f"✓ LOCO Dataset ({mode}, cell={held_out_cell}): {self.n_samples:,} samples")
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'x_pert': self.X_pert[idx],
            'x_ctx': self.X_ctx[idx],
            'y': self.y[idx],
            'cell_id': self.cell_ids[idx],
            'is_ctx_dropout': torch.tensor(0, dtype=torch.float32)
        }


class OODDataset(Dataset):
    """
    Out-of-Distribution (OOD) 测试数据集
    
    将所有上下文设为零向量，测试模型的回退能力
    """
    
    def __init__(self, data: Dict, fold_ids: List[int]):
        """
        初始化OOD数据集
        
        参数：
            data: HCA-DR数据字典
            fold_ids: 使用的fold列表
        """
        mask = np.isin(data['folds'], fold_ids)
        
        self.X_pert = torch.FloatTensor(data['X_pert'][mask])
        self.X_ctx = torch.zeros_like(self.X_pert)  # 全部设为零
        self.y = torch.LongTensor(data['y'][mask])
        self.cell_ids = torch.LongTensor(data['cell_ids'][mask])
        
        self.n_samples = len(self.y)
        
        print(f"✓ OOD Dataset: {self.n_samples:,} samples (all contexts zeroed)")
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'x_pert': self.X_pert[idx],
            'x_ctx': self.X_ctx[idx],
            'y': self.y[idx],
            'cell_id': self.cell_ids[idx],
            'is_ctx_dropout': torch.tensor(1, dtype=torch.float32)  # 标记为OOD
        }


if __name__ == "__main__":
    # 测试数据集
    print("Testing HCA-DR Dataset...")
    
    # 创建测试数据
    n_samples = 1000
    n_genes = 978
    n_compounds = 100
    n_cells = 10
    
    test_data = {
        'X_pert': np.random.randn(n_samples, n_genes).astype(np.float32),
        'X_ctx': np.random.randn(n_samples, n_genes).astype(np.float32),
        'y': np.random.randint(0, n_compounds, n_samples).astype(np.int64),
        'cell_ids': np.random.randint(0, n_cells, n_samples).astype(np.int64),
        'folds': np.random.randint(0, 3, n_samples).astype(np.int32),
        'n_samples': n_samples,
        'n_compounds': n_compounds,
        'n_genes': n_genes,
        'n_cell_lines': n_cells
    }
    
    # 创建数据集
    dataset = HCADRDataset(test_data, fold_ids=[0, 1], mode="train")
    
    # 测试单个样本
    sample = dataset[0]
    print(f"\n✓ Sample shapes:")
    for key, value in sample.items():
        print(f"  {key}: {value.shape if hasattr(value, 'shape') else value}")
    
    # 测试DataLoader
    train_loader, val_loader = create_dataloaders(
        test_data,
        train_folds=[0, 1],
        val_folds=[2],
        batch_size=32,
        num_workers=0
    )
    
    batch = next(iter(train_loader))
    print(f"\n✓ Batch shapes:")
    for key, value in batch.items():
        print(f"  {key}: {value.shape}")
    
    # 测试Context Dropout
    dropout_count = 0
    for i in range(100):
        sample = dataset[i % len(dataset)]
        dropout_count += sample['is_ctx_dropout'].item()
    
    print(f"\n✓ Context Dropout rate: {dropout_count/100:.2f} (expected: ~0.15)")