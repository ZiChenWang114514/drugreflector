"""
测试预处理数据与训练系统的兼容性
"""
import pickle
from pathlib import Path

def test_data_compatibility(data_file):
    """验证数据格式"""
    print("="*80)
    print("🔍 Testing Data Compatibility")
    print("="*80)
    
    # 加载数据
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    # 检查必需字段
    required_fields = ['X', 'y', 'folds', 'compound_names', 'gene_names']
    
    print("\n✅ Required Fields:")
    for field in required_fields:
        if field in data:
            value = data[field]
            if hasattr(value, 'shape'):
                print(f"  {field:20s}: {type(value).__name__:15s} shape={value.shape}")
            elif isinstance(value, (list, tuple)):
                print(f"  {field:20s}: {type(value).__name__:15s} len={len(value)}")
            else:
                print(f"  {field:20s}: {type(value).__name__}")
        else:
            print(f"  {field:20s}: ❌ MISSING")
    
    # 检查可选字段
    optional_fields = ['metadata', 'sample_meta', 'pert_to_idx']
    print("\n📋 Optional Fields:")
    for field in optional_fields:
        if field in data:
            value = data[field]
            print(f"  {field:20s}: ✓ Present ({type(value).__name__})")
        else:
            print(f"  {field:20s}: - Not present")
    
    # 验证数据类型
    print("\n🔍 Data Type Validation:")
    
    # 检查gene_names和compound_names是否是list
    if isinstance(data['gene_names'], list):
        print(f"  gene_names: ✓ list (correct)")
    else:
        print(f"  gene_names: ⚠️  {type(data['gene_names']).__name__} (should be list)")
    
    if isinstance(data['compound_names'], list):
        print(f"  compound_names: ✓ list (correct)")
    else:
        print(f"  compound_names: ⚠️  {type(data['compound_names']).__name__} (should be list)")
    
    # 验证数据维度
    print("\n📊 Data Dimensions:")
    print(f"  Samples: {len(data['X']):,}")
    print(f"  Genes: {data['X'].shape[1]}")
    print(f"  Compounds: {len(data['compound_names']):,}")
    print(f"  Labels: {len(data['y']):,}")
    print(f"  Folds: {len(data['folds']):,}")
    
    # 检查一致性
    print("\n✅ Consistency Checks:")
    checks = [
        (len(data['X']) == len(data['y']), "X and y have same length"),
        (len(data['X']) == len(data['folds']), "X and folds have same length"),
        (data['X'].shape[1] == len(data['gene_names']), "X columns match gene_names"),
        (max(data['y']) == len(data['compound_names']) - 1, "y labels match compound count"),
    ]
    
    for check, desc in checks:
        status = "✓" if check else "❌"
        print(f"  {status} {desc}")
    
    # Fold分布
    import numpy as np
    print("\n📈 Fold Distribution:")
    for fold_id in range(3):
        count = (data['folds'] == fold_id).sum()
        pct = count / len(data['folds']) * 100
        print(f"  Fold {fold_id}: {count:,} samples ({pct:.1f}%)")
    
    print("\n" + "="*80)
    print("✅ Compatibility Test Complete!")
    print("="*80)

if __name__ == "__main__":
    data_file = "E:/科研/Models/drugreflector/processed_data/training_data_lincs2020_paper_compliant.pkl"
    test_data_compatibility(data_file)