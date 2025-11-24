"""
PyTorch Dataset for Two-Tower training.
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, Optional

try:
    from chemprop.data import MoleculeDatapoint, MoleculeDataset
    from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
    from rdkit import Chem
    CHEMPROP_AVAILABLE = True
except ImportError:
    CHEMPROP_AVAILABLE = False


class TwoTowerDataset(Dataset):
    """
    Dataset combining transcriptome and chemical structure.
    
    Parameters
    ----------
    X : np.ndarray
        Gene expression matrix (n_samples, n_genes)
    y : np.ndarray
        Compound labels (n_samples,)
    compound_ids : np.ndarray
        Compound IDs for each sample (n_samples,)
    smiles_dict : Dict[str, str]
        Mapping from compound ID to SMILES
    fold_mask : np.ndarray, optional
        Boolean mask for samples to include
    """
    
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        compound_ids: np.ndarray,
        smiles_dict: Dict[str, str],
        fold_mask: Optional[np.ndarray] = None
    ):
        if not CHEMPROP_AVAILABLE:
            raise ImportError("Chemprop is required")
        
        # Apply fold mask
        if fold_mask is not None:
            self.X = X[fold_mask]
            self.y = y[fold_mask]
            self.compound_ids = compound_ids[fold_mask]
        else:
            self.X = X
            self.y = y
            self.compound_ids = compound_ids
        
        self.smiles_dict = smiles_dict
        
        # Create molecule featurizer
        self.mol_featurizer = SimpleMoleculeMolGraphFeaturizer()
        
        print(" Pre-featurizing and validating molecules...")
        self.mol_cache = {}
        self.mol_graph_cache = {}  # ������ 新增：缓存MolGraph
        unique_compounds = np.unique(self.compound_ids)
        
        n_valid = 0
        n_invalid_smiles = 0
        n_invalid_mol = 0
        n_invalid_graph = 0
        
        for comp_id in unique_compounds:
            smiles = smiles_dict.get(comp_id)
            
            # 检查点1: SMILES是否存在且有效
            if not smiles or pd.isna(smiles):
                n_invalid_smiles += 1
                continue
            
            try:
                # 检查点2: 能否转换为Mol对象
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    n_invalid_mol += 1
                    print(f"  ⚠️  Invalid SMILES for {comp_id}: {smiles}")
                    continue
                
                # 检查点3: 能否特征化
                mol_graph = self.mol_featurizer(mol)
                
                # 检查点4: 特征化结果是否有效
                if mol_graph is None:
                    n_invalid_graph += 1
                    print(f"  ⚠️  Featurization returned None for {comp_id}")
                    continue
                
                if not hasattr(mol_graph, 'V') or mol_graph.V is None:
                    n_invalid_graph += 1
                    print(f"  ⚠️  MolGraph.V is None for {comp_id}")
                    continue
                
                if not hasattr(mol_graph, 'E') or mol_graph.E is None:
                    n_invalid_graph += 1
                    print(f"  ⚠️  MolGraph.E is None for {comp_id}")
                    continue
                
                # 验证特征维度
                expected_atom_dim = self.mol_featurizer.atom_fdim
                expected_bond_dim = self.mol_featurizer.bond_fdim
                
                if mol_graph.V.shape[1] != expected_atom_dim:
                    n_invalid_graph += 1
                    print(f"  ⚠️  Wrong atom feature dim for {comp_id}: "
                          f"got {mol_graph.V.shape[1]}, expected {expected_atom_dim}")
                    continue
                
                if mol_graph.E.shape[1] != expected_bond_dim:
                    n_invalid_graph += 1
                    print(f"  ⚠️  Wrong bond feature dim for {comp_id}: "
                          f"got {mol_graph.E.shape[1]}, expected {expected_bond_dim}")
                    continue
                
                # 全部通过，缓存
                self.mol_cache[comp_id] = mol
                self.mol_graph_cache[comp_id] = mol_graph
                n_valid += 1
                
            except Exception as e:
                n_invalid_graph += 1
                print(f"  ⚠️  Error processing {comp_id}: {e}")
                continue
        
        # 报告统计
        print(f"\n Molecule Validation Summary:")
        print(f"    Total unique compounds: {len(unique_compounds)}")
        print(f"    ✓ Valid: {n_valid}")
        print(f"    ✗ Invalid SMILES: {n_invalid_smiles}")
        print(f"    ✗ Invalid Mol objects: {n_invalid_mol}")
        print(f"    ✗ Invalid graphs: {n_invalid_graph}")
        
        if n_valid == 0:
            raise ValueError("No valid molecules found! Check your SMILES data.")
        
        # 测试一个分子的完整流程
        if len(self.mol_graph_cache) > 0:
            test_comp_id = list(self.mol_graph_cache.keys())[0]
            test_graph = self.mol_graph_cache[test_comp_id]
            
            print(f"\n Test Molecule ({test_comp_id}):")
            print(f"    Atom features: {test_graph.V.shape}")
            print(f"    Bond features: {test_graph.E.shape}")
            print(f"    Edge index: {test_graph.edge_index.shape}")
            
            # ������ 测试能否创建BatchMolGraph
            try:
                from chemprop.data import BatchMolGraph
                test_batch = BatchMolGraph([test_graph])
                
                if test_batch.V is None:
                    raise ValueError("BatchMolGraph.V is None after creation!")
                
                print(f"    ✓ BatchMolGraph creation: SUCCESS")
                print(f"    BatchMolGraph.V: {test_batch.V.shape}")
                
            except Exception as e:
                raise RuntimeError(f"BatchMolGraph test failed: {e}")
        
        # Create valid sample mask
        self.valid_mask = np.array([
            comp_id in self.mol_cache 
            for comp_id in self.compound_ids
        ])
        
        n_valid_samples = self.valid_mask.sum()
        n_invalid_samples = len(self.valid_mask) - n_valid_samples
        
        print(f"\n Sample Validation:")
        print(f"    Total samples: {len(self.valid_mask)}")
        print(f"    ✓ Valid samples: {n_valid_samples}")
        print(f"    ✗ Invalid samples: {n_invalid_samples}")
        
        if n_valid_samples == 0:
            raise ValueError("No valid samples found!")
    
    def __len__(self):
        return self.valid_mask.sum()
    
    def __getitem__(self, idx):
        """
        Get item by index (only valid samples).
        
        Returns
        -------
        tuple
            (x_transcript, mol_graph, y_label)  # 直接返回MolGraph
        """
        # Map to valid index
        valid_indices = np.where(self.valid_mask)[0]
        actual_idx = valid_indices[idx]
        
        # Get transcriptome data
        x_transcript = torch.FloatTensor(self.X[actual_idx])
        y_label = torch.LongTensor([self.y[actual_idx]])[0]
        
        # 直接从缓存获取MolGraph (不是MoleculeDatapoint)
        comp_id = self.compound_ids[actual_idx]
        mol_graph = self.mol_graph_cache[comp_id]
        
        # 最后一次验证
        if mol_graph.V is None:
            raise RuntimeError(f"MolGraph.V is None for compound {comp_id} at idx {idx}")
        
        return x_transcript, mol_graph, y_label


def collate_two_tower(batch):
    """
    Custom collate function for batching.
    
    Parameters
    ----------
    batch : list
        List of (x_transcript, mol_graph, y_label) tuples
    
    Returns
    -------
    tuple
        (x_transcript_batch, bmg_batch, y_batch)
    """
    from chemprop.data import BatchMolGraph
    
    # Separate components
    x_transcripts = []
    mol_graphs = []
    y_labels = []
    
    for x_t, mol_graph, y in batch:
        # 在collate时再次验证
        if mol_graph is None:
            raise ValueError("Received None mol_graph in collate_fn")
        if not hasattr(mol_graph, 'V') or mol_graph.V is None:
            raise ValueError("mol_graph.V is None in collate_fn")
        
        x_transcripts.append(x_t)
        mol_graphs.append(mol_graph)
        y_labels.append(y)
    
    # Stack transcriptome data
    x_batch = torch.stack(x_transcripts)
    y_batch = torch.stack(y_labels)
    
    # 创建 BatchMolGraph (输入已经是MolGraph列表)
    try:
        bmg_batch = BatchMolGraph(mol_graphs)
        
        # 立即验证
        if bmg_batch.V is None:
            # 调试信息
            print(f"❌ BatchMolGraph.V is None!")
            print(f"   Number of graphs: {len(mol_graphs)}")
            for i, g in enumerate(mol_graphs):
                print(f"   Graph {i}: V={g.V.shape if g.V is not None else None}, "
                      f"E={g.E.shape if g.E is not None else None}")
            raise ValueError("BatchMolGraph.V is None after creation")
        
        if bmg_batch.E is None:
            raise ValueError("BatchMolGraph.E is None after creation")
        
        # 打印调试信息
        print(f"✓ Created BatchMolGraph: V={bmg_batch.V.shape}, E={bmg_batch.E.shape}")
        
    except Exception as e:
        print(f"❌ Error creating BatchMolGraph: {e}")
        print(f"   Batch size: {len(mol_graphs)}")
        for i, g in enumerate(mol_graphs):
            print(f"   Graph {i} valid: V={g.V is not None}, E={g.E is not None}")
        raise
    
    return x_batch, bmg_batch, y_batch